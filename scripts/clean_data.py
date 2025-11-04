"""Validate, clean and de-duplicate the raw FIRMS files.

Reads every file listed in raw/manifest.json, writes one Parquet file per
sensor and year with a common schema, and produces a data quality report
(valid, rejected, duplicated and missing records per file).

Original values are never altered: records are either kept as they are or
rejected with a documented reason. Low-confidence and non-vegetation
detections are kept in the cleaned files but flagged so that they are left
out of the gridded analysis.

    python -m scripts.clean_data
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml import config

REQUIRED = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "daynight"]
SATELLITES = {
    "Terra": "Terra",
    "Aqua": "Aqua",
    "N": "S-NPP",
    "N20": "NOAA-20",
    "1": "NOAA-20",
    "N21": "NOAA-21",
    "2": "NOAA-21",
}
EXPECTED_SATELLITES = {
    "modis": {"Terra", "Aqua"},
    "viirs_snpp": {"S-NPP"},
    "viirs_noaa20": {"NOAA-20"},
}
VIIRS_CONFIDENCE = {"l": "low", "n": "nominal", "h": "high"}
DEDUP_KEY = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
BBOX_MARGIN_DEG = 0.5


def country_bbox() -> tuple[float, float, float, float] | None:
    path = config.BOUNDARIES_DIR / f"{config.COUNTRY_ISO3}_ADM0.geojson"
    if not path.exists():
        return None
    from shapely.geometry import shape

    geojson = json.loads(path.read_text(encoding="utf-8"))
    bounds = np.array([shape(f["geometry"]).bounds for f in geojson["features"]])
    return (
        bounds[:, 0].min() - BBOX_MARGIN_DEG,
        bounds[:, 1].min() - BBOX_MARGIN_DEG,
        bounds[:, 2].max() + BBOX_MARGIN_DEG,
        bounds[:, 3].max() + BBOX_MARGIN_DEG,
    )


def clean_file(path: Path, sensor: str, year: int, bbox) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    report = {"sensor": sensor, "year": year, "raw_records": int(len(raw)), "rejected": {}}

    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"{path.name}: missing columns {missing_cols}")

    is_modis = sensor == "modis"
    bright_col, bright_ir_col = ("brightness", "bright_t31") if is_modis else ("bright_ti4", "bright_ti5")

    df = pd.DataFrame(
        {
            "latitude": pd.to_numeric(raw["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["longitude"], errors="coerce"),
            "acq_date": pd.to_datetime(raw["acq_date"], format="%Y-%m-%d", errors="coerce"),
            "acq_time": pd.to_numeric(raw["acq_time"], errors="coerce"),
            "satellite": raw["satellite"].str.strip().map(SATELLITES),
            "confidence": raw["confidence"].str.strip(),
            "frp": pd.to_numeric(raw["frp"], errors="coerce"),
            "brightness": pd.to_numeric(raw.get(bright_col), errors="coerce"),
            "brightness_ir": pd.to_numeric(raw.get(bright_ir_col), errors="coerce"),
            "scan": pd.to_numeric(raw.get("scan"), errors="coerce"),
            "track": pd.to_numeric(raw.get("track"), errors="coerce"),
            "daynight": raw["daynight"].str.strip().str.upper(),
            "version": raw["version"].str.strip() if "version" in raw else "",
            # near-real-time files carry no type column; treat as vegetation fire
            "type": pd.to_numeric(raw["type"], errors="coerce") if "type" in raw else 0,
        }
    )

    if is_modis:
        conf_pct = pd.to_numeric(df["confidence"], errors="coerce")
        conf_ok = conf_pct.between(0, 100)
        level = pd.cut(conf_pct, [-1, 29, 79, 100], labels=["low", "nominal", "high"]).astype(object)
    else:
        conf_pct = pd.Series(np.nan, index=df.index)
        level = df["confidence"].str.lower().map(VIIRS_CONFIDENCE)
        conf_ok = level.notna()

    hh, mm = df["acq_time"] // 100, df["acq_time"] % 100
    checks = {
        "missing_or_invalid_coordinates": df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180),
        "invalid_date": df["acq_date"].notna() & (df["acq_date"].dt.year == year),
        "invalid_time": df["acq_time"].notna() & hh.between(0, 23) & mm.between(0, 59),
        "unexpected_satellite": df["satellite"].isin(EXPECTED_SATELLITES[sensor]),
        "invalid_confidence": conf_ok,
        "invalid_frp": df["frp"].notna() & (df["frp"] >= 0),
        "invalid_brightness": df["brightness"].isna() | (df["brightness"] > 0),
        "invalid_daynight": df["daynight"].isin(["D", "N"]),
        "invalid_type": df["type"].isin([0, 1, 2, 3]),
    }
    if bbox is not None:
        west, south, east, north = bbox
        checks["outside_study_area"] = df["longitude"].between(west, east) & df["latitude"].between(south, north)

    keep = pd.Series(True, index=df.index)
    for reason, ok in checks.items():
        # count each record once, under the first rule it breaks
        failed = keep & ~ok
        if failed.any():
            report["rejected"][reason] = int(failed.sum())
        keep &= ok

    df["confidence_pct"] = conf_pct
    df["confidence_level"] = level
    df = df[keep]

    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEY, keep="first")
    report["duplicates_removed"] = int(before - len(df))

    low_conf = (
        df["confidence_pct"] < config.MODIS_MIN_CONFIDENCE
        if is_modis
        else df["confidence"].str.lower().isin(config.VIIRS_EXCLUDED_CONFIDENCE)
    )
    vegetation = df["type"].isin(config.ANALYSIS_TYPES)
    df["analysis"] = vegetation & ~low_conf

    report["valid_records"] = int(len(df))
    report["flagged_low_confidence"] = int((vegetation & low_conf).sum())
    report["flagged_non_vegetation_source"] = int((~vegetation).sum())
    report["analysis_records"] = int(df["analysis"].sum())
    report["missing_frp_in_raw"] = int((raw["frp"].str.strip() == "").sum())

    days_in_year = pd.date_range(date(year, 1, 1), date(year, 12, 31), freq="D")
    observed_days = pd.DatetimeIndex(df["acq_date"].unique())
    missing_days = days_in_year.difference(observed_days)
    report["days_without_detections"] = int(len(missing_days))
    report["days_without_detections_list"] = [d.strftime("%Y-%m-%d") for d in missing_days]
    report["date_min"] = df["acq_date"].min().strftime("%Y-%m-%d") if len(df) else None
    report["date_max"] = df["acq_date"].max().strftime("%Y-%m-%d") if len(df) else None
    report["versions"] = sorted(df["version"].unique().tolist())
    report["by_satellite"] = {k: int(v) for k, v in df["satellite"].value_counts().items()}

    out = df.assign(
        acq_date=df["acq_date"].dt.date,
        acq_time=df["acq_time"].astype("int16"),
        frp=df["frp"].astype("float32"),
        brightness=df["brightness"].astype("float32"),
        brightness_ir=df["brightness_ir"].astype("float32"),
        scan=df["scan"].astype("float32"),
        track=df["track"].astype("float32"),
        confidence_pct=df["confidence_pct"].astype("float32"),
        type=df["type"].astype("int8"),
        sensor=sensor,
        instrument=config.SENSORS[sensor]["instrument"],
    ).sort_values(["acq_date", "acq_time"], kind="stable")
    return out.reset_index(drop=True), report


def write_markdown(report: dict, manifest: dict) -> None:
    lines = [
        "# Data quality report",
        "",
        f"Generated by `scripts/clean_data.py` on {report['generated_at']}. Do not edit by hand.",
        "",
        f"Study area: {config.COUNTRY_NAME} ({config.COUNTRY_ISO3}). "
        "Source: NASA FIRMS country yearly summaries (standard processing).",
        "",
        "## Totals by sensor",
        "",
        "| Sensor | Product | Years | Files | Raw records | Rejected | Duplicates | Valid | Used in analysis |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sensor, tot in report["totals"].items():
        lines.append(
            f"| {sensor} | {config.SENSORS[sensor]['product']} | {tot['first_year']}-{tot['last_year']} | "
            f"{tot['files']} | {tot['raw_records']:,} | {tot['rejected']:,} | {tot['duplicates_removed']:,} | "
            f"{tot['valid_records']:,} | {tot['analysis_records']:,} |"
        )
    lines += [
        "",
        "Rejected: records failing a validation rule (coordinates, date, time, satellite, confidence, FRP, "
        "brightness, day/night flag, type, study-area bounding box). "
        "Used in analysis: valid records that are presumed vegetation fires (`type = 0`) and not low confidence "
        f"(MODIS confidence >= {config.MODIS_MIN_CONFIDENCE}; VIIRS confidence nominal or high).",
        "",
        "## Exclusions",
        "",
        "| Sensor | Valid | Low confidence | Non-vegetation source (type 1-3) | Share excluded from analysis |",
        "|---|---|---|---|---|",
    ]
    for sensor, tot in report["totals"].items():
        share = 1 - tot["analysis_records"] / tot["valid_records"] if tot["valid_records"] else 0
        lines.append(
            f"| {sensor} | {tot['valid_records']:,} | {tot['flagged_low_confidence']:,} | "
            f"{tot['flagged_non_vegetation_source']:,} | {share:.1%} |"
        )
    lines += ["", "## Rejections by rule", "", "| Sensor | Rule | Records |", "|---|---|---|"]
    any_rejection = False
    for sensor, tot in report["totals"].items():
        for reason, n in sorted(tot["rejected_by_rule"].items()):
            any_rejection = True
            lines.append(f"| {sensor} | {reason} | {n:,} |")
    if not any_rejection:
        lines.append("| - | no record failed validation | 0 |")

    lines += [
        "",
        "## Files",
        "",
        "| Sensor | Year | Raw | Valid | Analysis | Duplicates | Days without detections | Version(s) | Period | SHA-256 (first 12) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in report["files"]:
        entry = manifest["files"].get(f"{item['sensor']}/{item['year']}", {})
        lines.append(
            f"| {item['sensor']} | {item['year']} | {item['raw_records']:,} | {item['valid_records']:,} | "
            f"{item['analysis_records']:,} | {item['duplicates_removed']:,} | {item['days_without_detections']} | "
            f"{', '.join(item['versions'])} | {item['date_min']} to {item['date_max']} | "
            f"`{entry.get('sha256', '')[:12]}` |"
        )
    lines += [
        "",
        "Days without detections are days of the year with no valid record for that sensor over the whole "
        "country. In the wet season this is usually genuine absence of fire or persistent cloud; it can also "
        "be a sensor outage. These days are listed in `quality_report.json` and handled explicitly when "
        "MODIS-VIIRS pairs are built (see `docs/methodology.md`).",
        "",
    ]
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DIR / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sensors", nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="clean again files that were already processed")
    args = parser.parse_args(argv)

    manifest_path = config.RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No manifest found. Run scripts.download_firms first.")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bbox = country_bbox()

    # files cleaned in a previous run are reused unless their source changed
    report_path = config.QUALITY_DIR / "quality_report.json"
    previous = {}
    if report_path.exists():
        for item in json.loads(report_path.read_text(encoding="utf-8"))["files"]:
            previous[f"{item['sensor']}/{item['year']}"] = item

    reports = []
    for key in sorted(manifest["files"]):
        entry = manifest["files"][key]
        sensor, year = entry["sensor"], int(entry["year"])
        out = config.CLEAN_DIR / sensor / f"{year}.parquet"
        done = previous.get(key)
        unchanged = done is not None and done.get("source_sha256") == entry["sha256"] and out.exists()
        if unchanged and (not args.force or (args.sensors and sensor not in args.sensors)):
            reports.append(done)
            continue
        if args.sensors and sensor not in args.sensors:
            continue
        df, rep = clean_file(config.RAW_DIR / entry["file"], sensor, year, bbox)
        rep["source_sha256"] = entry["sha256"]
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        reports.append(rep)
        rejected = sum(rep["rejected"].values())
        print(
            f"{sensor} {year}: {rep['raw_records']:,} raw, {rep['valid_records']:,} valid, "
            f"{rejected:,} rejected, {rep['duplicates_removed']:,} duplicates, "
            f"{rep['analysis_records']:,} for analysis",
            flush=True,
        )

    totals: dict[str, dict] = {}
    for rep in reports:
        tot = totals.setdefault(
            rep["sensor"],
            {
                "files": 0, "raw_records": 0, "rejected": 0, "duplicates_removed": 0, "valid_records": 0,
                "analysis_records": 0, "flagged_low_confidence": 0, "flagged_non_vegetation_source": 0,
                "rejected_by_rule": {}, "first_year": rep["year"], "last_year": rep["year"],
            },
        )
        tot["files"] += 1
        tot["first_year"] = min(tot["first_year"], rep["year"])
        tot["last_year"] = max(tot["last_year"], rep["year"])
        tot["rejected"] += sum(rep["rejected"].values())
        for field in ("raw_records", "duplicates_removed", "valid_records", "analysis_records",
                      "flagged_low_confidence", "flagged_non_vegetation_source"):
            tot[field] += rep[field]
        for reason, n in rep["rejected"].items():
            tot["rejected_by_rule"][reason] = tot["rejected_by_rule"].get(reason, 0) + n

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "country": config.COUNTRY_NAME,
        "rules": {
            "modis_min_confidence": config.MODIS_MIN_CONFIDENCE,
            "viirs_excluded_confidence": list(config.VIIRS_EXCLUDED_CONFIDENCE),
            "analysis_types": list(config.ANALYSIS_TYPES),
            "dedup_key": DEDUP_KEY,
            "bbox": bbox,
        },
        "totals": totals,
        "files": reports,
    }
    config.QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, manifest)
    print(f"Quality report: {config.QUALITY_DIR / 'quality_report.json'} and docs/data_quality.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
