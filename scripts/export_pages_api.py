"""Export API snapshots + report files for the GitHub Pages static app.

Covers the control combinations the UI can request for national and province
views, plus district defaults, so every main field has data on Pages.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "app" / "static-api"
BASE = "http://127.0.0.1:8000"
METRICS = ("hfai", "count", "frp")
DIRS = ("above", "below", "both")
SEVS = ("moderate", "high", "extreme")


def key(path: str, params: dict | None = None, ext: str = ".json") -> str:
    params = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
    items = sorted((str(k), str(v)) for k, v in params.items())
    query = "&".join(f"{k}={v}" for k, v in items)
    raw = path.lstrip("/") + (f"__{query}" if query else "")
    return re.sub(r"[^\w.=\/-]+", "_", raw) + ext


def fetch(path: str, params: dict | None = None) -> bytes:
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None and v != ""})
    url = f"{BASE}{path}" + (f"?{query}" if query else "")
    with urllib.request.urlopen(url, timeout=300) as res:
        return res.read()


def save(path: str, params: dict | None = None, ext: str = ".json") -> Path | None:
    try:
        raw = fetch(path, params)
    except urllib.error.HTTPError as exc:
        print(f"  SKIP {path} {params} -> HTTP {exc.code}")
        return None
    dest = OUT / key(path, params, ext)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".json":
        data = json.loads(raw.decode("utf-8"))
        dest.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    else:
        dest.write_bytes(raw)
    print(f"  {dest.relative_to(OUT)}  ({dest.stat().st_size // 1024} KB)")
    return dest


def export_area_core(area_id: str, years: list[int], viirs_year: int, test: list[int], full_metrics: bool) -> None:
    metrics = METRICS if full_metrics else ("hfai",)
    save("/api/areas/resolve", {"area_id": area_id})

    for metric in metrics:
        save("/api/calendar", {"area_id": area_id, "granularity": "year", "metric": metric})
        save("/api/calendar", {"area_id": area_id, "granularity": "month", "metric": metric})
        save("/api/calendar", {"area_id": area_id, "granularity": "season", "metric": metric})
        save("/api/trends", {"area_id": area_id, "metric": metric})
        for basis in ("area", "month"):
            save("/api/critical", {
                "area_id": area_id, "metric": metric, "percentile": 90,
                "min_days": 3, "max_gap": 1, "basis": basis,
            })
        # Engine spans
        save("/api/harmonization", {"area_id": area_id, "metric": metric, "freq": "month"})
        save("/api/harmonization", {"area_id": area_id, "metric": metric, "freq": "year"})
        save("/api/harmonization", {"area_id": area_id, "metric": metric, "freq": "day"})
        save("/api/harmonization", {"area_id": area_id, "metric": metric, "freq": "month", "start": f"{viirs_year}-01-01"})
        save("/api/harmonization", {"area_id": area_id, "metric": metric, "freq": "year", "start": f"{viirs_year}-01-01"})
        save("/api/harmonization", {"area_id": area_id, "metric": metric, "freq": "day", "start": f"{viirs_year}-01-01"})
        save("/api/harmonization", {
            "area_id": area_id, "metric": metric, "freq": "month",
            "start": f"{test[0]}-01-01", "end": f"{test[1]}-12-31",
        })

    for year in years:
        save("/api/dashboard", {"area_id": area_id, "year": year})
        for metric in metrics:
            save("/api/calendar", {"area_id": area_id, "granularity": "day", "metric": metric, "year": year})
            save("/api/harmonization", {
                "area_id": area_id, "metric": metric, "freq": "month",
                "start": f"{year}-01-01", "end": f"{year}-12-31",
            })
        if year >= viirs_year:
            save("/api/anomalies/series", {"area_id": area_id, "year": year})

    for direction in DIRS:
        for severity in SEVS:
            save("/api/anomalies", {
                "area_id": area_id, "direction": direction, "min_severity": severity, "limit": 60,
            })
        # year-scoped lists for recent years
        for year in years[-5:]:
            if year < viirs_year:
                continue
            save("/api/anomalies", {
                "area_id": area_id, "start": f"{year}-01-01", "end": f"{year}-12-31",
                "direction": direction, "min_severity": "moderate", "limit": 60,
            })
            save("/api/anomalies/overview", {
                "level": "ADM1", "start": f"{year}-01-01", "end": f"{year}-12-31", "direction": direction,
            })


def export_map(iso3: str, last_year: int, peak_month: str, months: list[str]) -> None:
    month = f"{months.index(peak_month) + 1:02d}"
    end = f"{last_year}-{month}-07"
    for days in (1, 7, 30):
        # MapPage setSpan(days) keeps end and moves start back days-1
        from datetime import date, timedelta
        end_d = date.fromisoformat(end)
        start_d = end_d - timedelta(days=days - 1)
        start = start_d.isoformat()
        save("/api/fires", {
            "area_id": iso3, "start": start, "end": end,
            "sensors": "modis,viirs_snpp", "min_confidence": "low", "limit": 25000,
        })
        for metric in METRICS:
            mins = {"hfai": 0.1, "count": 0.2, "frp": 5}[metric]
            for layer in ("harmonized", "modis", "viirs"):
                save("/api/harmonization/grid", {
                    "area_id": iso3, "start": start, "end": end,
                    "layer": layer, "metric": metric,
                    "min_value": mins if layer == "harmonized" else 0,
                })


def export_reports(iso3: str, last_year: int, last_day: str) -> None:
    start = f"{last_year}-01-01"
    end = last_day
    for kind, ext, extra in (
        ("pdf", ".pdf", {}),
        ("series", ".csv", {"freq": "day"}),
        ("series", ".csv", {"freq": "month"}),
        ("series", ".csv", {"freq": "year"}),
        ("anomalies", ".csv", {}),
        ("grid", ".geojson", {}),
    ):
        params = {"area_id": iso3, "start": start, "end": end, "kind": kind, **extra}
        save("/api/reports", params, ext=ext)


def main() -> None:
    if OUT.exists():
        for old in OUT.rglob("*"):
            if old.is_file():
                old.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = json.loads(fetch("/api/meta").decode("utf-8"))
    first_year = int(str(meta["record"]["first_day"])[:4])
    last_year = int(str(meta["record"]["last_day"])[:4])
    years = list(range(first_year, last_year + 1))
    iso3 = meta["iso3"]
    viirs_year = int(str(meta["record"]["viirs_start"])[:4])
    test = meta["split"]["test"]
    last_day = meta["record"]["last_day"]
    trends = json.loads(fetch("/api/trends", {"area_id": iso3, "metric": "hfai"}).decode("utf-8"))
    months = trends["climatology"]["months"]
    peak = trends["climatology"]["peak_month"]
    print(f"Exporting static API for {iso3}, years {first_year}-{last_year}")

    save("/api/meta")
    save("/api/health")
    save("/api/areas")
    save("/api/areas/geojson", {"level": "ADM0"})
    save("/api/areas/geojson", {"level": "ADM1"})
    save("/api/model/metrics")
    save("/api/critical/regions", {"level": "ADM1"})
    save("/api/critical/regions", {"level": "ADM2"})

    areas = json.loads(fetch("/api/areas").decode("utf-8"))["areas"]
    provinces = [a for a in areas if a["level"] == "ADM1"]
    districts = [a for a in areas if a["level"] == "ADM2"]

    # National: full metric coverage
    export_area_core(iso3, years, viirs_year, test, full_metrics=True)
    export_map(iso3, last_year, peak, months)
    export_reports(iso3, last_year, last_day)

    # Provinces: full metric coverage + district polygons
    for prov in provinces:
        aid = prov["area_id"]
        save("/api/areas/geojson", {"level": "ADM2", "parent_id": aid})
        export_area_core(aid, years, viirs_year, test, full_metrics=True)

    # Districts: ensure picker/selection fields work (hfai-focused, all years for dashboard)
    for dist in districts:
        aid = dist["area_id"]
        save("/api/areas/resolve", {"area_id": aid})
        save("/api/calendar", {"area_id": aid, "granularity": "year", "metric": "hfai"})
        save("/api/calendar", {"area_id": aid, "granularity": "month", "metric": "hfai"})
        save("/api/calendar", {"area_id": aid, "granularity": "season", "metric": "hfai"})
        save("/api/trends", {"area_id": aid, "metric": "hfai"})
        save("/api/critical", {
            "area_id": aid, "metric": "hfai", "percentile": 90,
            "min_days": 3, "max_gap": 1, "basis": "area",
        })
        save("/api/anomalies", {
            "area_id": aid, "direction": "above", "min_severity": "moderate", "limit": 60,
        })
        for year in years:
            save("/api/dashboard", {"area_id": aid, "year": year})
            save("/api/calendar", {"area_id": aid, "granularity": "day", "metric": "hfai", "year": year})
            save("/api/harmonization", {
                "area_id": aid, "metric": "hfai", "freq": "month",
                "start": f"{year}-01-01", "end": f"{year}-12-31",
            })
        if last_year >= viirs_year:
            save("/api/anomalies/series", {"area_id": aid, "year": last_year})

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"Done -> {OUT} ({total / (1024 * 1024):.1f} MB, {n} files)")


if __name__ == "__main__":
    main()
