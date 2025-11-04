"""Build MODIS-VIIRS pairs (feature rows) for every MODIS-informed cell-day.

For each year, writes data/features/<grid>/informed_<year>.parquet with the
MODIS features, the VIIRS S-NPP target (from 2012) and the NOAA-20 values
(from 2020), plus sensor_days.parquet with the daily coverage of each sensor.

Days on which VIIRS S-NPP reports no detection anywhere in the country are not
used as training or evaluation pairs (`pair_valid = False`): the target cannot
be told apart from a gap in the record. When MODIS still sees at least
VIIRS_GAP_MIN_MODIS detections on such a day it is flagged as `viirs_gap`, and
the calendar shows VIIRS as missing rather than as zero.

    python -m scripts.build_pairs
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from ml import config
from ml.preprocessing import features as ft

VIIRS_GAP_MIN_MODIS = 10


def daily_totals(sensor: str, years, cell_km: float, column: str) -> pd.Series:
    df = ft.load_sensor(sensor, years, cell_km)
    if df.empty:
        return pd.Series(dtype="float64")
    return df.groupby("date")[column].sum()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell-km", type=float, default=config.CELL_KM)
    parser.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    args = parser.parse_args(argv)

    first = config.SENSORS["modis"]["first_year"]
    last = config.LAST_YEAR
    if args.years:
        first, last = max(first, args.years[0]), min(last, args.years[1])
    all_years = list(range(config.SENSORS["modis"]["first_year"], config.LAST_YEAR + 1))

    cells = ft.load_cells(args.cell_km)
    out_dir = config.FEATURES_DIR / config.grid_tag(args.cell_km)
    out_dir.mkdir(parents=True, exist_ok=True)

    # daily coverage of each sensor over the whole record
    days = pd.DataFrame(index=pd.date_range(f"{all_years[0]}-01-01", f"{all_years[-1]}-12-31", freq="D"))
    days["modis_count"] = daily_totals("modis", all_years, args.cell_km, "modis_fire_count")
    days["viirs_count"] = daily_totals("viirs_snpp", all_years, args.cell_km, "fire_count")
    days["noaa20_count"] = daily_totals("viirs_noaa20", all_years, args.cell_km, "fire_count")
    days = days.fillna(0)
    viirs_first = config.SENSORS["viirs_snpp"]["first_year"]
    # S-NPP science data starts on 20 January 2012
    viirs_start = days.index[(days["viirs_count"] > 0) & (days.index.year >= viirs_first)].min()
    days["viirs_available"] = days.index >= viirs_start
    days["pair_valid"] = days["viirs_available"] & (days["viirs_count"] > 0)
    # no VIIRS detection in the whole country while MODIS clearly sees fires: a gap in the VIIRS record
    days["viirs_gap"] = days["viirs_available"] & (days["viirs_count"] == 0) & (days["modis_count"] >= VIIRS_GAP_MIN_MODIS)
    days.rename_axis("date").reset_index().to_parquet(out_dir / "sensor_days.parquet", index=False)
    valid_days = set(days.index[days["pair_valid"]])

    report = []
    for year in range(first, last + 1):
        modis = ft.load_sensor("modis", [year - 1, year, year + 1], args.cell_km)
        first_day = int(ft.day_number([f"{year}-01-01"])[0])
        last_day = int(ft.day_number([f"{year}-12-31"])[0])
        rows = ft.build_informed(modis, cells, first_day, last_day)

        viirs = ft.load_sensor("viirs_snpp", [year], args.cell_km)
        noaa = ft.load_sensor("viirs_noaa20", [year], args.cell_km)
        rows = ft.attach_sensor(rows, viirs, "viirs")
        rows = ft.attach_sensor(rows, noaa, "noaa20")
        rows["pair_valid"] = rows["date"].isin(valid_days)
        rows.to_parquet(ft.features_path(year, args.cell_km), index=False)

        entry = {
            "year": year,
            "informed_cell_days": int(len(rows)),
            "with_modis_in_cell": int((rows["modis_fire_count"] > 0).sum()),
            "pair_valid_rows": int(rows["pair_valid"].sum()),
        }
        if not viirs.empty:
            total = float(viirs["fire_count"].sum())
            inside = float(rows["viirs_fire_count"].sum())
            entry.update(
                {
                    "with_viirs_in_cell": int((rows["viirs_fire_count"] > 0).sum()),
                    "viirs_detections": int(total),
                    "viirs_detections_in_informed": int(inside),
                    "viirs_share_in_informed": round(inside / total, 4) if total else None,
                    "days_excluded_no_viirs": int(
                        ((days.index.year == year) & days["viirs_available"] & ~days["pair_valid"]).sum()
                    ),
                }
            )
        report.append(entry)
        share = entry.get("viirs_share_in_informed")
        print(
            f"{year}: {len(rows):,} informed cell-days, {entry['with_modis_in_cell']:,} with MODIS in cell"
            + (f", {share:.1%} of VIIRS detections inside informed cell-days" if share is not None else ""),
            flush=True,
        )

    (out_dir / "pairs_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
