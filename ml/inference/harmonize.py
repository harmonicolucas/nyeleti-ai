"""Apply a trained harmonization model to the MODIS record.

For every MODIS-informed cell-day of every year this writes the estimated
VIIRS-equivalent detection count and FRP. The estimates are stored in their
own files and tables; the MODIS and VIIRS observations are never modified.

    python -m ml.inference.harmonize
    python -m ml.inference.harmonize --years 2001 2011
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ml import config
from ml.preprocessing import background as bgmod
from ml.preprocessing import features as ft


def active_version() -> str:
    registry = json.loads((config.ARTIFACTS_DIR / "registry.json").read_text(encoding="utf-8"))
    return registry["active"]


class Harmonizer:
    def __init__(self, version: str | None = None):
        self.version = version or active_version()
        self.dir: Path = config.ARTIFACTS_DIR / self.version
        self.meta = json.loads((self.dir / "model.json").read_text(encoding="utf-8"))
        self.cell_km = self.meta["cell_km"]
        self.boosters = {
            t: lgb.Booster(model_file=str(self.dir / f"harmonization_{t}.lgb.txt")) for t in ("count", "frp")
        }
        self.background = pd.read_parquet(self.dir / "background.parquet")
        self._baselines = None

    @property
    def baselines(self) -> dict:
        if self._baselines is None:
            self._baselines = joblib.load(self.dir / "baselines.joblib")
        return self._baselines

    def predict(self, rows: pd.DataFrame, target: str) -> np.ndarray:
        return np.clip(self.boosters[target].predict(rows[ft.FEATURES]), 0, None)

    def hfai(self, count, frp):
        """Harmonized Fire Activity Index from a VIIRS-scale count and FRP sum."""
        h = self.meta["hfai"]
        return h["weight_count"] * np.asarray(count) / h["count_ref"] + h["weight_frp"] * np.asarray(frp) / h["frp_ref"]


def harmonized_path(year: int, cell_km: float) -> Path:
    return config.HARMONIZED_DIR / config.grid_tag(cell_km) / f"harmonized_{year}.parquet"


def run(years, version: str | None = None) -> dict:
    model = Harmonizer(version)
    cells = ft.load_cells(model.cell_km)
    out_dir = config.HARMONIZED_DIR / config.grid_tag(model.cell_km)
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = []
    for year in years:
        rows = ft.load_features([year], model.cell_km)
        if rows.empty:
            continue
        out = rows[["date", "cell_id", "modis_fire_count"]].copy()
        out["est_count"] = model.predict(rows, "count").astype("float32")
        out["est_frp"] = model.predict(rows, "frp").astype("float32")
        out = bgmod.attach_background(out, model.background, cells)
        out["cell_id"] = out["cell_id"].astype("int64")
        out.to_parquet(harmonized_path(year, model.cell_km), index=False)
        totals.append({"year": year, "cell_days": int(len(out)), "est_count": float(out["est_count"].sum())})
        print(f"{year}: {len(out):,} cell-days harmonized, estimated count {out['est_count'].sum():,.0f}", flush=True)

    bgmod.cell_background(model.background, cells).to_parquet(out_dir / "background_cells.parquet", index=False)

    run_info = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model.version,
        "first_year": years[0],
        "last_year": years[-1],
        "cell_days": int(sum(t["cell_days"] for t in totals)),
        "years": totals,
    }
    log = model.dir / "runs.json"
    runs = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    runs.append(run_info)
    log.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return run_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=None)
    parser.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    args = parser.parse_args(argv)
    first, last = args.years or (config.SENSORS["modis"]["first_year"], config.LAST_YEAR)
    run(list(range(first, last + 1)), args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
