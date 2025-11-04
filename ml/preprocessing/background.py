"""Background rate: VIIRS activity where MODIS sees nothing nearby.

VIIRS detects many small fires in cell-days that are not MODIS-informed. No
model fed with MODIS data can place those fires, but leaving them out would
make any MODIS-based estimate systematically lower than VIIRS. They are
accounted for with a climatological rate: the mean VIIRS activity per
non-informed cell-day, by calendar month and by block of BACKGROUND_BLOCK_KM
(25 km) per side, measured on the training years only.

The rate has no year-to-year variability by construction. It corrects the
level of the harmonized series; all interannual signal comes from MODIS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml import config
from ml.preprocessing import features as ft


def load_sensor_days(cell_km: float | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.FEATURES_DIR / config.grid_tag(cell_km) / "sensor_days.parquet")


def fit_background(years, cell_km: float | None = None) -> pd.DataFrame:
    """Rate per block and month: one row per (block_id, month)."""
    cells = ft.load_cells(cell_km)
    block_of = cells.set_index("cell_id")["block_id"]
    cells_per_block = cells.groupby("block_id").size().rename("n_cells")

    days = load_sensor_days(cell_km)
    days = days[days["date"].dt.year.isin(years) & days["pair_valid"]]
    valid_dates = set(days["date"])
    days_per_month = days.groupby(days["date"].dt.month).size()

    informed = ft.load_features(years, cell_km, columns=["key", "date", "cell_id", "pair_valid"])
    informed = informed[informed["pair_valid"]]
    informed_keys = informed["key"].values
    n_informed = (
        pd.DataFrame({"block_id": block_of.reindex(informed["cell_id"]).values, "month": informed["date"].dt.month.values})
        .groupby(["block_id", "month"])
        .size()
        .rename("n_informed")
    )

    viirs = ft.load_sensor(config.REFERENCE_SENSOR, years, cell_km)
    viirs = viirs[viirs["date"].isin(valid_dates) & viirs["cell_id"].isin(block_of.index)]
    key = ft.make_key(ft.day_number(viirs["date"]), viirs["cell_id"].values)
    residual = viirs[~np.isin(key, informed_keys)]
    residual = (
        pd.DataFrame(
            {
                "block_id": block_of.reindex(residual["cell_id"]).values,
                "month": residual["date"].dt.month.values,
                "count": residual["fire_count"].values,
                "frp": residual["frp_sum"].values,
            }
        )
        .groupby(["block_id", "month"])[["count", "frp"]]
        .sum()
    )

    index = pd.MultiIndex.from_product([cells_per_block.index, range(1, 13)], names=["block_id", "month"])
    bg = pd.DataFrame(index=index).join(residual).join(n_informed).fillna(0.0).reset_index()
    bg["n_cell_days"] = bg["block_id"].map(cells_per_block) * bg["month"].map(days_per_month).fillna(0)
    bg["n_non_informed"] = bg["n_cell_days"] - bg["n_informed"]
    denom = bg["n_non_informed"].where(bg["n_non_informed"] > 0)
    bg["bg_count"] = (bg["count"] / denom).fillna(0.0)
    bg["bg_frp"] = (bg["frp"] / denom).fillna(0.0)
    return bg[["block_id", "month", "bg_count", "bg_frp", "n_non_informed", "n_informed", "count", "frp"]].rename(
        columns={"count": "residual_count", "frp": "residual_frp"}
    )


def cell_background(bg: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Expand block rates to one row per (cell_id, month)."""
    out = cells[["cell_id", "block_id"]].merge(bg[["block_id", "month", "bg_count", "bg_frp"]], on="block_id")
    out["month"] = out["month"].astype("int8")
    out[["bg_count", "bg_frp"]] = out[["bg_count", "bg_frp"]].astype("float32")
    return out[["cell_id", "month", "bg_count", "bg_frp"]]


def attach_background(rows: pd.DataFrame, bg: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Add the background rate each informed row would receive if it were not informed."""
    block = cells.set_index("cell_id")["block_id"].reindex(rows["cell_id"]).values
    lookup = bg.set_index(["block_id", "month"])[["bg_count", "bg_frp"]]
    idx = pd.MultiIndex.from_arrays([block, rows["date"].dt.month.values])
    values = lookup.reindex(idx).fillna(0.0)
    rows["bg_count"] = values["bg_count"].values.astype("float32")
    rows["bg_frp"] = values["bg_frp"].values.astype("float32")
    return rows
