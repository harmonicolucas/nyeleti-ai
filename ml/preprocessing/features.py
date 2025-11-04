"""MODIS-VIIRS pairs on the common grid.

The unit of analysis is one grid cell on one day (UTC). A MODIS point is never
matched to a VIIRS point: the same fire yields different numbers of detections
in each sensor, so both are aggregated to the cell and compared there.

A cell-day is "MODIS-informed" when MODIS detected at least one fire in the
cell or in one of its eight neighbours, on that day, the day before or the day
after. Informed cell-days are the rows the harmonization models work on, and
include many rows where the cell itself has no MODIS detection. Everything
else is handled by the background rate (see background.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml import config
from ml.preprocessing import grid

_KEY_SHIFT = 1 << 34  # cell ids stay below 2**34

MODIS_CELL_COLUMNS = [
    "modis_fire_count",
    "modis_frp_sum",
    "modis_frp_mean",
    "modis_frp_max",
    "modis_confidence_mean",
    "modis_brightness_mean",
    "modis_scan_mean",
    "modis_day_count",
    "modis_night_count",
    "modis_terra_count",
    "modis_aqua_count",
]
CONTEXT_COLUMNS = [
    "nbr_fire_count",
    "nbr_frp_sum",
    "prev_fire_count",
    "next_fire_count",
    "nbr_prev_fire_count",
    "nbr_next_fire_count",
]
STATIC_COLUMNS = ["month", "season", "cell_lat", "cell_lon"]
FEATURES = MODIS_CELL_COLUMNS + CONTEXT_COLUMNS + STATIC_COLUMNS

# Features a simple calibration can use (no missing values)
LINEAR_FEATURES = {
    "count": ["modis_fire_count", "nbr_fire_count", "prev_fire_count", "next_fire_count",
              "nbr_prev_fire_count", "nbr_next_fire_count"],
    "frp": ["modis_frp_sum", "nbr_frp_sum", "modis_fire_count", "nbr_fire_count"],
}
TARGETS = {"count": "viirs_fire_count", "frp": "viirs_frp_sum"}
RAW_MODIS = {"count": "modis_fire_count", "frp": "modis_frp_sum"}


def grid_dir(cell_km: float | None = None):
    return config.GRID_DIR / config.grid_tag(cell_km)


def load_cells(cell_km: float | None = None) -> pd.DataFrame:
    return pd.read_parquet(grid_dir(cell_km) / "grid_cells.parquet")


def load_sensor(sensor: str, years, cell_km: float | None = None) -> pd.DataFrame:
    frames = []
    for year in years:
        path = grid_dir(cell_km) / f"{sensor}_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def day_number(dates) -> np.ndarray:
    return pd.to_datetime(dates).values.astype("datetime64[D]").astype("int64")


def make_key(day, cell_id) -> np.ndarray:
    return np.asarray(day, dtype="int64") * _KEY_SHIFT + np.asarray(cell_id, dtype="int64")


def split_key(key) -> tuple[np.ndarray, np.ndarray]:
    key = np.asarray(key, dtype="int64")
    return key // _KEY_SHIFT, key % _KEY_SHIFT


def _sum_by_key(keys: np.ndarray, *values: np.ndarray):
    uniq, inverse = np.unique(keys, return_inverse=True)
    return uniq, [np.bincount(inverse, weights=v, minlength=len(uniq)) for v in values]


def season_of(month) -> np.ndarray:
    """Meteorological season: 0 DJF, 1 MAM, 2 JJA, 3 SON."""
    return (np.asarray(month) % 12) // 3


def build_informed(modis: pd.DataFrame, cells: pd.DataFrame, first_day: int, last_day: int) -> pd.DataFrame:
    """Feature rows for every MODIS-informed cell-day between two day numbers.

    `modis` must also contain the day before `first_day` and the day after
    `last_day` so that the +/- 1 day context is complete at the edges.
    """
    day = day_number(modis["date"])
    keep = (day >= first_day - 1) & (day <= last_day + 1)
    modis = modis.loc[keep].reset_index(drop=True)
    day = day[keep]
    ix, iy = grid.split_id(modis["cell_id"].values)
    count = modis["modis_fire_count"].values.astype("float64")
    frp = modis["modis_frp_sum"].values.astype("float64")

    # 3 x 3 block sums on the same day
    keys, c, f = [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            keys.append(make_key(day, grid.cell_id(ix + dx, iy + dy)))
            c.append(count)
            f.append(frp)
    block_key, (block_count, block_frp) = _sum_by_key(np.concatenate(keys), np.concatenate(c), np.concatenate(f))
    block = pd.DataFrame({"blk": block_count, "blk_frp": block_frp}, index=block_key)

    one_day = _KEY_SHIFT
    own_key = make_key(day, modis["cell_id"].values)
    own_count = pd.Series(count, index=own_key)

    informed = np.unique(np.concatenate([block_key, block_key + one_day, block_key - one_day]))
    informed_day, informed_cell = split_key(informed)
    inside = (informed_day >= first_day) & (informed_day <= last_day)
    informed, informed_day, informed_cell = informed[inside], informed_day[inside], informed_cell[inside]

    out = pd.DataFrame({"cell_id": informed_cell, "day": informed_day}, index=informed)
    centre = modis[MODIS_CELL_COLUMNS].set_index(own_key)
    out = out.join(centre)
    for col in ("modis_fire_count", "modis_frp_sum", "modis_day_count", "modis_night_count",
                "modis_terra_count", "modis_aqua_count"):
        out[col] = out[col].fillna(0)

    same = block.reindex(informed, fill_value=0.0)
    prev = block["blk"].reindex(informed - one_day, fill_value=0.0).values
    nxt = block["blk"].reindex(informed + one_day, fill_value=0.0).values
    own_prev = own_count.reindex(informed - one_day, fill_value=0.0).values
    own_next = own_count.reindex(informed + one_day, fill_value=0.0).values

    out["nbr_fire_count"] = same["blk"].values - out["modis_fire_count"].values
    out["nbr_frp_sum"] = np.clip(same["blk_frp"].values - out["modis_frp_sum"].values, 0, None)
    out["prev_fire_count"] = own_prev
    out["next_fire_count"] = own_next
    out["nbr_prev_fire_count"] = prev - own_prev
    out["nbr_next_fire_count"] = nxt - own_next

    cell_info = cells.set_index("cell_id")[["lat", "lon"]]
    out = out[out["cell_id"].isin(cell_info.index)]
    out["cell_lat"] = cell_info["lat"].reindex(out["cell_id"]).values
    out["cell_lon"] = cell_info["lon"].reindex(out["cell_id"]).values
    out["date"] = out["day"].values.astype("datetime64[D]").astype("datetime64[ns]")
    out["month"] = out["date"].dt.month
    out["season"] = season_of(out["month"])

    out = out.reset_index(names="key")
    out[FEATURES] = out[FEATURES].astype("float32")
    return out[["key", "date", "cell_id"] + FEATURES]


def attach_sensor(rows: pd.DataFrame, sensor_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add the count and FRP of another sensor for the same cell and day (0 when absent)."""
    if sensor_df.empty:
        rows[f"{prefix}_fire_count"] = np.float32(np.nan)
        rows[f"{prefix}_frp_sum"] = np.float32(np.nan)
        return rows
    key = make_key(day_number(sensor_df["date"]), sensor_df["cell_id"].values)
    lookup = pd.DataFrame(
        {"c": sensor_df["fire_count"].values, "f": sensor_df["frp_sum"].values}, index=key
    ).reindex(rows["key"].values, fill_value=0)
    rows[f"{prefix}_fire_count"] = lookup["c"].values.astype("float32")
    rows[f"{prefix}_frp_sum"] = lookup["f"].values.astype("float32")
    return rows


def features_path(year: int, cell_km: float | None = None):
    return config.FEATURES_DIR / config.grid_tag(cell_km) / f"informed_{year}.parquet"


def load_features(years, cell_km: float | None = None, columns=None) -> pd.DataFrame:
    frames = [pd.read_parquet(features_path(y, cell_km), columns=columns) for y in years
              if features_path(y, cell_km).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
