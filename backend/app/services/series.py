"""Daily activity series of a selection: observed MODIS, observed VIIRS, harmonized."""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

from backend.app.database import db
from backend.app.services.areas import Selection
from ml import config

SOURCES = ("modis", "viirs", "harmonized")
METRICS = ("hfai", "count", "frp")

_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_SIZE = 48


def _area_daily(area_id: str) -> pd.DataFrame:
    return db.cursor().execute(
        """
        SELECT date, modis_count, modis_frp, viirs_count, viirs_frp, noaa20_count,
               harmonized_count, harmonized_frp, model_count, background_count
        FROM calendar_activity WHERE area_id = ? ORDER BY date
        """,
        [area_id],
    ).df()


def _aoi_daily(cell_ids: np.ndarray) -> pd.DataFrame:
    meta = db.metadata()
    cur = db.cursor()
    cur.register("aoi_cells", pd.DataFrame({"cell_id": cell_ids}))
    obs = cur.execute(
        """
        SELECT date, sum(modis_count) AS modis_count, sum(modis_frp) AS modis_frp,
               sum(viirs_count) AS viirs_count, sum(viirs_frp) AS viirs_frp, sum(noaa20_count) AS noaa20_count
        FROM observed_activity WHERE cell_id IN (SELECT cell_id FROM aoi_cells) GROUP BY date
        """
    ).df()
    est = cur.execute(
        """
        SELECT date, sum(est_count) AS est_count, sum(est_frp) AS est_frp,
               sum(bg_count) AS bg_count, sum(bg_frp) AS bg_frp
        FROM harmonized_activity WHERE cell_id IN (SELECT cell_id FROM aoi_cells) GROUP BY date
        """
    ).df()
    bg = cur.execute(
        """
        SELECT month, sum(bg_count) AS bg_count, sum(bg_frp) AS bg_frp
        FROM background_activity WHERE cell_id IN (SELECT cell_id FROM aoi_cells) GROUP BY month
        """
    ).df()
    cur.unregister("aoi_cells")

    index = pd.date_range(meta["first_day"], meta["last_day"], freq="D", name="date")
    obs = obs.assign(date=pd.to_datetime(obs["date"])).set_index("date").reindex(index, fill_value=0)
    est = est.assign(date=pd.to_datetime(est["date"])).set_index("date").reindex(index, fill_value=0)
    bg = bg.set_index("month").reindex(range(1, 13), fill_value=0.0)
    bg_count = bg["bg_count"].reindex(index.month).values
    bg_frp = bg["bg_frp"].reindex(index.month).values

    out = pd.DataFrame(index=index)
    out["modis_count"] = obs["modis_count"].astype("float64")
    out["modis_frp"] = obs["modis_frp"].astype("float64")
    out["viirs_count"] = obs["viirs_count"].astype("float64")
    out["viirs_frp"] = obs["viirs_frp"].astype("float64")
    out["noaa20_count"] = obs["noaa20_count"].astype("float64")
    out["harmonized_count"] = est["est_count"].values + bg_count - est["bg_count"].values
    out["harmonized_frp"] = est["est_frp"].values + bg_frp - est["bg_frp"].values
    out["model_count"] = est["est_count"].values
    out["background_count"] = bg_count - est["bg_count"].values

    # no VIIRS value before the record starts or on days flagged as gaps
    flags = cur.execute("SELECT date, viirs_available AND NOT viirs_gap AS ok FROM sensor_days").df()
    viirs_ok = flags.assign(date=pd.to_datetime(flags["date"])).set_index("date")["ok"].reindex(index, fill_value=False)
    out.loc[~viirs_ok.values, ["viirs_count", "viirs_frp"]] = np.nan
    if meta.get("noaa20_start"):
        out.loc[index < pd.Timestamp(meta["noaa20_start"]), "noaa20_count"] = np.nan
    else:
        out["noaa20_count"] = np.nan
    return out.reset_index()


def daily(sel: Selection) -> pd.DataFrame:
    """Full daily record of the selection. Treat the result as read-only."""
    with _cache_lock:
        if sel.key in _cache:
            _cache.move_to_end(sel.key)
            return _cache[sel.key]

    df = _area_daily(sel.area_id) if sel.kind == "area" else _aoi_daily(sel.cell_ids)
    df["date"] = pd.to_datetime(df["date"])
    ref = db.metadata()["hfai"]
    for col in ("modis_count", "modis_frp", "viirs_count", "viirs_frp", "harmonized_count", "harmonized_frp"):
        df[col] = df[col].astype("float64")
    for source in SOURCES:
        df[f"{source}_hfai"] = config.hfai(df[f"{source}_count"], df[f"{source}_frp"], ref)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    with _cache_lock:
        _cache[sel.key] = df
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return df


def window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df


def values(series, digits: int = 3) -> list:
    """JSON-safe list: rounded floats, None for missing values."""
    arr = np.asarray(series, dtype="float64")
    return [None if np.isnan(v) else round(float(v), digits) for v in arr]


def number(value, digits: int = 3):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = float(value)
    return None if np.isnan(value) or np.isinf(value) else round(value, digits)


def split_of(year: int) -> str:
    """Role of a year for the harmonization model."""
    split = db.metadata()["split"]
    for name in ("train", "validation", "test"):
        first, last = split[name]
        if first <= year <= last:
            return name
    return "pre-viirs" if year < split["train"][0] else "other"
