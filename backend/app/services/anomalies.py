"""Unusual-activity detection for any selection, using the fitted Isolation Forest."""
from __future__ import annotations

import threading
from collections import OrderedDict
from functools import lru_cache

import pandas as pd
from fastapi import HTTPException

from backend.app.services import series
from backend.app.services.analysis import analysis_rows
from backend.app.services.areas import Selection
from ml.anomaly import detector as det

_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
_lock = threading.Lock()


@lru_cache(maxsize=1)
def detector() -> det.AnomalyDetector:
    try:
        return det.AnomalyDetector.load()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="Anomaly model not trained yet (python -m ml.training.train_anomaly)."
        ) from exc


def thresholds(sel: Selection) -> dict:
    return detector().thresholds_for(sel.level, sel.n_cells)


def scored(sel: Selection) -> pd.DataFrame:
    """Every day of the reference period with its anomaly score."""
    with _lock:
        if sel.key in _cache:
            _cache.move_to_end(sel.key)
            return _cache[sel.key]
    hist = analysis_rows(series.daily(sel))
    daily = pd.DataFrame(
        {
            "date": hist["date"].values,
            "hfai": hist["harmonized_hfai"].values,
            "count": hist["harmonized_count"].values,
            "frp": hist["harmonized_frp"].values,
        }
    )
    result = detector().score(det.build_features(daily), sel.level, sel.n_cells)
    with _lock:
        _cache[sel.key] = result
        while len(_cache) > 32:
            _cache.popitem(last=False)
    return result


def events(sel: Selection, start: str | None, end: str | None, min_severity: str, direction: str) -> list[dict]:
    rows = series.window(scored(sel), start, end)
    allowed = det.SEVERITY_ORDER[det.SEVERITY_ORDER.index(min_severity):]
    rows = rows[rows["severity"].isin(allowed)]
    if direction != "both":
        rows = rows[rows["direction"] == direction]
    return det.to_events(rows)
