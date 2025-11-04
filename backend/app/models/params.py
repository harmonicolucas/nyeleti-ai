"""Query parameter types shared by the API routers."""
from __future__ import annotations

from datetime import date
from enum import Enum

from fastapi import HTTPException, Query

from backend.app.database import db


class Metric(str, Enum):
    hfai = "hfai"
    count = "count"
    frp = "frp"


class Granularity(str, Enum):
    day = "day"
    month = "month"
    season = "season"
    year = "year"


class Frequency(str, Enum):
    day = "day"
    month = "month"
    year = "year"


class Layer(str, Enum):
    harmonized = "harmonized"
    modis = "modis"
    viirs = "viirs"


class Severity(str, Enum):
    moderate = "moderate"
    high = "high"
    extreme = "extreme"


class Direction(str, Enum):
    above = "above"
    below = "below"
    both = "both"


METRIC_LABELS = {
    "hfai": "Harmonized Fire Activity Index (HFAI)",
    "count": "Active fire detections",
    "frp": "Fire radiative power (MW)",
}
SOURCE_LABELS = {
    "modis": "MODIS original (observed, 1 km)",
    "viirs": "VIIRS S-NPP original (observed, 375 m)",
    "harmonized": "Harmonized (model estimate from MODIS, VIIRS scale)",
}


class Period:
    """Optional start and end dates, clipped to the record."""

    def __init__(
        self,
        start: date | None = Query(None, description="First day, YYYY-MM-DD"),
        end: date | None = Query(None, description="Last day, YYYY-MM-DD"),
    ):
        meta = db.metadata()
        self.start = (start.isoformat() if start else meta["first_day"])
        self.end = (end.isoformat() if end else meta["last_day"])
        if self.start > self.end:
            raise HTTPException(status_code=422, detail="start must not be after end")
        self.start = max(self.start, meta["first_day"])
        self.end = min(self.end, meta["last_day"])
