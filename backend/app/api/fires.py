from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import shapely
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.database import db
from backend.app.services import areas as area_service
from backend.app.services.areas import Selection

router = APIRouter(prefix="/api/fires", tags=["fires"])

SENSORS = ("modis", "viirs_snpp", "viirs_noaa20")
CONFIDENCE = {"low": ("low", "nominal", "high"), "nominal": ("nominal", "high"), "high": ("high",)}
COLUMNS = [
    "longitude", "latitude", "sensor", "satellite", "acq_date", "acq_time", "frp", "brightness",
    "confidence", "confidence_level", "daynight", "scan", "track", "type", "version",
]
MAX_DAYS = 366


@router.get("")
def fires(
    sel: Selection = Depends(area_service.selection),
    start: date | None = Query(None),
    end: date | None = Query(None),
    sensors: str = Query("modis,viirs_snpp", description="Comma separated: modis, viirs_snpp, viirs_noaa20"),
    min_confidence: str = Query("low", pattern="^(low|nominal|high)$"),
    daynight: str | None = Query(None, pattern="^[DN]$"),
    limit: int = Query(20000, ge=1, le=60000),
) -> dict:
    """Original MODIS and VIIRS detections, exactly as distributed by NASA FIRMS."""
    meta = db.metadata()
    last = end or date.fromisoformat(meta["last_day"])
    first = start or last - timedelta(days=6)
    if first > last:
        raise HTTPException(status_code=422, detail="start must not be after end")
    if (last - first).days >= MAX_DAYS:
        raise HTTPException(status_code=422, detail=f"Period too long: at most {MAX_DAYS} days per request")

    wanted = [s.strip() for s in sensors.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SENSORS]
    if unknown or not wanted:
        raise HTTPException(status_code=422, detail=f"Unknown sensor(s): {unknown or sensors}")

    west, south, east, north = sel.bounds
    sql = f"""
        SELECT {', '.join(COLUMNS)}
        FROM fire_detections
        WHERE acq_date BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?
          AND sensor IN ({', '.join('?' for _ in wanted)})
          AND confidence_level IN ({', '.join('?' for _ in CONFIDENCE[min_confidence])})
    """
    params = [first, last, west, east, south, north, *wanted, *CONFIDENCE[min_confidence]]
    if daynight:
        sql += " AND daynight = ?"
        params.append(daynight)
    frame = db.cursor().execute(sql, params).df()

    if not sel.is_country and len(frame):
        geom = area_service.selection_geometry(sel)
        frame = frame[shapely.contains_xy(geom, frame["longitude"].values, frame["latitude"].values)]

    total = len(frame)
    by_sensor = {k: int(v) for k, v in frame["sensor"].value_counts().items()}
    truncated = total > limit
    if truncated:
        keep = np.sort(np.random.default_rng(0).choice(total, limit, replace=False))
        frame = frame.iloc[keep]

    frame = frame.assign(
        acq_date=frame["acq_date"].astype(str),
        longitude=frame["longitude"].round(5),
        latitude=frame["latitude"].round(5),
        frp=frame["frp"].round(2),
        brightness=frame["brightness"].round(1),
        scan=frame["scan"].round(2),
        track=frame["track"].round(2),
    )
    rows = frame.astype(object).where(frame.notna(), None).values.tolist()
    return {
        "selection": sel.describe(),
        "start": first.isoformat(),
        "end": last.isoformat(),
        "total": total,
        "returned": len(rows),
        "truncated": truncated,
        "by_sensor": by_sensor,
        "columns": COLUMNS,
        "rows": rows,
        "source": "NASA FIRMS active fire archive, standard processing (MODIS 1 km, VIIRS 375 m)",
    }
