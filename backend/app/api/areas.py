from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query

from backend.app.database import db
from backend.app.services import areas as area_service
from backend.app.services.areas import Selection

router = APIRouter(prefix="/api/areas", tags=["areas"])


@router.get("")
def list_areas(
    level: str | None = Query(None, pattern="^ADM[012]$"),
    parent_id: str | None = None,
) -> dict:
    """Country, provinces and districts available for selection."""
    frame = area_service.areas_frame()
    if level:
        frame = frame[frame["level"] == level]
    if parent_id:
        frame = frame[frame["parent_id"] == parent_id]
    items = [
        {
            "area_id": r.area_id,
            "name": r.name,
            "level": r.level,
            "parent_id": r.parent_id,
            "bounds": [round(r.west, 4), round(r.south, 4), round(r.east, 4), round(r.north, 4)],
            "n_cells": int(r.n_cells),
        }
        for r in frame.itertuples()
    ]
    return {"areas": items}


@router.get("/geojson")
def areas_geojson(level: str = Query("ADM1", pattern="^ADM[012]$"), parent_id: str | None = None) -> dict:
    """Boundaries of one administrative level as a GeoJSON FeatureCollection."""
    sql = "SELECT area_id, name, level, parent_id, geometry FROM geographic_areas WHERE level = ?"
    params = [level]
    if parent_id:
        sql += " AND parent_id = ?"
        params.append(parent_id)
    features = [
        {
            "type": "Feature",
            "properties": {"area_id": area_id, "name": name, "level": lvl, "parent_id": parent},
            "geometry": json.loads(geometry),
        }
        for area_id, name, lvl, parent, geometry in db.cursor().execute(sql, params).fetchall()
    ]
    return {"type": "FeatureCollection", "features": features}


@router.get("/resolve")
def resolve(sel: Selection = Depends(area_service.selection)) -> dict:
    """Describe a selection: name, number of grid cells and area."""
    return sel.describe()
