"""Area of interest: an administrative unit or a shape drawn on the map."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import shapely
from fastapi import HTTPException, Query
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from backend.app.database import db
from ml import config
from ml.preprocessing import grid

MAX_AOI_VERTICES = 2000


@dataclass
class Selection:
    key: str
    kind: str  # "area" or "aoi"
    name: str
    level: str
    n_cells: int
    bounds: tuple[float, float, float, float]
    area_id: str | None = None
    geometry: BaseGeometry | None = None
    cell_ids: np.ndarray | None = field(default=None, repr=False)

    @property
    def is_country(self) -> bool:
        return self.kind == "area" and self.level == "ADM0"

    @property
    def area_km2(self) -> float:
        return self.n_cells * db.metadata()["cell_km"] ** 2

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "area_id": self.area_id,
            "name": self.name,
            "level": self.level,
            "n_cells": self.n_cells,
            "area_km2": round(self.area_km2),
            "bounds": [round(v, 4) for v in self.bounds],
        }


@lru_cache(maxsize=1)
def cells_frame() -> pd.DataFrame:
    return db.cursor().execute("SELECT cell_id, ix, iy, lon, lat, adm1_id, adm2_id FROM grid_cells").df()


@lru_cache(maxsize=1)
def areas_frame() -> pd.DataFrame:
    return db.cursor().execute(
        "SELECT area_id, name, level, parent_id, west, south, east, north, coalesce(n_cells, 0) AS n_cells "
        "FROM geographic_areas "
        "ORDER BY level, name"
    ).df()


@lru_cache(maxsize=256)
def area_geometry(area_id: str) -> BaseGeometry:
    row = db.cursor().execute("SELECT geometry FROM geographic_areas WHERE area_id = ?", [area_id]).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown area '{area_id}'")
    return shape(json.loads(row[0]))


def _area_selection(area_id: str) -> Selection:
    areas = areas_frame()
    match = areas[areas["area_id"] == area_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown area '{area_id}'")
    row = match.iloc[0]
    return Selection(
        key=f"area:{area_id}",
        kind="area",
        area_id=area_id,
        name=row["name"],
        level=row["level"],
        n_cells=int(row["n_cells"]) if pd.notna(row["n_cells"]) else 0,
        bounds=(row["west"], row["south"], row["east"], row["north"]),
    )


def _parse_geometry(aoi: str | None, bbox: str | None) -> BaseGeometry:
    try:
        if bbox:
            west, south, east, north = (float(v) for v in bbox.split(","))
            if not (west < east and south < north):
                raise ValueError("bbox must be west,south,east,north")
            return box(west, south, east, north)
        data = json.loads(aoi)
        if data.get("type") == "Feature":
            data = data["geometry"]
        if data.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("AOI must be a Polygon or MultiPolygon")
        geom = shape(data)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid area of interest: {exc}") from exc
    if shapely.get_num_coordinates(geom) > MAX_AOI_VERTICES:
        raise HTTPException(status_code=422, detail=f"AOI has more than {MAX_AOI_VERTICES} vertices")
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    if geom.is_empty:
        raise HTTPException(status_code=422, detail="AOI is empty")
    return geom


def _aoi_selection(geom: BaseGeometry) -> Selection:
    cells = cells_frame()
    west, south, east, north = geom.bounds
    margin = 0.06
    near = cells[
        cells["lon"].between(west - margin, east + margin) & cells["lat"].between(south - margin, north + margin)
    ]
    inside = shapely.contains_xy(geom, near["lon"].values, near["lat"].values)
    if not inside.any() and len(near):
        # AOI smaller than a cell: take the cells it touches
        w, s, e, n = grid.cell_bounds(near["ix"].values, near["iy"].values, db.metadata()["cell_km"])
        inside = shapely.intersects(geom, shapely.box(w, s, e, n))
    cell_ids = np.sort(near.loc[inside, "cell_id"].values)
    if len(cell_ids) == 0:
        raise HTTPException(status_code=404, detail=f"The selected area is outside {config.COUNTRY_NAME}")
    digest = hashlib.sha1(cell_ids.tobytes()).hexdigest()[:16]
    return Selection(
        key=f"aoi:{digest}",
        kind="aoi",
        name="Custom area",
        level="AOI",
        n_cells=int(len(cell_ids)),
        bounds=geom.bounds,
        geometry=geom,
        cell_ids=cell_ids,
    )


def resolve(area_id: str | None = None, aoi: str | None = None, bbox: str | None = None) -> Selection:
    if aoi or bbox:
        return _aoi_selection(_parse_geometry(aoi, bbox))
    return _area_selection(area_id or config.COUNTRY_ISO3)


def selection(
    area_id: str | None = Query(None, description="Administrative area id (see /api/areas). Default: the country."),
    aoi: str | None = Query(None, description="GeoJSON Polygon or MultiPolygon drawn by the user"),
    bbox: str | None = Query(None, description="west,south,east,north in degrees"),
) -> Selection:
    """FastAPI dependency shared by every endpoint that works on an area."""
    return resolve(area_id, aoi, bbox)


def selection_geometry(sel: Selection) -> BaseGeometry:
    return sel.geometry if sel.kind == "aoi" else area_geometry(sel.area_id)
