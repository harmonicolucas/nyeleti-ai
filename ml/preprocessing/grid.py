"""Common equal-area grid shared by every sensor.

Cells are squares of `cell_km` x `cell_km` in EASE-Grid 2.0 (EPSG:6933), so
every cell covers the same ground area. The projection is cylindrical, which
makes each cell a longitude/latitude rectangle: at Mozambique's latitudes a
nominal 5 km cell is about 5.5 km wide and 4.6 km tall (25 km2).
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

from ml import config

_OFFSET = 50_000
_MULT = 100_000

_to_grid = Transformer.from_crs("EPSG:4326", config.GRID_CRS, always_xy=True)
_to_geo = Transformer.from_crs(config.GRID_CRS, "EPSG:4326", always_xy=True)


def cell_size_m(cell_km: float | None = None) -> float:
    return (config.CELL_KM if cell_km is None else cell_km) * 1000.0


def cell_index(lon, lat, cell_km: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Column and row of the cell containing each point."""
    x, y = _to_grid.transform(np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64"))
    size = cell_size_m(cell_km)
    return np.floor(x / size).astype("int64"), np.floor(y / size).astype("int64")


def cell_id(ix, iy) -> np.ndarray:
    return (np.asarray(ix, dtype="int64") + _OFFSET) * _MULT + (np.asarray(iy, dtype="int64") + _OFFSET)


def split_id(cid) -> tuple[np.ndarray, np.ndarray]:
    cid = np.asarray(cid, dtype="int64")
    return cid // _MULT - _OFFSET, cid % _MULT - _OFFSET


def cell_bounds(ix, iy, cell_km: float | None = None):
    """West, south, east, north of each cell in degrees."""
    size = cell_size_m(cell_km)
    ix = np.asarray(ix, dtype="float64")
    iy = np.asarray(iy, dtype="float64")
    west, south = _to_geo.transform(ix * size, iy * size)
    east, north = _to_geo.transform((ix + 1) * size, (iy + 1) * size)
    return west, south, east, north


def cell_center(ix, iy, cell_km: float | None = None):
    size = cell_size_m(cell_km)
    ix = np.asarray(ix, dtype="float64")
    iy = np.asarray(iy, dtype="float64")
    return _to_geo.transform((ix + 0.5) * size, (iy + 0.5) * size)


def index_range(west: float, south: float, east: float, north: float, cell_km: float | None = None):
    """Inclusive ranges of columns and rows covering a bounding box."""
    (ix0, ix1), (iy0, iy1) = cell_index([west, east], [south, north], cell_km)
    return (int(ix0), int(ix1)), (int(iy0), int(iy1))
