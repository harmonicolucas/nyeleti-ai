"""Build the common grid and aggregate each sensor by cell and day.

Outputs, under data/grid/<cell size>/:
  areas.geojson         country, provinces and districts with stable ids
  grid_cells.parquet    every cell of the study area with its admin units
  <sensor>_<year>.parquet   one row per cell and day with detections
  aggregation_report.json   records in, records outside the grid, rows out

    python -m scripts.aggregate_grid
    python -m scripts.aggregate_grid --cell-km 10
"""
from __future__ import annotations

import argparse
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ml import config
from ml.preprocessing import grid

UNIVERSE_BUFFER_DEG = 0.03


def load_areas() -> gpd.GeoDataFrame:
    """Country, provinces and districts as one table with stable ids."""
    frames = []
    for level in ("ADM0", "ADM1", "ADM2"):
        gdf = gpd.read_file(config.BOUNDARIES_DIR / f"{config.COUNTRY_ISO3}_{level}.geojson")
        gdf = gdf.sort_values("shapeName", kind="stable").reset_index(drop=True)
        gdf["geometry"] = gdf.geometry.make_valid()
        if level == "ADM0":
            ids = [config.COUNTRY_ISO3]
            names = [config.COUNTRY_NAME]
        else:
            ids = [f"{config.COUNTRY_ISO3}-{level}-{i + 1:03d}" for i in range(len(gdf))]
            names = gdf["shapeName"].tolist()
        frames.append(
            gpd.GeoDataFrame(
                {"area_id": ids, "name": names, "level": level, "source_id": gdf.get("shapeID")},
                geometry=gdf.geometry,
                crs="EPSG:4326",
            )
        )
    areas = pd.concat(frames, ignore_index=True)

    # parent of each district = province containing its representative point
    provinces = areas[areas.level == "ADM1"]
    areas["parent_id"] = None
    areas.loc[areas.level == "ADM1", "parent_id"] = config.COUNTRY_ISO3
    districts = areas[areas.level == "ADM2"]
    points = gpd.GeoDataFrame(
        {"area_id": districts.area_id.values}, geometry=districts.geometry.representative_point().values, crs="EPSG:4326"
    )
    joined = gpd.sjoin_nearest(points, provinces[["area_id", "geometry"]], how="left", rsuffix="prov")
    joined = joined.drop_duplicates("area_id_left").set_index("area_id_left")["area_id_prov"]
    areas.loc[areas.level == "ADM2", "parent_id"] = areas.loc[areas.level == "ADM2", "area_id"].map(joined)
    return areas


def build_cells(areas: gpd.GeoDataFrame, cell_km: float) -> pd.DataFrame:
    country = areas.loc[areas.level == "ADM0", "geometry"].union_all().buffer(UNIVERSE_BUFFER_DEG)
    west, south, east, north = country.bounds
    (ix0, ix1), (iy0, iy1) = grid.index_range(west, south, east, north, cell_km)
    ix, iy = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1))
    ix, iy = ix.ravel(), iy.ravel()

    w, s, e, n = grid.cell_bounds(ix, iy, cell_km)
    shapely.prepare(country)
    inside = shapely.intersects(country, shapely.box(w, s, e, n))
    ix, iy = ix[inside], iy[inside]
    lon, lat = grid.cell_center(ix, iy, cell_km)

    cells = pd.DataFrame(
        {
            "cell_id": grid.cell_id(ix, iy),
            "ix": ix.astype("int32"),
            "iy": iy.astype("int32"),
            "lon": lon,
            "lat": lat,
        }
    )
    centres = gpd.GeoDataFrame(cells[["cell_id"]], geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    for level, column in (("ADM1", "adm1_id"), ("ADM2", "adm2_id")):
        units = areas.loc[areas.level == level, ["area_id", "geometry"]]
        # nearest unit: equals the containing unit for centres inside the
        # country and attaches coastal or border cells to the closest one
        joined = gpd.sjoin_nearest(centres, units, how="left").drop_duplicates("cell_id")
        cells[column] = cells["cell_id"].map(joined.set_index("cell_id")["area_id"])
    b = config.background_block_cells(cell_km)
    cells["block_id"] = grid.cell_id(cells["ix"] // b, cells["iy"] // b)
    return cells


def aggregate(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    df = df.assign(is_day=(df["daynight"] == "D").astype("int32"))
    keys = ["acq_date", "cell_id"]
    if sensor == "modis":
        df = df.assign(is_terra=(df["satellite"] == "Terra").astype("int32"))
        out = df.groupby(keys, sort=True).agg(
            modis_fire_count=("frp", "size"),
            modis_frp_sum=("frp", "sum"),
            modis_frp_max=("frp", "max"),
            modis_confidence_mean=("confidence_pct", "mean"),
            modis_brightness_mean=("brightness", "mean"),
            modis_scan_mean=("scan", "mean"),
            modis_day_count=("is_day", "sum"),
            modis_terra_count=("is_terra", "sum"),
        )
        out["modis_frp_mean"] = out["modis_frp_sum"] / out["modis_fire_count"]
        out["modis_night_count"] = out["modis_fire_count"] - out["modis_day_count"]
        out["modis_aqua_count"] = out["modis_fire_count"] - out["modis_terra_count"]
    else:
        out = df.groupby(keys, sort=True).agg(
            fire_count=("frp", "size"),
            frp_sum=("frp", "sum"),
            day_count=("is_day", "sum"),
        )
    out = out.reset_index().rename(columns={"acq_date": "date"})
    out["date"] = pd.to_datetime(out["date"])
    float_cols = out.select_dtypes("float").columns
    out[float_cols] = out[float_cols].astype("float32")
    int_cols = [c for c in out.columns if c.endswith("_count")]
    out[int_cols] = out[int_cols].astype("int32")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell-km", type=float, default=config.CELL_KM)
    parser.add_argument("--sensors", nargs="+", default=list(config.SENSORS))
    parser.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    args = parser.parse_args(argv)

    out_dir = config.GRID_DIR / config.grid_tag(args.cell_km)
    out_dir.mkdir(parents=True, exist_ok=True)

    areas = load_areas()
    areas.to_file(out_dir / "areas.geojson", driver="GeoJSON")
    cells = build_cells(areas, args.cell_km)
    cells.to_parquet(out_dir / "grid_cells.parquet", index=False)
    universe = set(cells["cell_id"].tolist())
    print(f"Grid {config.grid_tag(args.cell_km)}: {len(cells):,} cells, {areas.level.value_counts().to_dict()}")

    report = {"cell_km": args.cell_km, "cells": int(len(cells)), "files": []}
    for sensor in args.sensors:
        for path in sorted((config.CLEAN_DIR / sensor).glob("*.parquet")):
            year = int(path.stem)
            if args.years and not (args.years[0] <= year <= args.years[1]):
                continue
            df = pd.read_parquet(path)
            df = df[df["analysis"]]
            ix, iy = grid.cell_index(df["longitude"].values, df["latitude"].values, args.cell_km)
            df = df.assign(cell_id=grid.cell_id(ix, iy))
            in_grid = df["cell_id"].isin(universe)
            agg = aggregate(df[in_grid], sensor)
            agg.to_parquet(out_dir / f"{sensor}_{year}.parquet", index=False)
            report["files"].append(
                {
                    "sensor": sensor,
                    "year": year,
                    "analysis_records": int(len(df)),
                    "outside_grid": int((~in_grid).sum()),
                    "cell_days": int(len(agg)),
                }
            )
            print(
                f"{sensor} {year}: {len(df):,} detections -> {len(agg):,} cell-days "
                f"({int((~in_grid).sum())} outside the grid)",
                flush=True,
            )
    (out_dir / "aggregation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
