from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.app.database import db
from ml import config

router = APIRouter(prefix="/api", tags=["model"])


def _versions() -> list[dict]:
    rows = db.cursor().execute(
        "SELECT version, created_at, cell_km, is_active, meta, metrics FROM model_versions ORDER BY created_at DESC"
    ).fetchall()
    return [
        {
            "version": version,
            "created_at": created_at,
            "cell_km": cell_km,
            "is_active": bool(is_active),
            "meta": json.loads(meta),
            "metrics": json.loads(metrics) if metrics else None,
        }
        for version, created_at, cell_km, is_active, meta, metrics in rows
    ]


@router.get("/model/metrics")
def model_metrics(version: str | None = None) -> dict:
    """Description, parameters and evaluation of the harmonization model.

    All figures come from the evaluation script; test metrics are computed on
    years the model never saw during training or selection.
    """
    versions = _versions()
    if version:
        chosen = next((v for v in versions if v["version"] == version), None)
    else:
        chosen = next((v for v in versions if v["is_active"]), None)
    if chosen is None:
        raise HTTPException(status_code=404, detail="Model version not found")

    runs = db.cursor().execute(
        "SELECT run_at, model_version, first_year, last_year, cell_days FROM harmonization_runs ORDER BY run_at DESC"
    ).fetchall()
    meta = chosen["meta"]
    return {
        "version": chosen["version"],
        "is_active": chosen["is_active"],
        "created_at": chosen["created_at"],
        "model": {
            "algorithm": "LightGBM gradient boosted decision trees (regression)",
            "reference_sensor": "VIIRS S-NPP 375 m",
            "input_sensor": "MODIS Terra and Aqua 1 km",
            "unit": f"{meta['cell_km']:g} km equal-area cell x day ({meta['grid_crs']})",
            "features": meta["features"],
            "split": meta["split"],
            "rows": meta["rows"],
            "targets": {
                name: {
                    "target": info["target"],
                    "params": info["params"],
                    "trees": info["best_iteration"],
                    "search": info["search"],
                    "validation": info["validation"],
                    "feature_importance_gain": info["feature_importance_gain"],
                    "linear_baseline": info.get("linear_baseline"),
                }
                for name, info in meta["targets"].items()
            },
            "hfai": meta["hfai"],
            "background": meta["background"],
            "data_fingerprint": meta["data_fingerprint"],
            "libraries": meta["libraries"],
            "seed": meta["seed"],
        },
        "metrics": chosen["metrics"],
        "versions": [
            {k: v[k] for k in ("version", "created_at", "cell_km", "is_active")} for v in versions
        ],
        "runs": [
            {"run_at": r[0], "model_version": r[1], "first_year": r[2], "last_year": r[3], "cell_days": r[4]}
            for r in runs
        ],
    }


@router.get("/meta")
def meta() -> dict:
    """Data inventory: NASA files used, quality totals and coverage of the record."""
    info = db.metadata()
    files = db.cursor().execute(
        "SELECT sensor, year, product, url, records, bytes, sha256, date_min, date_max, versions, downloaded_at "
        "FROM data_files ORDER BY sensor, year"
    ).fetchall()
    keys = ["sensor", "year", "product", "url", "records", "bytes", "sha256", "date_min", "date_max",
            "versions", "downloaded_at"]
    counts = db.cursor().execute(
        "SELECT sensor, count(*), min(acq_date), max(acq_date), sum(analysis::INT) FROM fire_detections GROUP BY sensor"
    ).fetchall()
    return {
        "project": "NYELETI AI",
        "country": info["country"],
        "iso3": info["iso3"],
        "grid": {"cell_km": info["cell_km"], "crs": config.GRID_CRS},
        "record": {"first_day": info["first_day"], "last_day": info["last_day"],
                   "viirs_start": info["viirs_start"], "noaa20_start": info.get("noaa20_start"),
                   "analysis_first_year": config.ANALYSIS_FIRST_YEAR},
        "active_model": info["active_model"],
        "hfai": info["hfai"],
        "split": info["split"],
        "built_at": info["built_at"],
        "detections": [
            {"sensor": s, "records": int(n), "first": str(a), "last": str(b), "used_in_analysis": int(u)}
            for s, n, a, b, u in counts
        ],
        "quality": info.get("quality"),
        "anomaly_model": info.get("anomaly_model"),
        "files": [dict(zip(keys, row)) for row in files],
        "sources": [
            {"name": "NASA FIRMS", "url": "https://firms.modaps.eosdis.nasa.gov/"},
            {"name": "FIRMS country archive", "url": "https://firms.modaps.eosdis.nasa.gov/country/"},
            {"name": "geoBoundaries", "url": "https://www.geoboundaries.org/"},
        ],
    }
