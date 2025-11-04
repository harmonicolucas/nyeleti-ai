from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

from backend.app.database import db
from backend.app.models.params import METRIC_LABELS, SOURCE_LABELS, Frequency, Layer, Metric, Period
from backend.app.services import series
from backend.app.services.areas import Selection, selection
from backend.app.services.cells import cell_values
from ml.evaluation.metrics import regression_metrics
from ml.preprocessing import grid

router = APIRouter(prefix="/api/harmonization", tags=["harmonization"])

FREQ = {"day": "D", "month": "M", "year": "Y"}
MAX_GRID_CELLS = 12000


def _agreement(observed: pd.Series, estimate: pd.Series) -> dict | None:
    ok = observed.notna() & estimate.notna()
    if ok.sum() < 3:
        return None
    m = regression_metrics(observed[ok].values, estimate[ok].values)
    return {k: series.number(m[k]) for k in ("n", "mae", "rmse", "r2", "bias", "bias_pct", "spearman")}


@router.get("")
def harmonization(
    sel: Selection = Depends(selection),
    period: Period = Depends(),
    freq: Frequency = Frequency.month,
    metric: Metric = Metric.hfai,
) -> dict:
    """Original MODIS, original VIIRS and harmonized activity for the same area and period.

    Also reports how far MODIS and the harmonized estimate are from VIIRS over
    the part of the period where VIIRS exists, split by the role of each year
    in model development (training, validation, independent test).
    """
    df = series.window(series.daily(sel), period.start, period.end)
    cols = {s: f"{s}_{metric.value}" for s in series.SOURCES}
    extra = ["model_count", "background_count"] if metric is Metric.count else []
    bucket = df["date"].dt.to_period(FREQ[freq.value])
    grouped = df.groupby(bucket)[list(cols.values()) + extra].sum(min_count=1)
    # a period is only comparable with VIIRS if VIIRS covers all of its days
    complete = df.groupby(bucket)[cols["viirs"]].count() == df.groupby(bucket).size()
    grouped.loc[~complete, cols["viirs"]] = np.nan

    years = grouped.index.year
    agreement = {}
    for split in ("train", "validation", "test"):
        mask = np.array([series.split_of(int(y)) == split for y in years]) & complete.values
        if mask.sum() >= 3:
            agreement[split] = {
                "periods": int(mask.sum()),
                "modis_vs_viirs": _agreement(grouped.loc[mask, cols["viirs"]], grouped.loc[mask, cols["modis"]]),
                "harmonized_vs_viirs": _agreement(grouped.loc[mask, cols["viirs"]], grouped.loc[mask, cols["harmonized"]]),
            }

    meta = db.metadata()
    body = {
        "selection": sel.describe(),
        "start": period.start,
        "end": period.end,
        "freq": freq.value,
        "metric": metric.value,
        "metric_label": METRIC_LABELS[metric.value],
        "sources": SOURCE_LABELS,
        "periods": grouped.index.astype(str).tolist(),
        "split": [series.split_of(int(y)) for y in years],
        "series": {s: series.values(grouped[c]) for s, c in cols.items()},
        "difference": {
            "modis_minus_viirs": series.values(grouped[cols["modis"]] - grouped[cols["viirs"]]),
            "harmonized_minus_viirs": series.values(grouped[cols["harmonized"]] - grouped[cols["viirs"]]),
        },
        "agreement": agreement,
        "model_version": meta["active_model"],
        "viirs_start": meta["viirs_start"],
        "note": "Agreement is computed for this area at the chosen frequency, over periods fully covered by VIIRS. "
                "Only the test split is independent of model development.",
    }
    if extra:
        body["components"] = {
            "model": series.values(grouped["model_count"]),
            "background": series.values(grouped["background_count"]),
        }
    return body


@router.get("/grid")
def harmonization_grid(
    sel: Selection = Depends(selection),
    period: Period = Depends(),
    layer: Layer = Layer.harmonized,
    metric: Metric = Metric.hfai,
    min_value: float = Query(0.0, ge=0),
) -> dict:
    """Activity per grid cell over a period, for the map layer."""
    meta = db.metadata()
    frame = cell_values(sel, period.start, period.end, layer.value)
    frame["value"] = frame[metric.value]
    cells_in_selection = len(frame)
    total = float(frame["value"].sum())
    frame = frame[frame["value"] > min_value].sort_values("value", ascending=False)
    truncated = len(frame) > MAX_GRID_CELLS
    frame = frame.head(MAX_GRID_CELLS)

    west, south, east, north = grid.cell_bounds(frame["ix"].values, frame["iy"].values, meta["cell_km"])
    cells = np.column_stack(
        [np.round(west, 4), np.round(south, 4), np.round(east, 4), np.round(north, 4), np.round(frame["value"].values, 3)]
    ).tolist()
    return {
        "selection": sel.describe(),
        "start": period.start,
        "end": period.end,
        "layer": layer.value,
        "layer_label": SOURCE_LABELS[layer.value],
        "metric": metric.value,
        "cell_km": meta["cell_km"],
        "columns": ["west", "south", "east", "north", "value"],
        "cells": cells,
        "cells_in_selection": cells_in_selection,
        "truncated": truncated,
        "total": series.number(total),
        "max": series.number(frame["value"].max()) if len(frame) else 0,
        "note": "Harmonized cells are model estimates; MODIS and VIIRS cells are observations."
        if layer is Layer.harmonized
        else "Observed detections aggregated on the common grid.",
    }
