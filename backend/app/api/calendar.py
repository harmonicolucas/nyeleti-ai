from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.database import db
from backend.app.models.params import METRIC_LABELS, SOURCE_LABELS, Granularity, Metric
from backend.app.services import analysis, series
from backend.app.services import anomalies as anomaly_service
from backend.app.services.areas import Selection, selection

router = APIRouter(prefix="/api/calendar", tags=["calendar"])


@router.get("")
def calendar(
    sel: Selection = Depends(selection),
    granularity: Granularity = Granularity.month,
    metric: Metric = Metric.hfai,
    year: int | None = Query(None, description="Required for daily granularity"),
) -> dict:
    """Burning activity calendar of the selected area by day, month, season or year.

    Every response carries the three series side by side: original MODIS,
    original VIIRS S-NPP (from 2012) and the harmonized estimate.
    """
    df = series.daily(sel)
    meta = db.metadata()
    if granularity is Granularity.day:
        first, last = int(df["year"].min()), int(df["year"].max())
        year = year or last
        if not first <= year <= last:
            raise HTTPException(status_code=422, detail=f"year must be between {first} and {last}")
        body = analysis.calendar_days(df, metric.value, year)
        try:
            scored = anomaly_service.scored(sel)
            flagged = scored[(scored["date"].dt.year == year) & scored["severity"].notna()]
            body["anomalies"] = [
                {
                    "date": r.date.strftime("%Y-%m-%d"),
                    "severity": r.severity,
                    "direction": r.direction,
                    "pct_vs_reference": round(float(r.pct_vs_reference), 1),
                }
                for r in flagged.itertuples()
            ]
        except HTTPException:
            body["anomalies"] = []
    elif granularity is Granularity.month:
        body = analysis.calendar_months(df, metric.value)
    elif granularity is Granularity.season:
        body = analysis.calendar_seasons(df, metric.value)
    else:
        body = analysis.calendar_years(df, metric.value)

    body.update(
        {
            "selection": sel.describe(),
            "metric": metric.value,
            "metric_label": METRIC_LABELS[metric.value],
            "sources": SOURCE_LABELS,
            "viirs_start": meta["viirs_start"],
            "model_version": meta["active_model"],
            "note": "MODIS and VIIRS values are observations. Harmonized values are model estimates "
                    "on the VIIRS scale; before the VIIRS record starts they cannot be checked against VIIRS.",
        }
    )
    return body
