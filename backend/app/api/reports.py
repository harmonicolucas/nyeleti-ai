from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response

from backend.app.models.params import Frequency, Period
from backend.app.services import reports
from backend.app.services.areas import Selection, selection

router = APIRouter(prefix="/api/reports", tags=["reports"])


def _filename(sel: Selection, period: Period, suffix: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "-", sel.name).strip("-").lower() or "area"
    return f"nyeleti_{name}_{period.start}_{period.end}{suffix}"


@router.get("")
def report(
    sel: Selection = Depends(selection),
    period: Period = Depends(),
    kind: str = Query("pdf", pattern="^(pdf|series|anomalies|grid)$",
                      description="pdf: report; series: CSV time series; anomalies: CSV; grid: GeoJSON cells"),
    freq: Frequency = Frequency.day,
) -> Response:
    """Download a report or a data export for the selected area and period.

    Every export keeps observations and model estimates in separate,
    explicitly named columns.
    """
    if kind == "pdf":
        content = reports.pdf_report(sel, period.start, period.end)
        media, name = "application/pdf", _filename(sel, period, ".pdf")
    elif kind == "series":
        content = reports.series_csv(sel, period.start, period.end, freq.value)
        media, name = "text/csv", _filename(sel, period, f"_{freq.value}.csv")
    elif kind == "anomalies":
        content = reports.anomalies_csv(sel, period.start, period.end)
        media, name = "text/csv", _filename(sel, period, "_anomalies.csv")
    else:
        return JSONResponse(
            reports.grid_geojson(sel, period.start, period.end),
            media_type="application/geo+json",
            headers={"Content-Disposition": f'attachment; filename="{_filename(sel, period, "_grid.geojson")}"'},
        )
    return Response(content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})
