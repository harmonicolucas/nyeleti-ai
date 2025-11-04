from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.database import db
from backend.app.models.params import Direction, Period, Severity
from backend.app.services import anomalies as anomaly_service
from backend.app.services import series
from backend.app.services.areas import Selection, selection
from ml.anomaly import detector as det

router = APIRouter(prefix="/api/anomalies", tags=["anomalies"])

NOTE = (
    "Anomalies describe observed activity that departs from the area's own history for that time of year. "
    "They are computed on the harmonized series and are not forecasts."
)


@router.get("")
def anomalies(
    sel: Selection = Depends(selection),
    period: Period = Depends(),
    min_severity: Severity = Severity.moderate,
    direction: Direction = Direction.above,
    limit: int = Query(200, ge=1, le=2000),
) -> dict:
    """Periods of unusual fire activity in the selected area."""
    detector = anomaly_service.detector()
    events = anomaly_service.events(sel, period.start, period.end, min_severity.value, direction.value)
    scored = series.window(anomaly_service.scored(sel), period.start, period.end)
    flagged = scored[scored["severity"].notna()]
    counts = {name: int((flagged["severity"] == name).sum()) for name in det.SEVERITY_ORDER}
    events = sorted(events, key=lambda e: e["peak_score"], reverse=True)
    return {
        "selection": sel.describe(),
        "start": max(period.start, f"{detector.meta['first_year']}-01-01"),
        "end": period.end,
        "events": events[:limit],
        "total_events": len(events),
        "flagged_days": counts,
        "days_scored": int(len(scored)),
        "model": {
            "name": detector.meta["model"],
            "created_at": detector.meta["created_at"],
            "features": detector.meta["features"],
            "severity_quantiles": detector.meta["severity_quantiles"],
            "thresholds": anomaly_service.thresholds(sel),
            "threshold_basis": "quantiles of the scores of all areas of a similar administrative level",
            "reference": f"same +/-{detector.meta['window_days']}-day calendar window in the other years, "
                         f"{detector.meta['first_year']} onwards",
            "window_days": detector.meta["window_days"],
            "first_year": detector.meta["first_year"],
        },
        "note": NOTE,
    }


@router.get("/series")
def anomaly_series(sel: Selection = Depends(selection), year: int = Query(...)) -> dict:
    """Daily activity, seasonal reference and anomaly score for one year."""
    scored = anomaly_service.scored(sel)
    rows = scored[scored["date"].dt.year == year]
    return {
        "selection": sel.describe(),
        "year": year,
        "dates": rows["date"].dt.strftime("%Y-%m-%d").tolist(),
        "hfai": series.values(rows["hfai"]),
        "hfai_7d": series.values(rows["hfai_smooth"]),
        "reference": series.values(rows["reference"]),
        "score": series.values(rows["score"], 4),
        "severity": [None if s is None or s != s else s for s in rows["severity"].tolist()],
        "direction": rows["direction"].tolist(),
        "thresholds": anomaly_service.thresholds(sel),
        "note": NOTE,
    }


@router.get("/overview")
def overview(
    level: str = Query("ADM1", pattern="^ADM[12]$"),
    parent_id: str | None = None,
    period: Period = Depends(),
    direction: Direction = Direction.above,
) -> dict:
    """Flagged days per administrative unit, to compare regions."""
    sql = """
        SELECT g.area_id, g.name, g.parent_id,
               count(*) FILTER (WHERE a.severity IS NOT NULL) AS flagged_days,
               count(*) FILTER (WHERE a.severity = 'high') AS high_days,
               count(*) FILTER (WHERE a.severity = 'extreme') AS extreme_days,
               max(a.pct_vs_reference) AS max_pct_vs_reference,
               arg_max(a.date, a.score) AS peak_date
        FROM geographic_areas g
        LEFT JOIN fire_anomalies a
          ON a.area_id = g.area_id AND a.date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) {direction}
        WHERE g.level = ? {parent}
        GROUP BY g.area_id, g.name, g.parent_id
        ORDER BY flagged_days DESC, g.name
    """
    params = [period.start, period.end]
    direction_sql = ""
    if direction is not Direction.both:
        direction_sql = "AND a.direction = ?"
        params.append(direction.value)
    params.append(level)
    parent_sql = ""
    if parent_id:
        parent_sql = "AND g.parent_id = ?"
        params.append(parent_id)
    rows = db.cursor().execute(sql.format(direction=direction_sql, parent=parent_sql), params).fetchall()
    return {
        "level": level,
        "start": period.start,
        "end": period.end,
        "areas": [
            {
                "area_id": r[0],
                "name": r[1],
                "parent_id": r[2],
                "flagged_days": int(r[3]),
                "high_days": int(r[4]),
                "extreme_days": int(r[5]),
                "max_pct_vs_reference": series.number(r[6], 1),
                "peak_date": r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ],
        "note": NOTE,
    }
