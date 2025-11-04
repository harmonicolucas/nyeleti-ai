from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.database import db
from backend.app.models.params import METRIC_LABELS, SOURCE_LABELS, Metric
from backend.app.services import analysis, series
from backend.app.services import anomalies as anomaly_service
from backend.app.services.areas import Selection, selection
from ml import config

router = APIRouter(prefix="/api", tags=["trends"])


@router.get("/trends")
def trends(sel: Selection = Depends(selection), metric: Metric = Metric.hfai) -> dict:
    """Yearly totals, long-term trend, seasonality and timing of the fire season."""
    body = analysis.trends(series.daily(sel), metric.value)
    body.update(
        {
            "selection": sel.describe(),
            "metric_label": METRIC_LABELS[metric.value],
            "sources": SOURCE_LABELS,
            "note": f"The trend and the seasonal statistics use the harmonized series from "
                    f"{config.ANALYSIS_FIRST_YEAR}, the first full year with both MODIS platforms.",
        }
    )
    return body


@router.get("/critical")
def critical(
    sel: Selection = Depends(selection),
    metric: Metric = Metric.hfai,
    percentile: float = Query(90, ge=50, le=99.9),
    min_days: int = Query(3, ge=1, le=60),
    max_gap: int = Query(1, ge=0, le=7),
    basis: str = Query("area", pattern="^(area|month)$", description="area: one threshold; month: one per calendar month"),
) -> dict:
    """Historically critical burning periods of the selected area."""
    body = analysis.critical_periods(series.daily(sel), metric.value, percentile, min_days, max_gap, basis)
    body["selection"] = sel.describe()
    body["note"] = "Computed on the harmonized series (model estimates on the VIIRS scale)."
    return body


@router.get("/critical/regions")
def recurrent_regions(
    level: str = Query("ADM2", pattern="^ADM[12]$"),
    parent_id: str | None = None,
    top_share: float = Query(0.10, gt=0, le=0.5),
) -> dict:
    """Administrative units that are repeatedly among the most active of their level.

    Activity is the harmonized HFAI per 1000 km2, so that large and small
    units can be compared. A unit is a hotspot in a given year when it falls in
    the top `top_share` of all units of that level.
    """
    meta = db.metadata()
    ref = meta["hfai"]
    cell_area = meta["cell_km"] ** 2
    frame = db.cursor().execute(
        """
        SELECT c.area_id, g.name, g.parent_id, g.n_cells, year(c.date) AS year,
               sum(c.harmonized_count) AS count, sum(c.harmonized_frp) AS frp
        FROM calendar_activity c JOIN geographic_areas g USING (area_id)
        WHERE g.level = ? AND year(c.date) >= ? AND g.n_cells > 0
        GROUP BY c.area_id, g.name, g.parent_id, g.n_cells, year(c.date)
        """,
        [level, config.ANALYSIS_FIRST_YEAR],
    ).df()
    frame["hfai"] = config.hfai(frame["count"], frame["frp"], ref)
    frame["density"] = 1000.0 * frame["hfai"] / (frame["n_cells"] * cell_area)
    cutoff = frame.groupby("year")["density"].transform(lambda s: s.quantile(1 - top_share))
    frame["hotspot"] = frame["density"] >= cutoff
    n_years = frame["year"].nunique()

    summary = frame.groupby(["area_id", "name", "parent_id"]).agg(
        hotspot_years=("hotspot", "sum"),
        mean_density=("density", "mean"),
        max_density=("density", "max"),
        mean_hfai=("hfai", "mean"),
    ).reset_index()
    peak_year = frame.loc[frame.groupby("area_id")["density"].idxmax()].set_index("area_id")["year"]
    summary["peak_year"] = summary["area_id"].map(peak_year)
    summary["rank"] = summary["mean_density"].rank(ascending=False, method="min").astype(int)
    if parent_id:
        summary = summary[summary["parent_id"] == parent_id]
    summary = summary.sort_values(["hotspot_years", "mean_density"], ascending=False)

    names = {a: n for a, n in db.cursor().execute("SELECT area_id, name FROM geographic_areas").fetchall()}
    return {
        "level": level,
        "years": [config.ANALYSIS_FIRST_YEAR, int(frame["year"].max())],
        "n_years": int(n_years),
        "top_share": top_share,
        "definition": f"Hotspot year: annual harmonized HFAI per 1000 km2 within the top {top_share:.0%} "
                      f"of all {level} units that year.",
        "areas": [
            {
                "area_id": r.area_id,
                "name": r.name,
                "parent_id": r.parent_id,
                "parent_name": names.get(r.parent_id),
                "hotspot_years": int(r.hotspot_years),
                "recurrence_pct": series.number(100 * r.hotspot_years / n_years, 1),
                "mean_density": series.number(r.mean_density),
                "max_density": series.number(r.max_density),
                "mean_annual_hfai": series.number(r.mean_hfai, 1),
                "peak_year": int(r.peak_year),
                "rank": int(r.rank),
            }
            for r in summary.itertuples()
        ],
    }


@router.get("/dashboard")
def dashboard(
    sel: Selection = Depends(selection),
    year: int | None = Query(None, description="Year summarised in the indicators. Default: last year of the record."),
) -> dict:
    """Headline indicators of the selected area."""
    df = series.daily(sel)
    meta = db.metadata()
    last_year = int(df["year"].max())
    year = min(max(year or last_year, int(df["year"].min())), last_year)
    rows = df[df["year"] == year]
    hist = analysis.analysis_rows(df)

    def total(col):
        return series.number(rows[col].sum(min_count=1), 1)

    high_limit = float(np.percentile(hist["harmonized_hfai"].values, 90))
    high_days = int((rows["harmonized_hfai"] > high_limit).sum())
    yearly_high = (hist["harmonized_hfai"] > high_limit).groupby(hist["year"]).sum()

    try:
        events = anomaly_service.events(sel, f"{year}-01-01", f"{year}-12-31", "moderate", "above")
        scored = anomaly_service.scored(sel)
        flagged = scored[(scored["date"].dt.year == year) & scored["severity"].notna() & (scored["direction"] == "above")]
        anomaly_block = {
            "events": len(events),
            "flagged_days": int(len(flagged)),
            "top": sorted(events, key=lambda e: e["peak_score"], reverse=True)[:3],
        }
    except HTTPException:  # anomaly model not trained yet
        anomaly_block = None

    viirs_total = rows["viirs_count"].sum(min_count=1)
    observed_total = rows["modis_count"].sum() + (0 if np.isnan(viirs_total) else viirs_total)

    yearly = analysis.calendar_years(df, "hfai")
    yearly_count = analysis.calendar_years(df, "count")
    reference_mean = float(hist.groupby("year")["harmonized_hfai"].sum().mean())
    this_year = float(rows["harmonized_hfai"].sum())

    return {
        "selection": sel.describe(),
        "year": year,
        "years_available": [int(df["year"].min()), last_year],
        "observed": {
            "modis_detections": total("modis_count"),
            "viirs_detections": total("viirs_count"),
            "noaa20_detections": total("noaa20_count"),
            "total_detections": series.number(observed_total, 0),
            "modis_frp_mw": total("modis_frp"),
            "viirs_frp_mw": total("viirs_frp"),
        },
        "harmonized": {
            "detections_viirs_scale": total("harmonized_count"),
            "frp_viirs_scale_mw": total("harmonized_frp"),
            "hfai": total("harmonized_hfai"),
            "vs_reference_pct": series.number(100 * (this_year / reference_mean - 1), 1) if reference_mean else None,
            "reference": f"mean annual HFAI {config.ANALYSIS_FIRST_YEAR}-{last_year}",
        },
        "high_activity": {
            "days": high_days,
            "threshold_hfai": series.number(high_limit),
            "mean_days_per_year": series.number(float(yearly_high.mean()), 1),
            "definition": f"days with harmonized HFAI above the area's 90th percentile "
                          f"({config.ANALYSIS_FIRST_YEAR}-{last_year})",
        },
        "anomalies": anomaly_block,
        "yearly": {
            "years": yearly["years"],
            "hfai": yearly["series"],
            "count": yearly_count["series"],
            "viirs_complete": yearly["viirs_complete"],
            "modis_full_coverage": yearly["modis_full_coverage"],
        },
        "model_version": meta["active_model"],
        "viirs_start": meta["viirs_start"],
        "note": "Observed values are NASA FIRMS detections. Harmonized values are model estimates on the VIIRS scale. "
                "MODIS and VIIRS FRP are reported separately because they are not directly comparable.",
    }
