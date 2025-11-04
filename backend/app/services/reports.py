"""Exports: CSV series, GeoJSON grid and the PDF report of a selection."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import HTTPException
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.database import db
from backend.app.services import analysis, series
from backend.app.services import anomalies as anomaly_service
from backend.app.services.areas import Selection
from backend.app.services.cells import cell_values
from ml import config
from ml.preprocessing import grid

FREQ = {"day": "D", "month": "M", "year": "Y"}
COLOURS = {"modis": "#3b7dd8", "viirs": "#222222", "harmonized": "#d95f02"}
MAX_GEOJSON_CELLS = 20000


def series_csv(sel: Selection, start: str, end: str, freq: str) -> str:
    df = series.window(series.daily(sel), start, end)
    cols = [
        "modis_count", "modis_frp", "viirs_count", "viirs_frp", "harmonized_count", "harmonized_frp",
        "modis_hfai", "viirs_hfai", "harmonized_hfai",
    ]
    grouped = df.groupby(df["date"].dt.to_period(FREQ[freq]))[cols].sum(min_count=1)
    out = pd.DataFrame(
        {
            "period": grouped.index.astype(str),
            "modis_detections_observed": grouped["modis_count"].values,
            "modis_frp_mw_observed": grouped["modis_frp"].round(2).values,
            "viirs_snpp_detections_observed": grouped["viirs_count"].values,
            "viirs_snpp_frp_mw_observed": grouped["viirs_frp"].round(2).values,
            "harmonized_detections_estimate": grouped["harmonized_count"].round(2).values,
            "harmonized_frp_mw_estimate": grouped["harmonized_frp"].round(2).values,
            "hfai_modis_observed": grouped["modis_hfai"].round(3).values,
            "hfai_viirs_snpp_observed": grouped["viirs_hfai"].round(3).values,
            "hfai_harmonized_estimate": grouped["harmonized_hfai"].round(3).values,
        }
    )
    out["area"] = sel.name
    out["model_version"] = db.metadata()["active_model"]
    return out.to_csv(index=False)


def anomalies_csv(sel: Selection, start: str, end: str) -> str:
    events = anomaly_service.events(sel, start, end, "moderate", "both")
    frame = pd.DataFrame(
        [{k: v for k, v in e.items() if k != "explanation"} for e in events],
        columns=["start", "end", "days_flagged", "direction", "severity", "peak_date", "peak_score",
                 "peak_pct_vs_reference", "peak_hfai_7d", "reference_hfai", "seasonal_percentile"],
    )
    frame.insert(0, "area", sel.name)
    return frame.to_csv(index=False)


def grid_geojson(sel: Selection, start: str, end: str) -> dict:
    """Harmonized activity per cell with the observed counts alongside."""
    meta = db.metadata()
    est = cell_values(sel, start, end, "harmonized").set_index("cell_id")
    modis = cell_values(sel, start, end, "modis").set_index("cell_id")["count"]
    viirs = cell_values(sel, start, end, "viirs").set_index("cell_id")["count"]
    est["modis"] = modis.reindex(est.index).fillna(0)
    est["viirs"] = viirs.reindex(est.index).fillna(0)
    est = est[(est["count"] > 0.05) | (est["modis"] > 0) | (est["viirs"] > 0)]
    cells_before_cap = len(est)
    truncated = cells_before_cap > MAX_GEOJSON_CELLS
    if truncated:
        est = est.sort_values("hfai", ascending=False).head(MAX_GEOJSON_CELLS)
    viirs_available = end >= meta["viirs_start"]
    west, south, east, north = grid.cell_bounds(est["ix"].values, est["iy"].values, meta["cell_km"])
    features = []
    for i, (cell_id, row) in enumerate(est.iterrows()):
        w, s, e, n = round(west[i], 5), round(south[i], 5), round(east[i], 5), round(north[i], 5)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
                "properties": {
                    "cell_id": int(cell_id),
                    "harmonized_detections_estimate": round(float(row["count"]), 3),
                    "harmonized_frp_mw_estimate": round(float(row["frp"]), 2),
                    "hfai_harmonized_estimate": round(float(row["hfai"]), 3),
                    "modis_detections_observed": int(row["modis"]),
                    "viirs_snpp_detections_observed": int(row["viirs"]) if viirs_available else None,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "nyeleti_harmonized_activity",
        "metadata": {
            "area": sel.name,
            "start": start,
            "end": end,
            "cell_km": meta["cell_km"],
            "truncated": truncated,
            "cells_before_cap": int(cells_before_cap),
            "cell_cap": MAX_GEOJSON_CELLS,
            "crs_of_grid": config.GRID_CRS,
            "model_version": meta["active_model"],
            "note": "Properties ending in _estimate are model output; properties ending in _observed are "
                    "NASA FIRMS detections aggregated on the grid.",
        },
        "features": features,
    }


# ------------------------------------------------------------------------- PDF

def _png(fig: Figure) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    buffer.seek(0)
    return buffer


def _yearly_chart(trend: dict) -> io.BytesIO:
    fig = Figure(figsize=(7.2, 3.0))
    ax = fig.subplots()
    years = trend["years"]
    for source, label in (("modis", "MODIS observed"), ("viirs", "VIIRS S-NPP observed"),
                          ("harmonized", "Harmonized estimate")):
        y = [np.nan if v is None else v for v in trend["series"][source]]
        ax.plot(years, y, marker="o", ms=3, lw=1.3, color=COLOURS[source], label=label)
    ax.set_ylabel("HFAI per year")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return _png(fig)


def _season_chart(trend: dict) -> io.BytesIO:
    clim = trend["climatology"]
    fig = Figure(figsize=(7.2, 2.6))
    ax = fig.subplots()
    x = np.arange(12)
    ax.bar(x, clim["harmonized_mean"], color=COLOURS["harmonized"], alpha=0.85, label="Mean (harmonized)")
    ax.vlines(x, clim["harmonized_min"], clim["harmonized_max"], color="#444444", lw=1, label="Min-max across years")
    ax.set_xticks(x)
    ax.set_xticklabels(clim["months"], fontsize=8)
    ax.set_ylabel("HFAI per month")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _png(fig)


def _table(rows: list[list], widths=None, header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f6f6")]),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    table.setStyle(TableStyle(style))
    return table


def _fmt(value, digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not available"
    return f"{value:,.{digits}f}"


def pdf_report(sel: Selection, start: str, end: str) -> bytes:
    meta = db.metadata()
    df = series.daily(sel)
    rows = series.window(df, start, end)
    trend = analysis.trends(df, "hfai")
    critical = analysis.critical_periods(df, "hfai", 90, 3, 1, "area")
    try:
        events = sorted(
            anomaly_service.events(sel, start, end, "moderate", "above"), key=lambda e: e["peak_score"], reverse=True
        )
    except HTTPException:
        events = []
    model = db.cursor().execute("SELECT metrics FROM model_versions WHERE is_active").fetchone()
    metrics = json.loads(model[0]) if model and model[0] else None

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#555555"))
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, alignment=0, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title=f"NYELETI AI - {sel.name}", author="NYELETI AI")
    story = [
        Paragraph("NYELETI AI", h1),
        Paragraph("MODIS-VIIRS fire data harmonization and burning activity calendar", body),
        Spacer(1, 6),
        Paragraph(
            f"<b>Area:</b> {sel.name} ({sel.level}, about {sel.area_km2:,.0f} km2, {sel.n_cells:,} grid cells)<br/>"
            f"<b>Period:</b> {start} to {end}<br/>"
            f"<b>Harmonization model:</b> {meta['active_model']}<br/>"
            f"<b>Generated:</b> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
            body,
        ),
        Paragraph("How to read this report", h2),
        Paragraph(
            "Values labelled <b>observed</b> are active fire detections from NASA FIRMS (MODIS Collection 6.1 at "
            "1 km, VIIRS S-NPP at 375 m), aggregated on a common "
            f"{meta['cell_km']:g} km equal-area grid. Values labelled <b>estimate</b> come from the harmonization "
            "model, which converts MODIS activity to the VIIRS scale. Before "
            f"{meta['viirs_start']} there is no VIIRS record, so harmonized values for that period are model "
            "estimates that cannot be checked against VIIRS. HFAI is the Harmonized Fire Activity Index "
            "(see the methodology at the end).",
            body,
        ),
        Paragraph("Summary of the period", h2),
    ]

    def total(col):
        return rows[col].sum(min_count=1)

    story.append(
        _table(
            [
                ["Indicator", "Value", "Nature"],
                ["MODIS detections", _fmt(total("modis_count")), "observed"],
                ["VIIRS S-NPP detections", _fmt(total("viirs_count")), f"observed (from {meta['viirs_start']})"],
                ["Harmonized detections (VIIRS scale)", _fmt(total("harmonized_count")), "model estimate"],
                ["Harmonized HFAI", _fmt(total("harmonized_hfai"), 1), "model estimate"],
                ["MODIS FRP (MW)", _fmt(total("modis_frp")), "observed, MODIS scale"],
                ["VIIRS S-NPP FRP (MW)", _fmt(total("viirs_frp")), "observed, VIIRS scale"],
                ["Harmonized FRP (MW, VIIRS scale)", _fmt(total("harmonized_frp")), "model estimate"],
            ],
            widths=[7.5 * cm, 4 * cm, 5.5 * cm],
        )
    )
    story.append(Paragraph("MODIS and VIIRS FRP are not directly comparable and are reported separately.", small))

    story += [
        Paragraph("Fire activity by year", h2),
        Image(_yearly_chart(trend), width=17 * cm, height=7.1 * cm),
    ]
    if trend["trend"]:
        t = trend["trend"]
        verdict = "statistically significant" if t["significant"] else "not statistically significant"
        story.append(
            Paragraph(
                f"Trend of the harmonized series, {t['first_year']}-{t['last_year']}: "
                f"{t['pct_per_decade']:+.1f}% per decade (Theil-Sen), Mann-Kendall p = {t['p_value']:.3f}, {verdict}.",
                body,
            )
        )
    clim = trend["climatology"]
    story += [
        Paragraph("Seasonality", h2),
        Image(_season_chart(trend), width=17 * cm, height=6.1 * cm),
        Paragraph(
            f"Peak month: {clim['peak_month']}. The three most active months ({', '.join(clim['top3_months'])}) "
            f"hold {clim['top3_share_pct']}% of the annual harmonized activity "
            f"({clim['reference_years'][0]}-{clim['reference_years'][1]}).",
            body,
        ),
        PageBreak(),
        Paragraph("Critical burning periods", h2),
        Paragraph(critical["definition"], small),
        Spacer(1, 4),
    ]
    in_window = [p for p in critical["periods"] if start <= p["start"] <= end][:10]
    if in_window:
        story.append(
            _table(
                [["Start", "End", "Days", "Total HFAI", "Peak day", "Peak HFAI"]]
                + [[p["start"], p["end"], p["days"], _fmt(p["total"], 1), p["peak_date"], _fmt(p["peak_value"], 1)]
                   for p in in_window],
                widths=[3 * cm, 3 * cm, 1.8 * cm, 3 * cm, 3 * cm, 3 * cm],
            )
        )
    else:
        story.append(Paragraph("No critical period in the selected dates.", body))

    story += [Paragraph("Unusual activity", h2)]
    if events:
        story.append(
            _table(
                [["Start", "End", "Severity", "Peak day", "Vs seasonal reference", "Seasonal percentile"]]
                + [[e["start"], e["end"], e["severity"], e["peak_date"], f"{e['peak_pct_vs_reference']:+.0f}%",
                    f"{e['seasonal_percentile']:.1f}"] for e in events[:10]],
                widths=[2.7 * cm, 2.7 * cm, 2.2 * cm, 2.7 * cm, 3.6 * cm, 3.1 * cm],
            )
        )
        story.append(
            Paragraph(
                "Periods in which the harmonized activity departs from the area's own history for the same time "
                "of year (Isolation Forest on seasonal indicators). This is a description of observed activity, "
                "not a forecast.",
                small,
            )
        )
    else:
        story.append(Paragraph("No period of unusually high activity flagged in the selected dates.", body))

    story.append(Paragraph("Model evaluation", h2))
    if metrics:
        table = metrics["aggregated"]["test"]["province_day"]["hfai"]
        labels = metrics["labels"]
        test = metrics["test_years"]
        span = str(test[0]) if test[0] == test[1] else f"{test[0]}-{test[1]}"
        story.append(
            _table(
                [["Method", "MAE", "RMSE", "R2", "Bias %"]]
                + [[labels[m], _fmt(r["mae"], 3), _fmt(r["rmse"], 3), _fmt(r["r2"], 3), _fmt(r["bias_pct"], 1)]
                   for m, r in table.items()],
                widths=[8 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm],
            )
        )
        story.append(
            Paragraph(
                f"HFAI by province and day in the independent test period ({span}), against VIIRS S-NPP. "
                "These are national figures for the model, not specific to the area of this report.",
                small,
            )
        )
    else:
        story.append(Paragraph("Evaluation metrics are not available for this model version.", body))

    ref = meta["hfai"]
    story += [
        Paragraph("Methodology", h2),
        Paragraph(
            "1. MODIS (Terra, Aqua) and VIIRS S-NPP detections are downloaded from the NASA FIRMS archive, "
            "validated and de-duplicated; low-confidence detections and non-vegetation sources are excluded from "
            "the analysis.<br/>"
            f"2. Detections are aggregated by day on a {meta['cell_km']:g} km equal-area grid ({config.GRID_CRS}).<br/>"
            "3. A LightGBM model, trained on the years where both sensors exist, estimates the VIIRS detection "
            "count and FRP of each cell and day from the MODIS observations of the cell and its neighbourhood. "
            "A climatological background rate accounts for VIIRS activity where MODIS sees nothing nearby.<br/>"
            f"4. HFAI = {ref['weight_count']:g} x count / {ref['count_ref']:.2f} + {ref['weight_frp']:g} x FRP / "
            f"{ref['frp_ref']:.1f} MW, where the denominators are the mean count and FRP of an active VIIRS "
            f"cell-day in {ref['reference_years'][0]}-{ref['reference_years'][1]}.<br/>"
            f"5. Years before {config.ANALYSIS_FIRST_YEAR} have a single MODIS platform (Terra) for all or part of "
            "the year and are biased low; they are excluded from trends and seasonal references.",
            body,
        ),
        Spacer(1, 8),
        Paragraph(
            "Data: NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/). Boundaries: geoBoundaries. "
            "We acknowledge the use of data from NASA's Fire Information for Resource Management System (FIRMS), "
            "part of NASA's Earth Science Data and Information System (ESDIS).",
            small,
        ),
    ]
    doc.build(story)
    return buffer.getvalue()
