"""Figures for the documentation, drawn from the application database.

    python -m scripts.make_figures

Writes to docs/figures/:
  historical_activity.png   yearly activity 2001 onwards, observed and harmonized
  calendar_province.png     monthly burning calendar of the most active province
  anomaly_examples.png      two periods flagged by the anomaly model
"""
from __future__ import annotations

import json
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from ml import config
from ml.anomaly import detector as det

MODIS, VIIRS, HARMONIZED = "#1f77b4", "#111111", "#d95f02"
CLASS_COLORS = ["#e8e2da", "#f2c14e", "#f28e2b", "#d62728"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def daily_series(con, area_id: str, ref: dict) -> pd.DataFrame:
    df = con.execute(
        "SELECT date, modis_count, viirs_count, harmonized_count, harmonized_frp "
        "FROM calendar_activity WHERE area_id = ? ORDER BY date",
        [area_id],
    ).df()
    df["date"] = pd.to_datetime(df["date"])
    df["hfai"] = config.hfai(df["harmonized_count"], df["harmonized_frp"], ref)
    return df


def fig_history(con, meta: dict, path) -> None:
    df = daily_series(con, config.COUNTRY_ISO3, meta["hfai"])
    year = df["date"].dt.year
    totals = df.groupby(year)[["modis_count", "harmonized_count"]].sum()
    viirs = df.groupby(year)["viirs_count"].agg(["sum", "count"])
    days = df.groupby(year).size()
    viirs_full = viirs["sum"].where(viirs["count"] >= days - 5)

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
    ax.axvspan(totals.index.min() - 0.5, config.ANALYSIS_FIRST_YEAR - 0.5, color="#999999", alpha=0.15,
               label="partial MODIS coverage (Terra only until mid-2002)")
    ax.plot(totals.index, totals["harmonized_count"] / 1e3, color=HARMONIZED, marker="o", lw=2,
            label="Harmonized estimate (VIIRS scale)")
    ax.plot(viirs_full.index, viirs_full / 1e3, color=VIIRS, marker="s", lw=1.6, label="VIIRS S-NPP observed")
    ax.plot(totals.index, totals["modis_count"] / 1e3, color=MODIS, marker="o", lw=1.6, label="MODIS observed")
    ax.axvline(pd.Timestamp(meta["viirs_start"]).year - 0.5, color="#555555", ls=":", lw=1)
    ax.text(pd.Timestamp(meta["viirs_start"]).year - 0.4, ax.get_ylim()[1] * 0.03, "VIIRS record starts", fontsize=8, color="#555555")
    ax.set_ylabel("Detections per year (thousands)")
    ax.set_title(f"{meta['country']}: yearly active fire detections, observed and harmonized", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.text(0.01, 0.01, "VIIRS years with more than 5 missing days are not drawn. Before the VIIRS record the harmonized "
             "series is a model estimate that cannot be verified.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_calendar(con, meta: dict, path) -> str:
    area_id, name = con.execute(
        """
        SELECT g.area_id, g.name
        FROM calendar_activity c JOIN geographic_areas g USING (area_id)
        WHERE g.level = 'ADM1' AND year(c.date) >= ?
        GROUP BY g.area_id, g.name ORDER BY sum(c.harmonized_count) DESC LIMIT 1
        """,
        [config.ANALYSIS_FIRST_YEAR],
    ).fetchall()[0]
    df = daily_series(con, area_id, meta["hfai"])
    monthly = df.groupby([df["date"].dt.year.rename("year"), df["date"].dt.month.rename("month")])["hfai"].sum().unstack("month")
    reference = monthly.loc[monthly.index >= config.ANALYSIS_FIRST_YEAR].values.ravel()
    limits = np.percentile(reference, [50, 80, 95])
    classes = np.digitize(monthly.values, limits, right=True)

    fig, ax = plt.subplots(figsize=(9, 7.2), dpi=130)
    cmap = ListedColormap(CLASS_COLORS)
    ax.imshow(classes, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N), aspect="auto")
    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(len(monthly)), [f"{y}*" if y < config.ANALYSIS_FIRST_YEAR else str(y) for y in monthly.index])
    for i in range(classes.shape[0]):
        for j in range(12):
            value = monthly.values[i, j]
            text = f"{value / 1e3:.1f}k" if value >= 1000 else f"{value:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color="white" if classes[i, j] == 3 else "#222222")
    ax.set_xticks(np.arange(-0.5, 12), minor=True)
    ax.set_yticks(np.arange(-0.5, len(monthly)), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="both", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS]
    labels = ["Low", f"Medium (> {limits[0]:,.0f})", f"High (> {limits[1]:,.0f})", f"Very high (> {limits[2]:,.0f})"]
    ax.legend(handles, labels, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False)
    ax.set_title(f"Burning activity calendar, {name} province: monthly harmonized HFAI", fontsize=11)
    fig.text(0.01, 0.01, f"Classes: percentiles 50, 80 and 95 of the monthly values from {config.ANALYSIS_FIRST_YEAR}. "
             "* partial MODIS coverage. Harmonized values are model estimates on the VIIRS scale.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return name


def fig_anomalies(con, meta: dict, path) -> list[str]:
    detector = det.AnomalyDetector.load()
    picks = con.execute(
        """
        SELECT a.area_id, g.name, g.level, year(a.date) AS year
        FROM fire_anomalies a JOIN geographic_areas g USING (area_id)
        WHERE a.direction = 'above' AND g.level IN ('ADM0', 'ADM1')
        QUALIFY row_number() OVER (PARTITION BY g.level ORDER BY a.score DESC) = 1
        ORDER BY g.level
        """
    ).fetchall()
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3.6 * len(picks)), dpi=130, squeeze=False)
    titles = []
    for ax, (area_id, name, level, year) in zip(axes[:, 0], picks):
        df = daily_series(con, area_id, meta["hfai"])
        df = df[df["date"].dt.year >= config.ANALYSIS_FIRST_YEAR]
        daily = df.rename(columns={"harmonized_count": "count", "harmonized_frp": "frp"})[["date", "hfai", "count", "frp"]]
        scored = detector.score(det.build_features(daily), level)
        rows = scored[scored["date"].dt.year == year]
        ax.bar(rows["date"], rows["hfai"], width=1.0, color=HARMONIZED, alpha=0.25, label="daily harmonized HFAI")
        ax.plot(rows["date"], rows["reference"], color="#555555", ls="--", lw=1.3, label="seasonal reference (other years)")
        ax.plot(rows["date"], rows["hfai_smooth"], color=HARMONIZED, lw=2, label="7-day mean")
        for severity, colour in (("moderate", "#e6b800"), ("high", "#f28e2b"), ("extreme", "#d62728")):
            hit = rows[(rows["severity"] == severity) & (rows["direction"] == "above")]
            if len(hit):
                ax.scatter(hit["date"], hit["hfai_smooth"], color=colour, s=22, zorder=3, edgecolor="black", lw=0.4,
                           label=f"flagged: {severity}")
        peak = rows.loc[rows["score"].idxmax()]
        title = f"{name}, {year}: peak on {peak['date']:%d %b}, {peak['pct_vs_reference']:+.0f}% against the seasonal reference"
        titles.append(title)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("HFAI")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    fig.text(0.01, 0.005, "Isolation Forest on seasonal features of the harmonized series. Flags describe unusual past "
             "activity; they are not forecasts.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return titles


def main() -> int:
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM metadata").fetchall()}
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_history(con, meta, config.FIGURES_DIR / "historical_activity.png")
    print("historical_activity.png")
    name = fig_calendar(con, meta, config.FIGURES_DIR / "calendar_province.png")
    print(f"calendar_province.png ({name})")
    for title in fig_anomalies(con, meta, config.FIGURES_DIR / "anomaly_examples.png"):
        print(f"anomaly_examples.png: {title}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
