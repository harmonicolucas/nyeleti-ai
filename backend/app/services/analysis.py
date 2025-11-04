"""Calendar, trend and critical-period statistics computed from a daily series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from backend.app.database import db
from backend.app.services.series import SOURCES, number, split_of, values
from ml import config

CLASS_PERCENTILES = {"medium": 50, "high": 80, "very_high": 95}
CLASS_NAMES = ["low", "medium", "high", "very_high"]
SEASON_NAMES = ["DJF", "MAM", "JJA", "SON"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Years with full MODIS coverage, used as the historical reference."""
    return df[df["year"] >= config.ANALYSIS_FIRST_YEAR]


def class_thresholds(reference: np.ndarray) -> dict:
    reference = reference[~np.isnan(reference)]
    if len(reference) == 0:
        return {name: 0.0 for name in CLASS_PERCENTILES}
    return {name: float(np.percentile(reference, p)) for name, p in CLASS_PERCENTILES.items()}


def classify(value: np.ndarray, thresholds: dict) -> list:
    value = np.asarray(value, dtype="float64")
    idx = np.zeros(len(value), dtype="int64")
    for i, name in enumerate(("medium", "high", "very_high"), start=1):
        idx[value > thresholds[name]] = i
    return [None if np.isnan(v) else CLASS_NAMES[i] for v, i in zip(value, idx)]


def _columns(metric: str) -> dict:
    return {source: f"{source}_{metric}" for source in SOURCES}


def _sum(frame: pd.DataFrame, by, cols) -> pd.DataFrame:
    # min_count=1 keeps periods without any VIIRS record as missing instead of 0
    return frame.groupby(by)[cols].sum(min_count=1)


def calendar_days(df: pd.DataFrame, metric: str, year: int) -> dict:
    cols = _columns(metric)
    thresholds = class_thresholds(analysis_rows(df)[cols["harmonized"]].values)
    rows = df[df["year"] == year]
    return {
        "granularity": "day",
        "year": year,
        "dates": rows["date"].dt.strftime("%Y-%m-%d").tolist(),
        "series": {source: values(rows[col]) for source, col in cols.items()},
        "classes": classify(rows[cols["harmonized"]].values, thresholds),
        "thresholds": {k: number(v) for k, v in thresholds.items()},
        "class_basis": "Classes: percentiles (50, 80, 95) of the area's daily harmonized values, "
                       f"{config.ANALYSIS_FIRST_YEAR} onwards",
    }


def calendar_months(df: pd.DataFrame, metric: str) -> dict:
    cols = _columns(metric)
    monthly = _sum(df, ["year", "month"], list(cols.values()))
    years = sorted(df["year"].unique().tolist())
    index = pd.MultiIndex.from_product([years, range(1, 13)], names=["year", "month"])
    monthly = monthly.reindex(index)
    reference = monthly.loc[monthly.index.get_level_values("year") >= config.ANALYSIS_FIRST_YEAR, cols["harmonized"]]
    thresholds = class_thresholds(reference.values)

    def matrix(col):
        flat = values(monthly[col])
        return [flat[i * 12:(i + 1) * 12] for i in range(len(years))]

    classes = classify(monthly[cols["harmonized"]].values, thresholds)
    return {
        "granularity": "month",
        "years": years,
        "months": MONTH_NAMES,
        "series": {source: matrix(col) for source, col in cols.items()},
        "classes": [classes[i * 12:(i + 1) * 12] for i in range(len(years))],
        "thresholds": {k: number(v) for k, v in thresholds.items()},
        "class_basis": "Classes: percentiles (50, 80, 95) of the area's monthly harmonized values, "
                       f"{config.ANALYSIS_FIRST_YEAR} onwards",
    }


def calendar_seasons(df: pd.DataFrame, metric: str) -> dict:
    cols = _columns(metric)
    # December counts towards the DJF season of the following year
    season_year = df["year"] + (df["month"] == 12).astype(int)
    season = (df["month"] % 12) // 3
    grouped = df.groupby([season_year.rename("season_year"), season.rename("season")])
    totals = grouped[list(cols.values())].sum(min_count=1)
    n_days = grouped.size()
    expected = {0: 90, 1: 92, 2: 92, 3: 91}
    thresholds = class_thresholds(
        totals.loc[totals.index.get_level_values("season_year") >= config.ANALYSIS_FIRST_YEAR, cols["harmonized"]].values
    )
    rows = []
    classes = classify(totals[cols["harmonized"]].values, thresholds)
    for (year, s), label in zip(totals.index, classes):
        rows.append(
            {
                "season_year": int(year),
                "season": SEASON_NAMES[s],
                "complete": bool(n_days[(year, s)] >= expected[s]),
                "class": label,
                **{source: number(totals.loc[(year, s), col]) for source, col in cols.items()},
            }
        )
    return {
        "granularity": "season",
        "seasons": rows,
        "thresholds": {k: number(v) for k, v in thresholds.items()},
        "class_basis": "Classes: percentiles (50, 80, 95) of the area's seasonal harmonized totals, "
                       f"{config.ANALYSIS_FIRST_YEAR} onwards",
    }


def yearly_totals(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    cols = _columns(metric)
    totals = _sum(df, "year", list(cols.values())).rename(columns={v: k for k, v in cols.items()})
    totals["viirs_days"] = df.groupby("year")[cols["viirs"]].count()
    totals["days"] = df.groupby("year").size()
    return totals


def calendar_years(df: pd.DataFrame, metric: str) -> dict:
    totals = yearly_totals(df, metric)
    reference = totals.loc[totals.index >= config.ANALYSIS_FIRST_YEAR, "harmonized"]
    thresholds = class_thresholds(reference.values)
    meta = db.metadata()
    return {
        "granularity": "year",
        "years": [int(y) for y in totals.index],
        "series": {source: values(totals[source]) for source in SOURCES},
        "classes": classify(totals["harmonized"].values, thresholds),
        "viirs_complete": [bool(a == b) for a, b in zip(totals["viirs_days"], totals["days"])],
        "modis_full_coverage": [bool(y >= config.ANALYSIS_FIRST_YEAR) for y in totals.index],
        "split": [split_of(int(y)) for y in totals.index],
        "thresholds": {k: number(v) for k, v in thresholds.items()},
        "viirs_start": meta["viirs_start"],
    }


def trends(df: pd.DataFrame, metric: str) -> dict:
    cols = _columns(metric)
    totals = yearly_totals(df, metric)
    ref = totals[totals.index >= config.ANALYSIS_FIRST_YEAR]
    years = ref.index.values.astype("float64")
    y = ref["harmonized"].values

    trend = None
    if len(y) >= 5 and np.ptp(y) > 0:
        slope, intercept, low, high = stats.theilslopes(y, years, 0.95)
        tau = stats.kendalltau(years, y)
        mean = float(np.mean(y))
        trend = {
            "method": "Theil-Sen slope with Mann-Kendall test",
            "first_year": int(years[0]),
            "last_year": int(years[-1]),
            "slope_per_year": number(slope),
            "slope_ci95": [number(low), number(high)],
            "pct_per_decade": number(1000.0 * slope / mean, 1) if mean else None,
            "kendall_tau": number(tau.statistic),
            "p_value": number(tau.pvalue, 4),
            "significant": bool(tau.pvalue < 0.05),
            "fitted": values(intercept + slope * totals.index.values.astype("float64")),
        }

    hist = analysis_rows(df)
    monthly = _sum(hist, ["year", "month"], list(cols.values()))
    clim = monthly.groupby("month").agg(["mean", "min", "max"])
    harm_mean = clim[(cols["harmonized"], "mean")]
    annual = float(harm_mean.sum())
    order = harm_mean.sort_values(ascending=False)
    climatology = {
        "months": MONTH_NAMES,
        "harmonized_mean": values(harm_mean),
        "harmonized_min": values(clim[(cols["harmonized"], "min")]),
        "harmonized_max": values(clim[(cols["harmonized"], "max")]),
        "viirs_mean": values(clim[(cols["viirs"], "mean")]),
        "modis_mean": values(clim[(cols["modis"], "mean")]),
        "share_pct": values(100 * harm_mean / annual if annual else harm_mean * 0, 1),
        "peak_month": MONTH_NAMES[int(order.index[0]) - 1],
        "top3_months": [MONTH_NAMES[int(m) - 1] for m in order.index[:3]],
        "top3_share_pct": number(100 * order.iloc[:3].sum() / annual, 1) if annual else None,
        "reference_years": [config.ANALYSIS_FIRST_YEAR, int(df["year"].max())],
    }

    timing = []
    for year, rows in hist.groupby("year"):
        v = rows[cols["harmonized"]].values
        total = v.sum()
        if total <= 0:
            continue
        cum = np.cumsum(v) / total
        dates = rows["date"].values
        marks = {f"p{p}": pd.Timestamp(dates[int(np.searchsorted(cum, p / 100))]).strftime("%Y-%m-%d") for p in (10, 50, 90)}
        peak = int(np.argmax(v))
        timing.append(
            {
                "year": int(year),
                **marks,
                "p10_doy": int(pd.Timestamp(marks["p10"]).day_of_year),
                "p50_doy": int(pd.Timestamp(marks["p50"]).day_of_year),
                "p90_doy": int(pd.Timestamp(marks["p90"]).day_of_year),
                "peak_date": pd.Timestamp(dates[peak]).strftime("%Y-%m-%d"),
                "peak_value": number(v[peak]),
            }
        )

    return {
        "metric": metric,
        "years": [int(v) for v in totals.index],
        "series": {source: values(totals[source]) for source in SOURCES},
        "viirs_complete": [bool(a == b) for a, b in zip(totals["viirs_days"], totals["days"])],
        "modis_full_coverage": [bool(v >= config.ANALYSIS_FIRST_YEAR) for v in totals.index],
        "trend": trend,
        "climatology": climatology,
        "season_timing": timing,
    }


def _runs(flag: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Index ranges of consecutive flagged days, bridging gaps up to `max_gap` days."""
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def critical_periods(
    df: pd.DataFrame, metric: str, percentile: float, min_days: int, max_gap: int, basis: str
) -> dict:
    col = f"harmonized_{metric}"
    hist = analysis_rows(df)
    v = df[col].values.astype("float64")
    dates = pd.DatetimeIndex(df["date"])

    if basis == "month":
        by_month = hist.groupby("month")[col].quantile(percentile / 100)
        limit = by_month.reindex(df["month"]).values
        threshold_info = {"by_month": values(by_month.reindex(range(1, 13)))}
        basis_text = (
            f"{percentile:g}th percentile of the area's daily harmonized values in the same calendar month, "
            f"{config.ANALYSIS_FIRST_YEAR} onwards"
        )
    else:
        single = float(np.percentile(hist[col].values, percentile))
        limit = np.full(len(df), single)
        threshold_info = {"value": number(single)}
        basis_text = (
            f"{percentile:g}th percentile of all daily harmonized values of the area, "
            f"{config.ANALYSIS_FIRST_YEAR} onwards"
        )

    above = (v > limit) & (v > 0)
    periods = []
    for start, end in _runs(above, max_gap):
        n_above = int(above[start:end + 1].sum())
        if n_above < min_days:
            continue
        segment = v[start:end + 1]
        peak = start + int(np.argmax(segment))
        periods.append(
            {
                "start": dates[start].strftime("%Y-%m-%d"),
                "end": dates[end].strftime("%Y-%m-%d"),
                "year": int(dates[start].year),
                "days": int(end - start + 1),
                "days_above": n_above,
                "total": number(segment.sum()),
                "mean": number(segment.mean()),
                "peak_date": dates[peak].strftime("%Y-%m-%d"),
                "peak_value": number(v[peak]),
                "peak_over_threshold": number(v[peak] / limit[peak], 2) if limit[peak] > 0 else None,
            }
        )

    flagged = pd.DataFrame({"year": dates.year, "month": dates.month, "above": above, "date": dates})
    by_year = []
    period_frame = pd.DataFrame(periods)
    for year, rows in flagged.groupby("year"):
        hits = rows[rows["above"]]
        own = period_frame[period_frame["year"] == year] if len(period_frame) else period_frame
        by_year.append(
            {
                "year": int(year),
                "days_above": int(rows["above"].sum()),
                "periods": int(len(own)),
                "longest_days": int(own["days"].max()) if len(own) else 0,
                "first_day": hits["date"].min().strftime("%Y-%m-%d") if len(hits) else None,
                "last_day": hits["date"].max().strftime("%Y-%m-%d") if len(hits) else None,
                "modis_full_coverage": bool(year >= config.ANALYSIS_FIRST_YEAR),
            }
        )

    ref = flagged[flagged["year"] >= config.ANALYSIS_FIRST_YEAR]
    n_years = max(ref["year"].nunique(), 1)
    month_days = ref.groupby("month")["above"].sum().reindex(range(1, 13), fill_value=0) / n_years

    return {
        "metric": metric,
        "percentile": percentile,
        "min_days": min_days,
        "max_gap": max_gap,
        "basis": basis,
        "definition": (
            f"A day is above threshold when its harmonized {metric.upper()} exceeds the {basis_text}. "
            f"A critical period is a run with at least {min_days} such days, allowing gaps of up to "
            f"{max_gap} day(s)."
        ),
        "threshold": threshold_info,
        "periods": sorted(periods, key=lambda p: p["total"] or 0, reverse=True),
        "by_year": by_year,
        "mean_days_above_by_month": {"months": MONTH_NAMES, "days": values(month_days, 1)},
    }
