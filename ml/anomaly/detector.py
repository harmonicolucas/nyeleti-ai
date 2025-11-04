"""Unusual fire activity for a daily series of one area.

The detector compares each day with what is normal for that area at that time
of year, then scores the result with an Isolation Forest. It describes
observed activity; it does not forecast fires.

Seasonal reference of a day: the same calendar window (+/- 15 days) in all
other years of the record. The year being scored is left out of its own
reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml import config

WINDOW_DAYS = 15
SMOOTH_DAYS = 7
FLOOR_SHARE = 0.10  # floor added to ratios, as a share of the area's mean daily activity
MERGE_GAP_DAYS = 2

FEATURES = ["log_ratio", "seasonal_rank", "count_log_ratio", "frp_log_ratio", "day_spike", "level"]
SEVERITY_QUANTILES = {"moderate": 0.98, "high": 0.995, "extreme": 0.999}
SEVERITY_ORDER = ["moderate", "high", "extreme"]

ARTIFACT_DIR = config.ARTIFACTS_DIR / "anomaly"


def _doy(dates: pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 365-day calendar (29 February shares the slot of 28 February)."""
    doy = dates.day_of_year.values.copy()
    leap_after_feb = dates.is_leap_year & (doy > 59)
    doy[leap_after_feb] -= 1
    return doy - 1


def seasonal_reference(values: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-year-out seasonal mean and percentile rank of each value."""
    years = dates.year.values
    year_ids, year_idx = np.unique(years, return_inverse=True)
    doy = _doy(dates)
    n_years = len(year_ids)

    total = np.zeros((n_years, 365))
    count = np.zeros((n_years, 365))
    np.add.at(total, (year_idx, doy), values)
    np.add.at(count, (year_idx, doy), 1.0)
    with np.errstate(invalid="ignore"):
        matrix = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    offsets = np.arange(-WINDOW_DAYS, WINDOW_DAYS + 1)
    window_idx = (np.arange(365)[:, None] + offsets[None, :]) % 365
    windows = matrix[:, window_idx]  # years x 365 x window
    win_sum = np.nansum(windows, axis=2)
    win_n = np.sum(~np.isnan(windows), axis=2)
    all_sum, all_n = win_sum.sum(axis=0), win_n.sum(axis=0)
    other_n = all_n[None, :] - win_n
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_mean = np.where(other_n > 0, (all_sum[None, :] - win_sum) / other_n, np.nan)

    rank = np.full((n_years, 365), np.nan)
    for d in range(365):
        pool = windows[:, d, :]
        flat = np.sort(pool[~np.isnan(pool)])
        own_sorted = np.sort(np.where(np.isnan(pool), np.inf, pool), axis=1)
        v = matrix[:, d]
        ok = ~np.isnan(v)
        below_all = np.searchsorted(flat, v[ok], side="left")
        below_own = np.array([np.searchsorted(own_sorted[i], v[i], side="left") for i in np.flatnonzero(ok)])
        denom = len(flat) - win_n[ok, d]
        rank[ok, d] = np.where(denom > 0, (below_all - below_own) / np.maximum(denom, 1), np.nan)

    return ref_mean[year_idx, doy], rank[year_idx, doy]


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Anomaly features for one area.

    `daily` holds one row per day, in order, with columns date, hfai, count, frp
    (harmonized values).
    """
    dates = pd.DatetimeIndex(daily["date"])
    out = pd.DataFrame({"date": dates})
    smooth = {}
    for name in ("hfai", "count", "frp"):
        series = pd.Series(daily[name].values.astype("float64"))
        smooth[name] = series.rolling(SMOOTH_DAYS, min_periods=1).mean().values

    refs, floors = {}, {}
    rank = None
    for name in ("hfai", "count", "frp"):
        ref, r = seasonal_reference(smooth[name], dates)
        refs[name] = ref
        floors[name] = max(FLOOR_SHARE * float(np.mean(daily[name].values)), 1e-9)
        if name == "hfai":
            rank = r

    def log_ratio(value, ref, floor):
        return np.log((value + floor) / (np.nan_to_num(ref, nan=0.0) + floor))

    level_ref = max(float(np.quantile(smooth["hfai"], 0.99)), 1e-9)
    out["hfai"] = daily["hfai"].values
    out["hfai_smooth"] = smooth["hfai"]
    out["reference"] = refs["hfai"]
    out["log_ratio"] = log_ratio(smooth["hfai"], refs["hfai"], floors["hfai"])
    out["seasonal_rank"] = np.nan_to_num(rank, nan=0.5)
    out["count_log_ratio"] = log_ratio(smooth["count"], refs["count"], floors["count"])
    out["frp_log_ratio"] = log_ratio(smooth["frp"], refs["frp"], floors["frp"])
    out["day_spike"] = log_ratio(daily["hfai"].values.astype("float64"), smooth["hfai"], floors["hfai"])
    out["level"] = np.clip(smooth["hfai"] / level_ref, 0, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pct_vs_reference"] = 100.0 * (smooth["hfai"] / (refs["hfai"] + floors["hfai"]) - 1.0)
    return out


class AnomalyDetector:
    """Isolation Forest with severity thresholds per administrative level.

    Daily series of small areas are noisier than those of large ones, so the
    scores of a district are compared with other districts, those of a
    province with other provinces, and so on.
    """

    def __init__(self, forest, thresholds: dict, meta: dict | None = None):
        self.forest = forest
        self.thresholds = thresholds
        self.meta = meta or {}

    def thresholds_for(self, level: str, n_cells: int | None = None) -> dict:
        if level in self.thresholds:
            return self.thresholds[level]
        # user-drawn area: use the level whose typical size is closest
        sizes = self.meta.get("level_cells", {})
        if not sizes or not n_cells:
            return self.thresholds[sorted(self.thresholds)[-1]]
        nearest = min(sizes, key=lambda name: abs(np.log(max(sizes[name], 1)) - np.log(max(n_cells, 1))))
        return self.thresholds[nearest]

    @classmethod
    def load(cls, directory: Path | None = None) -> "AnomalyDetector":
        directory = directory or ARTIFACT_DIR
        meta = json.loads((directory / "anomaly.json").read_text(encoding="utf-8"))
        return cls(joblib.load(directory / "isolation_forest.joblib"), meta["thresholds"], meta)

    def raw_scores(self, features: pd.DataFrame) -> np.ndarray:
        # higher score = more unusual
        return -self.forest.score_samples(features[FEATURES].values)

    def score(self, features: pd.DataFrame, level: str, n_cells: int | None = None,
              scores: np.ndarray | None = None) -> pd.DataFrame:
        """Add the anomaly score, the severity and the direction of each day."""
        out = features.copy()
        out["score"] = self.raw_scores(out) if scores is None else scores
        limits = self.thresholds_for(level, n_cells)
        severity = np.full(len(out), None, dtype=object)
        for name in SEVERITY_ORDER:
            severity[out["score"].values >= limits[name]] = name
        out["severity"] = severity
        out["direction"] = np.where(out["log_ratio"] >= 0, "above", "below")
        return out


def explain(row) -> list[dict]:
    """The indicators behind an alert, in plain terms."""
    items = [
        {
            "key": "activity",
            "indicator": "7-day activity vs seasonal reference",
            "value": f"{row['pct_vs_reference']:+.0f}%",
            "detail": f"7-day mean HFAI {row['hfai_smooth']:.2f} against {row['reference']:.2f} "
                      "for this time of year in the other years",
            "mean": f"{row['hfai_smooth']:.2f}",
            "reference": f"{row['reference']:.2f}",
        },
        {
            "key": "percentile",
            "indicator": "Seasonal percentile",
            "value": f"{100 * row['seasonal_rank']:.1f}",
            "detail": "share of comparable days (same 31-day window, other years) with lower activity",
        },
        {
            "key": "count",
            "indicator": "Detections vs reference",
            "value": f"x{np.exp(row['count_log_ratio']):.2f}",
            "detail": "ratio of the 7-day harmonized detection count to its seasonal reference",
        },
        {
            "key": "frp",
            "indicator": "FRP vs reference",
            "value": f"x{np.exp(row['frp_log_ratio']):.2f}",
            "detail": "ratio of the 7-day harmonized FRP to its seasonal reference",
        },
        {
            "key": "spike",
            "indicator": "Day vs its own week",
            "value": f"x{np.exp(row['day_spike']):.2f}",
            "detail": "activity of the day relative to the mean of the last 7 days",
        },
    ]
    return items


def to_events(scored: pd.DataFrame) -> list[dict]:
    """Merge consecutive flagged days into events."""
    flagged = scored[scored["severity"].notna()]
    if flagged.empty:
        return []
    events = []
    current = None
    for row in flagged.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if (
            current is not None
            and row.direction == current["direction"]
            and (date - current["end"]).days <= MERGE_GAP_DAYS + 1
        ):
            current["rows"].append(row)
            current["end"] = date
        else:
            if current is not None:
                events.append(current)
            current = {"direction": row.direction, "start": date, "end": date, "rows": [row]}
    events.append(current)

    out = []
    for ev in events:
        rows = ev["rows"]
        peak = max(rows, key=lambda r: r.score)
        severity = max((r.severity for r in rows), key=SEVERITY_ORDER.index)
        out.append(
            {
                "start": ev["start"].strftime("%Y-%m-%d"),
                "end": ev["end"].strftime("%Y-%m-%d"),
                "days_flagged": len(rows),
                "direction": ev["direction"],
                "severity": severity,
                "peak_date": pd.Timestamp(peak.date).strftime("%Y-%m-%d"),
                "peak_score": float(peak.score),
                "peak_pct_vs_reference": float(peak.pct_vs_reference),
                "peak_hfai_7d": float(peak.hfai_smooth),
                "reference_hfai": float(peak.reference),
                "seasonal_percentile": float(100 * peak.seasonal_rank),
                "explanation": explain(peak._asdict()),
            }
        )
    return out
