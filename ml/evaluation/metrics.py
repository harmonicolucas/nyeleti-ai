"""Error metrics shared by training and evaluation."""
from __future__ import annotations

import numpy as np
from scipy import stats


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_pred, dtype="float64")
    err = p - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    mean_y = float(y.mean())
    if len(y) > 2 and np.ptp(y) > 0 and np.ptp(p) > 0:
        rho = float(stats.spearmanr(y, p).statistic)
    else:
        rho = float("nan")
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": float(err.mean()),
        "bias_pct": float(100.0 * err.mean() / mean_y) if mean_y else float("nan"),
        "spearman": rho,
        "mean_observed": mean_y,
        "mean_predicted": float(p.mean()),
    }


def detection_metrics(observed_high, predicted_high) -> dict:
    o = np.asarray(observed_high, dtype=bool)
    p = np.asarray(predicted_high, dtype=bool)
    tp = int(np.sum(o & p))
    fp = int(np.sum(~o & p))
    fn = int(np.sum(o & ~p))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "observed_high": int(o.sum()),
        "predicted_high": int(p.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
