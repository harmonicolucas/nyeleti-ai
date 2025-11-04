"""Reference methods the LightGBM model has to beat."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ml.preprocessing import features as ft


class LinearCalibration:
    """Ordinary least squares on the MODIS count/FRP features, clipped at zero."""

    def __init__(self, target: str):
        self.target = target
        self.columns = ft.LINEAR_FEATURES[target]
        self.model = LinearRegression()

    def fit(self, rows: pd.DataFrame, y: np.ndarray) -> "LinearCalibration":
        self.model.fit(rows[self.columns].values.astype("float64"), y)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(rows[self.columns].values.astype("float64")), 0, None)

    def describe(self) -> dict:
        return {
            "intercept": float(self.model.intercept_),
            "coefficients": {c: float(v) for c, v in zip(self.columns, self.model.coef_)},
        }


class QuantileMapping:
    """Empirical quantile mapping of the in-cell MODIS value onto the VIIRS distribution.

    Tied MODIS values (most rows have a small integer count) are mapped to the
    mean of the VIIRS quantiles they span, which keeps the mapped mean equal to
    the VIIRS mean on the training data.
    """

    def __init__(self, target: str, max_knots: int = 2000):
        self.target = target
        self.column = ft.RAW_MODIS[target]
        self.max_knots = max_knots

    def fit(self, rows: pd.DataFrame, y: np.ndarray) -> "QuantileMapping":
        xs = np.sort(rows[self.column].values.astype("float64"))
        ys = np.sort(np.asarray(y, dtype="float64"))
        n = len(xs)
        ux, first = np.unique(xs, return_index=True)
        last = np.append(first[1:], n)
        if len(ux) > self.max_knots:
            chunk = first // max(1, n // self.max_knots)
            _, chunk = np.unique(chunk, return_inverse=True)
        else:
            chunk = np.arange(len(ux))
        weights = (last - first).astype("float64")
        y_cum = np.concatenate([[0.0], np.cumsum(ys)])
        groups = pd.DataFrame({"chunk": chunk, "first": first, "last": last, "wx": ux * weights, "w": weights})
        agg = groups.groupby("chunk").agg(first=("first", "min"), last=("last", "max"), wx=("wx", "sum"), w=("w", "sum"))
        self.x_ = (agg["wx"] / agg["w"]).values
        self.y_ = ((y_cum[agg["last"].values] - y_cum[agg["first"].values]) / agg["w"].values)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.interp(rows[self.column].values.astype("float64"), self.x_, self.y_)


def raw_modis(rows: pd.DataFrame, target: str) -> np.ndarray:
    """No harmonization: the MODIS value taken as if it were the VIIRS value."""
    return rows[ft.RAW_MODIS[target]].values.astype("float64")
