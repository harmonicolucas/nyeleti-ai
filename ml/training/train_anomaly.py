"""Fit the Isolation Forest for unusual fire activity and score every area.

Input: the harmonized daily series of the country, provinces and districts
(table calendar_activity). Output: ml/artifacts/anomaly/ and the table
fire_anomalies with one row per flagged area-day.

One forest is fitted on all areas. Severity thresholds are quantiles of the
scores within each administrative level, because the daily series of a
district is much noisier than that of a province or of the country.

    python -m ml.training.train_anomaly
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml import config
from ml.anomaly import detector as det


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=4096)
    args = parser.parse_args(argv)

    con = duckdb.connect(str(config.DB_PATH))
    meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM metadata").fetchall()}
    calendar = con.execute(
        """
        SELECT area_id, date, harmonized_count AS count, harmonized_frp AS frp
        FROM calendar_activity WHERE year(date) >= ? ORDER BY area_id, date
        """,
        [config.ANALYSIS_FIRST_YEAR],
    ).df()
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["hfai"] = config.hfai(calendar["count"], calendar["frp"], meta["hfai"])

    areas = con.execute("SELECT area_id, level, coalesce(n_cells, 0) AS n_cells FROM geographic_areas").df()
    levels = dict(zip(areas["area_id"], areas["level"]))

    parts = []
    for area_id, daily in calendar.groupby("area_id", sort=False):
        if daily["hfai"].sum() <= 0:
            continue
        feats = det.build_features(daily)
        feats.insert(0, "area_id", area_id)
        feats.insert(1, "area_level", levels[area_id])
        parts.append(feats)
    features = pd.concat(parts, ignore_index=True)
    print(f"{len(features):,} area-days from {features['area_id'].nunique()} areas", flush=True)

    forest = IsolationForest(
        n_estimators=args.trees,
        max_samples=min(args.max_samples, len(features)),
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    ).fit(features[det.FEATURES].values)

    scores = -forest.score_samples(features[det.FEATURES].values)
    by_level = features.groupby("area_level").indices
    thresholds = {
        level: {name: float(np.quantile(scores[idx], q)) for name, q in det.SEVERITY_QUANTILES.items()}
        for level, idx in by_level.items()
    }
    detector = det.AnomalyDetector(forest, thresholds)
    scored = pd.concat(
        [detector.score(features.iloc[idx], level, scores=scores[idx]) for level, idx in by_level.items()],
        ignore_index=True,
    )
    flagged = scored[scored["severity"].notna()].drop(columns="area_level")
    flagged["date"] = flagged["date"].dt.date
    used = areas[areas["area_id"].isin(features["area_id"].unique())]
    level_cells = {level: float(group["n_cells"].median()) for level, group in used.groupby("level")}

    det.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, det.ARTIFACT_DIR / "isolation_forest.joblib")
    info = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "IsolationForest",
        "params": {"n_estimators": args.trees, "max_samples": args.max_samples, "random_state": config.RANDOM_SEED},
        "features": det.FEATURES,
        "window_days": det.WINDOW_DAYS,
        "smooth_days": det.SMOOTH_DAYS,
        "severity_quantiles": det.SEVERITY_QUANTILES,
        "thresholds": thresholds,
        "level_cells": level_cells,
        "first_year": config.ANALYSIS_FIRST_YEAR,
        "area_days": int(len(features)),
        "flagged": {k: int(v) for k, v in flagged["severity"].value_counts().items()},
        "harmonization_model": meta["active_model"],
    }
    (det.ARTIFACT_DIR / "anomaly.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    con.register("flagged_df", flagged)
    con.execute("CREATE OR REPLACE TABLE fire_anomalies AS SELECT * FROM flagged_df ORDER BY area_id, date")
    con.unregister("flagged_df")
    con.execute("DELETE FROM metadata WHERE key = 'anomaly_model'")
    con.execute("INSERT INTO metadata VALUES ('anomaly_model', ?)", [json.dumps(info)])
    con.execute("CHECKPOINT")
    con.close()
    print(f"Flagged {len(flagged):,} area-days: {info['flagged']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
