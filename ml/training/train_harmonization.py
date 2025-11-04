"""Train the MODIS -> VIIRS harmonization models.

Two LightGBM regressors are trained on MODIS-informed cell-days of the overlap
period: one for the VIIRS S-NPP detection count and one for the VIIRS FRP sum.
The split is temporal (config.TRAIN_YEARS / VALID_YEARS / TEST_YEARS); the
validation years pick the hyper-parameters and the number of trees, and the
test years are not read here at all.

The linear calibration, the quantile mapping and the background rate are
fitted on the same training rows and stored next to the models.

    python -m ml.training.train_harmonization
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ml import config
from ml.evaluation.metrics import regression_metrics
from ml.preprocessing import background as bgmod
from ml.preprocessing import features as ft
from ml.training.baselines import LinearCalibration, QuantileMapping

BASE_PARAMS = {
    "learning_rate": 0.08,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "metric": "rmse",
    "verbosity": -1,
    "seed": config.RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": max(1, (os.cpu_count() or 2) // 2),
}

# One binned dataset is shared by trials with different min_data_in_leaf, so
# features must not be pre-filtered for any particular value of it.
DATASET_PARAMS = {"max_bin": BASE_PARAMS["max_bin"], "feature_pre_filter": False}

SEARCH_SPACE = {
    "count": [
        {"objective": "poisson", "num_leaves": 63, "min_data_in_leaf": 200},
        {"objective": "poisson", "num_leaves": 127, "min_data_in_leaf": 100},
        {"objective": "tweedie", "tweedie_variance_power": 1.3, "num_leaves": 63, "min_data_in_leaf": 200},
        {"objective": "regression", "num_leaves": 63, "min_data_in_leaf": 200},
    ],
    "frp": [
        {"objective": "tweedie", "tweedie_variance_power": 1.5, "num_leaves": 63, "min_data_in_leaf": 200},
        {"objective": "poisson", "num_leaves": 63, "min_data_in_leaf": 200},
        {"objective": "regression", "num_leaves": 63, "min_data_in_leaf": 200},
    ],
}


def fit_booster(params: dict, train: lgb.Dataset, valid: lgb.Dataset, rounds: int, patience: int) -> lgb.Booster:
    return lgb.train(
        {**BASE_PARAMS, **params},
        train,
        num_boost_round=rounds,
        valid_sets=[valid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)],
    )


def train_target(target: str, train: pd.DataFrame, valid: pd.DataFrame, args) -> tuple[lgb.Booster, dict]:
    y_col = ft.TARGETS[target]
    x_valid, y_valid = valid[ft.FEATURES], valid[y_col].values

    # hyper-parameter search on a fixed random subset of the training rows
    rng = np.random.default_rng(config.RANDOM_SEED)
    if len(train) > args.search_rows:
        subset = train.iloc[np.sort(rng.choice(len(train), args.search_rows, replace=False))]
    else:
        subset = train
    d_subset = lgb.Dataset(subset[ft.FEATURES], label=subset[y_col].values, params=DATASET_PARAMS, free_raw_data=False)
    d_valid = lgb.Dataset(x_valid, label=y_valid, reference=d_subset, free_raw_data=False)

    trials = []
    for params in SEARCH_SPACE[target]:
        t0 = time.time()
        booster = fit_booster(params, d_subset, d_valid, args.search_rounds, 30)
        pred = np.clip(booster.predict(x_valid, num_iteration=booster.best_iteration), 0, None)
        m = regression_metrics(y_valid, pred)
        trials.append(
            {
                "params": params,
                "best_iteration": int(booster.best_iteration),
                "valid_rmse": m["rmse"],
                "valid_mae": m["mae"],
                "valid_bias": m["bias"],
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(f"  [{target}] {params} -> valid RMSE {m['rmse']:.4f}, MAE {m['mae']:.4f}, bias {m['bias']:+.4f}", flush=True)

    best = min(trials, key=lambda t: t["valid_rmse"])
    d_train = lgb.Dataset(train[ft.FEATURES], label=train[y_col].values, params=DATASET_PARAMS, free_raw_data=False)
    d_valid = lgb.Dataset(x_valid, label=y_valid, reference=d_train, free_raw_data=False)
    booster = fit_booster(best["params"], d_train, d_valid, args.rounds, 40)
    pred = np.clip(booster.predict(x_valid, num_iteration=booster.best_iteration), 0, None)
    info = {
        "target": y_col,
        "params": {**BASE_PARAMS, **best["params"]},
        "best_iteration": int(booster.best_iteration),
        "search": trials,
        "validation": regression_metrics(y_valid, pred),
        "feature_importance_gain": dict(
            zip(ft.FEATURES, [float(v) for v in booster.feature_importance(importance_type="gain")])
        ),
    }
    print(f"  [{target}] final model: {booster.best_iteration} trees, valid RMSE {info['validation']['rmse']:.4f}", flush=True)
    return booster, info


def data_fingerprint() -> str:
    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for key in sorted(manifest["files"]):
        digest.update(manifest["files"][key]["sha256"].encode())
    return digest.hexdigest()[:16]


def hfai_reference(years, cell_km: float) -> dict:
    """Mean VIIRS count and FRP of an active cell-day in the training years."""
    viirs = ft.load_sensor(config.REFERENCE_SENSOR, years, cell_km)
    return {
        "count_ref": float(viirs["fire_count"].mean()),
        "frp_ref": float(viirs["frp_sum"].mean()),
        "weight_count": config.HFAI_WEIGHT_COUNT,
        "weight_frp": config.HFAI_WEIGHT_FRP,
        "reference_years": [years[0], years[-1]],
    }


def update_registry(version: str, meta: dict) -> None:
    path = config.ARTIFACTS_DIR / "registry.json"
    registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"versions": []}
    registry["versions"] = [v for v in registry["versions"] if v["version"] != version]
    registry["versions"].append({"version": version, "created_at": meta["created_at"], "cell_km": meta["cell_km"]})
    registry["active"] = version
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell-km", type=float, default=config.CELL_KM)
    parser.add_argument("--version", default=None)
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--search-rounds", type=int, default=300)
    parser.add_argument("--search-rows", type=int, default=1_500_000)
    parser.add_argument("--max-train-rows", type=int, default=6_000_000)
    parser.add_argument("--no-register", action="store_true", help="do not make this version the active one")
    args = parser.parse_args(argv)

    train_years = config.years_between(config.TRAIN_YEARS)
    valid_years = config.years_between(config.VALID_YEARS)
    columns = ["key", "date", "cell_id", "pair_valid"] + ft.FEATURES + list(ft.TARGETS.values())

    # Read year by year and thin each year by the same fraction, so the full
    # training period never has to sit in memory at once.
    pairs = json.loads(
        (config.FEATURES_DIR / config.grid_tag(args.cell_km) / "pairs_report.json").read_text(encoding="utf-8")
    )
    n_train_all = sum(p["pair_valid_rows"] for p in pairs if p["year"] in train_years)
    fraction = min(1.0, args.max_train_rows / n_train_all)
    rng = np.random.default_rng(config.RANDOM_SEED)
    parts = []
    for year in train_years:
        part = ft.load_features([year], args.cell_km, columns=columns)
        part = part[part["pair_valid"]]
        if fraction < 1.0:
            part = part.iloc[np.sort(rng.choice(len(part), int(round(len(part) * fraction)), replace=False))]
        parts.append(part)
    train = pd.concat(parts, ignore_index=True)
    del parts
    valid = ft.load_features(valid_years, args.cell_km, columns=columns)
    valid = valid[valid["pair_valid"]].reset_index(drop=True)
    print(f"Training rows: {len(train):,} of {n_train_all:,} ({train_years[0]}-{train_years[-1]}); "
          f"validation rows: {len(valid):,} ({valid_years[0]}-{valid_years[-1]})", flush=True)

    created = datetime.now(timezone.utc)
    version = args.version or f"hfai-lgbm-{config.grid_tag(args.cell_km)}-{created:%Y%m%d-%H%M}"
    out_dir = config.ARTIFACTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_info, baselines = {}, {}
    for target in ("count", "frp"):
        booster, info = train_target(target, train, valid, args)
        booster.save_model(str(out_dir / f"harmonization_{target}.lgb.txt"), num_iteration=booster.best_iteration)
        targets_info[target] = info

        y = train[ft.TARGETS[target]].values
        linear = LinearCalibration(target).fit(train, y)
        quantile = QuantileMapping(target).fit(train, y)
        baselines[target] = {"linear": linear, "quantile_mapping": quantile}
        targets_info[target]["linear_baseline"] = linear.describe()
    joblib.dump(baselines, out_dir / "baselines.joblib")

    background = bgmod.fit_background(train_years, args.cell_km)
    background.to_parquet(out_dir / "background.parquet", index=False)

    meta = {
        "version": version,
        "created_at": created.isoformat(timespec="seconds"),
        "cell_km": args.cell_km,
        "grid_crs": config.GRID_CRS,
        "reference_sensor": config.REFERENCE_SENSOR,
        "country": config.COUNTRY_NAME,
        "features": ft.FEATURES,
        "split": {"train": config.TRAIN_YEARS, "validation": config.VALID_YEARS, "test": config.TEST_YEARS},
        "rows": {"train_available": int(n_train_all), "train_used": int(len(train)), "validation": int(len(valid))},
        "targets": targets_info,
        "hfai": hfai_reference(train_years, args.cell_km),
        "background": {
            "block_cells": config.background_block_cells(args.cell_km),
            "block_km": config.BACKGROUND_BLOCK_KM,
            "fitted_on": config.TRAIN_YEARS,
            "residual_viirs_detections": float(background["residual_count"].sum()),
        },
        "data_fingerprint": data_fingerprint(),
        "libraries": {"lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "seed": config.RANDOM_SEED,
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not args.no_register:
        update_registry(version, meta)
    print(f"Saved model version {version} to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
