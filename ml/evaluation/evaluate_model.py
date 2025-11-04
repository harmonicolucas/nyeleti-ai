"""Evaluate the harmonization model against the reference methods.

Everything reported as "test" comes from config.TEST_YEARS, which are never
read during training or hyper-parameter selection.

Two levels are evaluated:
  cell-day      MODIS-informed cell-days, where each method predicts the VIIRS
                value of the same cell and day;
  aggregated    province-day, province-month and country-day totals, compared
                with the full VIIRS record (including detections outside
                MODIS-informed cell-days). This is the scale the calendar uses.

Writes ml/artifacts/<version>/metrics.json, docs/results.md and the figures in
docs/figures/.

    python -m ml.evaluation.evaluate_model
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml import config
from ml.evaluation.metrics import detection_metrics, regression_metrics
from ml.inference.harmonize import Harmonizer
from ml.preprocessing import background as bgmod
from ml.preprocessing import features as ft
from ml.training.baselines import raw_modis

METHODS = ["modis_raw", "linear", "quantile_mapping", "lightgbm"]
CALIBRATED = ["linear", "quantile_mapping", "lightgbm"]
LABELS = {
    "modis_raw": "Original MODIS (no harmonization)",
    "linear": "Linear regression",
    "quantile_mapping": "Quantile mapping",
    "lightgbm": "LightGBM",
    "lightgbm_no_background": "LightGBM without background rate",
    "noaa20": "VIIRS NOAA-20 (second VIIRS, reference ceiling)",
}
HIGH_ACTIVITY_PERCENTILE = 90


def predict_methods(model: Harmonizer, rows: pd.DataFrame, methods) -> dict:
    preds = {"count": {}, "frp": {}}
    for target in ("count", "frp"):
        for method in methods:
            if method == "modis_raw":
                preds[target][method] = raw_modis(rows, target)
            elif method == "lightgbm":
                preds[target][method] = model.predict(rows, target)
            else:
                preds[target][method] = model.baselines[target][method].predict(rows)
    return preds


class AreaAggregator:
    """Province-day totals of the observations and of each method's estimate."""

    def __init__(self, model: Harmonizer):
        self.model = model
        self.cells = ft.load_cells(model.cell_km)
        self.adm1 = self.cells.set_index("cell_id")["adm1_id"]
        self.units = sorted(self.adm1.unique())
        bg_cells = bgmod.cell_background(model.background, self.cells)
        bg_cells["adm1_id"] = self.adm1.reindex(bg_cells["cell_id"]).values
        self.bg_area = bg_cells.groupby(["adm1_id", "month"])[["bg_count", "bg_frp"]].sum()
        self.days = bgmod.load_sensor_days(model.cell_km).set_index("date")

    def year_frame(self, year: int, methods) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        rows = ft.load_features([year], self.model.cell_km)
        rows = rows[rows["pair_valid"]].reset_index(drop=True)
        rows = bgmod.attach_background(rows, self.model.background, self.cells)
        preds = predict_methods(self.model, rows, methods)

        frame = pd.DataFrame(
            {
                "adm1_id": self.adm1.reindex(rows["cell_id"]).values,
                "date": rows["date"].values,
                "bg_count": rows["bg_count"].values.astype("float64"),
                "bg_frp": rows["bg_frp"].values.astype("float64"),
            }
        )
        for target in ("count", "frp"):
            for method in methods:
                frame[f"{method}_{target}"] = preds[target][method]
        informed = frame.groupby(["adm1_id", "date"]).sum()

        valid_dates = self.days.index[(self.days.index.year == year) & self.days["pair_valid"]]
        index = pd.MultiIndex.from_product([self.units, valid_dates], names=["adm1_id", "date"])
        informed = informed.reindex(index, fill_value=0.0)
        months = index.get_level_values("date").month
        bg_idx = pd.MultiIndex.from_arrays([index.get_level_values("adm1_id"), months])
        bg_total = self.bg_area.reindex(bg_idx).fillna(0.0)

        out = pd.DataFrame(index=index)
        for target in ("count", "frp"):
            extra = bg_total[f"bg_{target}"].values - informed[f"bg_{target}"].values
            for method in methods:
                value = informed[f"{method}_{target}"].values
                out[f"{method}_{target}"] = value if method == "modis_raw" else value + extra
            if "lightgbm" in methods:
                out[f"lightgbm_no_background_{target}"] = informed[f"lightgbm_{target}"].values

        viirs = ft.load_sensor(config.REFERENCE_SENSOR, [year], self.model.cell_km)
        viirs = viirs[viirs["cell_id"].isin(self.adm1.index)]
        observed = (
            pd.DataFrame(
                {
                    "adm1_id": self.adm1.reindex(viirs["cell_id"]).values,
                    "date": viirs["date"].values,
                    "viirs_count": viirs["fire_count"].values.astype("float64"),
                    "viirs_frp": viirs["frp_sum"].values.astype("float64"),
                }
            )
            .groupby(["adm1_id", "date"])
            .sum()
            .reindex(index, fill_value=0.0)
        )
        out["viirs_count"] = observed["viirs_count"].values
        out["viirs_frp"] = observed["viirs_frp"].values

        names = list(methods) + (["lightgbm_no_background"] if "lightgbm" in methods else []) + ["viirs"]
        for name in names:
            out[f"{name}_hfai"] = self.model.hfai(out[f"{name}_count"].values, out[f"{name}_frp"].values)
        return out.reset_index(), rows, preds


def cell_day_table(model: Harmonizer, rows: pd.DataFrame, preds: dict) -> dict:
    y_count = rows[ft.TARGETS["count"]].values.astype("float64")
    y_frp = rows[ft.TARGETS["frp"]].values.astype("float64")
    y_hfai = model.hfai(y_count, y_frp)
    table = {"count": {}, "frp": {}, "hfai": {}}
    for method in preds["count"]:
        table["count"][method] = regression_metrics(y_count, preds["count"][method])
        table["frp"][method] = regression_metrics(y_frp, preds["frp"][method])
        table["hfai"][method] = regression_metrics(y_hfai, model.hfai(preds["count"][method], preds["frp"][method]))
    return table


def aggregated_table(frame: pd.DataFrame, methods) -> dict:
    """Metrics of each method at province-day, province-month and country-day scale."""
    value_cols = [c for c in frame.columns if c not in ("adm1_id", "date")]
    levels = {
        "province_day": frame,
        "province_month": frame.groupby(["adm1_id", frame["date"].dt.to_period("M")])[value_cols].sum().reset_index(),
        "country_day": frame.groupby("date")[value_cols].sum().reset_index(),
    }
    out = {}
    for level, data in levels.items():
        out[level] = {}
        for quantity in ("hfai", "count", "frp"):
            out[level][quantity] = {
                m: regression_metrics(data[f"viirs_{quantity}"], data[f"{m}_{quantity}"]) for m in methods
            }
    return out


def high_activity_table(frame: pd.DataFrame, thresholds: pd.Series, methods) -> dict:
    limit = frame["adm1_id"].map(thresholds).values
    observed = frame["viirs_hfai"].values > limit
    return {m: detection_metrics(observed, frame[f"{m}_hfai"].values > limit) for m in methods}


# --------------------------------------------------------------------- figures

def fig_before_after(monthly: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = monthly["month"].dt.to_timestamp()
    panels = [
        ("modis_raw_count", "Original MODIS", "#1f77b4", "Before harmonization: MODIS and VIIRS S-NPP monthly detections"),
        ("lightgbm_count", "Harmonized estimate (LightGBM + background)", "#d95f02",
         "After harmonization: MODIS-based estimate on the VIIRS scale"),
    ]
    for ax, (col, label, color, title) in zip(axes, panels):
        ax.plot(x, monthly["viirs_count"] / 1e3, color="black", lw=1.2, label="VIIRS S-NPP observed")
        ax.plot(x, monthly[col] / 1e3, color=color, lw=1.2, label=label)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Detections per month (thousands)")
        ax.grid(alpha=0.25)
        for span, shade, name in (
            (config.VALID_YEARS, "#cccccc", "validation"),
            (config.TEST_YEARS, "#f2c9a1", "test"),
        ):
            ax.axvspan(pd.Timestamp(f"{span[0]}-01-01"), pd.Timestamp(f"{span[1]}-12-31"), color=shade, alpha=0.35,
                       label=f"{name} years")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle(f"{config.COUNTRY_NAME}: MODIS and VIIRS active fire detections on the common grid", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_scatter(province_month: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    obs = province_month["viirs_count"].values
    top = max(obs.max(), province_month["lightgbm_count"].max()) * 1.05
    for ax, col, title in (
        (axes[0], "modis_raw_count", "Before: original MODIS"),
        (axes[1], "lightgbm_count", "After: harmonized estimate"),
    ):
        ax.scatter(obs, province_month[col], s=22, alpha=0.75, color="#d95f02", edgecolor="none")
        ax.plot([0, top], [0, top], color="black", lw=1, ls="--", label="1:1")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("VIIRS S-NPP detections (province-month)")
        ax.set_ylabel("MODIS-based value")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    years = config.TEST_YEARS
    span = str(years[0]) if years[0] == years[1] else f"{years[0]}-{years[1]}"
    fig.suptitle(f"Independent test period ({span}): province-month totals", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_methods(aggregated: dict, path) -> None:
    table = aggregated["province_day"]["hfai"]
    methods = [m for m in METHODS if m in table]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, title in zip(axes, ("mae", "rmse", "bias"), ("MAE", "RMSE", "Bias")):
        values = [table[m][metric] for m in methods]
        colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d95f02"][: len(methods)]
        ax.bar(range(len(methods)), values, color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABELS[m].split(" (")[0].replace(" ", "\n") for m in methods], fontsize=8)
        ax.set_title(f"{title} (HFAI, province-day, test)", fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_importance(model: Harmonizer, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, target in zip(axes, ("count", "frp")):
        gain = pd.Series(model.meta["targets"][target]["feature_importance_gain"]).sort_values()
        gain = 100 * gain / gain.sum()
        ax.barh(gain.index, gain.values, color="#d95f02")
        ax.set_xlabel("Share of total gain (%)")
        ax.set_title(f"Feature importance: {ft.TARGETS[target]}", loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------- report

def _fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _metric_rows(table: dict, methods) -> list[str]:
    lines = ["| Method | MAE | RMSE | R² | Bias | Bias % | Spearman |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        if m not in table:
            continue
        r = table[m]
        lines.append(
            f"| {LABELS[m]} | {_fmt(r['mae'])} | {_fmt(r['rmse'])} | {_fmt(r['r2'])} | {_fmt(r['bias'])} | "
            f"{_fmt(r['bias_pct'], 1)} | {_fmt(r['spearman'])} |"
        )
    return lines


def verdict(metrics: dict) -> list[str]:
    """Plain statements derived from the numbers, nothing else."""
    lines = []
    for level, label in (("cell_day", "cell-day"), ("province_day", "province-day"), ("province_month", "province-month")):
        table = metrics["cell_day"]["test"]["hfai"] if level == "cell_day" else metrics["aggregated"]["test"][level]["hfai"]
        lgbm = table["lightgbm"]
        for metric in ("mae", "rmse"):
            beaten = [LABELS[m].split(" (")[0] for m in METHODS[:-1] if lgbm[metric] < table[m][metric]]
            lost = [LABELS[m].split(" (")[0] for m in METHODS[:-1] if lgbm[metric] >= table[m][metric]]
            text = f"- HFAI, {label}, {metric.upper()}: LightGBM {_fmt(lgbm[metric])}"
            if beaten:
                text += "; lower than " + ", ".join(f"{b} ({_fmt(table[m][metric])})" for b, m in
                                                   [(LABELS[m].split(' (')[0], m) for m in METHODS[:-1] if lgbm[metric] < table[m][metric]])
            if lost:
                text += "; not lower than " + ", ".join(f"{b} ({_fmt(table[m][metric])})" for b, m in
                                                       [(LABELS[m].split(' (')[0], m) for m in METHODS[:-1] if lgbm[metric] >= table[m][metric]])
            lines.append(text + ".")
        biases = {m: abs(table[m]["bias_pct"]) for m in METHODS}
        best = min(biases, key=biases.get)
        lines.append(
            f"- HFAI, {label}, bias: LightGBM {_fmt(table['lightgbm']['bias_pct'], 1)} % of the observed mean; "
            f"original MODIS {_fmt(table['modis_raw']['bias_pct'], 1)} %; smallest absolute bias: {LABELS[best].split(' (')[0]}."
        )
    return lines


def write_results(metrics: dict, model: Harmonizer) -> None:
    test = metrics["test_years"]
    span = str(test[0]) if test[0] == test[1] else f"{test[0]}-{test[1]}"
    cell = metrics["cell_day"]["test"]
    agg = metrics["aggregated"]["test"]
    all_methods = METHODS + ["lightgbm_no_background"]
    lines = [
        "# Results",
        "",
        f"Generated by `ml/evaluation/evaluate_model.py` on {metrics['evaluated_at']} for model "
        f"`{metrics['version']}`. Do not edit by hand: re-run the script to refresh the numbers.",
        "",
        f"Study area: {config.COUNTRY_NAME}. Grid: {model.cell_km:g} km equal-area cells, daily. "
        f"Training {config.TRAIN_YEARS[0]}-{config.TRAIN_YEARS[1]}, validation "
        f"{config.VALID_YEARS[0]}-{config.VALID_YEARS[1]}, independent test {span}. "
        "Reference: VIIRS S-NPP 375 m.",
        "",
        "HFAI is the Harmonized Fire Activity Index defined in `docs/methodology.md`. Bias is the mean of "
        "(estimate - VIIRS); a negative bias means the method underestimates VIIRS.",
        "",
        "## Summary",
        "",
        *verdict(metrics),
        "",
        f"## 1. Cell-day level, test {span}",
        "",
        f"{cell['hfai']['lightgbm']['n']:,} MODIS-informed cell-days. Each method predicts the VIIRS value of the "
        "same cell and day.",
        "",
        "### VIIRS detection count",
        "",
        *_metric_rows(cell["count"], METHODS),
        "",
        "### VIIRS FRP sum (MW)",
        "",
        *_metric_rows(cell["frp"], METHODS),
        "",
        "### HFAI",
        "",
        *_metric_rows(cell["hfai"], METHODS),
        "",
    ]
    if metrics.get("ceiling"):
        c = metrics["ceiling"]
        lines += [
            "### Reference ceiling: a second VIIRS",
            "",
            "NOAA-20 carries the same VIIRS instrument as S-NPP, about 50 minutes apart. Using its count as a "
            "prediction of the S-NPP count on the same cell-days shows how much two near-identical sensors "
            "disagree at this scale, which bounds what any MODIS-based method can reach.",
            "",
            *_metric_rows({"noaa20": c["noaa20_as_predictor"], "lightgbm": c["lightgbm_same_rows"]}, ["noaa20", "lightgbm"]),
            "",
        ]
    for level, title, note in (
        ("province_day", "Province-day", "one value per province and day"),
        ("province_month", "Province-month", "daily values summed by month"),
        ("country_day", "Country-day", "daily national totals"),
    ):
        lines += [
            f"## Aggregated level: {title.lower()}, test {span}",
            "",
            f"{agg[level]['hfai']['lightgbm']['n']:,} values ({note}). Estimates are compared with the complete "
            "VIIRS record. Calibrated methods include the background rate; original MODIS does not.",
            "",
            "### HFAI",
            "",
            *_metric_rows(agg[level]["hfai"], all_methods),
            "",
            "### Detection count",
            "",
            *_metric_rows(agg[level]["count"], all_methods),
            "",
        ]
    high = metrics["high_activity"]
    lines += [
        f"## High-activity days, test {span}",
        "",
        f"{high['definition']}",
        "",
        "| Method | Observed high days | Flagged | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for m in all_methods:
        r = high["test"][m]
        lines.append(
            f"| {LABELS[m]} | {r['observed_high']:,} | {r['predicted_high']:,} | {_fmt(r['precision'])} | "
            f"{_fmt(r['recall'])} | {_fmt(r['f1'])} |"
        )
    lines += [
        "",
        "## National totals by year (overlap period)",
        "",
        "Detections on days with a valid VIIRS record. \"Harmonized\" is LightGBM plus the background rate.",
        "",
        "| Year | Split | MODIS | VIIRS S-NPP | Harmonized | MODIS / VIIRS | Harmonized / VIIRS |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in metrics["by_year"]:
        lines.append(
            f"| {r['year']} | {r['split']} | {r['modis_count']:,.0f} | {r['viirs_count']:,.0f} | "
            f"{r['harmonized_count']:,.0f} | {r['modis_ratio']:.3f} | {r['harmonized_ratio']:.3f} |"
        )
    lines += [
        "",
        "## Validation years",
        "",
        f"Cell-day HFAI, {config.VALID_YEARS[0]}-{config.VALID_YEARS[1]} (used for model selection, so not an "
        "independent estimate).",
        "",
        *_metric_rows(metrics["cell_day"]["validation"]["hfai"], METHODS),
        "",
        "## Figures",
        "",
        "- `figures/harmonization_before_after.png`: monthly national series before and after harmonization.",
        "- `figures/scatter_test_province_month.png`: province-month totals in the test period.",
        "- `figures/methods_province_day.png`: MAE, RMSE and bias by method.",
        "- `figures/feature_importance.png`: LightGBM gain by feature.",
        "",
    ]
    (config.DOCS_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------ main

def split_of(year: int) -> str:
    for name, span in (("train", config.TRAIN_YEARS), ("validation", config.VALID_YEARS), ("test", config.TEST_YEARS)):
        if span[0] <= year <= span[1]:
            return name
    return "other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=None)
    parser.add_argument("--no-docs", action="store_true", help="only write metrics.json")
    args = parser.parse_args(argv)

    model = Harmonizer(args.version)
    aggregator = AreaAggregator(model)
    overlap = [y for y in range(config.TRAIN_YEARS[0], config.TEST_YEARS[1] + 1)]

    frames, cell_tables, by_year = {}, {}, []
    valid_rows, valid_preds = [], []
    ceiling = None
    for year in overlap:
        split = split_of(year)
        methods = METHODS if split in ("validation", "test") else ["modis_raw", "lightgbm"]
        frame, rows, preds = aggregator.year_frame(year, methods)
        frames[year] = frame
        by_year.append(
            {
                "year": year,
                "split": split,
                "modis_count": float(frame["modis_raw_count"].sum()),
                "viirs_count": float(frame["viirs_count"].sum()),
                "harmonized_count": float(frame["lightgbm_count"].sum()),
                "modis_ratio": float(frame["modis_raw_count"].sum() / frame["viirs_count"].sum()),
                "harmonized_ratio": float(frame["lightgbm_count"].sum() / frame["viirs_count"].sum()),
                "harmonized_hfai": float(frame["lightgbm_hfai"].sum()),
                "viirs_hfai": float(frame["viirs_hfai"].sum()),
            }
        )
        print(f"{year} ({split}): MODIS/VIIRS {by_year[-1]['modis_ratio']:.3f}, "
              f"harmonized/VIIRS {by_year[-1]['harmonized_ratio']:.3f}", flush=True)
        if split == "validation":
            valid_rows.append(rows[list(ft.TARGETS.values())])
            valid_preds.append(preds)
        if split == "test":
            cell_tables.setdefault("rows", []).append(rows)
            cell_tables.setdefault("preds", []).append(preds)

    def stack(pred_list):
        return {t: {m: np.concatenate([p[t][m] for p in pred_list]) for m in pred_list[0][t]} for t in ("count", "frp")}

    test_rows = pd.concat(cell_tables["rows"], ignore_index=True)
    test_preds = stack(cell_tables["preds"])
    cell_day = {
        "test": cell_day_table(model, test_rows, test_preds),
        "validation": cell_day_table(model, pd.concat(valid_rows, ignore_index=True), stack(valid_preds)),
    }

    noaa = test_rows["noaa20_fire_count"]
    if noaa.notna().any():
        y = test_rows[ft.TARGETS["count"]].values
        ceiling = {
            "noaa20_as_predictor": regression_metrics(y, noaa.values),
            "lightgbm_same_rows": cell_day["test"]["count"]["lightgbm"],
        }

    test_years = config.years_between(config.TEST_YEARS)
    test_frame = pd.concat([frames[y] for y in test_years], ignore_index=True)
    all_methods = METHODS + ["lightgbm_no_background"]
    aggregated = {"test": aggregated_table(test_frame, all_methods)}

    train_frame = pd.concat([frames[y] for y in config.years_between(config.TRAIN_YEARS)], ignore_index=True)
    thresholds = train_frame.groupby("adm1_id")["viirs_hfai"].quantile(HIGH_ACTIVITY_PERCENTILE / 100)
    high = {
        "definition": (
            f"A province-day is a high-activity day when its observed VIIRS HFAI exceeds the "
            f"{HIGH_ACTIVITY_PERCENTILE}th percentile of that province's daily VIIRS HFAI in the training years. "
            "The same threshold is applied to each method's estimate."
        ),
        "percentile": HIGH_ACTIVITY_PERCENTILE,
        "test": high_activity_table(test_frame, thresholds, all_methods),
    }

    metrics = {
        "version": model.version,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_years": list(config.TEST_YEARS),
        "labels": LABELS,
        "cell_day": cell_day,
        "ceiling": ceiling,
        "aggregated": aggregated,
        "high_activity": high,
        "by_year": by_year,
    }
    (model.dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not args.no_docs:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        everything = pd.concat(frames.values(), ignore_index=True)
        monthly = (
            everything.groupby(everything["date"].dt.to_period("M"))[["modis_raw_count", "lightgbm_count", "viirs_count"]]
            .sum()
            .rename_axis("month")
            .reset_index()
        )
        monthly.assign(month=monthly["month"].astype(str)).to_csv(model.dir / "monthly_national.csv", index=False)
        cols = ["viirs_count", "modis_raw_count", "lightgbm_count"]
        province_month = test_frame.groupby(["adm1_id", test_frame["date"].dt.to_period("M")])[cols].sum().reset_index()
        fig_before_after(monthly, config.FIGURES_DIR / "harmonization_before_after.png")
        fig_scatter(province_month, config.FIGURES_DIR / "scatter_test_province_month.png")
        fig_methods(aggregated["test"], config.FIGURES_DIR / "methods_province_day.png")
        fig_importance(model, config.FIGURES_DIR / "feature_importance.png")
        write_results(metrics, model)

    t = aggregated["test"]["province_day"]["hfai"]
    print("Province-day HFAI, test: " + "; ".join(f"{m} RMSE {t[m]['rmse']:.3f} bias {t[m]['bias_pct']:+.1f}%" for m in METHODS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
