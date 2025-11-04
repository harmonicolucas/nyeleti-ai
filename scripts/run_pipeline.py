"""Run the whole pipeline, from the NASA FIRMS download to the application database.

    python -m scripts.run_pipeline                  # everything
    python -m scripts.run_pipeline --from pairs     # resume from a given step
    python -m scripts.run_pipeline --only calendar  # a single step

Every step can also be run on its own; this script only fixes the order.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from ml import config

STEPS = [
    ("download", "scripts.download_firms", "NASA FIRMS archive files"),
    ("boundaries", "scripts.download_boundaries", "administrative boundaries"),
    ("clean", "scripts.clean_data", "validation, duplicates, unified schema"),
    ("grid", "scripts.aggregate_grid", "cell x day aggregation on the common grid"),
    ("pairs", "scripts.build_pairs", "MODIS-VIIRS pairs and model features"),
    ("train", "ml.training.train_harmonization", "LightGBM harmonization model"),
    ("evaluate", "ml.evaluation.evaluate_model", "metrics on the held-out years"),
    ("harmonize", "ml.inference.harmonize", "harmonized estimates for the full record"),
    ("calendar", "scripts.build_calendar", "application database and calendar"),
    ("anomaly", "ml.training.train_anomaly", "Isolation Forest anomaly model"),
    ("figures", "scripts.make_figures", "figures for the documentation"),
]


def main() -> None:
    names = [name for name, _, _ in STEPS]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="first", choices=names, default=names[0])
    parser.add_argument("--to", dest="last", choices=names, default=names[-1])
    parser.add_argument("--only", choices=names)
    args = parser.parse_args()

    if args.only:
        selected = [args.only]
    else:
        selected = names[names.index(args.first): names.index(args.last) + 1]

    started = time.time()
    for name, module, title in STEPS:
        if name not in selected:
            continue
        print(f"\n=== {name}: {title} ===", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, "-m", module], cwd=config.ROOT)
        if result.returncode != 0:
            sys.exit(f"step '{name}' failed with exit code {result.returncode}")
        print(f"--- {name} done in {time.time() - t0:.0f} s", flush=True)
    print(f"\nPipeline finished in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
