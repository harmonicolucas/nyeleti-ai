"""Shared configuration for the data pipeline, the models and the API."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("NYELETI_DATA_DIR", ROOT / "ml" / "data"))
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
GRID_DIR = DATA_DIR / "grid"
FEATURES_DIR = DATA_DIR / "features"
HARMONIZED_DIR = DATA_DIR / "harmonized"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
QUALITY_DIR = DATA_DIR / "quality"
DB_PATH = Path(os.environ.get("NYELETI_DB", DATA_DIR / "db" / "nyeleti.duckdb"))

ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

# Study area. The pipeline is keyed on the FIRMS country name and the ISO3
# code used by geoBoundaries, so another country only needs these two values.
COUNTRY_NAME = os.environ.get("NYELETI_COUNTRY", "Mozambique")
COUNTRY_ISO3 = os.environ.get("NYELETI_ISO3", "MOZ")

FIRMS_ARCHIVE = "https://firms.modaps.eosdis.nasa.gov/data/country"

# key -> FIRMS archive slug, instrument and first full year available
SENSORS = {
    "modis": {
        "slug": "modis",
        "instrument": "MODIS",
        "product": "MCD14ML (MODIS Collection 6.1, standard processing)",
        "first_year": 2001,
    },
    "viirs_snpp": {
        "slug": "viirs-snpp",
        "instrument": "VIIRS",
        "product": "VNP14IMGTML (VIIRS S-NPP 375 m, standard processing)",
        "first_year": 2012,
    },
    "viirs_noaa20": {
        "slug": "viirs-jpss1",
        "instrument": "VIIRS",
        "product": "VJ114IMGTML (VIIRS NOAA-20 375 m, standard processing)",
        "first_year": 2020,
    },
}
LAST_YEAR = int(os.environ.get("NYELETI_LAST_YEAR", 2025))

# VIIRS S-NPP is the reference sensor: it has the longest 375 m record and a
# stable afternoon orbit close to Aqua's.
REFERENCE_SENSOR = "viirs_snpp"

# Common grid: EASE-Grid 2.0 global cylindrical equal-area (EPSG:6933).
GRID_CRS = "EPSG:6933"
CELL_KM = float(os.environ.get("NYELETI_CELL_KM", 5))
BACKGROUND_BLOCK_KM = 25  # side of the blocks used for the background rate

# Detections used for analysis: presumed vegetation fires, not low confidence.
MODIS_MIN_CONFIDENCE = 30
VIIRS_EXCLUDED_CONFIDENCE = ("l",)
ANALYSIS_TYPES = (0,)

# Temporal split (by acquisition year, never by row)
TRAIN_YEARS = (2012, 2022)
VALID_YEARS = (2023, 2024)
TEST_YEARS = (2025, 2025)

# Harmonized Fire Activity Index weights (count and FRP components)
HFAI_WEIGHT_COUNT = 0.5
HFAI_WEIGHT_FRP = 0.5

# Aqua joined Terra in July 2002. Before that MODIS has half the overpasses, so
# statistics that compare years (trends, seasonal references) start in 2003.
ANALYSIS_FIRST_YEAR = 2003

RANDOM_SEED = 42


def hfai(count, frp, ref: dict):
    """Harmonized Fire Activity Index from a detection count and an FRP sum.

    HFAI = w_count * count / count_ref + w_frp * frp / frp_ref, where the
    references are the mean count and FRP of an active VIIRS cell-day in the
    training years. The index is additive over cells and days.
    """
    return ref["weight_count"] * count / ref["count_ref"] + ref["weight_frp"] * frp / ref["frp_ref"]


def background_block_cells(cell_km: float | None = None) -> int:
    """Cells per side of a background block, so that blocks keep the same size on any grid."""
    km = CELL_KM if cell_km is None else cell_km
    return max(1, round(BACKGROUND_BLOCK_KM / km))


def grid_tag(cell_km: float | None = None) -> str:
    km = CELL_KM if cell_km is None else cell_km
    return f"{km:g}km"


def years_between(span: tuple[int, int]) -> list[int]:
    return list(range(span[0], span[1] + 1))
