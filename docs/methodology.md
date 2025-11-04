# Methodology

This document describes what the pipeline does to the NASA FIRMS records, in
the order in which it happens. Numbers quoted here come from the generated
reports (`data_quality.md`, `results.md`, `resolution_comparison.md`) and
refer to the Mozambique case study.

## 1. The problem

MODIS (Terra from 2000, Aqua from 2002) gives the long record of active fires,
at 1 km. VIIRS (S-NPP from 2012, NOAA-20 from 2018 in orbit, 2020 in the
country archive) gives a 375 m record that sees many more small and cool
fires. On the Mozambique grid VIIRS S-NPP reports between 4.5 and 5.6 times
as many detections per year as MODIS (MODIS / VIIRS between 0.18 and 0.22), and
the ratio is not constant: it changes with the season, with fire size and from
year to year. A series that switches from MODIS to VIIRS in 2012 therefore
shows a jump that has nothing to do with fire activity.

The harmonization expresses the MODIS record on the VIIRS S-NPP scale, so the
whole period can be read with a single yardstick. The originals are never
modified: every table keeps MODIS observed, VIIRS observed and the harmonized
estimate as separate columns.

## 2. Data preparation

1. **Download** (`scripts/download_firms.py`). Yearly country files from the
   FIRMS archive. Each file is recorded in `ml/data/raw/manifest.json` with
   its URL, product, SHA-256, record count, covered period and download time.
   Files already present are skipped, so the step is incremental.
2. **Cleaning** (`scripts/clean_data.py`). Column names and types are
   unified across sensors; records are checked for valid coordinates, date
   inside the file's year, time, satellite, confidence, non-negative FRP,
   positive brightness, day/night flag, fire type and position inside the
   study bounding box (0.5 degree margin). Exact duplicates (same latitude,
   longitude, date, time, satellite) are dropped. Rejected rows are counted by
   rule in the quality report; nothing is silently corrected.
3. **Analysis subset.** Only presumed vegetation fires (`type = 0`) that are
   not low confidence are used: MODIS confidence of 30 or more, VIIRS nominal
   or high. Static sources (volcanoes, gas flares, industrial sites) and
   offshore detections are excluded. All valid records, including the excluded
   ones, stay in the cleaned files and in the map layer, with their flags.
4. **Common grid** (`scripts/aggregate_grid.py`). Detections are assigned to
   square cells of an equal-area grid (EPSG:6933). The operational size is
   5 km; the choice is examined in `resolution_comparison.md`. The universe is
   every cell that intersects the country boundary (33,442 cells at 5 km).
   For each sensor, cell and UTC day the pipeline stores the number of
   detections, the FRP sum, mean and maximum, mean confidence, mean brightness
   temperature, mean scan size, day and night counts and, for MODIS, the Terra
   and Aqua counts.

**Choice of the cell size.** The same pipeline was run at 1, 5 and 10 km and
scored on the 2025 province-day detection counts, which mean the same thing
at every resolution:

| Cell size | RMSE | R² | Bias | VIIRS detections inside MODIS-informed cell-days |
|---|---|---|---|---|
| 1 km | 205.2 | 0.790 | +2.5 % | 36.4 % |
| 5 km | 159.0 | 0.874 | -1.3 % | 75.7 % |
| 10 km | 150.4 | 0.887 | -3.3 % | 90.2 % |

At 1 km most VIIRS detections fall where MODIS reports nothing nearby, and
the estimate leans on the background rate. 10 km gives a slightly lower error
than 5 km, with a larger bias and a quarter of the spatial detail. 5 km was
kept as the operational grid for the map and the district calendars; the
gain in accuracy from 1 km to 5 km is large, the difference between 5 km and
10 km is small. Note that the neighbourhood that makes a cell-day "informed"
(section 3) is three cells wide, so it grows with the cell: this comparison
changes the cell size and the neighbourhood together.

Points are never matched one to one. A single fire produces a different
number of detections in each sensor, at different times, so the comparison is
made between cell-day aggregates.

## 3. MODIS-VIIRS pairs

A cell-day is **MODIS-informed** when MODIS detected at least one fire in the
cell or in one of its eight neighbours, on that day, the day before or the day
after (`ml/preprocessing/features.py`). These are the rows the models work on.
At 5 km they contain 76 to 79 % of the VIIRS detections of a year (69 % in
2020, see section 8), although only about 7 % of the rows have a MODIS
detection in the cell itself.

Each row carries 21 features:

- the MODIS aggregates of the cell (count, FRP sum, mean and maximum,
  confidence, brightness, scan size, day/night and Terra/Aqua counts);
- the context: count and FRP in the eight neighbours on the same day, count in
  the cell and in the neighbours on the previous and on the following day;
- month, season, latitude and longitude of the cell centre.

The targets are the VIIRS S-NPP detection count and FRP sum of the same cell
and day (zero when VIIRS saw nothing there).

The use of the following day makes this a reprocessing method, not a real-time
one: the estimate for day *d* needs the MODIS data of *d + 1*.

**Days without a usable VIIRS target.** When VIIRS S-NPP reports no detection
anywhere in the country, the day is not used for training or evaluation: a
true zero cannot be told apart from a gap in the record. If MODIS has ten or
more detections on such a day it is flagged as a VIIRS gap, and the calendar
shows VIIRS as missing instead of zero (40 days between 2012 and 2025).

## 4. Temporal split

| Role | Years | Use |
|---|---|---|
| Training | 2012-2022 | model fitting, baselines, background rate, HFAI references |
| Validation | 2023-2024 | choice of objective, tree size and number of trees |
| Test | 2025 | reported metrics only |

The split is by year, never by row, so no day of the test period is seen
during training. The specification planned 2025-2026 for testing; the FIRMS
country archive did not yet include 2026 when the project was built, so the
test period is the year 2025.

## 5. Harmonization model

`ml/training/train_harmonization.py` fits two LightGBM regressors, one per
target. For each target a small set of candidates (Poisson, Tweedie and
squared-error objectives, two tree sizes) is tried on a fixed random subset of
the training rows and scored on the validation years; the best candidate is
refitted on the training rows with early stopping on the validation RMSE. On
the development machine training uses a random sample of 6,000,000 of the
12,848,114 training pairs (memory limit), with a fixed seed.

Selected models: Tweedie objective with variance power 1.3 for the count (336
trees) and 1.5 for the FRP (176 trees), 63 leaves, at least 200 rows per leaf.
Predictions are clipped at zero.

### Baselines

Fitted on the same training rows and evaluated on the same test rows:

- **Original MODIS**: the MODIS value used as if it were the VIIRS value.
- **Linear regression** on the MODIS count/FRP of the cell, its neighbours and
  the adjacent days.
- **Quantile mapping** of the in-cell MODIS value onto the VIIRS distribution.

### Background rate

Between a fifth and a quarter of the VIIRS detections fall in cell-days with
no MODIS detection anywhere near. No method fed with MODIS data can place
those fires, and ignoring them would leave every aggregated estimate well
below VIIRS (bias of -23.5 % for LightGBM alone on the 2025 province totals).
They are accounted for with a climatological rate
(`ml/preprocessing/background.py`): the mean VIIRS count and FRP per
non-informed cell-day, by calendar month and by 25 km block, measured on the
training years only.

The harmonized value of an area on a day is therefore

    sum over MODIS-informed cells of the model estimate
    + sum over the other cells of the background rate

The background has no year-to-year variability by construction: it corrects
the level, and all interannual signal comes from MODIS. The calibrated
baselines receive the same background when aggregated, so the comparison is
fair. In the low season the background is most of the harmonized value, which
is why January-April values look alike from year to year.

## 6. HFAI

The Harmonized Fire Activity Index combines count and FRP in one number:

    HFAI = 0.5 * count / count_ref + 0.5 * FRP / FRP_ref

`count_ref` and `FRP_ref` are the mean detection count (2.41) and the mean FRP
sum (22.2 MW) of an active VIIRS S-NPP cell-day in the training years, at 5 km.
An average active VIIRS cell-day has HFAI = 1. The index is linear, so it adds
up over cells and days, and it can be computed from observed VIIRS values or
from harmonized estimates with the same formula. The references are stored
with the model and depend on the cell size, so HFAI values from grids of
different resolution are not comparable.

## 7. Evaluation

`ml/evaluation/evaluate_model.py` reads the test year only at this stage and
writes `docs/results.md`, `metrics.json` and the figures. Metrics: MAE, RMSE,
R², bias (absolute and as a share of the observed mean), Spearman correlation,
and precision, recall and F1 for high-activity days (province-days above the
90th percentile of that province's VIIRS HFAI in the training years).

Results are reported at four levels: cell-day, province-day, province-month
and country-day. As a yardstick for the cell-day level, the NOAA-20 VIIRS
count is scored as a "prediction" of the S-NPP count on the same rows: two
copies of the same instrument 50 minutes apart reach R² = 0.45 at 5 km, which
bounds what a MODIS-based estimate can achieve at that scale.

## 8. Estimates before 2012

Between 2001 and January 2012 there is no VIIRS observation. The harmonized
values of that period are model estimates on the VIIRS scale and cannot be
verified; the interface and the exports label them as such. Two further
caveats apply:

- 2001 and the first half of 2002 have Terra only (and 115 days without any
  detection in 2001), so the MODIS input itself is incomplete. Trend and
  seasonal statistics start in 2003.
- The harmonized series inherits every gap of MODIS. The clearest case inside
  the overlap period is 16 August to 1 September 2020, when the archive has
  no Aqua detection at all, at the height of the fire season: that year has
  the lowest MODIS / VIIRS ratio (0.177) and the harmonized total is 11.5 %
  below VIIRS. Similar gaps before 2012 would lower the estimate in the same
  way and cannot be detected by comparison with VIIRS.
- The model assumes the MODIS-VIIRS relationship learnt in 2012-2022 holds
  earlier. Changes in MODIS calibration or orbit would break that assumption;
  the yearly MODIS / VIIRS ratio (0.18 to 0.22 over 2012-2025, see
  `results.md`) gives an idea of how stable it has been since 2012.

## 9. Burning activity calendar

`scripts/build_calendar.py` loads everything into the application database
and builds one row per area and day for the country, the 10 provinces and the
districts: MODIS observed, VIIRS observed, harmonized count, FRP and HFAI. The
calendar classifies each day, month, season or year into four classes (low,
medium, high, very high) with the 50th, 80th and 95th percentiles of the same
area and the same time step from 2003 onwards, so a class always means "for
this place". Areas drawn by the user are aggregated on request from the grid.

Critical burning periods are runs of days whose harmonized value exceeds a
chosen percentile of the area's own record, either over all days or within
the same calendar month. The percentile, the minimum length and the gap
allowed inside a run are adjustable in the interface.

## 10. Anomaly detection

`ml/anomaly/detector.py` describes each day of an area's harmonized series
relative to the same time of year in the other years (a window of ±15 days,
leaving the scored year out): log ratio of the 7-day mean to its seasonal
reference for HFAI, count and FRP, seasonal percentile rank, single-day spike
and activity level. An Isolation Forest (200 trees) is fitted on these
features for all administrative areas from 2003 onwards (1,419,769 area-days).

Severity classes are quantiles of the score (98 %, 99.5 %, 99.9 %) computed
separately for the country, the provinces and the districts, because small
areas are noisier than large ones. Consecutive flagged days are merged into
events, and each event reports its peak, its departure from the seasonal
reference and which features drove the score.

The model is unsupervised: there is no labelled set of "true" anomalies, so no
accuracy figure is given. It flags unusual past activity and is not a
forecast.

## 11. Reproducibility

- `python -m scripts.run_pipeline` rebuilds everything from the NASA files.
- Random seeds are fixed (`RANDOM_SEED = 42`); LightGBM runs in deterministic
  mode.
- Each model directory stores its parameters, the search log, the validation
  and test metrics, the HFAI references, the background table and a
  fingerprint of the input files (hash of the manifest checksums).
- Reports in `docs/` that say "generated by" are written by the scripts and
  are not edited by hand.
