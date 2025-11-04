# Model card

Two models are part of NYELETI AI. Both are trained by scripts in this
repository on NASA FIRMS data only; neither uses a pre-trained component.

---

## 1. Harmonization model (`hfai-lgbm-5km-20261115-1941`)

### Purpose

Estimate, from MODIS active fire detections, what VIIRS S-NPP would have
reported for the same 5 km cell and day: detection count and FRP sum. Summed
over cells (plus a background rate) the estimates give a fire activity record
on a single scale from 2001 to 2025.

### Not intended for

- Locating or confirming individual fires. The estimate of a single cell-day
  is weak (see below); the model is meant to be aggregated.
- Real-time monitoring or forecasting. It needs the MODIS data of the
  following day and standard-processing inputs.
- Burned area, emissions or damage estimates. Active fire detections are not
  burned area.
- Areas outside Mozambique without retraining.

### Data

| | |
|---|---|
| Input | MODIS MCD14ML Collection 6.1 (Terra, Aqua), vegetation fires, confidence >= 30 |
| Target | VIIRS S-NPP VNP14IMGTML, vegetation fires, nominal or high confidence |
| Unit | MODIS-informed cell-day on a 5 km equal-area grid (EPSG:6933), UTC days |
| Training | 2012-2022: 12,848,114 pairs available, 6,000,000 sampled |
| Validation | 2023-2024: 2,281,155 pairs |
| Test | 2025: 1,081,853 pairs |

### Algorithm

LightGBM gradient-boosted trees, one model per target.

| Target | Objective | Trees | Leaves | Min rows per leaf | Validation RMSE |
|---|---|---|---|---|---|
| VIIRS detection count | Tweedie, variance power 1.3 | 336 | 63 | 200 | 1.280 |
| VIIRS FRP sum (MW) | Tweedie, variance power 1.5 | 176 | 63 | 200 | 18.51 |

Learning rate 0.08, feature fraction 0.8, bagging fraction 0.7, L2 = 1,
127 bins, early stopping on the validation years. Objective and tree size were
chosen among four candidates for the count and three for the FRP, by
validation RMSE; the full search log is in `model.json`.

21 features: MODIS aggregates of the cell, of its eight neighbours and of the
adjacent days, month, season and cell position (list in `methodology.md`).
Most of the gain comes from the MODIS FRP sum, the MODIS mean FRP and the MODIS
count of the cell (`figures/feature_importance.png`).

A climatological background rate, by 25 km block and calendar month, accounts
for VIIRS detections in cell-days with no MODIS detection nearby. It is part
of every harmonized total.

### Performance on the independent test year (2025)

Full tables: `results.md`. Reference: VIIRS S-NPP observed.

HFAI by province and day (3,570 values):

| Method | MAE | RMSE | R² | Bias |
|---|---|---|---|---|
| Original MODIS | 40.9 | 122.6 | 0.572 | -57.6 % |
| Linear regression | 28.2 | 79.7 | 0.819 | +0.3 % |
| Quantile mapping | 31.0 | 92.2 | 0.758 | -2.4 % |
| **LightGBM + background** | **24.4** | **65.0** | **0.880** | -1.1 % |

HFAI by province and month (120 values): RMSE 595 and R² 0.981, against 925
and 0.955 for the linear regression and 2,818 and 0.578 for original MODIS.

National daily totals (357 days): RMSE 319 and R² 0.933, against 405 and 0.893
for the linear regression.

High-activity days by province (325 observed): precision 0.815, recall 0.717,
F1 0.763. Original MODIS on the same thresholds: precision 0.941, recall
0.342, F1 0.501.

National detection total for 2025: harmonized 601,937 against 609,745 observed
by VIIRS (ratio 0.987); MODIS observed 122,267 (ratio 0.201).

### Where it is weak

- **Single cell-days.** R² is 0.24 for the count and 0.23 for HFAI. Original
  MODIS even has the lowest MAE at this level (0.191 against 0.234 for HFAI),
  because most cell-days are zero in both sensors and predicting small
  positive values everywhere costs absolute error. For comparison, NOAA-20,
  the same instrument as the target 50 minutes apart, reaches R² = 0.45 on the
  same rows. The value of the model is in the aggregates.
- **Rank correlation.** Spearman correlation at province-day level is 0.845,
  not better than the linear regression (0.854) or original MODIS (0.842).
  The model corrects the scale and the size of the peaks; MODIS already ranks
  days well.
- **Bias.** The linear regression has the smallest absolute bias on the test
  year (+0.3 % against -1.1 %).
- **MODIS gaps.** With Aqua missing from 16 August to 1 September 2020 the
  harmonized national total of 2020 is 11.5 % below VIIRS. Over 2012-2025 the
  yearly harmonized / VIIRS ratio ranges from 0.885 to 1.047.
- **Low season.** Values are small and mostly background; relative errors are
  large. Example: Sofala, January to April 2024, harmonized is about half of
  VIIRS.
- **Before 2012.** No VIIRS exists to check the estimates. 2001-2002 have
  incomplete MODIS coverage.
- **FRP.** VIIRS FRP per cell-day is noisier than the count (cell-day R² 0.19).

### Uncertainty

The model gives point estimates; no prediction interval is produced. The
tables above and the yearly ratios in `results.md` are the available measure
of error. For yearly national totals the observed range of the harmonized /
VIIRS ratio (0.885 to 1.047 over 14 years, 11 of them used for training) is a
reasonable guide.

### Files

`ml/artifacts/hfai-lgbm-5km-20261115-1941/`: `harmonization_count.lgb.txt`,
`harmonization_frp.lgb.txt`, `baselines.joblib`, `background.parquet`,
`model.json` (parameters, search, references, input fingerprint),
`metrics.json`.

---

## 2. Anomaly model (Isolation Forest)

### Purpose

Point out days and periods when the harmonized activity of an area was
unusual for that area and that time of year, between 2003 and 2025. It is a
screening aid for analysts, and it does not forecast.

### Data and features

Daily harmonized series of the country, 10 provinces and 158 districts, 2003
to 2025: 1,419,769 area-days. Six features per day: log ratio of the 7-day
mean to the seasonal reference for HFAI, for the count and for the FRP;
seasonal percentile rank; single-day spike; activity level relative to the
area's own 99th percentile. The seasonal reference of a day is the mean of the
same ±15-day window in all other years.

### Algorithm

scikit-learn `IsolationForest`, 200 trees, 4,096 samples per tree, seed 42.
Severity is a quantile of the score within the administrative level:
moderate above 98 %, high above 99.5 %, extreme above 99.9 %. By construction
2 % of the area-days of each level are flagged (28,398 in total: 21,297
moderate, 5,679 high, 1,422 extreme). For an area drawn by the user the
thresholds of the level with the closest typical size are used.

### Validation

There is no labelled reference of anomalous fire days, so the model has no
accuracy figure. What has been checked:

- the flagged periods coincide with visible departures from the seasonal
  reference (`figures/anomaly_examples.png`);
- the strongest national event, 29 August to 6 September 2008 (+166 % against
  the seasonal reference at the peak), falls in the year with the highest
  MODIS and harmonized totals of the record.

### Limits

- The share of flagged days is fixed by the quantiles: the model ranks, it
  does not decide how many anomalies exist.
- It works on the harmonized series and inherits its errors. A MODIS outage
  can appear as a "below normal" period.
- Small districts have few detections; a single cluster of fires can flag a
  day there.
- The reference includes all other years, also later ones, so the model is
  retrospective.

### Files

`ml/artifacts/anomaly/isolation_forest.joblib`, `anomaly.json`.
