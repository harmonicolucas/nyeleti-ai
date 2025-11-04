"""Checks on the building blocks that do not need the downloaded data."""
import numpy as np
import pandas as pd
import pytest

from ml import config
from ml.anomaly import detector as det
from ml.evaluation import metrics
from ml.preprocessing import features as ft
from ml.preprocessing import grid
from ml.training.baselines import QuantileMapping


def test_cell_id_round_trip():
    ix = np.array([-1200, 0, 3411, 6999])
    iy = np.array([-3100, -2, 0, 4500])
    back_ix, back_iy = grid.split_id(grid.cell_id(ix, iy))
    assert (back_ix == ix).all() and (back_iy == iy).all()


@pytest.mark.parametrize("cell_km", [1, 5, 10])
def test_point_falls_in_its_cell(cell_km):
    lon = np.array([32.57, 35.0, 40.6])
    lat = np.array([-25.97, -17.5, -10.9])
    ix, iy = grid.cell_index(lon, lat, cell_km)
    west, south, east, north = grid.cell_bounds(ix, iy, cell_km)
    assert ((west <= lon) & (lon < east)).all()
    assert ((south <= lat) & (lat < north)).all()


def test_cells_have_equal_area():
    # same width in degrees everywhere, height shrinking towards the equator
    w1, s1, e1, n1 = grid.cell_bounds(np.array([6500]), np.array([-600]), 5)
    w2, s2, e2, n2 = grid.cell_bounds(np.array([6500]), np.array([-250]), 5)
    assert (e1 - w1) == pytest.approx(e2 - w2)
    assert (n1 - s1) > (n2 - s2)
    # areas on a sphere; the grid is defined on the ellipsoid, hence the tolerance
    area1 = (e1 - w1) * (np.sin(np.radians(n1)) - np.sin(np.radians(s1)))
    area2 = (e2 - w2) * (np.sin(np.radians(n2)) - np.sin(np.radians(s2)))
    assert area1 == pytest.approx(area2, rel=5e-3)


def test_key_round_trip():
    day = np.array([11323, 19000, 20450])
    cell = np.array([grid.cell_id(-1200, -3100), grid.cell_id(0, 0), grid.cell_id(6999, 4500)]).ravel()
    back_day, back_cell = ft.split_key(ft.make_key(day, cell))
    assert (back_day == day).all() and (back_cell == cell).all()


def test_background_blocks_keep_their_size():
    assert config.background_block_cells(5) == 5
    assert config.background_block_cells(10) * 10 == pytest.approx(config.BACKGROUND_BLOCK_KM, abs=5)
    assert config.background_block_cells(1) == 25


def test_hfai_is_additive():
    ref = {"weight_count": 0.5, "weight_frp": 0.5, "count_ref": 2.4, "frp_ref": 22.0}
    count = np.array([1.0, 4.0, 0.0])
    frp = np.array([10.0, 80.0, 0.0])
    assert config.hfai(count, frp, ref).sum() == pytest.approx(config.hfai(count.sum(), frp.sum(), ref))
    assert config.hfai(ref["count_ref"], ref["frp_ref"], ref) == pytest.approx(1.0)


def _modis_rows(records):
    rows = pd.DataFrame(records, columns=["date", "cell_id", "modis_fire_count", "modis_frp_sum"])
    rows["date"] = pd.to_datetime(rows["date"])
    for col in ft.MODIS_CELL_COLUMNS:
        if col not in rows:
            rows[col] = 0.0
    return rows


def test_informed_cell_days_and_context():
    centre = int(grid.cell_id(100, -400))
    east = int(grid.cell_id(101, -400))
    far = int(grid.cell_id(140, -400))
    ixs, iys = np.meshgrid(np.arange(95, 145), np.arange(-405, -395))
    cells = pd.DataFrame({"cell_id": grid.cell_id(ixs.ravel(), iys.ravel()), "lat": -18.0, "lon": 35.0})
    modis = _modis_rows([("2020-08-10", centre, 3, 60.0), ("2020-08-11", east, 2, 25.0)])
    first = int(ft.day_number(["2020-08-01"])[0])
    last = int(ft.day_number(["2020-08-31"])[0])
    rows = ft.build_informed(modis, cells, first, last)

    # two detections one cell apart on consecutive days: 9 cells x 3 days and 9 x 3, overlapping
    assert rows["date"].min() == pd.Timestamp("2020-08-09")
    assert rows["date"].max() == pd.Timestamp("2020-08-12")
    assert far not in set(rows["cell_id"])

    row = rows[(rows["cell_id"] == centre) & (rows["date"] == "2020-08-10")].iloc[0]
    assert row["modis_fire_count"] == 3
    assert row["nbr_fire_count"] == 0
    assert row["next_fire_count"] == 0
    assert row["nbr_next_fire_count"] == 2

    row = rows[(rows["cell_id"] == east) & (rows["date"] == "2020-08-10")].iloc[0]
    assert row["modis_fire_count"] == 0
    assert row["nbr_fire_count"] == 3
    assert row["nbr_frp_sum"] == pytest.approx(60.0)
    assert row["next_fire_count"] == 2


def test_regression_metrics():
    m = metrics.regression_metrics([0, 2, 4, 6], [1, 3, 5, 7])
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(1.0)
    assert m["bias_pct"] == pytest.approx(100.0 / 3.0)
    assert m["r2"] == pytest.approx(0.8)
    assert m["spearman"] == pytest.approx(1.0)


def test_detection_metrics():
    m = metrics.detection_metrics([1, 1, 1, 0, 0], [1, 1, 0, 1, 0])
    assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (2, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_quantile_mapping_keeps_the_mean():
    rng = np.random.default_rng(3)
    modis = rng.poisson(0.4, 20000).astype("float32")
    viirs = rng.poisson(1.0 + 3.0 * modis).astype("float64")
    rows = pd.DataFrame({"modis_fire_count": modis})
    mapped = QuantileMapping("count").fit(rows, viirs).predict(rows)
    assert mapped.mean() == pytest.approx(viirs.mean(), rel=1e-6)
    assert (np.diff(mapped[np.argsort(modis, kind="stable")]) >= -1e-9).all()


def test_seasonal_reference_leaves_the_year_out():
    dates = pd.date_range("2010-01-01", "2014-12-31", freq="D")
    values = np.ones(len(dates))
    values[dates.year == 2012] = 10.0
    ref, rank = det.seasonal_reference(values, dates)
    assert ref[dates.year == 2012] == pytest.approx(1.0)
    assert ref[dates.year == 2010] == pytest.approx(13.0 / 4.0)
    assert rank[dates.year == 2012] == pytest.approx(1.0)


def test_anomaly_features_flag_an_unusual_year():
    dates = pd.date_range("2005-01-01", "2014-12-31", freq="D")
    season = 50.0 * np.exp(-0.5 * ((dates.day_of_year.values - 250) / 25.0) ** 2) + 1.0
    hfai = season.copy()
    burst = (dates >= "2011-05-01") & (dates <= "2011-05-20")
    hfai[burst] += 40.0
    daily = pd.DataFrame({"date": dates, "hfai": hfai, "count": hfai * 2.0, "frp": hfai * 20.0})
    features = det.build_features(daily)
    assert list(features.columns[:1]) == ["date"]
    assert set(det.FEATURES) <= set(features.columns)
    assert features.loc[burst, "log_ratio"].mean() > 1.0
    # an ordinary fire season, far from the burst, stays well below it
    ordinary = (dates.year == 2008) & (dates.month >= 8)
    assert features.loc[ordinary, "log_ratio"].abs().max() < 0.3
