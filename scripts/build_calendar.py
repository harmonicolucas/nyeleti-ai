"""Load everything into the analysis database and build the activity calendar.

Tables (DuckDB file at config.DB_PATH):
  fire_detections      cleaned MODIS and VIIRS detections, as observed
  geographic_areas     country, provinces, districts
  grid_cells           cells of the common grid and their admin units
  area_cells           membership of cells in areas
  observed_activity    MODIS / VIIRS counts and FRP per cell and day
  harmonized_activity  model estimates per MODIS-informed cell and day
  background_activity  background rate per cell and month
  calendar_activity    daily series per area: observed and harmonized
  sensor_days          daily national totals and coverage flags per sensor
  model_versions, harmonization_runs, data_files, metadata

Observations and model output live in separate tables and columns; nothing
observed is ever overwritten by an estimate.

    python -m scripts.build_calendar
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd
from shapely.geometry import shape

from ml import config
from ml.inference.harmonize import Harmonizer
from ml.preprocessing import background as bgmod


def posix(path) -> str:
    return str(path).replace("\\", "/")


def row_count(con, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchall()[0][0]


def load_detections(con) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE fire_detections AS
        SELECT sensor, instrument, satellite, acq_date, acq_time, latitude, longitude, frp,
               brightness, brightness_ir, scan, track, confidence, confidence_level, confidence_pct,
               daynight, version, type, analysis
        FROM read_parquet('{posix(config.CLEAN_DIR)}/*/*.parquet', union_by_name = true)
        ORDER BY acq_date, sensor
        """
    )


def load_areas(con, grid_dir) -> None:
    geojson = json.loads((grid_dir / "areas.geojson").read_text(encoding="utf-8"))
    rows = []
    for feature in geojson["features"]:
        props = feature["properties"]
        west, south, east, north = shape(feature["geometry"]).bounds
        rows.append(
            {
                "area_id": props["area_id"],
                "name": props["name"],
                "level": props["level"],
                "parent_id": props.get("parent_id"),
                "west": west, "south": south, "east": east, "north": north,
                "geometry": json.dumps(feature["geometry"]),
            }
        )
    areas = pd.DataFrame(rows)
    con.register("areas_df", areas)
    con.execute("CREATE OR REPLACE TABLE geographic_areas AS SELECT * FROM areas_df")
    con.unregister("areas_df")

    con.execute(f"CREATE OR REPLACE TABLE grid_cells AS SELECT * FROM read_parquet('{posix(grid_dir)}/grid_cells.parquet')")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE area_cells AS
        SELECT '{config.COUNTRY_ISO3}' AS area_id, cell_id FROM grid_cells
        UNION ALL SELECT adm1_id, cell_id FROM grid_cells
        UNION ALL SELECT adm2_id, cell_id FROM grid_cells
        """
    )
    con.execute(
        """
        ALTER TABLE geographic_areas ADD COLUMN n_cells INTEGER;
        UPDATE geographic_areas g SET n_cells = c.n
        FROM (SELECT area_id, count(*) AS n FROM area_cells GROUP BY area_id) c WHERE c.area_id = g.area_id;
        """
    )


def load_activity(con, grid_dir, harmonized_dir, version: str) -> None:
    g = posix(grid_dir)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE observed_activity AS
        SELECT date, cell_id,
               sum(modis_count)::INTEGER AS modis_count, sum(modis_frp)::FLOAT AS modis_frp,
               sum(viirs_count)::INTEGER AS viirs_count, sum(viirs_frp)::FLOAT AS viirs_frp,
               sum(noaa20_count)::INTEGER AS noaa20_count, sum(noaa20_frp)::FLOAT AS noaa20_frp
        FROM (
            SELECT date::DATE AS date, cell_id, modis_fire_count AS modis_count, modis_frp_sum AS modis_frp,
                   0 AS viirs_count, 0 AS viirs_frp, 0 AS noaa20_count, 0 AS noaa20_frp
            FROM read_parquet('{g}/modis_*.parquet')
            UNION ALL
            SELECT date::DATE, cell_id, 0, 0, fire_count, frp_sum, 0, 0 FROM read_parquet('{g}/viirs_snpp_*.parquet')
            UNION ALL
            SELECT date::DATE, cell_id, 0, 0, 0, 0, fire_count, frp_sum FROM read_parquet('{g}/viirs_noaa20_*.parquet')
        )
        GROUP BY date, cell_id
        ORDER BY date, cell_id
        """
    )
    h = posix(harmonized_dir)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE harmonized_activity AS
        SELECT date::DATE AS date, cell_id, modis_fire_count::INTEGER AS modis_count,
               est_count, est_frp, bg_count, bg_frp, '{version}' AS model_version
        FROM read_parquet('{h}/harmonized_*.parquet')
        ORDER BY date, cell_id
        """
    )
    con.execute(
        f"CREATE OR REPLACE TABLE background_activity AS SELECT * FROM read_parquet('{h}/background_cells.parquet')"
    )


def build_calendar(con, days: pd.DataFrame, noaa20_start: str | None) -> None:
    first_day = days["date"].min().strftime("%Y-%m-%d")
    last_day = days["date"].max().strftime("%Y-%m-%d")
    con.register("days_df", days.assign(date=days["date"].dt.date))
    con.execute("CREATE OR REPLACE TABLE sensor_days AS SELECT * FROM days_df ORDER BY date")
    con.unregister("days_df")
    noaa_expr = (
        f"CASE WHEN d.date >= DATE '{noaa20_start}' THEN coalesce(o.noaa20_count, 0) END"
        if noaa20_start
        else "NULL::INTEGER"
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE calendar_activity AS
        WITH days AS (
            SELECT range::DATE AS date
            FROM range(DATE '{first_day}', DATE '{last_day}' + INTERVAL 1 DAY, INTERVAL 1 DAY)
        ),
        obs AS (
            SELECT a.area_id, o.date,
                   sum(o.modis_count) AS modis_count, sum(o.modis_frp) AS modis_frp,
                   sum(o.viirs_count) AS viirs_count, sum(o.viirs_frp) AS viirs_frp,
                   sum(o.noaa20_count) AS noaa20_count
            FROM observed_activity o JOIN area_cells a USING (cell_id)
            GROUP BY a.area_id, o.date
        ),
        est AS (
            SELECT a.area_id, h.date,
                   sum(h.est_count) AS est_count, sum(h.est_frp) AS est_frp,
                   sum(h.bg_count) AS bg_count, sum(h.bg_frp) AS bg_frp
            FROM harmonized_activity h JOIN area_cells a USING (cell_id)
            GROUP BY a.area_id, h.date
        ),
        bg AS (
            SELECT a.area_id, b.month, sum(b.bg_count) AS bg_count, sum(b.bg_frp) AS bg_frp
            FROM background_activity b JOIN area_cells a USING (cell_id)
            GROUP BY a.area_id, b.month
        )
        SELECT g.area_id, d.date,
               coalesce(o.modis_count, 0)::INTEGER AS modis_count,
               coalesce(o.modis_frp, 0)::FLOAT AS modis_frp,
               CASE WHEN s.viirs_available AND NOT s.viirs_gap THEN coalesce(o.viirs_count, 0)::INTEGER END AS viirs_count,
               CASE WHEN s.viirs_available AND NOT s.viirs_gap THEN coalesce(o.viirs_frp, 0)::FLOAT END AS viirs_frp,
               ({noaa_expr})::INTEGER AS noaa20_count,
               (coalesce(e.est_count, 0) + coalesce(bg.bg_count, 0) - coalesce(e.bg_count, 0))::FLOAT AS harmonized_count,
               (coalesce(e.est_frp, 0) + coalesce(bg.bg_frp, 0) - coalesce(e.bg_frp, 0))::FLOAT AS harmonized_frp,
               coalesce(e.est_count, 0)::FLOAT AS model_count,
               (coalesce(bg.bg_count, 0) - coalesce(e.bg_count, 0))::FLOAT AS background_count
        FROM geographic_areas g
        CROSS JOIN days d
        JOIN sensor_days s ON s.date = d.date
        LEFT JOIN obs o ON o.area_id = g.area_id AND o.date = d.date
        LEFT JOIN est e ON e.area_id = g.area_id AND e.date = d.date
        LEFT JOIN bg ON bg.area_id = g.area_id AND bg.month = month(d.date)
        ORDER BY g.area_id, d.date
        """
    )


def load_registry(con, model: Harmonizer) -> None:
    registry = json.loads((config.ARTIFACTS_DIR / "registry.json").read_text(encoding="utf-8"))
    versions, runs = [], []
    for item in registry["versions"]:
        directory = config.ARTIFACTS_DIR / item["version"]
        if not (directory / "model.json").exists():
            continue
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        metrics_path = directory / "metrics.json"
        versions.append(
            {
                "version": item["version"],
                "created_at": meta["created_at"],
                "cell_km": meta["cell_km"],
                "is_active": item["version"] == registry["active"],
                "meta": json.dumps(meta),
                "metrics": metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else None,
            }
        )
        runs_path = directory / "runs.json"
        if runs_path.exists():
            for run in json.loads(runs_path.read_text(encoding="utf-8")):
                runs.append({k: run[k] for k in ("run_at", "model_version", "first_year", "last_year", "cell_days")})
    con.register("versions_df", pd.DataFrame(versions))
    con.execute("CREATE OR REPLACE TABLE model_versions AS SELECT * FROM versions_df")
    con.unregister("versions_df")
    runs_df = pd.DataFrame(runs, columns=["run_at", "model_version", "first_year", "last_year", "cell_days"])
    con.register("runs_df", runs_df)
    con.execute("CREATE OR REPLACE TABLE harmonization_runs AS SELECT * FROM runs_df")
    con.unregister("runs_df")

    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    files = pd.DataFrame(
        [
            {
                "sensor": e["sensor"], "year": e["year"], "product": e["product"], "url": e["url"],
                "records": e["records"], "bytes": e["bytes"], "sha256": e["sha256"],
                "date_min": e["date_min"], "date_max": e["date_max"],
                "versions": ", ".join(e["versions"]), "downloaded_at": e["downloaded_at"],
            }
            for e in manifest["files"].values()
        ]
    )
    con.register("files_df", files)
    con.execute("CREATE OR REPLACE TABLE data_files AS SELECT * FROM files_df ORDER BY sensor, year")
    con.unregister("files_df")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=None)
    parser.add_argument("--skip-detections", action="store_true", help="keep the existing fire_detections table")
    args = parser.parse_args(argv)

    model = Harmonizer(args.version)
    grid_dir = config.GRID_DIR / config.grid_tag(model.cell_km)
    harmonized_dir = config.HARMONIZED_DIR / config.grid_tag(model.cell_km)
    days = bgmod.load_sensor_days(model.cell_km)
    first_day = days["date"].min().strftime("%Y-%m-%d")
    last_day = days["date"].max().strftime("%Y-%m-%d")
    viirs_start = days.loc[days["viirs_available"], "date"].min().strftime("%Y-%m-%d")
    noaa_days = days.loc[days["noaa20_count"] > 0, "date"]
    noaa20_start = noaa_days.min().strftime("%Y-%m-%d") if len(noaa_days) else None

    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH))
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute(f"PRAGMA temp_directory='{posix(config.DB_PATH.parent / 'tmp')}'")

    if not args.skip_detections:
        load_detections(con)
        print(f"fire_detections: {row_count(con, 'fire_detections'):,} rows", flush=True)
    load_areas(con, grid_dir)
    load_activity(con, grid_dir, harmonized_dir, model.version)
    for table in ("observed_activity", "harmonized_activity", "background_activity"):
        print(f"{table}: {row_count(con, table):,} rows", flush=True)
    build_calendar(con, days, noaa20_start)
    print(f"calendar_activity: {row_count(con, 'calendar_activity'):,} rows", flush=True)
    load_registry(con, model)

    quality_path = config.QUALITY_DIR / "quality_report.json"
    metadata = {
        "country": config.COUNTRY_NAME,
        "iso3": config.COUNTRY_ISO3,
        "cell_km": model.cell_km,
        "first_day": first_day,
        "last_day": last_day,
        "viirs_start": viirs_start,
        "viirs_gap_days": int(days["viirs_gap"].sum()),
        "noaa20_start": noaa20_start,
        "active_model": model.version,
        "hfai": model.meta["hfai"],
        "split": model.meta["split"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quality": json.loads(quality_path.read_text(encoding="utf-8"))["totals"] if quality_path.exists() else None,
    }
    meta_df = pd.DataFrame([{"key": k, "value": json.dumps(v)} for k, v in metadata.items()])
    con.register("meta_df", meta_df)
    con.execute("CREATE OR REPLACE TABLE metadata AS SELECT * FROM meta_df")
    con.unregister("meta_df")
    con.execute("CHECKPOINT")
    con.close()
    print(f"Database ready: {config.DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
