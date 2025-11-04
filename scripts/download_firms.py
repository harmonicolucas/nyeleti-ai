"""Download NASA FIRMS active fire archives (MODIS and VIIRS) for the study country.

Source: FIRMS yearly country summaries (standard-processing science data),
https://firms.modaps.eosdis.nasa.gov/country/

Every file is recorded in raw/manifest.json with its URL, product, checksum,
record count and covered period, so each number shown later can be traced back
to a NASA file. Files already present are skipped.

    python -m scripts.download_firms
    python -m scripts.download_firms --sensors modis viirs_snpp --years 2012 2025
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

from ml import config

MANIFEST = config.RAW_DIR / "manifest.json"


def archive_url(sensor: str, year: int) -> str:
    slug = config.SENSORS[sensor]["slug"]
    return f"{config.FIRMS_ARCHIVE}/{slug}/{year}/{slug}_{year}_{config.COUNTRY_NAME}.csv"


def target_path(sensor: str, year: int) -> Path:
    return config.RAW_DIR / sensor / f"{sensor}_{year}_{config.COUNTRY_ISO3}.csv"


def describe(path: Path) -> dict:
    """Checksum, record count and coverage of a downloaded CSV."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    n = 0
    dates_min, dates_max = None, None
    versions, satellites = set(), set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        for row in reader:
            n += 1
            d = row.get("acq_date")
            if d:
                dates_min = d if dates_min is None or d < dates_min else dates_min
                dates_max = d if dates_max is None or d > dates_max else dates_max
            versions.add(row.get("version", ""))
            satellites.add(row.get("satellite", ""))
    return {
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "records": n,
        "columns": columns,
        "date_min": dates_min,
        "date_max": dates_max,
        "versions": sorted(v for v in versions if v),
        "satellites": sorted(s for s in satellites if s),
    }


def fetch(sensor: str, year: int, retries: int = 4) -> dict | None:
    url = archive_url(sensor, year)
    dest = target_path(sensor, year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as resp:
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                expected = int(resp.headers.get("content-length", 0))
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            if expected and tmp.stat().st_size != expected:
                raise IOError(f"incomplete download ({tmp.stat().st_size} of {expected} bytes)")
            tmp.replace(dest)
            break
        except (requests.RequestException, IOError) as exc:
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {dest.name}: {exc}", flush=True)
            time.sleep(3 * attempt)

    entry = {
        "sensor": sensor,
        "year": year,
        "file": str(dest.relative_to(config.RAW_DIR)).replace("\\", "/"),
        "url": url,
        "source": "NASA FIRMS country yearly summary",
        "product": config.SENSORS[sensor]["product"],
        "instrument": config.SENSORS[sensor]["instrument"],
        "country": config.COUNTRY_NAME,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entry.update(describe(dest))
    return entry


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"files": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sensors", nargs="+", default=list(config.SENSORS), choices=list(config.SENSORS))
    parser.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    jobs = []
    for sensor in args.sensors:
        first = config.SENSORS[sensor]["first_year"]
        last = config.LAST_YEAR
        if args.years:
            first, last = max(first, args.years[0]), min(last, args.years[1])
        for year in range(first, last + 1):
            key = f"{sensor}/{year}"
            if not args.force and key in manifest["files"] and target_path(sensor, year).exists():
                continue
            jobs.append((sensor, year))

    if not jobs:
        print("Nothing to download, manifest is up to date.")
        return 0

    print(f"Downloading {len(jobs)} file(s) from {config.FIRMS_ARCHIVE}")
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, s, y): (s, y) for s, y in jobs}
        for fut in as_completed(futures):
            sensor, year = futures[fut]
            try:
                entry = fut.result()
            except Exception as exc:  # network errors after all retries
                failed.append((sensor, year, str(exc)))
                print(f"  FAILED {sensor} {year}: {exc}", flush=True)
                continue
            if entry is None:
                print(f"  not published: {sensor} {year}", flush=True)
                continue
            manifest["files"][f"{sensor}/{year}"] = entry
            save_manifest(manifest)
            print(
                f"  {sensor} {year}: {entry['records']:,} records, "
                f"{entry['bytes'] / 1e6:.1f} MB, {entry['date_min']} to {entry['date_max']}",
                flush=True,
            )

    total = sum(e["records"] for e in manifest["files"].values())
    print(f"Manifest: {len(manifest['files'])} files, {total:,} records")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
