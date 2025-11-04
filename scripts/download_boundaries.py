"""Download administrative boundaries (country, provinces, districts) from geoBoundaries.

    python -m scripts.download_boundaries
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import requests

from ml import config

API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{level}/"
LEVELS = ("ADM0", "ADM1", "ADM2")


def main() -> int:
    config.BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    sources = {}
    for level in LEVELS:
        meta = requests.get(API.format(iso3=config.COUNTRY_ISO3, level=level), timeout=60).json()
        url = meta.get("simplifiedGeometryGeoJSON") or meta["gjDownloadURL"]
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        geojson = resp.json()
        out = config.BOUNDARIES_DIR / f"{config.COUNTRY_ISO3}_{level}.geojson"
        out.write_text(json.dumps(geojson), encoding="utf-8")
        sources[level] = {
            "boundary_id": meta["boundaryID"],
            "source": meta["boundarySource"],
            "license": meta["boundaryLicense"],
            "year": meta["boundaryYearRepresented"],
            "units": len(geojson["features"]),
            "url": url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"{level}: {len(geojson['features'])} unit(s) from {meta['boundarySource']}")
    (config.BOUNDARIES_DIR / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
