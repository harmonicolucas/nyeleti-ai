"""Export a curated API snapshot for the GitHub Pages static app."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "app" / "static-api"
BASE = "http://127.0.0.1:8000"


def key(path: str, params: dict | None = None) -> str:
    params = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
    items = sorted((str(k), str(v)) for k, v in params.items())
    query = "&".join(f"{k}={v}" for k, v in items)
    raw = path.lstrip("/") + (f"__{query}" if query else "")
    return re.sub(r"[^\w.=\/-]+", "_", raw) + ".json"


def fetch(path: str, params: dict | None = None) -> object:
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None and v != ""})
    url = f"{BASE}{path}" + (f"?{query}" if query else "")
    with urllib.request.urlopen(url, timeout=180) as res:
        return json.loads(res.read().decode("utf-8"))


def save(path: str, params: dict | None = None) -> Path:
    data = fetch(path, params)
    dest = OUT / key(path, params)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {dest.name}  ({dest.stat().st_size // 1024} KB)")
    return dest


def main() -> None:
    if OUT.exists():
        for old in OUT.glob("*.json"):
            old.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = fetch("/api/meta")
    year = int(str(meta["record"]["last_day"])[:4])
    iso3 = meta["iso3"]
    viirs_year = str(meta["record"]["viirs_start"])[:4]
    test = meta["split"]["test"]
    trends = fetch("/api/trends", {"area_id": iso3, "metric": "hfai"})
    months = trends["climatology"]["months"]
    peak = trends["climatology"]["peak_month"]
    month = f"{months.index(peak) + 1:02d}"
    map_start = f"{year}-{month}-01"
    map_end = f"{year}-{month}-07"
    print(f"Exporting static API for {iso3}, year {year}, map {map_start}→{map_end}")

    save("/api/meta")
    save("/api/health")
    save("/api/areas")
    save("/api/areas/geojson", {"level": "ADM0"})
    save("/api/areas/geojson", {"level": "ADM1"})
    save("/api/areas/resolve", {"area_id": iso3})
    save("/api/model/metrics")

    for metric in ("hfai", "count", "frp"):
        save("/api/dashboard", {"area_id": iso3, "year": year})
        save("/api/calendar", {"area_id": iso3, "granularity": "year", "metric": metric})
        save("/api/calendar", {"area_id": iso3, "granularity": "month", "metric": metric})
        save("/api/calendar", {"area_id": iso3, "granularity": "season", "metric": metric})
        save("/api/calendar", {"area_id": iso3, "granularity": "day", "metric": metric, "year": year})
        save("/api/harmonization", {
            "area_id": iso3, "metric": metric, "freq": "month",
            "start": f"{year}-01-01", "end": f"{year}-12-31",
        })
        save("/api/harmonization", {"area_id": iso3, "metric": metric, "freq": "month", "start": f"{viirs_year}-01-01"})
        save("/api/harmonization", {"area_id": iso3, "metric": metric, "freq": "year", "start": f"{viirs_year}-01-01"})
        save("/api/harmonization", {"area_id": iso3, "metric": metric, "freq": "day", "start": f"{viirs_year}-01-01"})
        save("/api/harmonization", {
            "area_id": iso3, "metric": metric, "freq": "month",
            "start": f"{test[0]}-01-01", "end": f"{test[1]}-12-31",
        })
        save("/api/trends", {"area_id": iso3, "metric": metric})
        save("/api/critical", {
            "area_id": iso3, "metric": metric, "percentile": 90,
            "min_days": 3, "max_gap": 1, "basis": "area",
        })

    # dashboard is independent of metric — saved once above; keep one call only next time
    save("/api/critical/regions", {"level": "ADM1"})
    save("/api/critical/regions", {"level": "ADM2"})

    save("/api/anomalies", {
        "area_id": iso3, "direction": "above", "min_severity": "moderate", "limit": 60,
    })
    save("/api/anomalies", {
        "area_id": iso3, "start": f"{year}-01-01", "end": f"{year}-12-31",
        "direction": "above", "min_severity": "moderate", "limit": 60,
    })
    save("/api/anomalies/series", {"area_id": iso3, "year": year})
    save("/api/anomalies/overview", {
        "level": "ADM1", "start": f"{year}-01-01", "end": f"{year}-12-31", "direction": "above",
    })

    save("/api/fires", {
        "area_id": iso3, "start": map_start, "end": map_end,
        "sensors": "modis,viirs_snpp", "min_confidence": "low", "limit": 25000,
    })
    for metric in ("hfai", "count", "frp"):
        save("/api/harmonization/grid", {
            "area_id": iso3, "start": map_start, "end": map_end,
            "layer": "harmonized", "metric": metric, "min_value": 0.1 if metric == "hfai" else (0.2 if metric == "count" else 5),
        })
        save("/api/harmonization/grid", {
            "area_id": iso3, "start": map_start, "end": map_end,
            "layer": "modis", "metric": metric, "min_value": 0,
        })
        save("/api/harmonization/grid", {
            "area_id": iso3, "start": map_start, "end": map_end,
            "layer": "viirs", "metric": metric, "min_value": 0,
        })

    areas = fetch("/api/areas")["areas"]
    provinces = [a for a in areas if a.get("level") == "ADM1"][:4]
    for prov in provinces:
        aid = prov["area_id"]
        save("/api/areas/resolve", {"area_id": aid})
        save("/api/dashboard", {"area_id": aid, "year": year})
        save("/api/calendar", {"area_id": aid, "granularity": "day", "metric": "hfai", "year": year})
        save("/api/calendar", {"area_id": aid, "granularity": "year", "metric": "hfai"})
        save("/api/calendar", {"area_id": aid, "granularity": "month", "metric": "hfai"})
        save("/api/harmonization", {
            "area_id": aid, "metric": "hfai", "freq": "month",
            "start": f"{year}-01-01", "end": f"{year}-12-31",
        })
        save("/api/areas/geojson", {"level": "ADM2", "parent_id": aid})
        save("/api/trends", {"area_id": aid, "metric": "hfai"})
        save("/api/anomalies", {
            "area_id": aid, "direction": "above", "min_severity": "moderate", "limit": 60,
        })
        save("/api/anomalies/series", {"area_id": aid, "year": year})

    print(f"Done → {OUT} ({sum(p.stat().st_size for p in OUT.glob('*.json')) // 1024} KB total)")


if __name__ == "__main__":
    main()
