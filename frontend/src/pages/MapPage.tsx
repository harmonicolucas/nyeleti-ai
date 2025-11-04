import maplibregl, { type GeoJSONSource, type Map as MlMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import { Async, Field, PageHead, Panel, fmt } from "../components/ui";
import { selectionParams, type Area, type Metric, type SelectionInfo } from "../services/api";
import { useApi, useApp } from "../state";

interface FiresData {
  total: number;
  returned: number;
  truncated: boolean;
  by_sensor: Record<string, number>;
  columns: string[];
  rows: (string | number | null)[][];
}

interface GridData {
  cells: number[][];
  max: number;
  total: number;
  truncated: boolean;
  layer_label: string;
  note: string;
}

interface TrendsLite {
  climatology: { peak_month: string; months: string[] };
}

type GridLayer = "none" | "harmonized" | "modis" | "viirs";

const SENSORS = [
  { id: "modis", label: "MODIS (Terra, Aqua)", color: "#58a6ff", radius: 4.2 },
  { id: "viirs_snpp", label: "VIIRS S-NPP", color: "#34d3b4", radius: 2.6 },
  { id: "viirs_noaa20", label: "VIIRS NOAA-20", color: "#e879f9", radius: 2.6 },
] as const;

// harmonized cells below these totals are left out of the map layer
const MIN_ESTIMATE: Record<Metric, number> = { hfai: 0.1, count: 0.2, frp: 5 };

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    bluemarble: {
      type: "raster",
      tiles: [
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/" +
          "GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg",
      ],
      tileSize: 256,
      maxzoom: 8,
      attribution: "NASA Blue Marble (GIBS)",
    },
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#0a0908" } },
    {
      id: "base-bluemarble", type: "raster", source: "bluemarble",
      paint: { "raster-brightness-max": 0.5, "raster-saturation": -0.45, "raster-contrast": 0.1 },
    },
    {
      id: "base-osm", type: "raster", source: "osm", layout: { visibility: "none" },
      paint: { "raster-brightness-max": 0.55, "raster-saturation": -0.7 },
    },
  ],
};

function gibsTiles(date: string): string[] {
  return [
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/" +
      `${date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
  ];
}

function addDays(date: string, days: number): string {
  const d = new Date(`${date}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function source(map: MlMap, id: string): GeoJSONSource {
  return map.getSource(id) as GeoJSONSource;
}

export function MapPage() {
  const { selection, setSelection, meta, metric, t } = useApp();
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const [ready, setReady] = useState(false);

  const lastYear = meta.record.last_day.slice(0, 4);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [sensors, setSensors] = useState<string[]>(["modis", "viirs_snpp"]);
  const [confidence, setConfidence] = useState("low");
  const [daynight, setDaynight] = useState("");
  const [showPoints, setShowPoints] = useState(true);
  const [gridLayer, setGridLayer] = useState<GridLayer>("harmonized");
  const [imagery, setImagery] = useState(false);
  const [base, setBase] = useState<"bluemarble" | "osm">("bluemarble");
  const [drawing, setDrawing] = useState(false);
  const [vertices, setVertices] = useState<[number, number][]>([]);

  const drawingRef = useRef(drawing);
  drawingRef.current = drawing;
  const selectRef = useRef<(id: string) => void>(() => {});

  const sel = selectionParams(selection);
  const areas = useApi<{ areas: Area[] }>("/api/areas");
  const provinces = useApi<GeoJSON.FeatureCollection>("/api/areas/geojson", { level: "ADM1" });
  const provinceId =
    selection.kind === "area"
      ? selection.level === "ADM1"
        ? selection.areaId
        : areas.data?.areas.find((a) => a.area_id === selection.areaId)?.parent_id ?? null
      : null;
  const districts = useApi<GeoJSON.FeatureCollection>(
    provinceId && selection.kind === "area" && selection.level !== "ADM0" ? "/api/areas/geojson" : null,
    { level: "ADM2", parent_id: provinceId },
  );
  const info = useApi<SelectionInfo>("/api/areas/resolve", sel);

  // open on the first week of the country's peak fire month, where there is something to see
  const peak = useApi<TrendsLite>("/api/trends", { area_id: meta.iso3, metric: "hfai" });
  useEffect(() => {
    if (!peak.data || start) return;
    const month = String(peak.data.climatology.months.indexOf(peak.data.climatology.peak_month) + 1).padStart(2, "0");
    setStart(`${lastYear}-${month}-01`);
    setEnd(`${lastYear}-${month}-07`);
  }, [peak.data, start, lastYear]);

  const period = start && end && start <= end ? { start, end } : null;
  const fires = useApi<FiresData>(period && showPoints && sensors.length ? "/api/fires" : null, {
    ...sel, ...period, sensors: sensors.join(","), min_confidence: confidence, daynight, limit: 25000,
  });
  const grid = useApi<GridData>(period && gridLayer !== "none" ? "/api/harmonization/grid" : null, {
    ...sel, ...period, layer: gridLayer, metric,
    min_value: gridLayer === "harmonized" ? MIN_ESTIMATE[metric] : 0,
  });

  selectRef.current = (areaId: string) => {
    const area = areas.data?.areas.find((a) => a.area_id === areaId);
    if (area) setSelection({ kind: "area", areaId: area.area_id, name: area.name, level: area.level });
  };

  useEffect(() => {
    if (!container.current) return;
    const map = new maplibregl.Map({
      container: container.current,
      style: STYLE,
      center: [35.5, -18.5],
      zoom: 4.6,
      attributionControl: { compact: true },
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

    map.on("load", () => {
      for (const id of ["grid", "provinces", "districts", "selection", "fires", "draw"]) {
        map.addSource(id, { type: "geojson", data: EMPTY });
      }
      map.addLayer({
        id: "grid", type: "fill", source: "grid",
        paint: {
          "fill-color": ["interpolate", ["linear"], ["get", "n"], 0, "#3b1d0a", 0.35, "#b45309", 0.7, "#f59e0b", 1, "#fde68a"],
          "fill-opacity": 0.72,
        },
      });
      map.addLayer({ id: "provinces-fill", type: "fill", source: "provinces", paint: { "fill-color": "#ffffff", "fill-opacity": 0 } });
      map.addLayer({ id: "provinces-line", type: "line", source: "provinces", paint: { "line-color": "#8a7f76", "line-width": 1 } });
      map.addLayer({ id: "districts-fill", type: "fill", source: "districts", paint: { "fill-color": "#ffffff", "fill-opacity": 0 } });
      map.addLayer({ id: "districts-line", type: "line", source: "districts", paint: { "line-color": "#6f665e", "line-width": 0.7, "line-dasharray": [2, 2] } });
      map.addLayer({ id: "selection-line", type: "line", source: "selection", paint: { "line-color": "#f59e0b", "line-width": 2.2 } });
      for (const s of SENSORS) {
        map.addLayer({
          id: `fires-${s.id}`, type: "circle", source: "fires", filter: ["==", ["get", "sensor"], s.id],
          paint: {
            "circle-color": s.color,
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, s.radius * 0.55, 8, s.radius, 12, s.radius * 2.2],
            "circle-opacity": 0.85,
            "circle-stroke-color": "#0b0a09",
            "circle-stroke-width": s.id === "modis" ? 0.8 : 0.3,
          },
        });
      }
      map.addLayer({ id: "draw-fill", type: "fill", source: "draw", filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": "#f59e0b", "fill-opacity": 0.12 } });
      map.addLayer({ id: "draw-line", type: "line", source: "draw", paint: { "line-color": "#f59e0b", "line-width": 2, "line-dasharray": [2, 1] } });
      map.addLayer({ id: "draw-points", type: "circle", source: "draw", filter: ["==", ["geometry-type"], "Point"], paint: { "circle-color": "#f59e0b", "circle-radius": 4 } });
      setReady(true);
    });

    const fireLayers = SENSORS.map((s) => `fires-${s.id}`);
    map.on("click", (ev) => {
      if (drawingRef.current) {
        setVertices((prev) => [...prev, [ev.lngLat.lng, ev.lngLat.lat]]);
        return;
      }
      const hit = map.queryRenderedFeatures(ev.point, { layers: fireLayers })[0];
      if (hit) {
        const p = hit.properties as Record<string, string | number>;
        const time = String(p.acq_time).padStart(4, "0");
        const rows = [
          ["Sensor", `${p.satellite} (${p.sensor === "modis" ? "MODIS 1 km" : "VIIRS 375 m"})`],
          ["Date (UTC)", `${p.acq_date} ${time.slice(0, 2)}:${time.slice(2)}`],
          ["FRP", `${p.frp} MW`],
          ["Brightness", `${p.brightness} K`],
          ["Confidence", `${p.confidence} (${p.confidence_level})`],
          ["Day / night", p.daynight === "D" ? "Day" : "Night"],
          ["Pixel (scan x track)", `${p.scan} x ${p.track} km`],
          ["Position", `${Number(p.latitude).toFixed(4)}, ${Number(p.longitude).toFixed(4)}`],
          ["Product version", String(p.version)],
        ];
        new maplibregl.Popup({ maxWidth: "300px" })
          .setLngLat(ev.lngLat)
          .setHTML(`<div class="popup-grid">${rows.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("")}</div>`)
          .addTo(map);
        return;
      }
      const district = map.queryRenderedFeatures(ev.point, { layers: ["districts-fill"] })[0];
      const province = map.queryRenderedFeatures(ev.point, { layers: ["provinces-fill"] })[0];
      const target = district ?? province;
      if (target?.properties?.area_id) selectRef.current(String(target.properties.area_id));
    });
    map.on("mousemove", (ev) => {
      if (drawingRef.current) return;
      const over = map.queryRenderedFeatures(ev.point, { layers: [...fireLayers, "provinces-fill"] }).length > 0;
      map.getCanvas().style.cursor = over ? "pointer" : "";
    });

    return () => {
      map.remove();
      mapRef.current = null;
      setReady(false);
    };
  }, []);

  // boundaries
  useEffect(() => {
    if (ready && provinces.data) source(mapRef.current!, "provinces").setData(provinces.data);
  }, [ready, provinces.data]);
  useEffect(() => {
    if (ready) source(mapRef.current!, "districts").setData(districts.data ?? EMPTY);
  }, [ready, districts.data]);

  // outline of the selection and zoom to it
  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current!;
    let outline: GeoJSON.FeatureCollection = EMPTY;
    if (selection.kind === "aoi") {
      outline = { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: selection.geometry }] };
    } else if (selection.level !== "ADM0") {
      const pool = [...(provinces.data?.features ?? []), ...(districts.data?.features ?? [])];
      const feature = pool.find((f) => f.properties?.area_id === selection.areaId);
      if (feature) outline = { type: "FeatureCollection", features: [feature] };
    }
    source(map, "selection").setData(outline);
  }, [ready, selection, provinces.data, districts.data]);

  useEffect(() => {
    if (!ready || !info.data) return;
    const [w, s, e, n] = info.data.bounds;
    mapRef.current!.fitBounds([[w, s], [e, n]], { padding: 40, duration: 600, maxZoom: 10 });
  }, [ready, info.data]);

  // detections
  const fireCollection = useMemo<GeoJSON.FeatureCollection>(() => {
    if (!fires.data || !showPoints) return EMPTY;
    const cols = fires.data.columns;
    return {
      type: "FeatureCollection",
      features: fires.data.rows.map((row) => {
        const props: Record<string, string | number | null> = {};
        cols.forEach((c, i) => (props[c] = row[i]));
        return {
          type: "Feature",
          properties: props,
          geometry: { type: "Point", coordinates: [row[0] as number, row[1] as number] },
        };
      }),
    };
  }, [fires.data, showPoints]);
  useEffect(() => {
    if (ready) source(mapRef.current!, "fires").setData(sensors.length ? fireCollection : EMPTY);
  }, [ready, fireCollection, sensors.length]);

  // gridded activity
  useEffect(() => {
    if (!ready) return;
    const data = gridLayer === "none" ? null : grid.data;
    if (!data) {
      source(mapRef.current!, "grid").setData(EMPTY);
      return;
    }
    const top = Math.log1p(Math.max(data.max, 1e-9));
    source(mapRef.current!, "grid").setData({
      type: "FeatureCollection",
      features: data.cells.map(([w, s, e, n, v]) => ({
        type: "Feature",
        properties: { v, n: Math.log1p(v) / top },
        geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
      })),
    });
  }, [ready, grid.data, gridLayer]);

  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current!;
    map.setLayoutProperty("base-bluemarble", "visibility", base === "bluemarble" ? "visible" : "none");
    map.setLayoutProperty("base-osm", "visibility", base === "osm" ? "visible" : "none");
  }, [ready, base]);

  // NASA GIBS imagery for the last day of the period
  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current!;
    if (map.getLayer("gibs")) map.removeLayer("gibs");
    if (map.getSource("gibs")) map.removeSource("gibs");
    if (imagery && end) {
      map.addSource("gibs", { type: "raster", tiles: gibsTiles(end), tileSize: 256, maxzoom: 9, attribution: "NASA GIBS / Worldview" });
      map.addLayer({ id: "gibs", type: "raster", source: "gibs", paint: { "raster-opacity": 0.9 } }, "grid");
    }
  }, [ready, imagery, end]);

  // drawing
  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current!;
    map.getCanvas().style.cursor = drawing ? "crosshair" : "";
    if (drawing) map.doubleClickZoom.disable();
    else map.doubleClickZoom.enable();
    const features: GeoJSON.Feature[] = vertices.map((v) => ({
      type: "Feature", properties: {}, geometry: { type: "Point", coordinates: v },
    }));
    if (vertices.length >= 3) {
      features.push({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...vertices, vertices[0]]] } });
    } else if (vertices.length === 2) {
      features.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: vertices } });
    }
    source(map, "draw").setData({ type: "FeatureCollection", features: drawing ? features : [] });
  }, [ready, drawing, vertices]);

  const finishDrawing = () => {
    if (vertices.length >= 3) {
      setSelection({
        kind: "aoi", name: t("customArea"), level: "AOI",
        geometry: { type: "Polygon", coordinates: [[...vertices, vertices[0]]] },
      });
    }
    setDrawing(false);
    setVertices([]);
  };
  const cancelDrawing = () => {
    setDrawing(false);
    setVertices([]);
  };

  const toggleSensor = (id: string) =>
    setSensors((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
  const setSpan = (days: number) => end && setStart(addDays(end, -(days - 1)));

  return (
    <>
      <PageHead title={t("nav.map")} text={t("map.intro")} />
      <div className="map-layout">
        <div className="map-side">
          <Panel title={t("map.aoi")}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button className="btn primary" onClick={() => { setVertices([]); setDrawing(true); }} disabled={drawing}>
                {t("map.draw")}
              </button>
            </div>
            <Async state={info}>
              {(d) => (
                <p className="note">
                  <strong style={{ color: "var(--text)" }}>{d.kind === "aoi" ? t("customArea") : d.name}</strong><br />
                  {fmt(d.area_km2)} km² · {fmt(d.n_cells)} {t("map.cells")}
                </p>
              )}
            </Async>
            <p className="note">{t("map.aoiHint")}</p>
          </Panel>

          <Panel title={t("period")}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <Field label={t("from")}>
                <input type="date" value={start} min={meta.record.first_day} max={meta.record.last_day}
                  onChange={(e) => setStart(e.target.value)} />
              </Field>
              <Field label={t("to")}>
                <input type="date" value={end} min={meta.record.first_day} max={meta.record.last_day}
                  onChange={(e) => setEnd(e.target.value)} />
              </Field>
            </div>
            <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
              {[1, 7, 30].map((d) => (
                <button key={d} className="btn" onClick={() => setSpan(d)}>{d} {t(d === 1 ? "day" : "days")}</button>
              ))}
            </div>
          </Panel>

          <Panel title={t("map.detections")} sub={t("observed")}>
            <label className="check">
              <input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} />
              {t("map.showDetections")}
            </label>
            {SENSORS.map((s) => (
              <label className="check" key={s.id}>
                <input type="checkbox" checked={sensors.includes(s.id)} onChange={() => toggleSensor(s.id)} />
                <span className="swatch" style={{ background: s.color, width: s.id === "modis" ? 12 : 8, height: s.id === "modis" ? 12 : 8 }} />
                {s.label}
                <span style={{ marginLeft: "auto", color: "var(--muted)" }}>
                  {fires.data && showPoints && sensors.includes(s.id) ? fmt(fires.data.by_sensor[s.id] ?? 0) : ""}
                </span>
              </label>
            ))}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
              <Field label={t("map.minConfidence")}>
                <select value={confidence} onChange={(e) => setConfidence(e.target.value)}>
                  <option value="low">{t("conf.all")}</option>
                  <option value="nominal">{t("conf.nominal")}</option>
                  <option value="high">{t("conf.high")}</option>
                </select>
              </Field>
              <Field label={t("map.dayNight")}>
                <select value={daynight} onChange={(e) => setDaynight(e.target.value)}>
                  <option value="">{t("map.both")}</option>
                  <option value="D">{t("map.day")}</option>
                  <option value="N">{t("map.night")}</option>
                </select>
              </Field>
            </div>
            {fires.error && <p className="note warn">{fires.error}</p>}
            {fires.data?.truncated && (
              <p className="note warn">{t("map.truncated", { shown: fmt(fires.data.returned), total: fmt(fires.data.total) })}</p>
            )}
          </Panel>

          <Panel title={t("map.gridLayer")}>
            <Field label={t("map.layer")}>
              <select value={gridLayer} onChange={(e) => setGridLayer(e.target.value as GridLayer)}>
                <option value="harmonized">{t("src.harmonized")} ({t("estimate")})</option>
                <option value="viirs">{t("src.viirs")} ({t("observed")})</option>
                <option value="modis">{t("src.modis")} ({t("observed")})</option>
                <option value="none">{t("map.noLayer")}</option>
              </select>
            </Field>
            {gridLayer !== "none" && grid.data && (
              <>
                <div className="ramp" />
                <div className="ramp-labels"><span>0</span><span>{fmt(grid.data.max, 1)} ({metric.toUpperCase()}, log)</span></div>
                <p className="note">{t(gridLayer === "harmonized" ? "map.gridNoteHarmonized" : "map.gridNoteObserved")}</p>
              </>
            )}
            {grid.error && <p className="note warn">{grid.error}</p>}
          </Panel>

          <Panel title={t("map.base")}>
            <Field label={t("map.baseMap")}>
              <select value={base} onChange={(e) => setBase(e.target.value as "bluemarble" | "osm")}>
                <option value="bluemarble">NASA Blue Marble</option>
                <option value="osm">OpenStreetMap</option>
              </select>
            </Field>
            <label className="check" style={{ marginTop: 6 }}>
              <input type="checkbox" checked={imagery} onChange={(e) => setImagery(e.target.checked)} />
              {t("map.imagery")}
            </label>
          </Panel>
        </div>

        <div className="map-wrap">
          <div ref={container} className="map-canvas" />
          {drawing && (
            <div className="map-banner">
              {t("map.drawing", { n: vertices.length })}
              <button onClick={finishDrawing} disabled={vertices.length < 3}>{t("map.finish")}</button>
              <button onClick={cancelDrawing}>{t("cancel")}</button>
            </div>
          )}
          <div className="map-overlay">
            <div className="legend">
              {SENSORS.filter((s) => sensors.includes(s.id)).map((s) => (
                <span key={s.id}><i className="dot" style={{ background: s.color }} />{s.label}</span>
              ))}
            </div>
            <div style={{ color: "var(--muted)", marginTop: 4 }}>
              {period ? `${period.start} → ${period.end}` : ""}
              {(fires.loading || grid.loading) && ` · ${t("loading")}`}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
