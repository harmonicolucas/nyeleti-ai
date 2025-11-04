export type Metric = "hfai" | "count" | "frp";
export type Source = "modis" | "viirs" | "harmonized";
export type Num = number | null;

export type Selection =
  | { kind: "area"; areaId: string; name: string; level: string }
  | { kind: "aoi"; geometry: GeoJSON.Polygon; name: string; level: "AOI" };

export interface SelectionInfo {
  kind: string;
  area_id: string | null;
  name: string;
  level: string;
  n_cells: number;
  area_km2: number;
  bounds: [number, number, number, number];
}

export interface Area {
  area_id: string;
  name: string;
  level: string;
  parent_id: string | null;
  bounds: [number, number, number, number];
  n_cells: number;
}

export type SeriesBySource = Record<Source, Num[]>;

export interface Meta {
  country: string;
  iso3: string;
  grid: { cell_km: number; crs: string };
  record: { first_day: string; last_day: string; viirs_start: string; noaa20_start: string | null; analysis_first_year: number };
  active_model: string;
  hfai: { count_ref: number; frp_ref: number; weight_count: number; weight_frp: number; reference_years: number[] };
  split: Record<string, [number, number]>;
  built_at: string;
  detections: { sensor: string; records: number; first: string; last: string; used_in_analysis: number }[];
  quality: Record<string, Record<string, any>> | null;
  anomaly_model: Record<string, any> | null;
  files: Record<string, any>[];
  sources: { name: string; url: string }[];
}

const BASE = import.meta.env.VITE_API_URL ?? "";
const STATIC_API = import.meta.env.VITE_STATIC_API === "1";
const cache = new Map<string, Promise<unknown>>();

export type Params = Record<string, string | number | boolean | null | undefined>;

export function selectionParams(sel: Selection): Params {
  return sel.kind === "area" ? { area_id: sel.areaId } : { aoi: JSON.stringify(sel.geometry) };
}

function staticApiPath(path: string, params: Params = {}): string {
  const items = Object.entries(params)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([key, value]) => [key, String(value)] as const)
    .sort(([a], [b]) => a.localeCompare(b));
  const query = items.map(([key, value]) => `${key}=${value}`).join("&");
  const raw = `${path.replace(/^\//, "")}${query ? `__${query}` : ""}`;
  const safe = raw.replace(/[^\w.=\/-]+/g, "_");
  return `${import.meta.env.BASE_URL}static-api/${safe}.json`;
}

export function buildUrl(path: string, params: Params = {}): string {
  if (STATIC_API && path.startsWith("/api/")) return staticApiPath(path, params);
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== "") query.set(key, String(value));
  }
  const text = query.toString();
  return `${BASE}${path}${text ? `?${text}` : ""}`;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function get<T>(path: string, params: Params = {}): Promise<T> {
  const url = buildUrl(path, params);
  const hit = cache.get(url);
  if (hit) return hit as Promise<T>;
  const request = fetch(url).then(async (res) => {
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* keep the status text */
      }
      throw new ApiError(res.status, detail);
    }
    return res.json() as Promise<T>;
  });
  cache.set(url, request);
  request.catch(() => cache.delete(url));
  if (cache.size > 200) cache.delete(cache.keys().next().value as string);
  return request;
}
