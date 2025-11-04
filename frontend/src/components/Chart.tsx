import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "react";

export const COLORS = {
  modis: "#2563eb",
  viirs: "#0d9488",
  harmonized: "#c2410c",
  muted: "#5b6778",
  grid: "#d5dbe3",
  accent: "#1f6b5c",
  danger: "#dc2626",
};

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#334155", size: 11.5, family: "Segoe UI, IBM Plex Sans, system-ui, sans-serif" },
  margin: { l: 52, r: 16, t: 10, b: 36 },
  hovermode: "x unified",
  hoverlabel: { bgcolor: "#ffffff", bordercolor: "#d5dbe3", font: { color: "#1a2332" } },
  legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } },
  xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid, linecolor: COLORS.grid },
  yaxis: { gridcolor: COLORS.grid, zerolinecolor: "#cbd5e1", linecolor: COLORS.grid, rangemode: "tozero" },
};

function isPlain(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

// Plotly writes the axis type and range back into the layout it is given,
// so every chart needs a copy of its own, never a reference to BASE_LAYOUT.
function merge(base: any, extra: any): any {
  const out: any = {};
  for (const [key, value] of Object.entries(base)) out[key] = isPlain(value) ? merge(value, {}) : value;
  for (const [key, value] of Object.entries(extra ?? {})) {
    if (value === undefined) continue;
    out[key] = isPlain(value) ? merge(isPlain(out[key]) ? out[key] : {}, value) : value;
  }
  return out;
}

interface Props {
  data: any[];
  layout?: Record<string, any>;
  height?: number;
  onPointClick?: (point: { x: any; y: any; curveNumber: number; pointIndex: number }) => void;
}

/** Thin wrapper around Plotly. Memoise `data` and `layout` in the caller. */
export function Chart({ data, layout, height = 300, onPointClick }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const clickRef = useRef(onPointClick);
  clickRef.current = onPointClick;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    Plotly.react(el, data, merge(BASE_LAYOUT, { ...layout, height }), { displayModeBar: false, responsive: true });
    const node = el as any;
    node.removeAllListeners?.("plotly_click");
    node.on?.("plotly_click", (ev: any) => {
      const p = ev.points?.[0];
      if (p && clickRef.current) clickRef.current({ x: p.x, y: p.y, curveNumber: p.curveNumber, pointIndex: p.pointIndex });
    });
  }, [data, layout, height]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} style={{ width: "100%", height }} />;
}
