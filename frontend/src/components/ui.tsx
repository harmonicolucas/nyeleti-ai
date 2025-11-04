import type { ReactNode } from "react";
import { useApp, type Loaded } from "../state";

export function Panel(props: { title?: string; sub?: string; tools?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${props.className ?? ""}`}>
      {(props.title || props.tools) && (
        <div className="panel-head">
          {props.title && <h2>{props.title}</h2>}
          {props.sub && <span className="sub">{props.sub}</span>}
          {props.tools && <div className="tools">{props.tools}</div>}
        </div>
      )}
      {props.children}
    </section>
  );
}

export function PageHead(props: { title: string; text: string; tools?: ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <h1>{props.title}</h1>
        <p>{props.text}</p>
      </div>
      {props.tools && <div className="tools">{props.tools}</div>}
    </div>
  );
}

export function Kpi(props: { label: string; value: string; kind?: "observed" | "estimate"; extra?: ReactNode }) {
  const { t } = useApp();
  return (
    <div className="panel kpi">
      <div className="label">{props.label}</div>
      <div className="value">{props.value}</div>
      {props.kind && <span className={`tag ${props.kind}`}>{t(props.kind)}</span>}
      {props.extra && <div className="extra">{props.extra}</div>}
    </div>
  );
}

export function Segmented<T extends string | number>(props: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={props.label}>
      {props.options.map((o) => (
        <button
          key={String(o.value)}
          className={o.value === props.value ? "active" : ""}
          aria-pressed={o.value === props.value}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field(props: { label: string; children: ReactNode }) {
  return (
    <div className="field">
      <label>{props.label}</label>
      {props.children}
    </div>
  );
}

/** Renders the loading and error states of a request, then the content. */
export function Async<T>(props: { state: Loaded<T>; children: (data: T) => ReactNode; height?: number }) {
  const { t } = useApp();
  const { data, loading, error } = props.state;
  if (error) return <div className="state error">{error}</div>;
  if (!data) return <div className="state" style={{ minHeight: props.height }}>{loading ? t("loading") : t("noData")}</div>;
  return <div style={{ opacity: loading ? 0.55 : 1, transition: "opacity .15s" }}>{props.children(data)}</div>;
}

export function SourceLegend() {
  const { t } = useApp();
  return (
    <div className="legend">
      <span><i style={{ background: "var(--modis)" }} />{t("src.modis")}</span>
      <span><i style={{ background: "var(--viirs)" }} />{t("src.viirs")}</span>
      <span><i style={{ background: "var(--harmonized)" }} />{t("src.harmonized")}</span>
    </div>
  );
}

export function ClassLegend(props: { thresholds?: Record<string, number | null> }) {
  const { t } = useApp();
  const th = props.thresholds;
  const fmtTh = (v?: number | null) => (v == null ? "" : ` > ${fmt(v, v < 10 ? 2 : 0)}`);
  return (
    <div className="legend">
      <span><i className="cls-low" />{t("cls.low")}</span>
      <span><i className="cls-medium" />{t("cls.medium")}{fmtTh(th?.medium)}</span>
      <span><i className="cls-high" />{t("cls.high")}{fmtTh(th?.high)}</span>
      <span><i className="cls-very_high" />{t("cls.very_high")}{fmtTh(th?.very_high)}</span>
    </div>
  );
}

export function fmt(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function fmtCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e4) return `${(value / 1e3).toFixed(1)}k`;
  return fmt(value, abs < 100 && !Number.isInteger(value) ? 1 : 0);
}

export function signed(value: number | null | undefined, digits = 0, unit = "%"): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return `${value > 0 ? "+" : ""}${fmt(value, digits)}${unit}`;
}
