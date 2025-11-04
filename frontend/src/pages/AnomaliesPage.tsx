import { useEffect, useMemo, useState } from "react";
import { Chart, COLORS } from "../components/Chart";
import { Async, Field, PageHead, Panel, Segmented, fmt, signed } from "../components/ui";
import { selectionParams, type Num } from "../services/api";
import { useApi, useApp } from "../state";
import type { AnomalyEvent } from "./Dashboard";

interface AnomalyList {
  events: AnomalyEvent[];
  total_events: number;
  flagged_days: Record<string, number>;
  days_scored: number;
  model: {
    name: string;
    created_at: string;
    features: string[];
    severity_quantiles: Record<string, number>;
    thresholds: Record<string, number>;
    window_days: number;
    first_year: number;
  };
  note: string;
}

interface AnomalySeries {
  dates: string[];
  hfai: Num[];
  hfai_7d: Num[];
  reference: Num[];
  score: Num[];
  severity: (string | null)[];
  direction: string[];
  thresholds: Record<string, number>;
}

interface Overview {
  areas: {
    area_id: string; name: string; flagged_days: number; high_days: number; extreme_days: number;
    max_pct_vs_reference: Num; peak_date: string | null;
  }[];
}

const SEVERITY_COLORS: Record<string, string> = { moderate: "#facc15", high: "#fb923c", extreme: "#ef4444" };

export function AnomaliesPage() {
  const { selection, setSelection, meta, t } = useApp();
  const sel = selectionParams(selection);
  const lastYear = Number(meta.record.last_day.slice(0, 4));
  const firstYear = meta.record.analysis_first_year;
  const years = Array.from({ length: lastYear - firstYear + 1 }, (_, i) => lastYear - i);

  const [year, setYear] = useState(lastYear);
  const [scope, setScope] = useState<"year" | "record">("record");
  const [direction, setDirection] = useState("above");
  const [severity, setSeverity] = useState("moderate");
  const [picked, setPicked] = useState<string | null>(null);

  const period = scope === "year" ? { start: `${year}-01-01`, end: `${year}-12-31` } : {};
  const list = useApi<AnomalyList>("/api/anomalies", { ...sel, ...period, direction, min_severity: severity, limit: 60 });
  const daily = useApi<AnomalySeries>("/api/anomalies/series", { ...sel, year });
  const overview = useApi<Overview>("/api/anomalies/overview", {
    level: "ADM1", start: `${year}-01-01`, end: `${year}-12-31`, direction,
  });

  useEffect(() => setPicked(null), [selection, direction, severity, scope]);

  const event = list.data?.events.find((e) => e.start === picked) ?? list.data?.events[0] ?? null;

  // over the full record, open the year of the most unusual period
  const top = list.data?.events[0]?.peak_date ?? null;
  useEffect(() => {
    if (scope === "record" && top) setYear(Number(top.slice(0, 4)));
  }, [scope, top]);

  const traces = useMemo(() => {
    if (!daily.data) return [];
    const d = daily.data;
    const out: any[] = [
      { x: d.dates, y: d.hfai, name: t("anomalies.dailyValue"), type: "bar", marker: { color: "rgba(251,122,30,.28)" } },
      { x: d.dates, y: d.reference, name: t("anomalies.reference"), type: "scatter", mode: "lines",
        line: { color: COLORS.muted, width: 1.6, dash: "dot" } },
      { x: d.dates, y: d.hfai_7d, name: t("anomalies.mean7"), type: "scatter", mode: "lines",
        line: { color: COLORS.harmonized, width: 2.2 } },
    ];
    for (const name of ["moderate", "high", "extreme"]) {
      const idx = d.dates.map((_, i) => i).filter((i) => d.severity[i] === name);
      if (!idx.length) continue;
      out.push({
        x: idx.map((i) => d.dates[i]), y: idx.map((i) => d.hfai_7d[i]), name: t(`sev.${name}`),
        type: "scatter", mode: "markers",
        marker: { color: SEVERITY_COLORS[name], size: 8, line: { color: "#0b0a09", width: 1 },
          symbol: idx.map((i) => (d.direction[i] === "above" ? "triangle-up" : "triangle-down")) },
      });
    }
    return out;
  }, [daily.data, t]);
  const layout = useMemo(() => {
    const shapes = event && event.start.startsWith(String(year))
      ? [{ type: "rect", xref: "x", yref: "paper", x0: event.start, x1: event.end, y0: 0, y1: 1,
          fillcolor: "rgba(245,158,11,.12)", line: { width: 0 } }]
      : [];
    return { yaxis: { title: { text: "HFAI" } }, shapes, bargap: 0 };
  }, [event, year]);

  const scoreTraces = useMemo(() => {
    if (!daily.data) return [];
    return [{ x: daily.data.dates, y: daily.data.score, type: "scatter", mode: "lines", name: t("anomalies.score"),
      line: { color: COLORS.accent, width: 1.4 } }];
  }, [daily.data, t]);
  const scoreLayout = useMemo(() => {
    const th = daily.data?.thresholds ?? {};
    return {
      showlegend: false,
      yaxis: { title: { text: t("anomalies.score") }, rangemode: "normal" },
      shapes: Object.entries(th).map(([name, value]) => ({
        type: "line", xref: "paper", x0: 0, x1: 1, y0: value, y1: value,
        line: { color: SEVERITY_COLORS[name], width: 1, dash: "dash" },
      })),
    };
  }, [daily.data, t]);

  return (
    <>
      <PageHead
        title={t("nav.anomalies")}
        text={t("anomalies.intro")}
        tools={
          <>
            <Field label={t("year")}>
              <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
                {years.map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            </Field>
            <Segmented label={t("anomalies.direction")} value={direction} onChange={setDirection}
              options={[
                { value: "above", label: t("anomalies.above") },
                { value: "below", label: t("anomalies.below") },
                { value: "both", label: t("anomalies.both") },
              ]} />
          </>
        }
      />

      <Panel title={t("anomalies.yearSeries", { year })} sub={t("anomalies.notForecast")}>
        <Async state={daily} height={300}>
          {() => (
            <>
              <Chart data={traces} layout={layout} height={300} />
              <Chart data={scoreTraces} layout={scoreLayout} height={130} />
            </>
          )}
        </Async>
        <p className="note">{t("anomalies.chartNote")}</p>
      </Panel>

      <div className="grid two">
        <Panel
          title={t("anomalies.events")}
          sub={list.data ? `${list.data.total_events}` : undefined}
          tools={
            <>
              <Segmented label={t("period")} value={scope} onChange={setScope}
                options={[{ value: "year", label: String(year) }, { value: "record", label: t("anomalies.fullRecord") }]} />
              <Segmented label={t("severity")} value={severity} onChange={setSeverity}
                options={["moderate", "high", "extreme"].map((s) => ({ value: s, label: `≥ ${t(`sev.${s}`)}` }))} />
            </>
          }
        >
          <Async state={list} height={240}>
            {(d) =>
              d.events.length === 0 ? (
                <p className="note">{t("anomalies.none")}</p>
              ) : (
                <div style={{ maxHeight: 460, overflowY: "auto", paddingRight: 4 }}>
                  {d.events.map((e) => (
                    <div
                      key={`${e.start}-${e.direction}`}
                      className={`event${event?.start === e.start ? " selected" : ""}`}
                      onClick={() => { setPicked(e.start); setYear(Number(e.peak_date.slice(0, 4))); }}
                    >
                      <div className={`pct ${e.direction}`}>{signed(e.peak_pct_vs_reference)}</div>
                      <div>
                        <div>
                          <span className={`tag ${e.severity}`}>{t(`sev.${e.severity}`)}</span>{" "}
                          {e.start === e.end ? e.start : `${e.start} → ${e.end}`}
                        </div>
                        <div className="meta">
                          {t("anomalies.eventMeta", { days: e.days_flagged, peak: e.peak_date, score: e.peak_score.toFixed(3) })}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )
            }
          </Async>
        </Panel>

        <Panel title={t("anomalies.why")} sub={event ? event.peak_date : undefined}>
          {event ? (
            <>
              <table className="data">
                <thead>
                  <tr><th className="left">{t("anomalies.indicator")}</th><th>{t("value")}</th></tr>
                </thead>
                <tbody>
                  {event.explanation.map((x) => (
                    <tr key={x.key}>
                      <td className="left">
                        {t(`anomalies.ind.${x.key}`)}
                        <div style={{ color: "var(--muted)", fontSize: 11.5 }}>
                          {t(`anomalies.ind.${x.key}.detail`, { mean: x.mean ?? "", reference: x.reference ?? "" })}
                        </div>
                      </td>
                      <td>{x.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="note">{t("anomalies.whyNote")}</p>
            </>
          ) : (
            <p className="note">{t("anomalies.none")}</p>
          )}
          {list.data && (
            <dl className="def-list" style={{ marginTop: 10 }}>
              <dt>{t("model")}</dt><dd>{list.data.model.name}</dd>
              <dt>{t("anomalies.reference")}</dt><dd>{t("anomalies.referenceText", { days: list.data.model.window_days, year: list.data.model.first_year })}</dd>
              <dt>{t("anomalies.severityRule")}</dt>
              <dd>
                {Object.entries(list.data.model.severity_quantiles)
                  .map(([k, q]) => `${t(`sev.${k}`)}: ${t("anomalies.topShare", { pct: fmt(100 * (1 - q), 1) })}`)
                  .join(" · ")}
              </dd>
              <dt>{t("anomalies.trained")}</dt><dd>{list.data.model.created_at}</dd>
            </dl>
          )}
        </Panel>
      </div>

      <Panel title={t("anomalies.byProvince", { year })} sub={t("anomalies.byProvinceSub")}>
        <Async state={overview} height={200}>
          {(d) => (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="left">{t("province")}</th><th>{t("anomalies.flaggedDays")}</th>
                    <th>{t("sev.high")}</th><th>{t("sev.extreme")}</th>
                    <th>{t("anomalies.maxDeparture")}</th><th>{t("peakDay")}</th>
                  </tr>
                </thead>
                <tbody>
                  {d.areas.map((a) => (
                    <tr key={a.area_id} className="clickable"
                      onClick={() => setSelection({ kind: "area", areaId: a.area_id, name: a.name, level: "ADM1" })}>
                      <td className="left">{a.name}</td>
                      <td>{a.flagged_days}</td>
                      <td>{a.high_days}</td>
                      <td>{a.extreme_days}</td>
                      <td>{signed(a.max_pct_vs_reference)}</td>
                      <td>{a.peak_date ?? "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Async>
      </Panel>
    </>
  );
}
