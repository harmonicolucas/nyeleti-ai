import { useMemo, useState } from "react";
import { Chart, COLORS } from "../components/Chart";
import { Async, Field, Kpi, PageHead, Panel, fmt, fmtCompact, signed } from "../components/ui";
import { selectionParams, type Num, type SelectionInfo, type SeriesBySource } from "../services/api";
import { useApi, useApp } from "../state";

interface DashboardData {
  selection: SelectionInfo;
  year: number;
  years_available: [number, number];
  observed: Record<string, Num>;
  harmonized: { detections_viirs_scale: Num; frp_viirs_scale_mw: Num; hfai: Num; vs_reference_pct: Num; reference: string };
  high_activity: { days: number; threshold_hfai: Num; mean_days_per_year: Num; definition: string };
  anomalies: { events: number; flagged_days: number; top: AnomalyEvent[] } | null;
  note: string;
}

export interface AnomalyEvent {
  start: string;
  end: string;
  days_flagged: number;
  direction: "above" | "below";
  severity: "moderate" | "high" | "extreme";
  peak_date: string;
  peak_score: number;
  peak_pct_vs_reference: number;
  peak_hfai_7d: number;
  reference_hfai: number;
  seasonal_percentile: number;
  explanation: { key: string; indicator: string; value: string; detail: string; mean?: string; reference?: string }[];
}

interface YearCalendar {
  years: number[];
  series: SeriesBySource;
  viirs_complete: boolean[];
  modis_full_coverage: boolean[];
  metric_label: string;
}

interface Harmonization {
  periods: string[];
  series: SeriesBySource;
  metric_label: string;
}

export function sourceTraces(x: (string | number)[], series: SeriesBySource, t: (k: string) => string, type: "scatter" | "bar") {
  const names: [keyof SeriesBySource, string][] = [
    ["modis", COLORS.modis],
    ["viirs", COLORS.viirs],
    ["harmonized", COLORS.harmonized],
  ];
  return names.map(([source, color]) => ({
    x,
    y: series[source],
    name: t(`src.${source}`),
    type,
    mode: type === "scatter" ? "lines+markers" : undefined,
    marker: { color, size: 5 },
    line: { color, width: source === "harmonized" ? 2.4 : 1.6 },
    connectgaps: false,
  }));
}

export function Dashboard() {
  const { selection, metric, meta, t } = useApp();
  const lastYear = Number(meta.record.last_day.slice(0, 4));
  const firstYear = Number(meta.record.first_day.slice(0, 4));
  const [year, setYear] = useState(lastYear);
  const sel = selectionParams(selection);

  const dash = useApi<DashboardData>("/api/dashboard", { ...sel, year });
  const yearly = useApi<YearCalendar>("/api/calendar", { ...sel, granularity: "year", metric });
  const monthly = useApi<Harmonization>("/api/harmonization", {
    ...sel, metric, freq: "month", start: `${year}-01-01`, end: `${year}-12-31`,
  });

  const yearlyData = useMemo(
    () => (yearly.data ? sourceTraces(yearly.data.years, yearly.data.series, t, "scatter") : []),
    [yearly.data, t],
  );
  const yearlyLayout = useMemo(
    () => ({
      yaxis: { title: { text: t(`metricLabel.${metric}`) } },
      shapes: [{ type: "line", x0: year, x1: year, yref: "paper", y0: 0, y1: 1, line: { color: COLORS.accent, width: 1, dash: "dot" } }],
    }),
    [year, t, metric],
  );
  const monthlyData = useMemo(
    () => (monthly.data ? sourceTraces(monthly.data.periods, monthly.data.series, t, "bar") : []),
    [monthly.data, t],
  );
  const monthlyLayout = useMemo(
    () => ({ barmode: "group", yaxis: { title: { text: t(`metricLabel.${metric}`) } }, xaxis: { type: "category" } }),
    [t, metric],
  );

  const years = Array.from({ length: lastYear - firstYear + 1 }, (_, i) => lastYear - i);

  return (
    <>
      <PageHead
        title={t("nav.dashboard")}
        text={t("dashboard.intro")}
        tools={
          <Field label={t("year")}>
            <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
              {years.map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </Field>
        }
      />
      <Async state={dash} height={220}>
        {(d) => (
          <>
            <div className="grid kpis">
              <Kpi label={t("kpi.total")} value={fmtCompact(d.observed.total_detections)} kind="observed"
                extra={t("kpi.totalExtra")} />
              <Kpi label={t("kpi.modis")} value={fmtCompact(d.observed.modis_detections)} kind="observed"
                extra="Terra + Aqua, 1 km" />
              <Kpi label={t("kpi.viirs")} value={fmtCompact(d.observed.viirs_detections)} kind="observed"
                extra={d.observed.viirs_detections === null ? t("kpi.noViirs", { date: meta.record.viirs_start }) : "S-NPP, 375 m"} />
              <Kpi label={t("kpi.harmonized")} value={fmtCompact(d.harmonized.hfai)} kind="estimate"
                extra={<>
                  {signed(d.harmonized.vs_reference_pct, 0)} {t("kpi.vsMean")}<br />
                  ≈ {fmtCompact(d.harmonized.detections_viirs_scale)} {t("kpi.viirsScale")}
                </>} />
              <Kpi label={t("kpi.frp")} value={`${fmtCompact(d.observed.modis_frp_mw)} MW`} kind="observed"
                extra={<>MODIS · VIIRS: {fmtCompact(d.observed.viirs_frp_mw)} MW<br />{t("kpi.frpNote")}</>} />
              <Kpi label={t("kpi.highDays")} value={fmt(d.high_activity.days)}
                extra={t("kpi.highDaysExtra", { mean: fmt(d.high_activity.mean_days_per_year, 1) })} />
              <Kpi label={t("kpi.anomalies")} value={d.anomalies ? fmt(d.anomalies.events) : "-"}
                extra={d.anomalies ? t("kpi.anomaliesExtra", { days: d.anomalies.flagged_days }) : t("kpi.noAnomalyModel")} />
            </div>
            <p className="note">{t("dashboard.note", { year: d.year, name: d.selection.name, km2: fmt(d.selection.area_km2) })}</p>
          </>
        )}
      </Async>

      <div className="grid two" style={{ marginTop: 14 }}>
        <Panel title={t("dashboard.byYear")} sub={t("dashboard.byYearSub")}>
          <Async state={yearly} height={300}>
            {() => <Chart data={yearlyData} layout={yearlyLayout} onPointClick={(p) => setYear(Number(p.x))} />}
          </Async>
          <p className="note">{t("coverageNote", { year: meta.record.analysis_first_year, viirs: meta.record.viirs_start })}</p>
        </Panel>
        <Panel title={t("dashboard.byMonth", { year })} sub={t("dashboard.byMonthSub")}>
          <Async state={monthly} height={300}>
            {() => <Chart data={monthlyData} layout={monthlyLayout} />}
          </Async>
        </Panel>
      </div>

      {dash.data?.anomalies && dash.data.anomalies.top.length > 0 && (
        <Panel title={t("dashboard.topAnomalies", { year })} sub={t("anomalies.notForecast")}>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>{t("period")}</th><th>{t("severity")}</th><th>{t("peakDay")}</th>
                  <th>{t("vsReference")}</th><th>{t("seasonalPercentile")}</th>
                </tr>
              </thead>
              <tbody>
                {dash.data.anomalies.top.map((e) => (
                  <tr key={e.start}>
                    <td>{e.start} → {e.end}</td>
                    <td><span className={`tag ${e.severity}`}>{t(`sev.${e.severity}`)}</span></td>
                    <td>{e.peak_date}</td>
                    <td>{signed(e.peak_pct_vs_reference)}</td>
                    <td>{fmt(e.seasonal_percentile, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </>
  );
}
