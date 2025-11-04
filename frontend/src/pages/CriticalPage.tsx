import { useMemo, useState } from "react";
import { Chart, COLORS } from "../components/Chart";
import { Async, Field, PageHead, Panel, Segmented, fmt } from "../components/ui";
import { selectionParams, type Num } from "../services/api";
import { useApi, useApp } from "../state";

interface Critical {
  definition: string;
  threshold: { value?: Num; by_month?: Num[] };
  periods: {
    start: string; end: string; year: number; days: number; days_above: number; total: Num; mean: Num;
    peak_date: string; peak_value: Num; peak_over_threshold: Num;
  }[];
  by_year: {
    year: number; days_above: number; periods: number; longest_days: number;
    first_day: string | null; last_day: string | null; modis_full_coverage: boolean;
  }[];
  mean_days_above_by_month: { months: string[]; days: Num[] };
  note: string;
}

interface Trends {
  trend: {
    method: string; first_year: number; last_year: number; slope_per_year: Num; slope_ci95: [Num, Num];
    pct_per_decade: Num; kendall_tau: Num; p_value: Num; significant: boolean; fitted: Num[];
  } | null;
  years: number[];
  series: Record<string, Num[]>;
  climatology: {
    months: string[]; harmonized_mean: Num[]; harmonized_min: Num[]; harmonized_max: Num[]; share_pct: Num[];
    peak_month: string; top3_months: string[]; top3_share_pct: Num; reference_years: [number, number];
  };
  season_timing: { year: number; p10: string; p50: string; p90: string; p10_doy: number; p50_doy: number; p90_doy: number; peak_date: string }[];
  metric_label: string;
  note: string;
}

interface Regions {
  definition: string;
  n_years: number;
  years: [number, number];
  areas: {
    area_id: string; name: string; parent_name: string | null; hotspot_years: number; recurrence_pct: Num;
    mean_density: Num; peak_year: number; rank: number;
  }[];
}

export function CriticalPage() {
  const { selection, setSelection, metric, t } = useApp();
  const sel = selectionParams(selection);
  const [percentile, setPercentile] = useState(90);
  const [minDays, setMinDays] = useState(3);
  const [maxGap, setMaxGap] = useState(1);
  const [basis, setBasis] = useState("area");
  const [level, setLevel] = useState("ADM2");

  const critical = useApi<Critical>("/api/critical", { ...sel, metric, percentile, min_days: minDays, max_gap: maxGap, basis });
  const trends = useApi<Trends>("/api/trends", { ...sel, metric });
  const regions = useApi<Regions>("/api/critical/regions", { level });

  const yearTraces = useMemo(() => {
    if (!critical.data) return [];
    const rows = critical.data.by_year;
    return [
      { x: rows.map((r) => r.year), y: rows.map((r) => r.days_above), type: "bar", name: t("critical.daysAbove"),
        marker: { color: rows.map((r) => (r.modis_full_coverage ? COLORS.harmonized : "rgba(251,122,30,.35)")) } },
      { x: rows.map((r) => r.year), y: rows.map((r) => r.longest_days), type: "scatter", mode: "lines+markers",
        name: t("critical.longest"), line: { color: COLORS.accent, width: 1.6 }, marker: { size: 5 } },
    ];
  }, [critical.data, t]);
  const yearLayout = useMemo(() => ({ yaxis: { title: { text: t("days") } } }), [t]);

  const monthTraces = useMemo(() => {
    if (!trends.data || !critical.data) return [];
    const c = trends.data.climatology;
    const months = t("months").split(",");
    return [
      { x: months, y: c.harmonized_max, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x: months, y: c.harmonized_min, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
        fillcolor: "rgba(251,122,30,.16)", name: t("critical.range") },
      { x: months, y: c.harmonized_mean, type: "scatter", mode: "lines+markers", name: t("critical.monthMean"),
        line: { color: COLORS.harmonized, width: 2.2 } },
      { x: months, y: critical.data.mean_days_above_by_month.days, type: "bar",
        yaxis: "y2", name: t("critical.daysAbovePerYear"), marker: { color: "rgba(245,158,11,.45)" } },
    ];
  }, [trends.data, critical.data, t]);
  const monthLayout = useMemo(
    () => ({
      yaxis: { title: { text: t(`metricLabel.${metric}`) } },
      yaxis2: { overlaying: "y", side: "right", showgrid: false, rangemode: "tozero", title: { text: t("critical.daysAbovePerYear") } },
      margin: { r: 52 },
    }),
    [t, metric],
  );

  const timingTraces = useMemo(() => {
    if (!trends.data) return [];
    const rows = trends.data.season_timing;
    return [
      { x: rows.map((r) => r.year), y: rows.map((r) => r.p90_doy - r.p10_doy), base: rows.map((r) => r.p10_doy), type: "bar",
        name: t("critical.core"), marker: { color: "rgba(251,122,30,.5)" },
        customdata: rows.map((r) => [r.p10, r.p90]), hovertemplate: "%{customdata[0]} → %{customdata[1]}<extra></extra>" },
      { x: rows.map((r) => r.year), y: rows.map((r) => r.p50_doy), type: "scatter", mode: "markers", name: t("critical.midpoint"),
        marker: { color: "#fff", size: 6 }, customdata: rows.map((r) => r.p50), hovertemplate: "%{customdata}<extra></extra>" },
    ];
  }, [trends.data, t]);
  const timingLayout = useMemo(
    () => ({
      hovermode: "closest",
      yaxis: { rangemode: "normal", tickvals: [121, 152, 182, 213, 244, 274, 305, 335],
        ticktext: ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] },
    }),
    [],
  );

  const tr = trends.data?.trend;

  return (
    <>
      <PageHead title={t("nav.critical")} text={t("critical.intro")} />

      <Panel title={t("critical.settings")}>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-end" }}>
          <Field label={`${t("critical.percentile")}: P${percentile}`}>
            <input type="range" min={75} max={99} step={1} value={percentile}
              onChange={(e) => setPercentile(Number(e.target.value))} />
          </Field>
          <Field label={t("critical.minDays")}>
            <input type="number" min={1} max={60} value={minDays}
              onChange={(e) => setMinDays(Math.min(60, Math.max(1, Number(e.target.value) || 1)))} />
          </Field>
          <Field label={t("critical.maxGap")}>
            <input type="number" min={0} max={7} value={maxGap}
              onChange={(e) => setMaxGap(Math.min(7, Math.max(0, Number(e.target.value) || 0)))} />
          </Field>
          <Segmented label={t("critical.basis")} value={basis} onChange={setBasis}
            options={[{ value: "area", label: t("critical.basisArea") }, { value: "month", label: t("critical.basisMonth") }]} />
        </div>
        {critical.data && (
          <p className="note">
            {critical.data.definition}
            {critical.data.threshold.value != null && ` ${t("critical.threshold")}: ${fmt(critical.data.threshold.value, 2)}.`}
          </p>
        )}
      </Panel>

      <div className="grid two">
        <Panel title={t("critical.byYear")}>
          <Async state={critical} height={280}>{() => <Chart data={yearTraces} layout={yearLayout} height={280} />}</Async>
        </Panel>
        <Panel title={t("critical.seasonality")}
          sub={trends.data ? `${trends.data.climatology.reference_years.join("–")}` : undefined}>
          <Async state={trends} height={280}>{() => <Chart data={monthTraces} layout={monthLayout} height={280} />}</Async>
          {trends.data && (
            <p className="note">
              {t("critical.peakNote", {
                peak: trends.data.climatology.peak_month,
                top: trends.data.climatology.top3_months.join(", "),
                share: fmt(trends.data.climatology.top3_share_pct, 0),
              })}
            </p>
          )}
        </Panel>
      </div>

      <div className="grid two">
        <Panel title={t("critical.periods")} sub={critical.data ? String(critical.data.periods.length) : undefined}>
          <Async state={critical} height={260}>
            {(d) => (
              <div className="table-scroll" style={{ maxHeight: 380 }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th className="left">{t("period")}</th><th>{t("days")}</th><th>{t("critical.total")}</th>
                      <th>{t("peakDay")}</th><th>{t("critical.peakOver")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.periods.slice(0, 80).map((p) => (
                      <tr key={p.start}>
                        <td className="left">{p.start} → {p.end}</td>
                        <td>{p.days}</td>
                        <td>{fmt(p.total, 0)}</td>
                        <td>{p.peak_date}</td>
                        <td>{p.peak_over_threshold == null ? "–" : `×${fmt(p.peak_over_threshold, 1)}`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Async>
          <p className="note">{t("critical.periodsNote")}</p>
        </Panel>

        <Panel title={t("critical.timing")} sub={t("critical.timingSub")}>
          <Async state={trends} height={260}>{() => <Chart data={timingTraces} layout={timingLayout} height={260} />}</Async>
          {tr && (
            <p className="note">
              {t("critical.trend", {
                first: tr.first_year, last: tr.last_year, pct: fmt(tr.pct_per_decade, 1),
                p: fmt(tr.p_value, 3), tau: fmt(tr.kendall_tau, 2),
              })}{" "}
              {tr.significant ? t("critical.trendSig") : t("critical.trendNotSig")}
            </p>
          )}
        </Panel>
      </div>

      <Panel
        title={t("critical.regions")}
        sub={regions.data ? `${regions.data.years.join("–")}` : undefined}
        tools={
          <Segmented label={t("critical.level")} value={level} onChange={setLevel}
            options={[{ value: "ADM2", label: t("district") }, { value: "ADM1", label: t("province") }]} />
        }
      >
        <Async state={regions} height={260}>
          {(d) => (
            <>
              <div className="table-scroll" style={{ maxHeight: 420 }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th className="left">{level === "ADM2" ? t("district") : t("province")}</th>
                      {level === "ADM2" && <th className="left">{t("province")}</th>}
                      <th>{t("critical.hotspotYears")}</th><th>{t("critical.recurrence")}</th>
                      <th>{t("critical.density")}</th><th>{t("critical.rank")}</th><th>{t("critical.peakYear")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.areas.slice(0, 40).map((a) => (
                      <tr key={a.area_id} className="clickable"
                        onClick={() => setSelection({ kind: "area", areaId: a.area_id, name: a.name, level })}>
                        <td className="left">{a.name}</td>
                        {level === "ADM2" && <td className="left">{a.parent_name ?? "–"}</td>}
                        <td>{a.hotspot_years} / {d.n_years}</td>
                        <td>{fmt(a.recurrence_pct, 0)}%</td>
                        <td>{fmt(a.mean_density, 1)}</td>
                        <td>{a.rank}</td>
                        <td>{a.peak_year}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="note">{d.definition} {t("critical.regionsNote")}</p>
            </>
          )}
        </Async>
      </Panel>
    </>
  );
}
