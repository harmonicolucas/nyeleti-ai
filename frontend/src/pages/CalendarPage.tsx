import { useMemo, useState } from "react";
import { classOf, thresholdsOf, type ActivityClass, type Thresholds } from "../calendar/classes";
import { MonthMatrix } from "../calendar/MonthMatrix";
import { YearHeatmap } from "../calendar/YearHeatmap";
import { Chart, COLORS } from "../components/Chart";
import { Async, ClassLegend, Field, PageHead, Panel, Segmented, fmt, signed } from "../components/ui";
import { selectionParams, type Num, type SeriesBySource, type Source } from "../services/api";
import { useApi, useApp } from "../state";
import { sourceTraces } from "./Dashboard";

type Granularity = "day" | "month" | "season" | "year";

interface Common {
  metric_label: string;
  thresholds: Thresholds;
  class_basis: string;
  viirs_start: string;
  note: string;
}

interface DayCalendar extends Common {
  year: number;
  dates: string[];
  series: SeriesBySource;
  classes: (ActivityClass | null)[];
  anomalies: { date: string; severity: string; direction: string; pct_vs_reference: number }[];
}

interface MonthCalendar extends Common {
  years: number[];
  months: string[];
  series: Record<Source, Num[][]>;
  classes: (ActivityClass | null)[][];
}

interface SeasonCalendar extends Common {
  seasons: ({ season_year: number; season: string; complete: boolean; class: ActivityClass | null } & Record<Source, Num>)[];
}

interface YearCalendar extends Common {
  years: number[];
  series: SeriesBySource;
  classes: (ActivityClass | null)[];
  viirs_complete: boolean[];
  modis_full_coverage: boolean[];
  split: (string | null)[];
}

const SEASONS = ["DJF", "MAM", "JJA", "SON"];

function useBasis(step: "day" | "month" | "season"): string {
  const { t, meta } = useApp();
  return t(`calendar.basis.${step}`, { year: meta.record.analysis_first_year });
}

function useClasses(source: Source, values: Num[], apiClasses: (ActivityClass | null)[], apiThresholds: Thresholds) {
  return useMemo(() => {
    if (source === "harmonized") return { classes: apiClasses, thresholds: apiThresholds, own: false };
    const thresholds = thresholdsOf(values);
    return { classes: values.map((v) => classOf(v, thresholds)), thresholds, own: true };
  }, [source, values, apiClasses, apiThresholds]);
}

function DayBlock(props: { data: DayCalendar; source: Source; unit: string; picked: string | null; onPick: (d: string) => void }) {
  const { data, source } = props;
  const { classes, thresholds, own } = useClasses(source, data.series[source], data.classes, data.thresholds);
  const { t } = useApp();
  const basis = useBasis("day");
  const flagged = useMemo(() => new Set(data.anomalies.map((a) => a.date)), [data.anomalies]);
  return (
    <>
      <YearHeatmap
        dates={data.dates} values={data.series[source]} classes={classes}
        flagged={source === "harmonized" ? flagged : undefined}
        selected={props.picked} unit={props.unit} onPick={props.onPick}
      />
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "center", marginTop: 6 }}>
        <ClassLegend thresholds={thresholds} />
        {source === "harmonized" && flagged.size > 0 && (
          <span className="note" style={{ margin: 0 }}>□ {t("calendar.flagged", { n: flagged.size })}</span>
        )}
      </div>
      <p className="note">{own ? t("calendar.ownBasis") : basis}</p>
    </>
  );
}

function DayView(props: { source: Source; initialYear: number | null }) {
  const { selection, metric, meta, t } = useApp();
  const sel = selectionParams(selection);
  const lastYear = Number(meta.record.last_day.slice(0, 4));
  const firstYear = Number(meta.record.first_day.slice(0, 4));
  const years = Array.from({ length: lastYear - firstYear + 1 }, (_, i) => lastYear - i);
  const [year, setYear] = useState(props.initialYear ?? lastYear);
  const [other, setOther] = useState<number | "">("");
  const [picked, setPicked] = useState<string | null>(null);

  const main = useApi<DayCalendar>("/api/calendar", { ...sel, granularity: "day", metric, year });
  const second = useApi<DayCalendar>(other === "" ? null : "/api/calendar", { ...sel, granularity: "day", metric, year: other });
  const unit = metric === "frp" ? "MW" : metric === "count" ? t("detections") : "HFAI";

  const compare = useMemo(() => {
    if (!main.data) return [];
    const doy = (dates: string[]) => dates.map((_, i) => i + 1);
    const smooth = (v: Num[]) =>
      v.map((_, i) => {
        const part = v.slice(Math.max(0, i - 6), i + 1).filter((x): x is number => x !== null);
        return part.length ? part.reduce((a, b) => a + b, 0) / part.length : null;
      });
    const traces: any[] = [{
      x: doy(main.data.dates), y: smooth(main.data.series[props.source]), name: String(main.data.year),
      type: "scatter", mode: "lines", line: { color: COLORS[props.source], width: 2.2 },
    }];
    if (second.data && other !== "") {
      traces.push({
        x: doy(second.data.dates), y: smooth(second.data.series[props.source]), name: String(second.data.year),
        type: "scatter", mode: "lines", line: { color: COLORS.muted, width: 1.6, dash: "dot" },
      });
    }
    return traces;
  }, [main.data, second.data, other, props.source]);
  const compareLayout = useMemo(
    () => ({
      xaxis: { title: { text: t("calendar.dayOfYear") }, tickvals: [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
        ticktext: t("months").split(",") },
      yaxis: { title: { text: `${unit} (${t("calendar.mean7")})` } },
    }),
    [unit, t],
  );

  const detail = useMemo(() => {
    if (!picked) return null;
    const pool = [main.data, second.data].find((d) => d?.dates.includes(picked));
    if (!pool) return null;
    const i = pool.dates.indexOf(picked);
    return {
      date: picked,
      modis: pool.series.modis[i],
      viirs: pool.series.viirs[i],
      harmonized: pool.series.harmonized[i],
      cls: pool.classes[i],
      anomaly: pool.anomalies.find((a) => a.date === picked),
    };
  }, [picked, main.data, second.data]);

  return (
    <>
      <Panel
        title={`${t("calendar.daily")} · ${year}`}
        sub={t(`src.${props.source}`)}
        tools={
          <>
            <Field label={t("year")}>
              <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
                {years.map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            </Field>
            <Field label={t("calendar.compareWith")}>
              <select value={other} onChange={(e) => setOther(e.target.value ? Number(e.target.value) : "")}>
                <option value="">-</option>
                {years.filter((y) => y !== year).map((y) => <option key={y} value={y}>{y}</option>)}
              </select>
            </Field>
          </>
        }
      >
        <Async state={main} height={170}>
          {(d) => <DayBlock data={d} source={props.source} unit={unit} picked={picked} onPick={setPicked} />}
        </Async>
        {props.source === "viirs" && `${year}-12-31` < meta.record.viirs_start && (
          <p className="note warn">{t("calendar.noViirs", { date: meta.record.viirs_start })}</p>
        )}
      </Panel>

      {other !== "" && (
        <Panel title={`${t("calendar.daily")} · ${other}`} sub={t(`src.${props.source}`)}>
          <Async state={second} height={170}>
            {(d) => <DayBlock data={d} source={props.source} unit={unit} picked={picked} onPick={setPicked} />}
          </Async>
        </Panel>
      )}

      <div className="grid two">
        <Panel title={t("calendar.seasonCurve")} sub={other === "" ? String(year) : `${year} vs ${other}`}>
          <Async state={main} height={260}>
            {() => <Chart data={compare} layout={compareLayout} height={260} />}
          </Async>
        </Panel>
        <Panel title={t("calendar.dayDetail")}>
          {detail ? (
            <dl className="def-list">
              <dt>{t("date")}</dt><dd>{detail.date}</dd>
              <dt>{t("src.modis")}</dt>
              <dd>{fmt(detail.modis, 1)} {unit} <span className="tag observed">{t("observed")}</span></dd>
              <dt>{t("src.viirs")}</dt>
              <dd>
                {detail.viirs === null ? t("calendar.viirsMissing") : `${fmt(detail.viirs, 1)} ${unit}`}{" "}
                <span className="tag observed">{t("observed")}</span>
              </dd>
              <dt>{t("src.harmonized")}</dt>
              <dd>{fmt(detail.harmonized, 1)} {unit} <span className="tag estimate">{t("estimate")}</span></dd>
              <dt>{t("calendar.class")}</dt>
              <dd>{detail.cls ? t(`cls.${detail.cls}`) : "-"}</dd>
              <dt>{t("nav.anomalies")}</dt>
              <dd>
                {detail.anomaly ? (
                  <>
                    <span className={`tag ${detail.anomaly.severity}`}>{t(`sev.${detail.anomaly.severity}`)}</span>{" "}
                    {signed(detail.anomaly.pct_vs_reference)} {t("vsReference").toLowerCase()}
                  </>
                ) : t("calendar.notFlagged")}
              </dd>
            </dl>
          ) : (
            <p className="note">{t("calendar.pickDay")}</p>
          )}
        </Panel>
      </div>
    </>
  );
}

function MonthView(props: { source: Source; onOpenYear: (y: number) => void }) {
  const { selection, metric, meta, t } = useApp();
  const basis = useBasis("month");
  const state = useApi<MonthCalendar>("/api/calendar", { ...selectionParams(selection), granularity: "month", metric });
  const view = useMemo(() => {
    if (!state.data) return null;
    const d = state.data;
    const values = d.series[props.source];
    let classes = d.classes;
    let thresholds = d.thresholds;
    if (props.source !== "harmonized") {
      thresholds = thresholdsOf(values.flat());
      classes = values.map((row) => row.map((v) => classOf(v, thresholds)));
    }
    const order = d.years.map((_, i) => i).reverse();
    return {
      rows: order.map((i) => ({
        key: d.years[i], label: String(d.years[i]),
        note: d.years[i] < meta.record.analysis_first_year ? t("calendar.partialModis") : undefined,
      })),
      values: order.map((i) => values[i]),
      classes: order.map((i) => classes[i]),
      thresholds,
    };
  }, [state.data, props.source, meta, t]);

  return (
    <Panel title={t("calendar.monthly")} sub={`${t(`src.${props.source}`)} · ${t(`metricLabel.${metric}`)}`}>
      <Async state={state} height={420}>
        {() => view && (
          <>
            <MonthMatrix rows={view.rows} columns={t("months").split(",")} values={view.values} classes={view.classes}
              onPickRow={(y) => props.onOpenYear(Number(y))} />
            <div style={{ marginTop: 8 }}><ClassLegend thresholds={view.thresholds} /></div>
            <p className="note">
              {props.source === "harmonized" ? basis : t("calendar.ownBasis")}. {t("calendar.openYear")}{" "}
              * {t("calendar.partialModis")}.
            </p>
          </>
        )}
      </Async>
    </Panel>
  );
}

function SeasonView(props: { source: Source }) {
  const { selection, metric, t } = useApp();
  const basis = useBasis("season");
  const state = useApi<SeasonCalendar>("/api/calendar", { ...selectionParams(selection), granularity: "season", metric });
  const view = useMemo(() => {
    if (!state.data) return null;
    const d = state.data;
    const years = [...new Set(d.seasons.map((s) => s.season_year))].sort((a, b) => b - a);
    const thresholds = props.source === "harmonized" ? d.thresholds : thresholdsOf(d.seasons.map((s) => s[props.source]));
    const find = (y: number, s: string) => d.seasons.find((r) => r.season_year === y && r.season === s);
    return {
      rows: years.map((y) => ({
        key: y, label: String(y),
        note: SEASONS.some((s) => find(y, s) && !find(y, s)!.complete) ? t("calendar.partialSeason") : undefined,
      })),
      values: years.map((y) => SEASONS.map((s) => find(y, s)?.[props.source] ?? null)),
      classes: years.map((y) =>
        SEASONS.map((s) => {
          const row = find(y, s);
          if (!row) return null;
          return props.source === "harmonized" ? row.class : classOf(row[props.source], thresholds);
        }),
      ),
      thresholds,
    };
  }, [state.data, props.source, t]);

  return (
    <Panel title={t("calendar.seasonal")} sub={`${t(`src.${props.source}`)} · ${t(`metricLabel.${metric}`)}`}>
      <Async state={state} height={420}>
        {() => view && (
          <>
            <MonthMatrix rows={view.rows} columns={SEASONS} values={view.values} classes={view.classes} />
            <div style={{ marginTop: 8 }}><ClassLegend thresholds={view.thresholds} /></div>
            <p className="note">
              {props.source === "harmonized" ? basis : t("calendar.ownBasis")}. {t("calendar.seasonNote")}{" "}
              * {t("calendar.partialSeason")}.
            </p>
          </>
        )}
      </Async>
    </Panel>
  );
}

function YearView(props: { onOpenYear: (y: number) => void }) {
  const { selection, metric, meta, t } = useApp();
  const state = useApi<YearCalendar>("/api/calendar", { ...selectionParams(selection), granularity: "year", metric });
  const data = useMemo(() => (state.data ? sourceTraces(state.data.years, state.data.series, t, "bar") : []), [state.data, t]);
  const layout = useMemo(
    () => ({ barmode: "group", yaxis: { title: { text: t(`metricLabel.${metric}`) } }, xaxis: { dtick: 2 } }),
    [t, metric],
  );
  return (
    <Panel title={t("calendar.yearly")} sub={t(`metricLabel.${metric}`)}>
      <Async state={state} height={320}>
        {(d) => (
          <>
            <Chart data={data} layout={layout} height={320} onPointClick={(p) => props.onOpenYear(Number(p.x))} />
            <div className="table-scroll" style={{ maxHeight: 340 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th className="left">{t("year")}</th><th>{t("src.modis")}</th><th>{t("src.viirs")}</th>
                    <th>{t("src.harmonized")}</th><th>{t("calendar.class")}</th><th className="left">{t("calendar.role")}</th>
                  </tr>
                </thead>
                <tbody>
                  {d.years.map((_, k) => {
                    const i = d.years.length - 1 - k;
                    return (
                      <tr key={d.years[i]} className="clickable" onClick={() => props.onOpenYear(d.years[i])}>
                        <td className="left">{d.years[i]}{d.modis_full_coverage[i] ? "" : " *"}</td>
                        <td>{fmt(d.series.modis[i], 0)}</td>
                        <td>{d.series.viirs[i] === null ? "-" : `${fmt(d.series.viirs[i], 0)}${d.viirs_complete[i] ? "" : " †"}`}</td>
                        <td>{fmt(d.series.harmonized[i], 0)}</td>
                        <td>{d.classes[i] ? <span className={`tag cls-${d.classes[i]}`}>{t(`cls.${d.classes[i]}`)}</span> : "-"}</td>
                        <td className="left">{d.split[i] ? t(`split.${d.split[i]}`) : t("split.none")}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="note">
              * {t("calendar.partialModis")}. † {t("calendar.partialViirs")}.{" "}
              {t("coverageNote", { year: meta.record.analysis_first_year, viirs: meta.record.viirs_start })}
            </p>
          </>
        )}
      </Async>
    </Panel>
  );
}

export function CalendarPage() {
  const { t } = useApp();
  const [granularity, setGranularity] = useState<Granularity>("month");
  const [source, setSource] = useState<Source>("harmonized");
  const [dayKey, setDayKey] = useState(0);
  const [openYear, setOpenYear] = useState<number | null>(null);

  const open = (year: number) => {
    setOpenYear(year);
    setDayKey((k) => k + 1);
    setGranularity("day");
  };

  return (
    <>
      <PageHead
        title={t("nav.calendar")}
        text={t("calendar.intro")}
        tools={
          <>
            <Segmented
              label={t("calendar.granularity")} value={granularity} onChange={setGranularity}
              options={[
                { value: "day", label: t("calendar.day") },
                { value: "month", label: t("calendar.month") },
                { value: "season", label: t("calendar.season") },
                { value: "year", label: t("year") },
              ]}
            />
            {granularity !== "year" && (
              <Segmented
                label={t("calendar.source")} value={source} onChange={setSource}
                options={[
                  { value: "harmonized", label: t("src.harmonized") },
                  { value: "modis", label: "MODIS" },
                  { value: "viirs", label: "VIIRS" },
                ]}
              />
            )}
          </>
        }
      />
      {granularity === "day" && <DayView key={dayKey} source={source} initialYear={openYear} />}
      {granularity === "month" && <MonthView source={source} onOpenYear={open} />}
      {granularity === "season" && <SeasonView source={source} />}
      {granularity === "year" && <YearView onOpenYear={open} />}
    </>
  );
}
