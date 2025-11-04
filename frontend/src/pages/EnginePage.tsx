import { useMemo, useState } from "react";
import { Chart, COLORS } from "../components/Chart";
import { Async, Field, PageHead, Panel, Segmented, fmt, signed } from "../components/ui";
import { selectionParams, type Num, type SeriesBySource } from "../services/api";
import { useApi, useApp } from "../state";
import { sourceTraces } from "./Dashboard";

interface Scores {
  n: number; mae: number; rmse: number; r2: number; bias: number; bias_pct: number; spearman: number;
}
type ByMethod = Record<string, Scores>;

interface Agreement {
  periods: number;
  modis_vs_viirs: Record<string, Num> | null;
  harmonized_vs_viirs: Record<string, Num> | null;
}

interface Harmonization {
  periods: string[];
  split: (string | null)[];
  series: SeriesBySource;
  difference: { modis_minus_viirs: Num[]; harmonized_minus_viirs: Num[] };
  agreement: Record<string, Agreement>;
  components?: { model: Num[]; background: Num[] };
  metric_label: string;
  model_version: string;
  note: string;
}

interface ModelInfo {
  version: string;
  created_at: string;
  model: {
    algorithm: string; reference_sensor: string; input_sensor: string; unit: string; features: string[];
    split: Record<string, [number, number]>;
    rows: { train_available: number; train_used: number; validation: number };
    targets: Record<string, {
      target: string; params: Record<string, string | number>; trees: number;
      search: { params: Record<string, string | number>; best_iteration: number; valid_rmse: number; valid_mae: number; valid_bias: number }[];
      validation: Scores; feature_importance_gain: Record<string, number>;
    }>;
    hfai: { count_ref: number; frp_ref: number; weight_count: number; weight_frp: number; reference_years: number[] };
    background: { block_cells: number; fitted_on: number[]; residual_viirs_detections: number };
    data_fingerprint: string;
    libraries: Record<string, string>;
    seed: number;
  };
  metrics: {
    evaluated_at: string;
    test_years: [number, number];
    labels: Record<string, string>;
    cell_day: Record<string, Record<string, ByMethod>>;
    ceiling: { noaa20_as_predictor: Scores; lightgbm_same_rows: Scores } | null;
    aggregated: Record<string, Record<string, Record<string, ByMethod>>>;
    high_activity: {
      definition: string;
      test: Record<string, { observed_high: number; predicted_high: number; precision: number; recall: number; f1: number }>;
    };
    by_year: { year: number; split: string; modis_count: number; viirs_count: number; harmonized_count: number;
      modis_ratio: number; harmonized_ratio: number }[];
  } | null;
  versions: { version: string; created_at: string; cell_km: number; is_active: boolean }[];
  runs: { run_at: string; model_version: string; first_year: number; last_year: number; cell_days: number }[];
}

const METHOD_ORDER = ["modis_raw", "linear", "quantile_mapping", "lightgbm", "lightgbm_no_background"];
const LEVELS = ["cell_day", "province_day", "province_month", "country_day"] as const;
type Level = (typeof LEVELS)[number];

function digits(v: number): number {
  const abs = Math.abs(v);
  return abs >= 100 ? 0 : abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
}

export function EnginePage() {
  const { selection, metric, meta, t } = useApp();
  const sel = selectionParams(selection);
  const [freq, setFreq] = useState("month");
  const [span, setSpan] = useState<"overlap" | "all" | "test">("overlap");
  const [level, setLevel] = useState<Level>("province_day");

  const test = meta.split.test;
  const range =
    span === "all"
      ? {}
      : span === "test"
        ? { start: `${test[0]}-01-01`, end: `${test[1]}-12-31` }
        : { start: `${meta.record.viirs_start.slice(0, 4)}-01-01` };
  const effFreq = span === "test" && freq === "year" ? "month" : freq;
  const harm = useApi<Harmonization>("/api/harmonization", { ...sel, metric, freq: effFreq, ...range });
  const model = useApi<ModelInfo>("/api/model/metrics");

  const seriesTraces = useMemo(
    () => (harm.data ? sourceTraces(harm.data.periods, harm.data.series, t, "scatter") : []),
    [harm.data, t],
  );
  const xaxis = useMemo(() => ({ type: effFreq === "year" ? "linear" : "date", dtick: effFreq === "year" ? 2 : undefined }), [effFreq]);
  const seriesLayout = useMemo(
    () => ({ xaxis, yaxis: { title: { text: t(`metricLabel.${metric}`) } } }),
    [xaxis, t, metric],
  );
  const diffTraces = useMemo(() => {
    if (!harm.data) return [];
    return [
      { x: harm.data.periods, y: harm.data.difference.modis_minus_viirs, type: "bar", name: t("engine.modisMinus"),
        marker: { color: COLORS.modis } },
      { x: harm.data.periods, y: harm.data.difference.harmonized_minus_viirs, type: "bar", name: t("engine.harmMinus"),
        marker: { color: COLORS.harmonized } },
    ];
  }, [harm.data, t]);
  const diffLayout = useMemo(
    () => ({ barmode: "group", xaxis, yaxis: { title: { text: t("engine.error") }, rangemode: "normal" } }),
    [t, xaxis],
  );

  const metrics = model.data?.metrics ?? null;
  const table = useMemo(() => {
    if (!metrics) return null;
    const key = metric;
    const rows: ByMethod | undefined =
      level === "cell_day" ? metrics.cell_day.test?.[key] : metrics.aggregated.test?.[level]?.[key];
    if (!rows) return null;
    const names = METHOD_ORDER.filter((m) => rows[m]);
    const best = (field: "mae" | "rmse") =>
      names.reduce((a, b) => (rows[a][field] <= rows[b][field] ? a : b));
    return { rows, names, bestMae: best("mae"), bestRmse: best("rmse") };
  }, [metrics, level, metric]);

  const ratioTraces = useMemo(() => {
    if (!metrics) return [];
    const y = metrics.by_year;
    return [
      { x: y.map((r) => r.year), y: y.map((r) => r.modis_ratio), name: "MODIS / VIIRS", type: "scatter", mode: "lines+markers",
        line: { color: COLORS.modis, width: 1.8 } },
      { x: y.map((r) => r.year), y: y.map((r) => r.harmonized_ratio), name: t("engine.harmRatio"), type: "scatter",
        mode: "lines+markers", line: { color: COLORS.harmonized, width: 2.2 } },
    ];
  }, [metrics, t]);
  const ratioLayout = useMemo(() => {
    const split = meta.split;
    return {
      yaxis: { title: { text: t("engine.ratio") }, range: [0, 1.25] },
      shapes: [
        { type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: COLORS.viirs, width: 1, dash: "dash" } },
        { type: "rect", xref: "x", yref: "paper", x0: split.validation[0] - 0.5, x1: split.validation[1] + 0.5, y0: 0, y1: 1,
          fillcolor: "rgba(255,255,255,.04)", line: { width: 0 } },
        { type: "rect", xref: "x", yref: "paper", x0: split.test[0] - 0.5, x1: split.test[1] + 0.5, y0: 0, y1: 1,
          fillcolor: "rgba(245,158,11,.10)", line: { width: 0 } },
      ],
    };
  }, [meta, t]);

  const importance = useMemo(() => {
    const target = model.data?.model.targets.count;
    if (!target) return [];
    const entries = Object.entries(target.feature_importance_gain).sort((a, b) => a[1] - b[1]);
    const total = entries.reduce((s, e) => s + e[1], 0) || 1;
    return [{ y: entries.map((e) => e[0]), x: entries.map((e) => (100 * e[1]) / total), type: "bar", orientation: "h",
      marker: { color: COLORS.harmonized } }];
  }, [model.data]);
  const importanceLayout = useMemo(
    () => ({ showlegend: false, hovermode: "closest", margin: { l: 150 }, xaxis: { title: { text: t("engine.gainShare") } },
      yaxis: { rangemode: "normal" } }),
    [t],
  );

  const steps = [
    ["1", t("engine.step1"), t("engine.step1Text")],
    ["2", t("engine.step2"), t("engine.step2Text", { km: meta.grid.cell_km })],
    ["3", t("engine.step3"), t("engine.step3Text")],
    ["4", t("engine.step4"), t("engine.step4Text")],
    ["5", t("engine.step5"), t("engine.step5Text")],
    ["6", t("engine.step6"), t("engine.step6Text")],
  ];
  const splitName = (s: string) => t(`split.${s}`);

  return (
    <>
      <PageHead title={t("nav.engine")} text={t("engine.intro")} />

      <Panel title={t("engine.pipeline")}>
        <div className="pipeline">
          {steps.map(([n, title, text]) => (
            <div className="step" key={n}>
              <span className="n">{n}</span>
              <b>{title}</b>
              <span>{text}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel
        title={t("engine.threeSeries")}
        sub={t(`metricLabel.${metric}`)}
        tools={
          <>
            <Segmented label={t("period")} value={span} onChange={setSpan}
              options={[
                { value: "overlap", label: t("engine.overlap") },
                { value: "test", label: `${t("split.test")} ${test[0] === test[1] ? test[0] : test.join("-")}` },
                { value: "all", label: t("engine.fullRecord") },
              ]} />
            <Segmented label={t("reports.frequency")} value={effFreq} onChange={setFreq}
              options={[
                { value: "day", label: t("calendar.day") },
                { value: "month", label: t("calendar.month") },
                { value: "year", label: t("year") },
              ]} />
          </>
        }
      >
        <Async state={harm} height={300}>
          {(d) => (
            <>
              <Chart data={seriesTraces} layout={seriesLayout} height={300} />
              <Chart data={diffTraces} layout={diffLayout} height={190} />
              <p className="note">{t("engine.seriesNote", { date: meta.record.viirs_start })}</p>
              {Object.keys(d.agreement).length > 0 && (
                <div className="table-scroll">
                  <table className="data">
                    <thead>
                      <tr>
                        <th className="left">{t("engine.years")}</th><th className="left">{t("engine.series")}</th>
                        <th>n</th><th>MAE</th><th>RMSE</th><th>R²</th><th>{t("engine.bias")}</th><th>Spearman</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(["train", "validation", "test"] as const).filter((s) => d.agreement[s]).flatMap((s) =>
                        (["modis_vs_viirs", "harmonized_vs_viirs"] as const).map((k) => {
                          const m = d.agreement[s][k];
                          return (
                            <tr key={s + k} className={k === "harmonized_vs_viirs" ? "best" : ""}>
                              <td className="left">
                                {k === "modis_vs_viirs" && (
                                  <span className={`tag ${s === "test" ? "test" : ""}`}>
                                    {splitName(s)} {meta.split[s].join("-")}
                                  </span>
                                )}
                              </td>
                              <td className="left">{k === "modis_vs_viirs" ? t("src.modis") : t("src.harmonized")}</td>
                              <td>{m ? fmt(m.n) : "-"}</td>
                              <td>{m ? fmt(m.mae, digits(m.mae ?? 0)) : "-"}</td>
                              <td>{m ? fmt(m.rmse, digits(m.rmse ?? 0)) : "-"}</td>
                              <td>{m ? fmt(m.r2, 3) : "-"}</td>
                              <td>{m ? signed(m.bias_pct, 1) : "-"}</td>
                              <td>{m ? fmt(m.spearman, 3) : "-"}</td>
                            </tr>
                          );
                        }),
                      )}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="note">{t("engine.agreementNote")}</p>
            </>
          )}
        </Async>
      </Panel>

      <Async state={model} height={300}>
        {(m) => (
          <>
            <div className="grid two">
              <Panel
                title={t("engine.testMetrics", { years: m.metrics ? [...new Set(m.metrics.test_years)].join("-") : "" })}
                sub={t("engine.independent")}
                tools={
                  <Field label={t("engine.level")}>
                    <select value={level} onChange={(e) => setLevel(e.target.value as Level)}>
                      {LEVELS.map((l) => <option key={l} value={l}>{t(`engine.level.${l}`)}</option>)}
                    </select>
                  </Field>
                }
              >
                {table && m.metrics ? (
                  <>
                    <div className="table-scroll">
                      <table className="data">
                        <thead>
                          <tr>
                            <th className="left">{t("engine.method")}</th>
                            <th>MAE</th><th>RMSE</th><th>R²</th><th>{t("engine.bias")}</th><th>Spearman</th>
                          </tr>
                        </thead>
                        <tbody>
                          {table.names.map((name) => {
                            const r = table.rows[name];
                            return (
                              <tr key={name} className={name === "lightgbm" ? "best" : ""}>
                                <td className="left">{m.metrics!.labels[name]}</td>
                                <td>{fmt(r.mae, digits(r.mae))}{name === table.bestMae ? " ●" : ""}</td>
                                <td>{fmt(r.rmse, digits(r.rmse))}{name === table.bestRmse ? " ●" : ""}</td>
                                <td>{fmt(r.r2, 3)}</td>
                                <td>{signed(r.bias_pct, 1)}</td>
                                <td>{fmt(r.spearman, 3)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    <p className="note">
                      {t("engine.metricsNote", { n: fmt(table.rows[table.names[0]].n), metric: metric.toUpperCase() })}
                      {level === "cell_day" && ` ${t("engine.cellNote")}`}
                    </p>
                    {level === "cell_day" && metric === "count" && m.metrics.ceiling && (
                      <p className="note">
                        {t("engine.ceiling", {
                          r2: fmt(m.metrics.ceiling.noaa20_as_predictor.r2, 3),
                          rmse: fmt(m.metrics.ceiling.noaa20_as_predictor.rmse, 3),
                          ours: fmt(m.metrics.ceiling.lightgbm_same_rows.r2, 3),
                        })}
                      </p>
                    )}
                  </>
                ) : (
                  <p className="note">{t("engine.noMetrics")}</p>
                )}
              </Panel>

              <Panel title={t("engine.highActivity")} sub={t("engine.independent")}>
                {m.metrics ? (
                  <>
                    <div className="table-scroll">
                      <table className="data">
                        <thead>
                          <tr>
                            <th className="left">{t("engine.method")}</th><th>{t("engine.observedHigh")}</th>
                            <th>{t("engine.flagged")}</th><th>{t("engine.precision")}</th>
                            <th>{t("engine.recall")}</th><th>F1</th>
                          </tr>
                        </thead>
                        <tbody>
                          {METHOD_ORDER.filter((n) => m.metrics!.high_activity.test[n]).map((name) => {
                            const r = m.metrics!.high_activity.test[name];
                            return (
                              <tr key={name} className={name === "lightgbm" ? "best" : ""}>
                                <td className="left">{m.metrics!.labels[name]}</td>
                                <td>{r.observed_high}</td>
                                <td>{r.predicted_high}</td>
                                <td>{fmt(r.precision, 3)}</td>
                                <td>{fmt(r.recall, 3)}</td>
                                <td>{fmt(r.f1, 3)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    <p className="note">{m.metrics.high_activity.definition}</p>
                  </>
                ) : (
                  <p className="note">{t("engine.noMetrics")}</p>
                )}
              </Panel>
            </div>

            <div className="grid two">
              <Panel title={t("engine.ratioTitle")} sub={meta.country}>
                {m.metrics && <Chart data={ratioTraces} layout={ratioLayout} height={280} />}
                <p className="note">{t("engine.ratioNote")}</p>
              </Panel>
              <Panel title={t("engine.importance")} sub={m.model.targets.count?.target}>
                <Chart data={importance} layout={importanceLayout} height={420} />
              </Panel>
            </div>

            <div className="grid two">
              <Panel title={t("engine.modelCard")}>
                <dl className="def-list">
                  <dt>{t("engine.version")}</dt><dd><code>{m.version}</code></dd>
                  <dt>{t("engine.algorithm")}</dt><dd>{m.model.algorithm}</dd>
                  <dt>{t("engine.input")}</dt><dd>{m.model.input_sensor}</dd>
                  <dt>{t("engine.reference")}</dt><dd>{m.model.reference_sensor}</dd>
                  <dt>{t("engine.unit")}</dt><dd>{m.model.unit}</dd>
                  <dt>{t("engine.split")}</dt>
                  <dd>
                    {(["train", "validation", "test"] as const)
                      .map((s) => `${splitName(s)} ${m.model.split[s][0] === m.model.split[s][1] ? m.model.split[s][0] : m.model.split[s].join("-")}`)
                      .join(" · ")}
                  </dd>
                  <dt>{t("engine.rows")}</dt>
                  <dd>
                    {t("engine.rowsText", {
                      used: fmt(m.model.rows.train_used), available: fmt(m.model.rows.train_available),
                      validation: fmt(m.model.rows.validation),
                    })}
                  </dd>
                  {Object.entries(m.model.targets).map(([name, target]) => (
                    <span key={name} style={{ display: "contents" }}>
                      <dt>{target.target}</dt>
                      <dd>
                        {target.params.objective}
                        {target.params.tweedie_variance_power ? ` (p = ${target.params.tweedie_variance_power})` : ""},{" "}
                        {target.trees} {t("engine.trees")}, {target.params.num_leaves} {t("engine.leaves")},{" "}
                        {t("engine.validRmse")} {fmt(target.validation.rmse, 3)}
                      </dd>
                    </span>
                  ))}
                  <dt>{t("engine.trained")}</dt><dd>{m.created_at}</dd>
                  <dt>{t("engine.evaluated")}</dt><dd>{m.metrics?.evaluated_at ?? "-"}</dd>
                  <dt>{t("engine.fingerprint")}</dt><dd><code>{m.model.data_fingerprint}</code></dd>
                  <dt>{t("engine.libraries")}</dt>
                  <dd>{Object.entries(m.model.libraries).map(([k, v]) => `${k} ${v}`).join(" · ")} · seed {m.model.seed}</dd>
                </dl>
              </Panel>

              <Panel title={t("engine.hfai")}>
                <p style={{ margin: "0 0 8px", fontFamily: "ui-monospace, Consolas, monospace", fontSize: 13 }}>
                  HFAI = {m.model.hfai.weight_count} × N / {fmt(m.model.hfai.count_ref, 3)} + {m.model.hfai.weight_frp} × FRP /{" "}
                  {fmt(m.model.hfai.frp_ref, 2)}
                </p>
                <p className="note">
                  {t("engine.hfaiText", {
                    count: fmt(m.model.hfai.count_ref, 3), frp: fmt(m.model.hfai.frp_ref, 2),
                    years: m.model.hfai.reference_years.join("-"),
                  })}
                </p>
                <p className="note">
                  {t("engine.backgroundText", {
                    block: m.model.background.block_cells * meta.grid.cell_km,
                    years: m.model.background.fitted_on.join("-"),
                  })}
                </p>
                <h3 style={{ fontSize: 13, margin: "14px 0 6px" }}>{t("engine.runs")}</h3>
                <div className="table-scroll" style={{ maxHeight: 180 }}>
                  <table className="data">
                    <thead>
                      <tr>
                        <th className="left">{t("engine.runAt")}</th><th className="left">{t("engine.version")}</th>
                        <th>{t("engine.years")}</th><th>{t("engine.cellDays")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {m.runs.map((r) => (
                        <tr key={r.run_at}>
                          <td className="left">{r.run_at}</td>
                          <td className="left"><code>{r.model_version}</code></td>
                          <td>{r.first_year}-{r.last_year}</td>
                          <td>{fmt(r.cell_days)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {m.versions.length > 1 && (
                  <p className="note">
                    {t("engine.otherVersions")}: {m.versions.filter((v) => !v.is_active).map((v) => v.version).join(", ")}
                  </p>
                )}
              </Panel>
            </div>
          </>
        )}
      </Async>
    </>
  );
}
