import { useState } from "react";
import { Field, PageHead, Panel, Segmented } from "../components/ui";
import { buildUrl, selectionParams } from "../services/api";
import { useApp } from "../state";

export function ReportsPage() {
  const { selection, meta, t } = useApp();
  const lastYear = meta.record.last_day.slice(0, 4);
  const [start, setStart] = useState(`${lastYear}-01-01`);
  const [end, setEnd] = useState(meta.record.last_day);
  const [freq, setFreq] = useState("day");

  const valid = start && end && start <= end;
  const base = { ...selectionParams(selection), start, end };
  const link = (kind: string, extra: Record<string, string> = {}) => buildUrl("/api/reports", { ...base, kind, ...extra });
  const name = selection.kind === "aoi" ? t("customArea") : selection.name;

  const cards: { kind: string; title: string; text: string; action: string; extra: Record<string, string> }[] = [
    { kind: "pdf", title: t("reports.pdf"), text: t("reports.pdfText"), action: t("reports.download", { format: "PDF" }), extra: {} },
    { kind: "series", title: t("reports.series"), text: t("reports.seriesText"), action: t("reports.download", { format: "CSV" }), extra: { freq } },
    { kind: "anomalies", title: t("reports.anomalies"), text: t("reports.anomaliesText"), action: t("reports.download", { format: "CSV" }), extra: {} },
    { kind: "grid", title: t("reports.grid"), text: t("reports.gridText"), action: t("reports.download", { format: "GeoJSON" }), extra: {} },
  ];

  return (
    <>
      <PageHead title={t("nav.reports")} text={t("reports.intro")} />
      <Panel title={t("reports.scope")} sub={name}>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-end" }}>
          <Field label={t("from")}>
            <input type="date" value={start} min={meta.record.first_day} max={meta.record.last_day}
              onChange={(e) => setStart(e.target.value)} />
          </Field>
          <Field label={t("to")}>
            <input type="date" value={end} min={meta.record.first_day} max={meta.record.last_day}
              onChange={(e) => setEnd(e.target.value)} />
          </Field>
          <Field label={t("reports.frequency")}>
            <Segmented value={freq} onChange={setFreq}
              options={[
                { value: "day", label: t("calendar.day") },
                { value: "month", label: t("calendar.month") },
                { value: "year", label: t("year") },
              ]} />
          </Field>
        </div>
        {!valid && <p className="note warn">{t("reports.invalid")}</p>}
        <p className="note">{t("reports.scopeNote")}</p>
      </Panel>

      <Panel title={t("reports.exports")}>
        <div className="export-grid">
          {cards.map((c) => (
            <div className="export-card" key={c.kind}>
              <strong>{c.title}</strong>
              <p>{c.text}</p>
              {valid ? (
                <a className="btn primary" href={link(c.kind, c.extra)} target="_blank" rel="noreferrer" download>
                  {c.action}
                </a>
              ) : (
                <button className="btn" disabled>{c.action}</button>
              )}
            </div>
          ))}
        </div>
      </Panel>

      <Panel title={t("reports.contents")}>
        <dl className="def-list">
          <dt>{t("model")}</dt><dd>{meta.active_model}</dd>
          <dt>{t("reports.grid5")}</dt><dd>{meta.grid.cell_km} km · {meta.grid.crs}</dd>
          <dt>{t("reports.dataThrough")}</dt><dd>{meta.record.first_day} → {meta.record.last_day}</dd>
          <dt>{t("reports.columns")}</dt><dd>{t("reports.columnsText")}</dd>
          <dt>{t("reports.method")}</dt><dd>{t("reports.methodText")}</dd>
        </dl>
      </Panel>
    </>
  );
}
