import { useState } from "react";
import { PageHead, Panel, Segmented, fmt } from "../components/ui";
import { useApp } from "../state";

const SENSOR_NAMES: Record<string, string> = {
  modis: "MODIS (Terra, Aqua)",
  viirs_snpp: "VIIRS S-NPP",
  viirs_noaa20: "VIIRS NOAA-20",
};

export function DataPage() {
  const { meta, t } = useApp();
  const sensors = Object.keys(SENSOR_NAMES).filter((s) => meta.files.some((f) => f.sensor === s));
  const [sensor, setSensor] = useState(sensors[0] ?? "modis");
  const files = meta.files.filter((f) => f.sensor === sensor);
  const quality = meta.quality ?? {};

  return (
    <>
      <PageHead title={t("nav.data")} text={t("data.intro")} />

      <Panel title={t("data.quality")} sub={t("data.qualitySub")}>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th className="left">{t("data.sensor")}</th><th className="left">{t("data.product")}</th>
                <th>{t("data.years")}</th><th>{t("data.files")}</th><th>{t("data.raw")}</th><th>{t("data.rejected")}</th>
                <th>{t("data.duplicates")}</th><th>{t("data.valid")}</th><th>{t("data.lowConf")}</th>
                <th>{t("data.nonVeg")}</th><th>{t("data.analysis")}</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(quality).map(([name, q]) => (
                <tr key={name}>
                  <td className="left">{SENSOR_NAMES[name] ?? name}</td>
                  <td className="left">{meta.files.find((f) => f.sensor === name)?.product ?? "-"}</td>
                  <td>{q.first_year}-{q.last_year}</td>
                  <td>{q.files}</td>
                  <td>{fmt(q.raw_records)}</td>
                  <td>{fmt(q.rejected)}</td>
                  <td>{fmt(q.duplicates_removed)}</td>
                  <td>{fmt(q.valid_records)}</td>
                  <td>{fmt(q.flagged_low_confidence)}</td>
                  <td>{fmt(q.flagged_non_vegetation_source)}</td>
                  <td>{fmt(q.analysis_records)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">{t("data.qualityNote")}</p>
      </Panel>

      <div className="grid two">
        <Panel title={t("data.record")}>
          <dl className="def-list">
            <dt>{t("data.country")}</dt><dd>{meta.country} ({meta.iso3})</dd>
            <dt>{t("data.period")}</dt><dd>{meta.record.first_day} → {meta.record.last_day}</dd>
            <dt>{t("data.viirsStart")}</dt><dd>{meta.record.viirs_start}</dd>
            <dt>{t("data.noaaStart")}</dt><dd>{meta.record.noaa20_start ?? "-"}</dd>
            <dt>{t("data.fullModis")}</dt><dd>{meta.record.analysis_first_year}</dd>
            <dt>{t("data.grid")}</dt><dd>{meta.grid.cell_km} km · {meta.grid.crs}</dd>
            <dt>{t("data.built")}</dt><dd>{meta.built_at}</dd>
          </dl>
          <p className="note">{t("data.recordNote", { year: meta.record.analysis_first_year })}</p>
        </Panel>
        <Panel title={t("data.sources")}>
          <dl className="def-list">
            {meta.sources.map((s) => (
              <span key={s.url} style={{ display: "contents" }}>
                <dt>{s.name}</dt>
                <dd><a href={s.url} target="_blank" rel="noreferrer">{s.url}</a></dd>
              </span>
            ))}
          </dl>
          <p className="note">{t("data.sourcesNote")}</p>
        </Panel>
      </div>

      <Panel
        title={t("data.manifest")}
        sub={`${files.length} ${t("data.files").toLowerCase()}`}
        tools={
          <Segmented value={sensor} onChange={setSensor}
            options={sensors.map((s) => ({ value: s, label: SENSOR_NAMES[s] }))} />
        }
      >
        <div className="table-scroll" style={{ maxHeight: 420 }}>
          <table className="data">
            <thead>
              <tr>
                <th className="left">{t("year")}</th><th>{t("data.records")}</th><th>{t("data.firstDate")}</th>
                <th>{t("data.lastDate")}</th><th className="left">{t("data.version")}</th>
                <th>{t("data.downloaded")}</th><th className="left">SHA-256</th><th className="left">{t("data.file")}</th>
              </tr>
            </thead>
            <tbody>
              {files.map((f) => (
                <tr key={f.url}>
                  <td className="left">{f.year}</td>
                  <td>{fmt(f.records)}</td>
                  <td>{f.date_min}</td>
                  <td>{f.date_max}</td>
                  <td className="left">{String(f.versions)}</td>
                  <td>{String(f.downloaded_at).slice(0, 10)}</td>
                  <td className="left"><code>{String(f.sha256).slice(0, 12)}</code></td>
                  <td className="left"><a href={f.url} target="_blank" rel="noreferrer">{String(f.url).split("/").pop()}</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
