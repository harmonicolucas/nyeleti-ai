import { useEffect, useState } from "react";
import { AreaPicker } from "./components/AreaPicker";
import { Field, Segmented } from "./components/ui";
import { AnomaliesPage } from "./pages/AnomaliesPage";
import { CalendarPage } from "./pages/CalendarPage";
import { CriticalPage } from "./pages/CriticalPage";
import { Dashboard } from "./pages/Dashboard";
import { DataPage } from "./pages/DataPage";
import { EnginePage } from "./pages/EnginePage";
import { MapPage } from "./pages/MapPage";
import { ReportsPage } from "./pages/ReportsPage";
import { get, type Meta, type Metric } from "./services/api";
import type { Lang } from "./services/i18n";
import { AppProvider, useApp } from "./state";

const PAGES = [
  { id: "dashboard", component: Dashboard },
  { id: "map", component: MapPage },
  { id: "calendar", component: CalendarPage },
  { id: "engine", component: EnginePage },
  { id: "anomalies", component: AnomaliesPage },
  { id: "critical", component: CriticalPage },
  { id: "reports", component: ReportsPage },
  { id: "data", component: DataPage },
] as const;

type PageId = (typeof PAGES)[number]["id"];

function pageFromHash(): PageId {
  const id = window.location.hash.replace("#", "");
  return PAGES.some((p) => p.id === id) ? (id as PageId) : "dashboard";
}

function Shell() {
  const { t, metric, setMetric, lang, setLang, meta, selection } = useApp();
  const [page, setPage] = useState<PageId>(pageFromHash);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const Page = PAGES.find((p) => p.id === page)!.component;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-name">NYELETI <span>AI</span></div>
          <div className="brand-tag">{t("tagline")}</div>
        </div>
        <nav className="nav" aria-label="Modules">
          {PAGES.map((p, i) => (
            <button
              key={p.id}
              className={p.id === page ? "active" : ""}
              aria-current={p.id === page ? "page" : undefined}
              onClick={() => {
                window.location.hash = p.id;
              }}
            >
              <span className="num">{i + 1}</span>
              {t(`nav.${p.id}`)}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          {t("dataFrom")} NASA FIRMS<br />
          {meta.record.first_day.slice(0, 4)}–{meta.record.last_day.slice(0, 4)} · {meta.grid.cell_km} km<br />
          {t("model")}: <code>{meta.active_model}</code>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <AreaPicker />
          <div className="field">
            <span className="label">{t("selectedArea")}</span>
            <strong>{selection.kind === "aoi" ? t("customArea") : selection.name}</strong>
          </div>
          <div className="spacer" />
          <Field label={t("metric")}>
            <Segmented<Metric>
              value={metric}
              onChange={setMetric}
              label={t("metric")}
              options={[
                { value: "hfai", label: "HFAI" },
                { value: "count", label: t("detections") },
                { value: "frp", label: "FRP" },
              ]}
            />
          </Field>
          <Field label={t("language")}>
            <Segmented<Lang>
              value={lang}
              onChange={setLang}
              label={t("language")}
              options={[
                { value: "en", label: "EN" },
                { value: "pt", label: "PT" },
              ]}
            />
          </Field>
        </header>
        <main className="content">
          <Page />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    get<Meta>("/api/meta").then(setMeta).catch((err: Error) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="splash">
        <div>
          <h1>NYELETI AI</h1>
          <p>The API is not reachable or the database has not been built yet.</p>
          <p>{error}</p>
          <code>
            python -m scripts.run_pipeline
            <br />
            uvicorn backend.app.main:app --port 8000
          </code>
        </div>
      </div>
    );
  }
  if (!meta) return <div className="splash"><div><h1>NYELETI AI</h1><p>Loading…</p></div></div>;

  return (
    <AppProvider meta={meta}>
      <Shell />
    </AppProvider>
  );
}
