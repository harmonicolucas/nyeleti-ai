import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { get, type Meta, type Metric, type Params, type Selection } from "./services/api";
import { translate, type Lang } from "./services/i18n";

interface AppState {
  meta: Meta;
  selection: Selection;
  setSelection: (sel: Selection) => void;
  resetSelection: () => void;
  metric: Metric;
  setMetric: (m: Metric) => void;
  lang: Lang;
  setLang: (l: Lang) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const Context = createContext<AppState | null>(null);

export function AppProvider({ meta, children }: { meta: Meta; children: ReactNode }) {
  const country: Selection = useMemo(
    () => ({ kind: "area", areaId: meta.iso3, name: meta.country, level: "ADM0" }),
    [meta],
  );
  const [selection, setSelection] = useState<Selection>(country);
  const [metric, setMetric] = useState<Metric>("hfai");
  const [lang, setLang] = useState<Lang>(() => (localStorage.getItem("nyeleti.lang") as Lang) || "en");

  useEffect(() => {
    localStorage.setItem("nyeleti.lang", lang);
    document.documentElement.lang = lang;
  }, [lang]);

  const value = useMemo<AppState>(
    () => ({
      meta,
      selection,
      setSelection,
      resetSelection: () => setSelection(country),
      metric,
      setMetric,
      lang,
      setLang,
      t: (key, vars) => translate(lang, key, vars),
    }),
    [meta, selection, metric, lang, country],
  );
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useApp(): AppState {
  const ctx = useContext(Context);
  if (!ctx) throw new Error("useApp must be used inside AppProvider");
  return ctx;
}

export interface Loaded<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/** Fetch an endpoint whenever its parameters change. Pass `null` as path to skip. */
export function useApi<T>(path: string | null, params: Params = {}): Loaded<T> {
  const key = path ? `${path}?${JSON.stringify(params)}` : null;
  const [state, setState] = useState<Loaded<T>>({ data: null, loading: Boolean(path), error: null });

  useEffect(() => {
    if (!path) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let active = true;
    setState((prev) => ({ data: prev.data, loading: true, error: null }));
    get<T>(path, params)
      .then((data) => active && setState({ data, loading: false, error: null }))
      .catch((err: Error) => active && setState({ data: null, loading: false, error: err.message }));
    return () => {
      active = false;
    };
  }, [key]);

  return state;
}
