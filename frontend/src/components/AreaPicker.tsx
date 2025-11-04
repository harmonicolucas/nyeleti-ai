import { useMemo } from "react";
import type { Area } from "../services/api";
import { useApi, useApp } from "../state";
import { Field } from "./ui";

/** Country > province > district selector. A drawn area shows up as a chip. */
export function AreaPicker() {
  const { selection, setSelection, resetSelection, meta, t } = useApp();
  const { data } = useApi<{ areas: Area[] }>("/api/areas");
  const areas = data?.areas ?? [];

  const provinces = useMemo(() => areas.filter((a) => a.level === "ADM1"), [areas]);
  const current = selection.kind === "area" ? areas.find((a) => a.area_id === selection.areaId) : undefined;
  const provinceId = current?.level === "ADM1" ? current.area_id : current?.level === "ADM2" ? current.parent_id ?? "" : "";
  const districts = useMemo(
    () => areas.filter((a) => a.level === "ADM2" && a.parent_id === provinceId && a.n_cells > 0),
    [areas, provinceId],
  );
  const districtId = current?.level === "ADM2" ? current.area_id : "";

  const pick = (id: string) => {
    const area = areas.find((a) => a.area_id === id);
    if (area) setSelection({ kind: "area", areaId: area.area_id, name: area.name, level: area.level });
  };

  return (
    <>
      <Field label={t("province")}>
        <select
          value={provinceId}
          onChange={(e) => (e.target.value ? pick(e.target.value) : resetSelection())}
          aria-label={t("province")}
        >
          <option value="">{t("allCountry", { country: meta.country })}</option>
          {provinces.map((p) => (
            <option key={p.area_id} value={p.area_id}>{p.name}</option>
          ))}
        </select>
      </Field>
      <Field label={t("district")}>
        <select
          value={districtId}
          disabled={!provinceId}
          onChange={(e) => pick(e.target.value || provinceId)}
          aria-label={t("district")}
        >
          <option value="">{provinceId ? t("allDistricts") : t("chooseProvince")}</option>
          {districts.map((d) => (
            <option key={d.area_id} value={d.area_id}>{d.name}</option>
          ))}
        </select>
      </Field>
      {selection.kind === "aoi" && (
        <span className="chip">
          {t("customArea")}
          <button onClick={resetSelection} aria-label={t("clearArea")} title={t("clearArea")}>✕</button>
        </span>
      )}
    </>
  );
}
