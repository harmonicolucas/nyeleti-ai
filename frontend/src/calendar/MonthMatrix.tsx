import type { Num } from "../services/api";
import type { ActivityClass } from "./classes";

interface Props {
  rows: { label: string; key: string | number; note?: string }[];
  columns: string[];
  values: Num[][];
  classes: (ActivityClass | null)[][];
  onPickRow?: (key: string | number) => void;
}

function short(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `${Math.round(value / 1e3)}k`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return abs >= 10 ? value.toFixed(0) : value.toFixed(1);
}

/** Periods (rows) by sub-periods (columns), coloured by activity class. */
export function MonthMatrix({ rows, columns, values, classes, onPickRow }: Props) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="matrix">
        <thead>
          <tr>
            <th />
            {columns.map((c) => <th key={c}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={row.key}>
              <th
                className="year"
                style={{ cursor: onPickRow ? "pointer" : undefined }}
                title={row.note}
                onClick={() => onPickRow?.(row.key)}
              >
                {row.label}{row.note ? " *" : ""}
              </th>
              {columns.map((c, i) => {
                const value = values[r]?.[i] ?? null;
                const cls = classes[r]?.[i];
                return (
                  <td key={c} className={value === null || !cls ? "empty" : `cls-${cls}`}>
                    {value === null ? "·" : short(value)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
