import { useMemo } from "react";
import type { Num } from "../services/api";
import { useApp } from "../state";
import type { ActivityClass } from "./classes";

const CELL = 18;
const LEFT = 30;
const TOP = 18;

const FILL: Record<ActivityClass, string> = {
  low: "var(--low)",
  medium: "var(--medium)",
  high: "var(--high)",
  very_high: "var(--very-high)",
};

interface Props {
  dates: string[];
  values: Num[];
  classes: (ActivityClass | null)[];
  flagged?: Set<string>;
  selected?: string | null;
  unit: string;
  onPick?: (date: string) => void;
}

/** One year as a grid of days: a column per week, a row per weekday. */
export function YearHeatmap({ dates, values, classes, flagged, selected, unit, onPick }: Props) {
  const { t } = useApp();
  const months = t("months").split(",");
  const weekdays = t("weekdays").split(",");
  const cells = useMemo(() => {
    if (!dates.length) return [];
    const first = new Date(`${dates[0]}T00:00:00Z`);
    const offset = (first.getUTCDay() + 6) % 7;
    return dates.map((date, i) => {
      const slot = i + offset;
      return { date, i, week: Math.floor(slot / 7), weekday: slot % 7, month: Number(date.slice(5, 7)) - 1, day: Number(date.slice(8, 10)) };
    });
  }, [dates]);

  if (!cells.length) return null;
  const weeks = cells[cells.length - 1].week + 1;
  const monthStarts = cells.filter((c) => c.day === 1);

  return (
    <div className="year-heat">
      <svg width={LEFT + weeks * CELL + 4} height={TOP + 7 * CELL + 4} role="img">
        {monthStarts.map((c) => (
          <text key={c.month} x={LEFT + c.week * CELL + (c.weekday === 0 ? 0 : CELL)} y={11}>{months[c.month]}</text>
        ))}
        {weekdays.map((name, row) => name && <text key={row} x={0} y={TOP + row * CELL + 13}>{name}</text>)}
        {cells.map((c) => {
          const cls = classes[c.i];
          const value = values[c.i];
          return (
            <rect
              key={c.date}
              className={`day${flagged?.has(c.date) || selected === c.date ? " flag" : ""}`}
              x={LEFT + c.week * CELL}
              y={TOP + c.weekday * CELL}
              width={CELL - 1}
              height={CELL - 1}
              fill={cls ? FILL[cls] : "#e8edf3"}
              onClick={() => onPick?.(c.date)}
            >
              <title>{`${c.date}: ${value === null ? t("noRecord") : `${value.toLocaleString("en-US", { maximumFractionDigits: 1 })} ${unit}`}`}</title>
            </rect>
          );
        })}
      </svg>
    </div>
  );
}
