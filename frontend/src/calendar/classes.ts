import type { Num } from "../services/api";

export type ActivityClass = "low" | "medium" | "high" | "very_high";
export type Thresholds = Record<"medium" | "high" | "very_high", number | null>;

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const pos = ((sorted.length - 1) * p) / 100;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Same class limits as the API (P50, P80, P95), computed on the values given. */
export function thresholdsOf(values: Num[]): Thresholds {
  const sorted = values.filter((v): v is number => v !== null).sort((a, b) => a - b);
  return { medium: percentile(sorted, 50), high: percentile(sorted, 80), very_high: percentile(sorted, 95) };
}

export function classOf(value: Num, th: Thresholds): ActivityClass | null {
  if (value === null) return null;
  if (value > (th.very_high ?? Infinity)) return "very_high";
  if (value > (th.high ?? Infinity)) return "high";
  if (value > (th.medium ?? Infinity)) return "medium";
  return "low";
}
