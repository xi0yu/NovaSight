export function formatNumberDraft(value: number, digits: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

export function resolveNumberDraft(
  draft: string,
  fallback: number,
  min: number,
  max: number,
  digits: number
): number {
  if (draft.trim() === "") {
    return fallback;
  }
  const parsed = Number(draft);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  const rounded = Number(parsed.toFixed(digits));
  return Math.min(max, Math.max(min, rounded));
}
