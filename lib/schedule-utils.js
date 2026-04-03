// schedule-utils.js
// Pure frontend calendar/schedule logic shared by patient-schedule.js and
// caregiver-calendar.js.  No React or RN imports — plain JS only.

// ── Refill options ────────────────────────────────────────────────────────────

export const REFILL_OPTIONS = [
  "Weekly",
  "Biweekly",
  "Monthly",
  "Every 60 days",
  "Every 90 days",
];

// Maps lowercased option string → number of days between refills.
export const REFILL_INTERVAL_DAYS = {
  "weekly":        7,
  "biweekly":     14,
  "monthly":      30,
  "every 60 days": 60,
  "every 90 days": 90,
};

// ── Frequency → weekly day patterns ──────────────────────────────────────────
// For weekly-class frequencies, lists which offsets (mod 7) from the start
// day are scheduled dose days.  null means every day (all daily variants).

const WEEKLY_PATTERNS = {
  "once a week":    [0],
  "twice a week":   [0, 3],
  "3 times a week": [0, 2, 4],
  "4 times a week": [0, 1, 3, 4],
};

// ── Helpers ───────────────────────────────────────────────────────────────────

// Returns the Date when a medication schedule began.
// Falls back to 2026-03-01 for mock data that pre-dates addedAt tracking.
export function getMedStartDate(med) {
  if (med && med.addedAt) {
    const d = new Date(med.addedAt);
    if (!isNaN(d)) return d;
  }
  return new Date(2026, 2, 1); // March 1 2026 — safe mock origin
}

// Returns true if `med` is scheduled on the date (year, month 0-indexed, day).
export function isScheduledOn(med, year, month, day) {
  const start = getMedStartDate(med);
  start.setHours(0, 0, 0, 0);
  const check = new Date(year, month, day);
  if (check < start) return false;

  const daysSince = Math.round((check - start) / 86400000);
  const key = String(med.frequency ?? "daily").toLowerCase().trim();

  // All "X times daily" and plain "daily" → scheduled every day
  if (key === "daily" || key.endsWith("times daily") || key === "twice daily") {
    return true;
  }

  const pattern = WEEKLY_PATTERNS[key];
  if (!pattern) return false;
  return pattern.includes(daysSince % 7);
}

// Returns the number of days until the next refill from the given date.
// Returns null when no refill is configured or the date is before start.
export function daysUntilRefill(med, year, month, day) {
  const interval = REFILL_INTERVAL_DAYS[String(med.refill ?? "").toLowerCase().trim()];
  if (!interval) return null;

  const start = getMedStartDate(med);
  start.setHours(0, 0, 0, 0);
  const check = new Date(year, month, day);

  const daysSince = Math.round((check - start) / 86400000);
  if (daysSince <= 0) return null;

  const mod = daysSince % interval;
  return mod === 0 ? 0 : interval - mod;
}

// Returns true on the exact refill-due date (daysUntil === 0).
export function isRefillDueOn(med, year, month, day) {
  return daysUntilRefill(med, year, month, day) === 0;
}

// Returns true on the 1–3 days immediately before a refill-due date.
export function isRefillWarnOn(med, year, month, day) {
  const d = daysUntilRefill(med, year, month, day);
  return d !== null && d >= 1 && d <= 3;
}

// ── Calendar grid builder ─────────────────────────────────────────────────────

// Builds a monthly grid as an array of week arrays (each with 7 items).
// Items are day numbers (1–daysInMonth) or null for padding cells.
export function buildCalendarGrid(year, month) {
  const firstDow  = new Date(year, month, 1).getDay(); // 0 = Sunday
  const daysCount = new Date(year, month + 1, 0).getDate();

  const cells = [];
  for (let i = 0; i < firstDow; i++) cells.push(null);
  for (let d = 1; d <= daysCount; d++) cells.push(d);
  while (cells.length % 7 !== 0) cells.push(null);

  const weeks = [];
  for (let i = 0; i < cells.length; i += 7) weeks.push(cells.slice(i, i + 7));
  return weeks;
}

// ── Display constants ─────────────────────────────────────────────────────────

export const MONTH_NAMES = [
  "January", "February", "March",    "April",   "May",      "June",
  "July",    "August",   "September","October",  "November", "December",
];

export const DAY_HEADERS = ["S", "M", "T", "W", "T", "F", "S"];
