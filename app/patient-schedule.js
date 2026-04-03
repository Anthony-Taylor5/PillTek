import React, { useState, useCallback, useMemo } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useFocusEffect, useLocalSearchParams } from "expo-router";
import { getPatientMedications, getCaregiverPatientMeds } from "../lib/medication-store";
import {
  buildCalendarGrid,
  isScheduledOn,
  isRefillDueOn,
  isRefillWarnOn,
  daysUntilRefill,
  MONTH_NAMES,
  DAY_HEADERS,
} from "../lib/schedule-utils";

// ── Mock medications ──────────────────────────────────────────────────────────
// Used when no real store data is available (patient hasn't done bottle setup yet).
// All fields match the shape produced by medication-details-entry + capture-bottles.
const MOCK_MEDS = [
  {
    id: "1", name: "Metformin 500mg",
    frequency: "Daily", times: ["8:00 AM"], time: "8:00 AM",
    refill: "Monthly", addedAt: "2026-03-01T00:00:00.000Z", status: "Pending",
  },
  {
    id: "2", name: "Lisinopril 10mg",
    frequency: "3 times a week", times: ["12:00 PM"], time: "12:00 PM",
    refill: "Monthly", addedAt: "2026-03-01T00:00:00.000Z", status: "Pending",
  },
  {
    id: "3", name: "Atorvastatin 20mg",
    frequency: "Daily", times: ["8:00 PM"], time: "8:00 PM",
    refill: "Every 60 days", addedAt: "2026-03-01T00:00:00.000Z", status: "Pending",
  },
];

// ── Constants ─────────────────────────────────────────────────────────────────
const _now  = new Date();
const TODAY = { year: _now.getFullYear(), month: _now.getMonth(), day: _now.getDate() };

// ── Component ─────────────────────────────────────────────────────────────────
export default function PatientSchedule() {
  // selfName is passed when self-managed users reach this screen from self-home.
  // When absent the screen reads from the patient's own medication store.
  const { selfName } = useLocalSearchParams();

  const [year,         setYear]         = useState(TODAY.year);
  const [month,        setMonth]        = useState(TODAY.month);
  const [selectedDay,  setSelectedDay]  = useState(TODAY.day);
  const [medications,  setMedications]  = useState(MOCK_MEDS);

  // Refresh medication list whenever the screen gains focus.
  useFocusEffect(
    useCallback(() => {
      let meds;
      if (selfName) {
        meds = getCaregiverPatientMeds(String(selfName)) ?? [];
      } else {
        meds = getPatientMedications();
      }
      setMedications(meds.length > 0 ? meds : MOCK_MEDS);
    }, [selfName])
  );

  // ── Navigation ──────────────────────────────────────────────────────────────
  const prevMonth = () => {
    setSelectedDay(null);
    if (month === 0) { setYear((y) => y - 1); setMonth(11); }
    else setMonth((m) => m - 1);
  };
  const nextMonth = () => {
    setSelectedDay(null);
    if (month === 11) { setYear((y) => y + 1); setMonth(0); }
    else setMonth((m) => m + 1);
  };

  // ── Calendar grid ────────────────────────────────────────────────────────────
  const weeks = useMemo(() => buildCalendarGrid(year, month), [year, month]);

  // ── Per-day helpers ──────────────────────────────────────────────────────────
  const medsOnDay = useCallback(
    (day) => day ? medications.filter((m) => isScheduledOn(m, year, month, day)) : [],
    [medications, year, month]
  );

  const hasMedDot = (day) => day && medsOnDay(day).length > 0;
  const hasRefillDot = (day) =>
    day &&
    medications.some(
      (m) => isRefillDueOn(m, year, month, day) || isRefillWarnOn(m, year, month, day)
    );

  // ── Selected-day detail ──────────────────────────────────────────────────────
  const selectedMeds = useMemo(() => medsOnDay(selectedDay), [selectedDay, medsOnDay]);

  const isToday = (day) =>
    day === TODAY.day && month === TODAY.month && year === TODAY.year;

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>

        <Text style={styles.sectionTitle}>my schedule</Text>
        <View style={styles.divider} />

        {/* Month navigation */}
        <View style={styles.monthNav}>
          <TouchableOpacity onPress={prevMonth} hitSlop={12} style={styles.navBtn}>
            <Text style={styles.navArrow}>‹</Text>
          </TouchableOpacity>
          <Text style={styles.monthLabel}>
            {MONTH_NAMES[month]} {year}
          </Text>
          <TouchableOpacity onPress={nextMonth} hitSlop={12} style={styles.navBtn}>
            <Text style={styles.navArrow}>›</Text>
          </TouchableOpacity>
        </View>

        {/* Calendar grid card */}
        <View style={styles.calCard}>
          {/* Day-of-week headers */}
          <View style={styles.weekRow}>
            {DAY_HEADERS.map((h, i) => (
              <View key={i} style={styles.dayCell}>
                <Text style={styles.dowHeader}>{h}</Text>
              </View>
            ))}
          </View>

          {/* Date rows */}
          {weeks.map((week, wi) => (
            <View key={wi} style={styles.weekRow}>
              {week.map((day, di) => {
                // Null cells are empty padding — render as a transparent spacer
                // so they never appear as coloured boxes.
                if (!day) {
                  return <View key={di} style={styles.dayCellEmpty} />;
                }

                const selected = day === selectedDay;
                const today    = isToday(day);
                const hasMed   = hasMedDot(day);
                const hasRef   = hasRefillDot(day);
                return (
                  <TouchableOpacity
                    key={di}
                    style={[
                      styles.dayCell,
                      selected && styles.dayCellSelected,
                      today && !selected && styles.dayCellToday,
                    ]}
                    onPress={() => setSelectedDay(day)}
                    activeOpacity={0.65}
                  >
                    <Text
                      style={[
                        styles.dayNum,
                        selected && styles.dayNumSelected,
                        today && !selected && styles.dayNumToday,
                      ]}
                    >
                      {day}
                    </Text>
                    {/* Indicator dots — hidden on selected cell (detail shown below) */}
                    {!selected && (
                      <View style={styles.dotRow}>
                        {hasMed && <View style={styles.medDot} />}
                        {hasRef && <View style={styles.refillDot} />}
                      </View>
                    )}
                  </TouchableOpacity>
                );
              })}
            </View>
          ))}
        </View>

        {/* Dot legend */}
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.medDot, styles.legendDot]} />
            <Text style={styles.legendText}>Medication scheduled</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.refillDot, styles.legendDot]} />
            <Text style={styles.legendText}>Refill due soon</Text>
          </View>
        </View>

        {/* Selected-day detail card */}
        <View style={styles.detailCard}>
          <Text style={styles.detailTitle}>
            {selectedDay
              ? `${MONTH_NAMES[month]} ${selectedDay}, ${year}`
              : "Tap a date to see medications"}
          </Text>
          <View style={styles.detailDivider} />

          {!selectedDay ? (
            <Text style={styles.emptyText}>Select a day above.</Text>
          ) : selectedMeds.length === 0 ? (
            <Text style={styles.emptyText}>No medications scheduled.</Text>
          ) : (
            selectedMeds.map((med, i) => {
              const refillDays = daysUntilRefill(med, year, month, selectedDay);
              const warnRefill = refillDays !== null && refillDays <= 3;
              const timeStr = Array.isArray(med.times) && med.times.length > 0
                ? med.times.join("  ·  ")
                : (med.time || "—");
              return (
                <View
                  key={i}
                  style={[styles.medEntry, i > 0 && styles.medEntryBorder]}
                >
                  <View style={{ flex: 1 }}>
                    <Text style={styles.medName}>{med.name}</Text>
                    <Text style={styles.medTime}>{timeStr}</Text>
                    {warnRefill && (
                      <Text style={styles.refillWarn}>
                        {refillDays === 0
                          ? "⚠ Refill due today"
                          : `⚠ Refill due in ${refillDays} day${refillDays === 1 ? "" : "s"}`}
                      </Text>
                    )}
                  </View>
                </View>
              );
            })
          )}
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  safe:      { flex: 1, backgroundColor: "#e8f5e9" },
  container: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 32 },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider:      { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },

  // Month navigation row
  monthNav: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 16,
    marginBottom: 10,
  },
  navBtn:    { padding: 6 },
  navArrow:  { fontSize: 26, color: "#366a53", fontWeight: "600" },
  monthLabel: { fontSize: 17, fontWeight: "600", color: "#111" },

  // Calendar grid card — same visual spec as slotCard in previous version
  calCard: {
    backgroundColor: "#fff",
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 8,
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },

  weekRow: { flexDirection: "row" },

  dayCell: {
    flex: 1,
    height: 48,
    alignItems: "center",
    justifyContent: "flex-start",
    paddingTop: 5,
    borderRadius: 8,
    marginVertical: 1,
    marginHorizontal: 1,
  },
  dayCellEmpty:    { flex: 1, height: 48, marginVertical: 1, marginHorizontal: 1 },
  dayCellSelected: { backgroundColor: "#366a53" },
  dayCellToday: {
    borderWidth: 1.5,
    borderColor: "#366a53",
  },

  dowHeader:     { fontSize: 11, fontWeight: "600", color: "#888" },
  dayNum:        { fontSize: 14, color: "#222" },
  dayNumSelected:{ color: "#fff", fontWeight: "700" },
  dayNumToday:   { color: "#366a53", fontWeight: "700" },
  dayNumEmpty:   { opacity: 0 },

  dotRow: {
    flexDirection: "row",
    gap: 2,
    marginTop: 2,
    height: 6,
    alignItems: "center",
  },
  medDot: {
    width: 5,
    height: 5,
    borderRadius: 3,
    backgroundColor: "#366a53",
  },
  refillDot: {
    width: 5,
    height: 5,
    borderRadius: 3,
    backgroundColor: "#e8871a",
  },

  // Legend
  legend: {
    flexDirection: "row",
    gap: 20,
    marginTop: 10,
    marginBottom: 16,
    justifyContent: "center",
  },
  legendItem: { flexDirection: "row", alignItems: "center", gap: 6 },
  legendDot:  { width: 8, height: 8, borderRadius: 4 },
  legendText: { fontSize: 12, color: "#555" },

  // Selected-day detail card
  detailCard: {
    backgroundColor: "#fff",
    borderRadius: 10,
    padding: 16,
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  detailTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#366a53",
    marginBottom: 10,
  },
  detailDivider: {
    borderBottomWidth: 1,
    borderBottomColor: "#d6ebd9",
    marginBottom: 8,
  },
  emptyText: { fontSize: 14, color: "#888", paddingVertical: 8 },

  medEntry: { paddingVertical: 10 },
  medEntryBorder: { borderTopWidth: 1, borderTopColor: "#d6ebd9" },
  medName: { fontSize: 15, fontWeight: "600", color: "#000" },
  medTime: { fontSize: 13, color: "#555", marginTop: 2 },
  refillWarn: { fontSize: 12, color: "#e8871a", fontWeight: "600", marginTop: 4 },
});
