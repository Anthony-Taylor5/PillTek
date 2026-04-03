import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ScrollView,
  Modal,
  Pressable,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { setLastCompletedMed } from "../lib/med-detail-store";
import { REFILL_OPTIONS } from "../lib/schedule-utils";

// ── Frequency options ────────────────────────────────────────────────────────

const FREQUENCY_OPTIONS = [
  "Once a week",
  "Twice a week",
  "3 times a week",
  "4 times a week",
  "Daily",
  "Twice daily",
  "3 times daily",
  "4 times daily",
];

// Returns how many time slots this frequency requires.
function getDoseCount(freq) {
  if (freq === "Twice daily")    return 2;
  if (freq === "3 times daily")  return 3;
  if (freq === "4 times daily")  return 4;
  return 1; // all weekly options and "Daily" get one time slot
}

// ── Time helpers ─────────────────────────────────────────────────────────────

function parseTimeString(str) {
  if (!str) return { hour: 8, minute: 0, period: "AM" };
  const m = str.match(/^(\d+):(\d+)\s*(AM|PM)$/i);
  if (!m) return { hour: 8, minute: 0, period: "AM" };
  return { hour: parseInt(m[1], 10), minute: parseInt(m[2], 10), period: m[3].toUpperCase() };
}

function formatTimeString(hour, minute, period) {
  return `${hour}:${String(minute).padStart(2, "0")} ${period}`;
}

const HOURS   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const MINUTES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55];

// ── Component ────────────────────────────────────────────────────────────────

export default function MedicationDetailsEntry() {
  const router = useRouter();
  const { medName } = useLocalSearchParams();

  const [dosage,    setDosage]    = useState("");
  const [frequency, setFrequency] = useState("Daily");
  const [refill,    setRefill]    = useState("");

  // Refill picker
  const [refillPickerVisible, setRefillPickerVisible] = useState(false);

  // One string slot per required dose ("" = not yet set)
  const [scheduledTimes, setScheduledTimes] = useState([""]);

  // Frequency picker
  const [freqPickerVisible, setFreqPickerVisible] = useState(false);

  // Time picker
  const [timePickerVisible, setTimePickerVisible] = useState(false);
  const [activeSlotIndex,   setActiveSlotIndex]   = useState(0);
  const [pickerHour,        setPickerHour]         = useState(8);
  const [pickerMinute,      setPickerMinute]       = useState(0);
  const [pickerPeriod,      setPickerPeriod]       = useState("AM");

  // ── Handlers ───────────────────────────────────────────────────────────────

  const handleFrequencySelect = (newFreq) => {
    setFrequency(newFreq);
    const count = getDoseCount(newFreq);
    setScheduledTimes((prev) => {
      const next = [...prev];
      while (next.length < count) next.push("");
      return next.slice(0, count);
    });
    setFreqPickerVisible(false);
  };

  const openTimePicker = (index) => {
    const parsed = parseTimeString(scheduledTimes[index]);
    setPickerHour(parsed.hour);
    setPickerMinute(parsed.minute);
    setPickerPeriod(parsed.period);
    setActiveSlotIndex(index);
    setTimePickerVisible(true);
  };

  const confirmTime = () => {
    const formatted = formatTimeString(pickerHour, pickerMinute, pickerPeriod);
    setScheduledTimes((prev) => {
      const next = [...prev];
      next[activeSlotIndex] = formatted;
      return next;
    });
    setTimePickerVisible(false);
  };

  const handleSave = () => {
    if (!dosage.trim()) {
      Alert.alert("Missing info", "Please enter the dosage.");
      return;
    }
    if (scheduledTimes.some((t) => !t)) {
      Alert.alert(
        "Missing info",
        scheduledTimes.length === 1
          ? "Please set the scheduled time."
          : `Please set all ${scheduledTimes.length} scheduled times.`
      );
      return;
    }

    setLastCompletedMed({
      name:      String(medName),
      dosage:    dosage.trim(),
      frequency,
      times:     scheduledTimes,           // array used by calendar
      time:      scheduledTimes.join(", "), // backward-compatible display string
      refill:    refill || "—",
      addedAt:   new Date().toISOString(), // used by calendar to anchor schedule
      status:    "Pending",
    });

    router.back();
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>

        <Text style={styles.sectionTitle}>{medName}</Text>
        <View style={styles.divider} />
        <Text style={styles.hint}>Enter the details for this medication.</Text>

        {/* Dosage */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Dosage</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. 500mg"
            value={dosage}
            onChangeText={setDosage}
          />
        </View>

        {/* Frequency — dropdown */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>How often</Text>
          <TouchableOpacity
            style={styles.pickerBtn}
            onPress={() => setFreqPickerVisible(true)}
            activeOpacity={0.7}
          >
            <Text style={styles.pickerBtnText}>{frequency}</Text>
            <Text style={styles.pickerChevron}>▾</Text>
          </TouchableOpacity>
        </View>

        {/* Scheduled times — one button per dose */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>
            {scheduledTimes.length === 1 ? "Scheduled time" : "Scheduled times"}
          </Text>
          {scheduledTimes.map((t, i) => (
            <TouchableOpacity
              key={i}
              style={[styles.timeBtn, i > 0 && { marginTop: 8 }]}
              onPress={() => openTimePicker(i)}
              activeOpacity={0.7}
            >
              <Text style={styles.timeBtnLabel}>
                {scheduledTimes.length > 1 ? `Time ${i + 1}` : "Time"}
              </Text>
              <Text style={[styles.timeBtnValue, !t && styles.timeBtnPlaceholder]}>
                {t || "Tap to set"}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Refill frequency — dropdown picker (same pattern as frequency) */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Refill frequency</Text>
          <TouchableOpacity
            style={styles.pickerBtn}
            onPress={() => setRefillPickerVisible(true)}
            activeOpacity={0.7}
          >
            <Text style={[styles.pickerBtnText, !refill && styles.pickerBtnPlaceholder]}>
              {refill || "Select refill frequency"}
            </Text>
            <Text style={styles.pickerChevron}>▾</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.submitBtn} onPress={handleSave}>
          <Text style={styles.submitText}>Save Medication</Text>
        </TouchableOpacity>
      </ScrollView>

      {/* ── Frequency picker modal ─────────────────────────────────────────── */}
      <Modal
        visible={freqPickerVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setFreqPickerVisible(false)}
      >
        <Pressable style={styles.modalBackdrop} onPress={() => setFreqPickerVisible(false)}>
          <Pressable style={styles.pickerCard} onPress={() => {}}>
            <Text style={styles.pickerCardTitle}>How often</Text>
            {FREQUENCY_OPTIONS.map((opt) => (
              <TouchableOpacity
                key={opt}
                style={[
                  styles.pickerOption,
                  opt === frequency && styles.pickerOptionSelected,
                ]}
                onPress={() => handleFrequencySelect(opt)}
              >
                <Text
                  style={[
                    styles.pickerOptionText,
                    opt === frequency && styles.pickerOptionTextSelected,
                  ]}
                >
                  {opt}
                </Text>
                {opt === frequency && (
                  <Text style={styles.pickerCheckmark}>✓</Text>
                )}
              </TouchableOpacity>
            ))}
          </Pressable>
        </Pressable>
      </Modal>

      {/* ── Refill picker modal ────────────────────────────────────────────── */}
      <Modal
        visible={refillPickerVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setRefillPickerVisible(false)}
      >
        <Pressable style={styles.modalBackdrop} onPress={() => setRefillPickerVisible(false)}>
          <Pressable style={styles.pickerCard} onPress={() => {}}>
            <Text style={styles.pickerCardTitle}>Refill frequency</Text>
            {REFILL_OPTIONS.map((opt) => (
              <TouchableOpacity
                key={opt}
                style={[
                  styles.pickerOption,
                  opt === refill && styles.pickerOptionSelected,
                ]}
                onPress={() => { setRefill(opt); setRefillPickerVisible(false); }}
              >
                <Text
                  style={[
                    styles.pickerOptionText,
                    opt === refill && styles.pickerOptionTextSelected,
                  ]}
                >
                  {opt}
                </Text>
                {opt === refill && (
                  <Text style={styles.pickerCheckmark}>✓</Text>
                )}
              </TouchableOpacity>
            ))}
          </Pressable>
        </Pressable>
      </Modal>

      {/* ── Time picker modal ──────────────────────────────────────────────── */}
      <Modal
        visible={timePickerVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setTimePickerVisible(false)}
      >
        <Pressable style={styles.modalBackdrop} onPress={() => setTimePickerVisible(false)}>
          <Pressable style={styles.timePickerCard} onPress={() => {}}>
            <Text style={styles.pickerCardTitle}>
              {scheduledTimes.length > 1 ? `Time ${activeSlotIndex + 1}` : "Scheduled time"}
            </Text>

            {/* Hour row */}
            <Text style={styles.timeSection}>Hour</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.hourRow}
            >
              {HOURS.map((h) => (
                <TouchableOpacity
                  key={h}
                  style={[styles.unitBtn, pickerHour === h && styles.unitBtnSelected]}
                  onPress={() => setPickerHour(h)}
                >
                  <Text style={[styles.unitBtnText, pickerHour === h && styles.unitBtnTextSelected]}>
                    {h}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            {/* Minute row */}
            <Text style={styles.timeSection}>Minute</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.hourRow}
            >
              {MINUTES.map((m) => (
                <TouchableOpacity
                  key={m}
                  style={[styles.unitBtn, pickerMinute === m && styles.unitBtnSelected]}
                  onPress={() => setPickerMinute(m)}
                >
                  <Text style={[styles.unitBtnText, pickerMinute === m && styles.unitBtnTextSelected]}>
                    :{String(m).padStart(2, "0")}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            {/* AM/PM row */}
            <Text style={styles.timeSection}>Period</Text>
            <View style={styles.periodRow}>
              {["AM", "PM"].map((p) => (
                <TouchableOpacity
                  key={p}
                  style={[styles.unitBtn, styles.periodBtn, pickerPeriod === p && styles.unitBtnSelected]}
                  onPress={() => setPickerPeriod(p)}
                >
                  <Text style={[styles.unitBtnText, pickerPeriod === p && styles.unitBtnTextSelected]}>
                    {p}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Preview */}
            <Text style={styles.timePreview}>
              {formatTimeString(pickerHour, pickerMinute, pickerPeriod)}
            </Text>

            <TouchableOpacity style={styles.doneBtn} onPress={confirmTime}>
              <Text style={styles.doneBtnText}>Done</Text>
            </TouchableOpacity>
          </Pressable>
        </Pressable>
      </Modal>
    </SafeAreaView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: { paddingHorizontal: 20, paddingTop: 24, paddingBottom: 40 },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 16 },
  hint: { fontSize: 14, color: "#555", marginBottom: 20, lineHeight: 20 },

  fieldGroup: { marginBottom: 20 },
  label: { fontSize: 14, color: "#555", marginBottom: 8 },

  input: {
    backgroundColor: "#fff",
    height: 48,
    borderRadius: 8,
    paddingHorizontal: 14,
    fontSize: 16,
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },

  // Frequency picker button — same size as a text input
  pickerBtn: {
    backgroundColor: "#fff",
    height: 48,
    borderRadius: 8,
    paddingHorizontal: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  pickerBtnText: { fontSize: 16, color: "#000" },
  pickerBtnPlaceholder: { color: "#aaa" },
  pickerChevron: { fontSize: 14, color: "#555" },

  // Scheduled time button
  timeBtn: {
    backgroundColor: "#fff",
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  timeBtnLabel: { fontSize: 14, color: "#555" },
  timeBtnValue: { fontSize: 16, color: "#000", fontWeight: "500" },
  timeBtnPlaceholder: { color: "#aaa", fontWeight: "400" },

  submitBtn: {
    marginTop: 12,
    backgroundColor: "#366a53",
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: "center",
    elevation: 2,
  },
  submitText: { color: "#fff", fontSize: 17, fontWeight: "600" },

  // ── Modal shared ─────────────────────────────────────────────────────────
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.35)",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  pickerCardTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#111",
    marginBottom: 12,
    textAlign: "center",
  },

  // ── Frequency picker ─────────────────────────────────────────────────────
  pickerCard: {
    width: "100%",
    backgroundColor: "#fff",
    borderRadius: 12,
    paddingVertical: 8,
    paddingHorizontal: 4,
    elevation: 6,
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 3 },
  },
  pickerOption: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 13,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  pickerOptionSelected: {
    backgroundColor: "#f0f7f2",
  },
  pickerOptionText: { fontSize: 16, color: "#111" },
  pickerOptionTextSelected: { color: "#366a53", fontWeight: "600" },
  pickerCheckmark: { fontSize: 16, color: "#366a53", fontWeight: "700" },

  // ── Time picker ──────────────────────────────────────────────────────────
  timePickerCard: {
    width: "100%",
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 20,
    elevation: 6,
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 3 },
  },
  timeSection: {
    fontSize: 12,
    color: "#555",
    fontWeight: "600",
    marginBottom: 8,
    marginTop: 4,
  },

  hourRow: { paddingBottom: 4 },

  minuteRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 4,
  },
  periodRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 4,
  },

  unitBtn: {
    width: 42,
    height: 42,
    borderRadius: 8,
    backgroundColor: "#f5f5f5",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 6,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  unitBtnSelected: {
    backgroundColor: "#366a53",
    borderColor: "#366a53",
  },
  unitBtnText: { fontSize: 15, color: "#333", fontWeight: "500" },
  unitBtnTextSelected: { color: "#fff", fontWeight: "700" },

  minuteBtn: { flex: 1, width: undefined, marginRight: 0 },
  periodBtn: { flex: 1, width: undefined, marginRight: 0 },

  timePreview: {
    fontSize: 22,
    fontWeight: "700",
    color: "#366a53",
    textAlign: "center",
    marginTop: 12,
    marginBottom: 16,
  },

  doneBtn: {
    backgroundColor: "#366a53",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  doneBtnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
});
