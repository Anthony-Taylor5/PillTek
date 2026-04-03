import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useLocalSearchParams, useRouter } from "expo-router";
import { fetchMedLogs } from "../lib/api";

// Mock log history keyed by "patientId_medId"
const MED_LOGS = {
  "1_m1": [
    { id: "1_m1_1", date: "Today",     time: "8:04 AM",  status: "Taken" },
    { id: "1_m1_2", date: "Yesterday", time: "8:02 AM",  status: "Taken" },
    { id: "1_m1_3", date: "Mar 22",    time: "8:10 AM",  status: "Taken" },
    { id: "1_m1_4", date: "Mar 21",    time: "8:05 AM",  status: "Taken" },
    { id: "1_m1_5", date: "Mar 20",    time: "—",        status: "Missed" },
  ],
  "1_m2": [
    { id: "1_m2_1", date: "Today",     time: "—",        status: "Pending" },
    { id: "1_m2_2", date: "Yesterday", time: "8:03 PM",  status: "Taken" },
    { id: "1_m2_3", date: "Mar 22",    time: "8:01 PM",  status: "Taken" },
  ],
  "2_m1": [
    { id: "2_m1_1", date: "Today",     time: "12:01 PM", status: "Taken" },
    { id: "2_m1_2", date: "Yesterday", time: "12:00 PM", status: "Taken" },
    { id: "2_m1_3", date: "Mar 22",    time: "—",        status: "Missed" },
    { id: "2_m1_4", date: "Mar 21",    time: "12:03 PM", status: "Taken" },
  ],
  "2_m2": [
    { id: "2_m2_1", date: "Today",     time: "8:00 PM",  status: "Taken" },
    { id: "2_m2_2", date: "Yesterday", time: "8:05 PM",  status: "Taken" },
    { id: "2_m2_3", date: "Mar 22",    time: "8:02 PM",  status: "Taken" },
  ],
  "3_m1": [
    { id: "3_m1_1", date: "Today",     time: "—",        status: "Missed" },
    { id: "3_m1_2", date: "Yesterday", time: "8:06 AM",  status: "Taken" },
    { id: "3_m1_3", date: "Mar 22",    time: "8:04 AM",  status: "Taken" },
  ],
  "3_m2": [
    { id: "3_m2_1", date: "Today",     time: "12:00 PM", status: "Taken" },
    { id: "3_m2_2", date: "Yesterday", time: "12:02 PM", status: "Taken" },
    { id: "3_m2_3", date: "Mar 22",    time: "12:00 PM", status: "Taken" },
  ],
  "3_m3": [
    { id: "3_m3_1", date: "Today",     time: "—",        status: "Pending" },
    { id: "3_m3_2", date: "Yesterday", time: "8:05 PM",  status: "Taken" },
    { id: "3_m3_3", date: "Mar 22",    time: "8:07 PM",  status: "Taken" },
  ],
};

// Visual-only pill bottle photo placeholder — no network, no camera
function BottleThumb({ status }) {
  const haPhoto = status === "Taken";
  return (
    <View style={[styles.thumb, !haPhoto && styles.thumbEmpty]}>
      <Text style={styles.thumbIcon}>{haPhoto ? "🧴" : "—"}</Text>
    </View>
  );
}

function statusColor(s) {
  if (s === "Taken")   return "#366a53";
  if (s === "Missed")  return "#c0392b";
  return "#888";
}

function LogEntryRow({ entry, onPress }) {
  return (
    <TouchableOpacity style={styles.entryRow} onPress={onPress} activeOpacity={0.65}>
      {/* Left: bottle thumbnail */}
      <BottleThumb status={entry.status} />

      {/* Middle: date + status */}
      <View style={styles.entryMid}>
        <Text style={styles.entryDate}>{entry.date}</Text>
        <Text style={[styles.entryStatus, { color: statusColor(entry.status) }]}>
          {entry.status}
        </Text>
      </View>

      {/* Right: time */}
      <Text style={styles.entryTime}>{entry.time}</Text>
    </TouchableOpacity>
  );
}

// Format a DB log row into the shape the UI components expect.
function dbRowToEntry(row) {
  const dateLabel = formatDateLabel(row.log_date);
  const timeLabel = row.taken_at
    ? new Date(row.taken_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '—';
  return {
    id:     row.id,
    date:   dateLabel,
    time:   timeLabel,
    status: row.status ?? 'Pending',
  };
}

function formatDateLabel(isoDate) {
  if (!isoDate) return '—';
  const d     = new Date(isoDate + 'T00:00:00');
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diff  = Math.round((today - d) / 86400000);
  if (diff === 0) return 'Today';
  if (diff === 1) return 'Yesterday';
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function MedLog() {
  const router = useRouter();
  const { patientId, patientName, medId, medName, medTime } = useLocalSearchParams();

  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // medId is a UUID when coming from a Supabase-backed patient-detail screen.
    const isUuid = String(medId ?? '').includes('-');
    if (!isUuid) {
      // Legacy integer ID — use mock data.
      const key = `${patientId}_${medId}`;
      setEntries(MED_LOGS[key] ?? []);
      return;
    }

    setLoading(true);
    fetchMedLogs(medId)
      .then((rows) => {
        setEntries(rows.length > 0 ? rows.map(dbRowToEntry) : []);
      })
      .catch((err) => {
        console.warn('[med-log] fetchMedLogs failed:', err);
        setEntries([]);
      })
      .finally(() => setLoading(false));
  }, [medId, patientId]);

  const takenCount  = entries.filter((e) => e.status === "Taken").length;
  const missedCount = entries.filter((e) => e.status === "Missed").length;

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>

        {/* Medication + patient identity */}
        <Text style={styles.medName}>{medName}</Text>
        <Text style={styles.subLabel}>{patientName} · {medTime}</Text>

        {/* Summary strip */}
        <View style={styles.summaryRow}>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNum}>{takenCount}</Text>
            <Text style={styles.summaryLabel}>Taken</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={[styles.summaryNum, missedCount > 0 && styles.colorMissed]}>
              {missedCount}
            </Text>
            <Text style={styles.summaryLabel}>Missed</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNum}>{entries.length}</Text>
            <Text style={styles.summaryLabel}>Total</Text>
          </View>
        </View>

        {/* Log header */}
        <Text style={styles.sectionTitle}>history</Text>
        <View style={styles.divider} />

        {loading && <ActivityIndicator color="#366a53" style={{ marginTop: 20 }} />}
        {!loading && entries.length === 0 ? (
          <Text style={styles.emptyText}>No log entries yet.</Text>
        ) : (
          <FlatList
            data={entries}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <LogEntryRow
                entry={item}
                onPress={() =>
                  router.push({
                    pathname: "/log-entry-detail",
                    params: {
                      entryId:     item.id,
                      medName,
                      patientName,
                      date:        item.date,
                      time:        item.time,
                      status:      item.status,
                    },
                  })
                }
              />
            )}
          />
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: { flex: 1, paddingHorizontal: 20, paddingTop: 20 },

  medName: { fontSize: 22, fontWeight: "700", color: "#000" },
  subLabel: { fontSize: 14, color: "#555", marginTop: 4, marginBottom: 20 },

  summaryRow: { flexDirection: "row", gap: 12, marginBottom: 24 },
  summaryCard: {
    flex: 1,
    backgroundColor: "#fff",
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: "center",
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  summaryNum: { fontSize: 26, fontWeight: "700", color: "#366a53" },
  summaryLabel: { fontSize: 12, color: "#555", marginTop: 2 },
  colorMissed: { color: "#c0392b" },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },
  emptyText: { marginTop: 24, fontSize: 15, color: "#888", textAlign: "center" },

  // Entry row
  entryRow: {
    flexDirection: "row",
    alignItems: "center",
    minHeight: 64,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#d6ebd9",
    gap: 14,
  },

  // Bottle thumbnail placeholder
  thumb: {
    width: 52,
    height: 52,
    borderRadius: 8,
    backgroundColor: "#c8e6c9",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  thumbEmpty: {
    backgroundColor: "#e0e0e0",
  },
  thumbIcon: { fontSize: 26 },

  entryMid: { flex: 1 },
  entryDate: { fontSize: 16, fontWeight: "600", color: "#000" },
  entryStatus: { fontSize: 13, marginTop: 3 },

  entryTime: { fontSize: 13, color: "#555", flexShrink: 0 },
});
