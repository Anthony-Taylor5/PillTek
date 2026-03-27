import React, { useState } from "react";
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useLocalSearchParams, useRouter } from "expo-router";

// Same mock data as logs.js — filtered here by patient name so each
// patient's log view only shows their own entries.
const ALL_LOGS = [
  { id: "1", patient: "Ahmad",    medName: "Metformin 500mg",   status: "Taken",  time: "8:04 AM",  date: "Today"     },
  { id: "2", patient: "Shahriar", medName: "Lisinopril 10mg",   status: "Taken",  time: "12:01 PM", date: "Today"     },
  { id: "3", patient: "Mina",     medName: "Atorvastatin 20mg", status: "Missed", time: "8:00 PM",  date: "Today"     },
  { id: "4", patient: "Ahmad",    medName: "Metformin 500mg",   status: "Taken",  time: "8:02 AM",  date: "Yesterday" },
  { id: "5", patient: "Shahriar", medName: "Lisinopril 10mg",   status: "Taken",  time: "12:00 PM", date: "Yesterday" },
  { id: "6", patient: "Mina",     medName: "Atorvastatin 20mg", status: "Taken",  time: "8:05 PM",  date: "Yesterday" },
  { id: "7", patient: "Ahmad",    medName: "Metformin 500mg",   status: "Taken",  time: "8:10 AM",  date: "Mar 22"    },
  { id: "8", patient: "Shahriar", medName: "Lisinopril 10mg",   status: "Missed", time: "12:00 PM", date: "Mar 22"    },
  { id: "9", patient: "Mina",     medName: "Vitamin D 1000IU",  status: "Taken",  time: "12:00 PM", date: "Mar 22"    },
  { id: "10", patient: "Ahmad",   medName: "Aspirin 81mg",      status: "Taken",  time: "8:03 PM",  date: "Mar 22"    },
  { id: "11", patient: "Shahriar",medName: "Atorvastatin 20mg", status: "Taken",  time: "8:00 PM",  date: "Mar 21"    },
  { id: "12", patient: "Mina",    medName: "Metformin 500mg",   status: "Missed", time: "—",        date: "Mar 21"    },
  { id: "13", patient: "Ahmad",   medName: "Metformin 500mg",   status: "Taken",  time: "8:01 AM",  date: "Mar 20"    },
  { id: "14", patient: "Mina",    medName: "Atorvastatin 20mg", status: "Taken",  time: "8:07 PM",  date: "Mar 15"    },
  { id: "15", patient: "Ahmad",   medName: "Aspirin 81mg",      status: "Missed", time: "—",        date: "Mar 10"    },
];

// Dates considered within "this week" for the mock dataset
const WEEK_DATES = new Set(["Today", "Yesterday", "Mar 22", "Mar 21", "Mar 20", "Mar 19", "Mar 18"]);

function filterByPeriod(logs, period) {
  if (period === "Day")  return logs.filter((l) => l.date === "Today");
  if (period === "Week") return logs.filter((l) => WEEK_DATES.has(l.date));
  return logs; // Month = all
}

function statusColor(s) {
  if (s === "Taken")  return "#366a53";
  if (s === "Missed") return "#c0392b";
  return "#888";
}

// Same bottle thumbnail as logs.js / med-log.js
function BottleThumb({ status }) {
  const hasPhoto = status === "Taken";
  return (
    <View style={[styles.thumb, !hasPhoto && styles.thumbEmpty]}>
      <Text style={styles.thumbIcon}>{hasPhoto ? "🧴" : "—"}</Text>
    </View>
  );
}

function LogRow({ item, onPress }) {
  return (
    <TouchableOpacity style={styles.row} onPress={onPress} activeOpacity={0.65}>
      <BottleThumb status={item.status} />

      <View style={styles.rowMid}>
        <Text style={styles.rowMed}>{item.medName}</Text>
        <Text style={[styles.rowStatus, { color: statusColor(item.status) }]}>
          {item.status}
        </Text>
      </View>

      <View style={styles.rowRight}>
        <Text style={styles.rowTime}>{item.time}</Text>
        <Text style={styles.rowDate}>{item.date}</Text>
      </View>
    </TouchableOpacity>
  );
}

export default function PatientLogView() {
  const router = useRouter();
  const { patientName } = useLocalSearchParams();

  // Default selection is Day
  const [period, setPeriod] = useState("Day");

  const patientLogs = ALL_LOGS.filter(
    (l) => l.patient.toLowerCase() === String(patientName ?? "").toLowerCase()
  );
  const filtered = filterByPeriod(patientLogs, period);

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>
        <Text style={styles.sectionTitle}>logs</Text>
        <Text style={styles.subLabel}>{patientName}</Text>
        <View style={styles.divider} />

        {/* Day / Week / Month tab selector */}
        <View style={styles.tabRow}>
          {["Day", "Week", "Month"].map((p) => (
            <TouchableOpacity
              key={p}
              style={[styles.tab, period === p && styles.tabActive]}
              onPress={() => setPeriod(p)}
              activeOpacity={0.7}
            >
              <Text style={[styles.tabText, period === p && styles.tabTextActive]}>
                {p}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {filtered.length === 0 ? (
          <Text style={styles.emptyText}>No log entries for this period.</Text>
        ) : (
          <FlatList
            data={filtered}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <LogRow
                item={item}
                onPress={() =>
                  router.push({
                    pathname: "/log-entry-detail",
                    params: {
                      entryId:     item.id,
                      medName:     item.medName,
                      patientName: item.patient,
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

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 2 },
  subLabel: { fontSize: 14, color: "#555", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },

  // Tab selector row — sits just below the divider
  tabRow: {
    flexDirection: "row",
    marginTop: 14,
    marginBottom: 10,
    gap: 8,
  },
  tab: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: "#fff",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#cde8d0",
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.04,
    shadowRadius: 2,
    shadowOffset: { width: 0, height: 1 },
  },
  tabActive: {
    backgroundColor: "#366a53",
    borderColor: "#366a53",
  },
  tabText: { fontSize: 14, fontWeight: "600", color: "#366a53" },
  tabTextActive: { color: "#fff" },

  emptyText: {
    marginTop: 32,
    fontSize: 15,
    color: "#888",
    textAlign: "center",
  },

  // Log row — same spec as logs.js
  row: {
    flexDirection: "row",
    alignItems: "center",
    minHeight: 64,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#d6ebd9",
    gap: 14,
  },

  thumb: {
    width: 52,
    height: 52,
    borderRadius: 8,
    backgroundColor: "#c8e6c9",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  thumbEmpty: { backgroundColor: "#e0e0e0" },
  thumbIcon: { fontSize: 26 },

  rowMid: { flex: 1 },
  rowMed: { fontSize: 15, fontWeight: "600", color: "#000" },
  rowStatus: { fontSize: 12, fontWeight: "600", marginTop: 3 },

  rowRight: { alignItems: "flex-end", flexShrink: 0 },
  rowTime: { fontSize: 13, color: "#555" },
  rowDate: { fontSize: 12, color: "#888", marginTop: 2 },
});
