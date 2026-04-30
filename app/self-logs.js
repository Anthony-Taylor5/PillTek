import React, { useState } from "react";
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";

// Mock log entries for an individual (self-managed) user — scoped entirely
// to their own medications.  No patient field; always "Me" when passed to
// log-entry-detail.  Replace with a Firestore query keyed by auth.currentUser.uid
// when the backend is wired.
const SELF_LOGS = [
  { id: "1",  medName: "Metformin 500mg",   status: "Taken",   time: "8:04 AM",  date: "Today"     },
  { id: "2",  medName: "Atorvastatin 20mg", status: "Pending", time: "8:00 PM",  date: "Today"     },
  { id: "3",  medName: "Metformin 500mg",   status: "Taken",   time: "8:02 AM",  date: "Yesterday" },
  { id: "4",  medName: "Lisinopril 10mg",   status: "Taken",   time: "12:01 PM", date: "Yesterday" },
  { id: "5",  medName: "Atorvastatin 20mg", status: "Taken",   time: "8:05 PM",  date: "Yesterday" },
  { id: "6",  medName: "Metformin 500mg",   status: "Taken",   time: "8:10 AM",  date: "Mar 22"    },
  { id: "7",  medName: "Lisinopril 10mg",   status: "Missed",  time: "—",        date: "Mar 22"    },
  { id: "8",  medName: "Atorvastatin 20mg", status: "Taken",   time: "8:07 PM",  date: "Mar 22"    },
  { id: "9",  medName: "Metformin 500mg",   status: "Taken",   time: "8:05 AM",  date: "Mar 21"    },
  { id: "10", medName: "Atorvastatin 20mg", status: "Taken",   time: "8:00 PM",  date: "Mar 21"    },
  { id: "11", medName: "Metformin 500mg",   status: "Missed",  time: "—",        date: "Mar 20"    },
  { id: "12", medName: "Lisinopril 10mg",   status: "Taken",   time: "12:00 PM", date: "Mar 20"    },
  { id: "13", medName: "Metformin 500mg",   status: "Taken",   time: "8:01 AM",  date: "Mar 15"    },
  { id: "14", medName: "Atorvastatin 20mg", status: "Taken",   time: "8:03 PM",  date: "Mar 15"    },
  { id: "15", medName: "Metformin 500mg",   status: "Taken",   time: "8:06 AM",  date: "Mar 10"    },
];

// Dates considered within "this week" for mock filtering
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

export default function SelfLogs() {
  const router = useRouter();

  // Default selection: Day
  const [period, setPeriod] = useState("Day");

  const filtered = filterByPeriod(SELF_LOGS, period);

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>
        <Text style={styles.sectionTitle}>my logs</Text>
        <View style={styles.divider} />

        {/* Day / Week / Month tab selector — same spec as patient-log-view.js */}
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
                      patientName: "Me",
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
  safe:      { flex: 1, backgroundColor: "#e8f5e9" },
  container: { flex: 1, paddingHorizontal: 20, paddingTop: 20 },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider:      { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },

  // Tab selector — identical spec to patient-log-view.js
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
  tabText:       { fontSize: 14, fontWeight: "600", color: "#366a53" },
  tabTextActive: { color: "#fff" },

  emptyText: {
    marginTop: 32,
    fontSize: 15,
    color: "#888",
    textAlign: "center",
  },

  // Log row — same spec as logs.js / patient-log-view.js
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
  thumbIcon:  { fontSize: 26 },

  rowMid:    { flex: 1 },
  rowMed:    { fontSize: 15, fontWeight: "600", color: "#000" },
  rowStatus: { fontSize: 12, fontWeight: "600", marginTop: 3 },

  rowRight: { alignItems: "flex-end", flexShrink: 0 },
  rowTime:  { fontSize: 13, color: "#555" },
  rowDate:  { fontSize: 12, color: "#888", marginTop: 2 },
});
