import React from "react";
import { View, Text, FlatList, TouchableOpacity, StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";

// Mock log entries for the patient's own medications today.
// Replace with Firestore query keyed by auth.currentUser.uid when backend is wired.
const TODAY_LOGS = [
  { id: "1", medName: "Metformin 500mg",   time: "8:00 AM",  status: "Taken"   },
  { id: "2", medName: "Lisinopril 10mg",   time: "12:00 PM", status: "Pending" },
  { id: "3", medName: "Atorvastatin 20mg", time: "8:00 PM",  status: "Pending" },
];

function statusColor(s) {
  if (s === "Taken")  return "#366a53";
  if (s === "Missed") return "#c0392b";
  return "#888";
}

function LogRow({ item, onPress }) {
  const hasPhoto = item.status === "Taken";
  return (
    <TouchableOpacity style={styles.row} onPress={onPress} activeOpacity={0.65}>
      {/* Bottle thumbnail — same visual spec as logs.js / med-log.js */}
      <View style={[styles.thumb, !hasPhoto && styles.thumbEmpty]}>
        <Text style={styles.thumbIcon}>{hasPhoto ? "🧴" : "—"}</Text>
      </View>

      {/* Medication name + status */}
      <View style={styles.rowMid}>
        <Text style={styles.rowMed}>{item.medName}</Text>
        <Text style={[styles.rowStatus, { color: statusColor(item.status) }]}>
          {item.status}
        </Text>
      </View>

      {/* Scheduled time */}
      <Text style={styles.rowTime}>{item.time}</Text>
    </TouchableOpacity>
  );
}

export default function PatientDailyLogs() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>
        <Text style={styles.sectionTitle}>today's log</Text>
        <View style={styles.divider} />

        <FlatList
          data={TODAY_LOGS}
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
                    date:        "Today",
                    time:        item.time,
                    status:      item.status,
                  },
                })
              }
            />
          )}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: { flex: 1, paddingHorizontal: 20, paddingTop: 20 },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },

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
  rowMed: { fontSize: 16, fontWeight: "600", color: "#000" },
  rowStatus: { fontSize: 13, fontWeight: "600", marginTop: 3 },

  rowTime: { fontSize: 13, color: "#555", flexShrink: 0 },
});
