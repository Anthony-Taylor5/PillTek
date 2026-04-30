import React from "react";
import { View, Text, FlatList, TouchableOpacity, StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";

const LOGS = [
  { id: "1", patient: "Ahmad",    medName: "Metformin 500mg",   status: "Taken",  time: "8:04 AM",  date: "Today"     },
  { id: "2", patient: "Shahriar", medName: "Lisinopril 10mg",   status: "Taken",  time: "12:01 PM", date: "Today"     },
  { id: "3", patient: "Mina",     medName: "Atorvastatin 20mg", status: "Missed", time: "8:00 PM",  date: "Today"     },
  { id: "4", patient: "Ahmad",    medName: "Metformin 500mg",   status: "Taken",  time: "8:02 AM",  date: "Yesterday" },
  { id: "5", patient: "Shahriar", medName: "Lisinopril 10mg",   status: "Taken",  time: "12:00 PM", date: "Yesterday" },
  { id: "6", patient: "Mina",     medName: "Atorvastatin 20mg", status: "Taken",  time: "8:05 PM",  date: "Yesterday" },
  { id: "7", patient: "Ahmad",    medName: "Metformin 500mg",   status: "Taken",  time: "8:10 AM",  date: "Mar 22"    },
  { id: "8", patient: "Shahriar", medName: "Lisinopril 10mg",   status: "Missed", time: "12:00 PM", date: "Mar 22"    },
];

function statusColor(s) {
  if (s === "Taken")  return "#366a53";
  if (s === "Missed") return "#c0392b";
  return "#888";
}

// Same thumb component as med-log.js — no network, no camera
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
      {/* Left: bottle thumbnail */}
      <BottleThumb status={item.status} />

      {/* Middle: patient name + medication */}
      <View style={styles.rowMid}>
        <Text style={styles.rowPatient}>{item.patient}</Text>
        <Text style={styles.rowMed}>{item.medName}</Text>
        <Text style={[styles.rowStatus, { color: statusColor(item.status) }]}>
          {item.status}
        </Text>
      </View>

      {/* Right: time + date */}
      <View style={styles.rowRight}>
        <Text style={styles.rowTime}>{item.time}</Text>
        <Text style={styles.rowDate}>{item.date}</Text>
      </View>
    </TouchableOpacity>
  );
}

export default function Logs() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>
        <Text style={styles.sectionTitle}>activity logs</Text>
        <View style={styles.divider} />
        <FlatList
          data={LOGS}
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

  // Bottle thumbnail — identical spec to med-log.js
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

  rowMid: { flex: 1 },
  rowPatient: { fontSize: 16, fontWeight: "600", color: "#000" },
  rowMed: { fontSize: 13, color: "#555", marginTop: 2 },
  rowStatus: { fontSize: 12, fontWeight: "600", marginTop: 3 },

  rowRight: { alignItems: "flex-end", flexShrink: 0 },
  rowTime: { fontSize: 13, color: "#555" },
  rowDate: { fontSize: 12, color: "#888", marginTop: 2 },
});
