import React from "react";
import { View, Text, FlatList, StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

const SCHEDULE = [
  {
    id: "morning",
    slot: "Morning",
    time: "8:00 AM",
    meds: [{ name: "Metformin 500mg", status: "Taken" }],
  },
  {
    id: "afternoon",
    slot: "Afternoon",
    time: "12:00 PM",
    meds: [{ name: "Lisinopril 10mg", status: "Taken" }],
  },
  {
    id: "evening",
    slot: "Evening",
    time: "8:00 PM",
    meds: [{ name: "Atorvastatin 20mg", status: "Pending" }],
  },
];

function ScheduleSlot({ slot }) {
  return (
    <View style={styles.slotCard}>
      <View style={styles.slotHeader}>
        <Text style={styles.slotName}>{slot.slot}</Text>
        <Text style={styles.slotTime}>{slot.time}</Text>
      </View>
      {slot.meds.map((med, i) => (
        <View key={i} style={styles.medRow}>
          <Text style={styles.medName}>{med.name}</Text>
          <Text style={[styles.medStatus, med.status === "Taken" ? styles.taken : med.status === "Missed" ? styles.missed : styles.pending]}>
            {med.status}
          </Text>
        </View>
      ))}
    </View>
  );
}

export default function PatientSchedule() {
  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>
        <Text style={styles.sectionTitle}>my schedule</Text>
        <View style={styles.divider} />
        <FlatList
          data={SCHEDULE}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => <ScheduleSlot slot={item} />}
          contentContainerStyle={{ paddingTop: 12 }}
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

  slotCard: {
    backgroundColor: "#fff",
    borderRadius: 10,
    padding: 16,
    marginTop: 12,
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  slotHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  slotName: { fontSize: 17, fontWeight: "600", color: "#366a53" },
  slotTime: { fontSize: 13, color: "#555" },

  medRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: "#d6ebd9",
  },
  medName: { fontSize: 15, color: "#000", flex: 1 },
  medStatus: { fontSize: 13, fontWeight: "600", marginLeft: 12 },
  taken: { color: "#366a53" },
  missed: { color: "#c0392b" },
  pending: { color: "#888" },
});
