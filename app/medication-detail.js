import React from "react";
import { View, Text, StyleSheet, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useLocalSearchParams } from "expo-router";

// Mock detail data keyed by medication id
const MED_DETAILS = {
  "1": {
    dosage: "500mg",
    frequency: "Once daily",
    instructions: "Take with food in the morning.",
    prescriber: "Dr. Smith",
    refillDate: "Apr 15, 2026",
    status: "Taken today",
  },
  "2": {
    dosage: "10mg",
    frequency: "Once daily",
    instructions: "Take at midday, with or without food.",
    prescriber: "Dr. Patel",
    refillDate: "May 1, 2026",
    status: "Taken today",
  },
  "3": {
    dosage: "20mg",
    frequency: "Once daily",
    instructions: "Take in the evening. Avoid grapefruit.",
    prescriber: "Dr. Smith",
    refillDate: "Apr 30, 2026",
    status: "Pending",
  },
};

export default function MedicationDetail() {
  const { id, name, time } = useLocalSearchParams();
  const details = MED_DETAILS[id] ?? {
    dosage: "—",
    frequency: "—",
    instructions: "No instructions on file.",
    prescriber: "—",
    refillDate: "—",
    status: "—",
  };

  const statusColor = details.status === "Taken today" ? "#366a53" : details.status === "Missed" ? "#c0392b" : "#888";

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>
        {/* Med identity */}
        <View style={styles.medHeader}>
          <Text style={styles.medName}>{name}</Text>
          <Text style={[styles.medStatus, { color: statusColor }]}>{details.status}</Text>
        </View>

        {/* Details */}
        <Text style={styles.sectionTitle}>details</Text>
        <View style={styles.divider} />

        {[
          { label: "Dosage", value: details.dosage },
          { label: "Frequency", value: details.frequency },
          { label: "Scheduled time", value: time },
          { label: "Prescriber", value: details.prescriber },
          { label: "Next refill", value: details.refillDate },
        ].map((row) => (
          <View key={row.label} style={styles.infoRow}>
            <Text style={styles.infoLabel}>{row.label}</Text>
            <Text style={styles.infoValue}>{row.value}</Text>
          </View>
        ))}

        {/* Instructions */}
        <Text style={[styles.sectionTitle, { marginTop: 28 }]}>instructions</Text>
        <View style={styles.divider} />
        <View style={styles.instructionsCard}>
          <Text style={styles.instructionsText}>{details.instructions}</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: { paddingHorizontal: 20, paddingTop: 24, paddingBottom: 40 },

  medHeader: { marginBottom: 28 },
  medName: { fontSize: 24, fontWeight: "700", color: "#000" },
  medStatus: { fontSize: 15, fontWeight: "600", marginTop: 6 },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },

  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    minHeight: 52,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: "#d6ebd9",
  },
  infoLabel: { fontSize: 18, color: "#000", flex: 1 },
  infoValue: { fontSize: 14, color: "#555", marginLeft: 12, flexShrink: 0 },

  instructionsCard: {
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
  instructionsText: { fontSize: 15, color: "#333", lineHeight: 22 },
});
