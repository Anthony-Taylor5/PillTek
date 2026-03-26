import React from "react";
import { View, Text, FlatList, StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useLocalSearchParams, useRouter } from "expo-router";
import DashboardRow from "../components/DashboardRow";

// Mock per-patient medication data
const PATIENT_MEDS = {
  "1": [ // Ahmad
    { id: "m1", name: "Metformin 500mg", time: "8:00 AM", status: "Taken" },
    { id: "m2", name: "Aspirin 81mg", time: "8:00 PM", status: "Pending" },
  ],
  "2": [ // Shahriar
    { id: "m1", name: "Lisinopril 10mg", time: "12:00 PM", status: "Taken" },
    { id: "m2", name: "Atorvastatin 20mg", time: "8:00 PM", status: "Taken" },
  ],
  "3": [ // Mina
    { id: "m1", name: "Metformin 500mg", time: "8:00 AM", status: "Missed" },
    { id: "m2", name: "Vitamin D 1000IU", time: "12:00 PM", status: "Taken" },
    { id: "m3", name: "Atorvastatin 20mg", time: "8:00 PM", status: "Pending" },
  ],
};

export default function PatientDetail() {
  const { id, name } = useLocalSearchParams();
  const router = useRouter();
  const meds = PATIENT_MEDS[id] ?? [];

  const takenCount = meds.filter((m) => m.status === "Taken").length;
  const missedCount = meds.filter((m) => m.status === "Missed").length;

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>
        {/* Patient identity */}
        <View style={styles.patientHeader}>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>{String(name)?.[0]?.toUpperCase() ?? "P"}</Text>
          </View>
          <Text style={styles.patientName}>{name}</Text>
        </View>

        {/* Summary row */}
        <View style={styles.summaryRow}>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNum}>{takenCount}</Text>
            <Text style={styles.summaryLabel}>Taken</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={[styles.summaryNum, missedCount > 0 && styles.missed]}>{missedCount}</Text>
            <Text style={styles.summaryLabel}>Missed</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNum}>{meds.length}</Text>
            <Text style={styles.summaryLabel}>Total</Text>
          </View>
        </View>

        {/* Medications */}
        <Text style={styles.sectionTitle}>medications</Text>
        <View style={styles.divider} />
        <FlatList
          data={meds}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <DashboardRow
              label={item.name}
              value={`${item.time} · ${item.status}`}
              onPress={() =>
                router.push({
                  pathname: "/med-log",
                  params: {
                    patientId: id,
                    patientName: name,
                    medId: item.id,
                    medName: item.name,
                    medTime: item.time,
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

  patientHeader: { flexDirection: "row", alignItems: "center", gap: 14, marginBottom: 20 },
  avatar: {
    width: 52,
    height: 52,
    borderRadius: 26,
    borderWidth: 1.5,
    borderColor: "#111",
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: { fontSize: 20, fontWeight: "700" },
  patientName: { fontSize: 22, fontWeight: "600", color: "#366a53" },

  summaryRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 24,
  },
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
  summaryNum: { fontSize: 28, fontWeight: "700", color: "#366a53" },
  summaryLabel: { fontSize: 12, color: "#555", marginTop: 2 },
  missed: { color: "#c0392b" },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },
});
