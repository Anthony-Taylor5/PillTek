import React, { useState, useCallback } from "react";
import { View, Text, FlatList, StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useFocusEffect } from "expo-router";
import DashboardRow from "../components/DashboardRow";
import { getPatientMedications } from "./medication-store";

const MOCK_MEDICATIONS = [
  { id: "1", name: "Metformin 500mg", time: "8:00 AM", status: "Taken" },
  { id: "2", name: "Lisinopril 10mg", time: "12:00 PM", status: "Taken" },
  { id: "3", name: "Atorvastatin 20mg", time: "8:00 PM", status: "Pending" },
];

export default function PatientMedications() {
  const router = useRouter();
  const [medications, setMedications] = useState(MOCK_MEDICATIONS);

  // Refresh from the store each time this screen comes into focus.
  // When the patient completes bottle setup, their medications are written
  // to medication-store and appear here on the next visit.
  useFocusEffect(
    useCallback(() => {
      const stored = getPatientMedications();
      if (stored.length > 0) {
        setMedications(stored);
      }
    }, [])
  );

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>
        <Text style={styles.sectionTitle}>my medications</Text>
        <View style={styles.divider} />
        <FlatList
          data={medications}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <DashboardRow
              label={item.name}
              value={item.time}
              onPress={() =>
                router.push({ pathname: "/medication-detail", params: { id: item.id, name: item.name, time: item.time } })
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
});
