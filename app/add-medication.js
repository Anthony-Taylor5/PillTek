import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ScrollView,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams, useFocusEffect } from "expo-router";
import { getLastCompletedMed, clearLastCompletedMed } from "./med-detail-store";

// Screen reached from the patient-detail 3-dot menu → "Add Medication".
// Lets the caregiver add one or more medications for an existing patient,
// then routes into capture-bottles and returns to patient-detail.

export default function AddMedication() {
  const router = useRouter();
  const { patientId, patientName, patientCode } = useLocalSearchParams();

  const [medicationInput, setMedicationInput] = useState("");

  // Full medication objects: { name, dosage, frequency, time, times, refill, status }
  const [medications, setMedications] = useState([]);

  // When returning from medication-details-entry, pick up the completed medication.
  useFocusEffect(
    useCallback(() => {
      const completed = getLastCompletedMed();
      if (completed) {
        setMedications((prev) => {
          if (prev.some((m) => m.name === completed.name)) return prev;
          return [...prev, completed];
        });
        clearLastCompletedMed();
      }
    }, [])
  );

  const handleAddDetails = () => {
    const trimmed = medicationInput.trim();
    if (!trimmed) return;
    if (medications.some((m) => m.name === trimmed)) {
      Alert.alert("Duplicate", "That medication is already in the list.");
      return;
    }
    setMedicationInput("");
    router.push({
      pathname: "/medication-details-entry",
      params: { medName: trimmed },
    });
  };

  const removeMedication = (medName) => {
    setMedications((prev) => prev.filter((m) => m.name !== medName));
  };

  const handleSubmit = () => {
    if (medications.length === 0) {
      Alert.alert("No medications", "Please add at least one medication.");
      return;
    }

    // Use replace (not push) so add-medication is removed from the back stack.
    // capture-bottles will then use router.back() to return cleanly to patient-detail.
    router.replace({
      pathname: "/capture-bottles",
      params: {
        patientName,
        medications: JSON.stringify(medications),
        returnTo: "/patient-detail",
        returnMode: "back",
      },
    });
  };

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>

        <Text style={styles.sectionTitle}>Add Medication</Text>
        <View style={styles.divider} />

        <Text style={styles.hint}>
          Adding medication for: <Text style={styles.hintBold}>{patientName}</Text>
        </Text>

        <View style={styles.inputRow}>
          <TextInput
            style={[styles.input, { flex: 1 }]}
            placeholder="e.g. Metformin"
            value={medicationInput}
            onChangeText={setMedicationInput}
            onSubmitEditing={handleAddDetails}
            returnKeyType="done"
          />
          <TouchableOpacity style={styles.addBtn} onPress={handleAddDetails}>
            <Text style={styles.addBtnText}>Add Details</Text>
          </TouchableOpacity>
        </View>

        {medications.map((med) => (
          <View key={med.name} style={styles.medRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.medRowText}>{med.name}</Text>
              <Text style={styles.medRowSub}>
                {med.dosage} · {med.frequency} · {med.time}
              </Text>
            </View>
            <TouchableOpacity onPress={() => removeMedication(med.name)} hitSlop={8}>
              <Text style={styles.removeText}>Remove</Text>
            </TouchableOpacity>
          </View>
        ))}

        <TouchableOpacity
          style={[styles.submitBtn, medications.length === 0 && styles.submitBtnDisabled]}
          onPress={handleSubmit}
          disabled={medications.length === 0}
        >
          <Text style={styles.submitText}>Next: Capture Bottle</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: { paddingHorizontal: 20, paddingTop: 24, paddingBottom: 40 },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 16 },

  hint: { fontSize: 14, color: "#555", marginBottom: 20, lineHeight: 20 },
  hintBold: { fontWeight: "600", color: "#366a53" },

  inputRow: { flexDirection: "row", gap: 10, marginBottom: 12 },
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
  addBtn: {
    backgroundColor: "#366a53",
    paddingHorizontal: 14,
    borderRadius: 8,
    justifyContent: "center",
  },
  addBtnText: { color: "#fff", fontWeight: "600", fontSize: 14 },

  medRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "#fff",
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginBottom: 8,
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  medRowText: { fontSize: 15, color: "#222", fontWeight: "600" },
  medRowSub: { fontSize: 12, color: "#666", marginTop: 2 },
  removeText: { fontSize: 14, color: "#c0392b" },

  submitBtn: {
    marginTop: 24,
    backgroundColor: "#366a53",
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: "center",
    elevation: 2,
  },
  submitBtnDisabled: { backgroundColor: "#aaa" },
  submitText: { color: "#fff", fontSize: 17, fontWeight: "600" },
});
