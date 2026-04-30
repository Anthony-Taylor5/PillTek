import React, { useState } from "react";
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
import { useRouter } from "expo-router";
import { auth } from "../firebaseConfig";

export default function MedicationEntry() {
  const router = useRouter();
  const [medicationInput, setMedicationInput] = useState("");
  const [medications, setMedications] = useState([]);

  const addMedication = () => {
    const trimmed = medicationInput.trim();
    if (!trimmed) return;
    if (medications.includes(trimmed)) {
      Alert.alert("Duplicate", "That medication is already in the list.");
      return;
    }
    setMedications((prev) => [...prev, trimmed]);
    setMedicationInput("");
  };

  const removeMedication = (med) => {
    setMedications((prev) => prev.filter((m) => m !== med));
  };

  const handleNext = () => {
    if (medications.length === 0) {
      Alert.alert("No medications", "Please add at least one medication.");
      return;
    }
    const u = auth.currentUser;
    const patientName = u?.displayName || u?.email || "patient";
    router.push({
      pathname: "/capture-bottles",
      params: {
        patientName,
        medications: JSON.stringify(medications),
        returnTo: "/patient-home",
      },
    });
  };

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.sectionTitle}>My Medications</Text>
        <View style={styles.divider} />
        <Text style={styles.hint}>
          Enter each medication name and tap Add. You'll then photograph each bottle.
        </Text>

        <View style={styles.inputRow}>
          <TextInput
            style={[styles.input, { flex: 1 }]}
            placeholder="e.g. Metformin 500mg"
            value={medicationInput}
            onChangeText={setMedicationInput}
            onSubmitEditing={addMedication}
            returnKeyType="done"
          />
          <TouchableOpacity style={styles.addBtn} onPress={addMedication}>
            <Text style={styles.addBtnText}>Add</Text>
          </TouchableOpacity>
        </View>

        {medications.map((med) => (
          <View key={med} style={styles.medRow}>
            <Text style={styles.medRowText}>{med}</Text>
            <TouchableOpacity onPress={() => removeMedication(med)} hitSlop={8}>
              <Text style={styles.removeText}>Remove</Text>
            </TouchableOpacity>
          </View>
        ))}

        <TouchableOpacity
          style={[styles.submitBtn, medications.length === 0 && styles.submitBtnDisabled]}
          onPress={handleNext}
          disabled={medications.length === 0}
        >
          <Text style={styles.submitText}>Next: Capture Bottles</Text>
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
    paddingHorizontal: 16,
    borderRadius: 8,
    justifyContent: "center",
  },
  addBtnText: { color: "#fff", fontWeight: "600", fontSize: 15 },

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
  medRowText: { fontSize: 15, color: "#222", flex: 1 },
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
