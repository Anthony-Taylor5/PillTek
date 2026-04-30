import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ScrollView,
  Modal,
  Pressable,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams, useFocusEffect } from "expo-router";
import { getLastCompletedMed, clearLastCompletedMed } from "../lib/med-detail-store";
import { setCaregiverPatientMeds } from "../lib/medication-store";
import { createMedications } from "../lib/api";

const LABELS = ["A", "B", "D", "F"];
const MAX_MEDS = 4;

export default function AddMedication() {
  const router = useRouter();
  const { patientId, patientName } = useLocalSearchParams();

  const [medicationInput, setMedicationInput] = useState("");
  const [medications, setMedications] = useState([]);
  const [labelPickerFor, setLabelPickerFor] = useState(null); // med name whose picker is open
  const [saving, setSaving] = useState(false);

  useFocusEffect(
    useCallback(() => {
      const completed = getLastCompletedMed();
      if (completed) {
        setMedications((prev) => {
          if (prev.some((m) => m.name === completed.name)) return prev;
          return [...prev, { ...completed, label: null }];
        });
        clearLastCompletedMed();
      }
    }, [])
  );

  const handleAddDetails = () => {
    const trimmed = medicationInput.trim();
    if (!trimmed) return;
    if (medications.length >= MAX_MEDS) {
      Alert.alert("Limit reached", `You can add up to ${MAX_MEDS} medications.`);
      return;
    }
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

  const assignLabel = (medName, label) => {
    setMedications((prev) =>
      prev.map((m) => (m.name === medName ? { ...m, label } : m))
    );
    setLabelPickerFor(null);
  };

  const usedLabels = (excludeName) =>
    medications.filter((m) => m.name !== excludeName && m.label).map((m) => m.label);

  const handleSubmit = async () => {
    if (medications.length === 0) {
      Alert.alert("No medications", "Please add at least one medication.");
      return;
    }
    const unlabeled = medications.filter((m) => !m.label);
    if (unlabeled.length > 0) {
      Alert.alert(
        "Missing labels",
        `Please assign a label to: ${unlabeled.map((m) => m.name).join(", ")}`
      );
      return;
    }

    setSaving(true);
    try {
      setCaregiverPatientMeds(patientName, medications);
      await createMedications(patientId, medications);
      router.back();
    } catch (err) {
      Alert.alert("Error", err.message ?? "Could not save medications.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>

        <Text style={styles.sectionTitle}>Add Medication</Text>
        <View style={styles.divider} />

        <Text style={styles.hint}>
          Adding medication for: <Text style={styles.hintBold}>{patientName}</Text>
        </Text>

        {medications.length < MAX_MEDS && (
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
        )}

        {medications.length >= MAX_MEDS && (
          <Text style={styles.limitNote}>Maximum of {MAX_MEDS} medications reached.</Text>
        )}

        {medications.map((med) => {
          const takenLabels = usedLabels(med.name);
          return (
            <View key={med.name} style={styles.medRow}>
              <View style={{ flex: 1 }}>
                <Text style={styles.medRowText}>{med.name}</Text>
                <Text style={styles.medRowSub}>
                  {med.dosage} · {med.frequency} · {med.time}
                </Text>
              </View>

              <TouchableOpacity
                style={[styles.labelBtn, med.label && styles.labelBtnSet]}
                onPress={() => setLabelPickerFor(med.name)}
                hitSlop={4}
              >
                <Text style={[styles.labelBtnText, med.label && styles.labelBtnTextSet]}>
                  {med.label ?? "Label"}
                </Text>
                <Text style={[styles.labelChevron, med.label && styles.labelBtnTextSet]}>▾</Text>
              </TouchableOpacity>

              <TouchableOpacity onPress={() => removeMedication(med.name)} hitSlop={8} style={{ marginLeft: 10 }}>
                <Text style={styles.removeText}>Remove</Text>
              </TouchableOpacity>

              {/* Label picker modal */}
              <Modal
                visible={labelPickerFor === med.name}
                transparent
                animationType="fade"
                onRequestClose={() => setLabelPickerFor(null)}
              >
                <Pressable style={styles.modalBackdrop} onPress={() => setLabelPickerFor(null)}>
                  <Pressable style={styles.pickerCard} onPress={() => {}}>
                    <Text style={styles.pickerCardTitle}>Bottle Label — {med.name}</Text>
                    {LABELS.map((lbl) => {
                      const taken = takenLabels.includes(lbl);
                      const selected = med.label === lbl;
                      return (
                        <TouchableOpacity
                          key={lbl}
                          style={[
                            styles.pickerOption,
                            selected && styles.pickerOptionSelected,
                            taken && styles.pickerOptionDisabled,
                          ]}
                          onPress={() => !taken && assignLabel(med.name, lbl)}
                          disabled={taken}
                        >
                          <Text
                            style={[
                              styles.pickerOptionText,
                              selected && styles.pickerOptionTextSelected,
                              taken && styles.pickerOptionTextDisabled,
                            ]}
                          >
                            {lbl}
                          </Text>
                          {selected && <Text style={styles.pickerCheckmark}>✓</Text>}
                          {taken && !selected && <Text style={styles.pickerTaken}>In use</Text>}
                        </TouchableOpacity>
                      );
                    })}
                  </Pressable>
                </Pressable>
              </Modal>
            </View>
          );
        })}

        <TouchableOpacity
          style={[styles.submitBtn, (medications.length === 0 || saving || medications.some((m) => !m.label)) && styles.submitBtnDisabled]}
          onPress={handleSubmit}
          disabled={medications.length === 0 || saving || medications.some((m) => !m.label)}
        >
          {saving
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.submitText}>Save Medications</Text>
          }
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

  limitNote: { fontSize: 13, color: "#888", marginBottom: 12, fontStyle: "italic" },

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

  labelBtn: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#bbb",
    borderRadius: 6,
    paddingVertical: 5,
    paddingHorizontal: 10,
    marginLeft: 8,
    gap: 4,
  },
  labelBtnSet: { borderColor: "#366a53", backgroundColor: "#f0f7f2" },
  labelBtnText: { fontSize: 14, color: "#666", fontWeight: "600" },
  labelBtnTextSet: { color: "#366a53" },
  labelChevron: { fontSize: 11, color: "#666" },

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

  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.35)",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  pickerCard: {
    width: "100%",
    backgroundColor: "#fff",
    borderRadius: 12,
    paddingVertical: 8,
    paddingHorizontal: 4,
    elevation: 6,
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 3 },
  },
  pickerCardTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#111",
    marginBottom: 12,
    textAlign: "center",
    paddingHorizontal: 12,
    paddingTop: 8,
  },
  pickerOption: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  pickerOptionSelected: { backgroundColor: "#f0f7f2" },
  pickerOptionDisabled: { backgroundColor: "#fafafa" },
  pickerOptionText: { fontSize: 18, color: "#111", fontWeight: "600" },
  pickerOptionTextSelected: { color: "#366a53" },
  pickerOptionTextDisabled: { color: "#ccc" },
  pickerCheckmark: { fontSize: 16, color: "#366a53", fontWeight: "700" },
  pickerTaken: { fontSize: 12, color: "#bbb" },
});
