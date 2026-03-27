import React, { useRef } from "react";
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
import { useState } from "react";
import { generatePatientCode, addPatient } from "./patient-store";

export default function AddPatient() {
  const router = useRouter();

  const [name, setName]   = useState("");
  const [dob, setDob]     = useState("");
  const [phone, setPhone] = useState("");

  // Patient code is generated once on mount and stays stable across re-renders.
  const patientCode = useRef(generatePatientCode());

  const handleSubmit = () => {
    if (!name.trim()) {
      Alert.alert("Missing info", "Please enter the patient's name.");
      return;
    }

    // Register the patient in the session store so home.js shows them immediately.
    addPatient({
      name: name.trim(),
      dob: dob.trim(),
      phone: phone.trim(),
      patientCode: patientCode.current,
    });

    Alert.alert(
      "Patient created",
      `${name.trim()} has been added.\n\nPatient Code: ${patientCode.current}\n\nShare this code with the patient so they can link their account to you.`,
      [{ text: "Done", onPress: () => router.replace("/home") }]
    );
  };

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>

        <Text style={styles.sectionTitle}>Patient details</Text>
        <View style={styles.divider} />

        {/* Unique patient code — shown so the caregiver can share it */}
        <View style={styles.codeCard}>
          <Text style={styles.codeLabel}>Patient Code</Text>
          <Text style={styles.codeValue}>{patientCode.current}</Text>
          <Text style={styles.codeHint}>
            Share this code with the patient. They will enter it when creating
            their account to link to you as their caregiver.
          </Text>
        </View>

        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Full name</Text>
          <TextInput
            style={styles.input}
            placeholder="Enter patient's name"
            value={name}
            onChangeText={setName}
            autoCapitalize="words"
          />
        </View>

        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Date of birth</Text>
          <TextInput
            style={styles.input}
            placeholder="MM/DD/YYYY"
            value={dob}
            onChangeText={setDob}
            keyboardType="numeric"
          />
        </View>

        <View style={styles.fieldGroup}>
          <Text style={styles.label}>Phone number</Text>
          <TextInput
            style={styles.input}
            placeholder="Enter phone number"
            value={phone}
            onChangeText={setPhone}
            keyboardType="phone-pad"
          />
        </View>

        <TouchableOpacity style={styles.submitBtn} onPress={handleSubmit}>
          <Text style={styles.submitText}>Create Patient</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: { paddingHorizontal: 20, paddingTop: 24, paddingBottom: 40 },

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 24 },

  codeCard: {
    backgroundColor: "#fff",
    borderRadius: 10,
    padding: 16,
    marginBottom: 24,
    borderLeftWidth: 4,
    borderLeftColor: "#366a53",
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  codeLabel: { fontSize: 12, color: "#555", fontWeight: "600", marginBottom: 4 },
  codeValue: { fontSize: 26, fontWeight: "700", color: "#366a53", letterSpacing: 2 },
  codeHint: { fontSize: 12, color: "#777", marginTop: 8, lineHeight: 17 },

  fieldGroup: { marginBottom: 20 },
  label: { fontSize: 14, color: "#555", marginBottom: 8 },
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

  submitBtn: {
    marginTop: 12,
    backgroundColor: "#366a53",
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: "center",
    elevation: 2,
  },
  submitText: { color: "#fff", fontSize: 17, fontWeight: "600" },
});
