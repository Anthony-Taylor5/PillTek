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

export default function AddPatient() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [dob, setDob] = useState("");
  const [phone, setPhone] = useState("");

  const handleSubmit = () => {
    if (!name.trim()) {
      Alert.alert("Missing info", "Please enter the patient's name.");
      return;
    }
    Alert.alert("Patient added", `${name.trim()} has been added to your patients.`, [
      { text: "OK", onPress: () => router.back() },
    ]);
  };

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.sectionTitle}>Patient details</Text>
        <View style={styles.divider} />

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
          <Text style={styles.submitText}>Add Patient</Text>
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
