import React, { useState } from "react";
import {
  Alert,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  ScrollView,
} from "react-native";
import { useRouter } from "expo-router";
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import { auth } from "../firebaseConfig";
import { getRole } from "./role-store";
import { setLinkedCaregiverCode } from "./patient-store";

export default function CreateAccount() {
  const router = useRouter();
  const role = getRole(); // "caregiver" | "patient" | "self"

  const [firstName, setFirstName]   = useState("");
  const [lastName, setLastName]     = useState("");
  const [email, setEmail]           = useState("");
  const [password, setPassword]     = useState("");
  const [phone, setPhone]           = useState("");
  const [linkCode, setLinkCode]     = useState(""); // patient-only: caregiver link code

  const isPatient = role === "patient";

  const handleCreateAccount = async () => {
    if (!firstName || !lastName || !phone || !email || !password) {
      Alert.alert("Missing info", "Please fill all fields.");
      return;
    }
    if (isPatient && !linkCode.trim()) {
      Alert.alert(
        "Missing info",
        "Please enter the Patient Code your caregiver provided."
      );
      return;
    }

    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email.trim(),
        password
      );

      await updateProfile(userCredential.user, {
        displayName: `${firstName.trim()} ${lastName.trim()}`,
      });

      // If the patient entered a caregiver link code, save it to the session store.
      // When the backend is wired, this is where you'd write the association to Firestore.
      if (isPatient && linkCode.trim()) {
        setLinkedCaregiverCode(linkCode.trim().toUpperCase());
      }

      // Route based on role
      if (role === "caregiver") {
        router.replace("/home");
      } else if (role === "self") {
        router.replace("/self-home");
      } else {
        // patient
        router.replace("/patient-home");
      }
    } catch (error: any) {
      Alert.alert("Sign up failed", error.code || "Unknown error");
      console.log("SIGNUP ERROR:", error);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Create Account</Text>

        <TextInput
          placeholder="First name"
          value={firstName}
          onChangeText={setFirstName}
          style={styles.input}
          autoCapitalize="words"
        />
        <TextInput
          placeholder="Last name"
          value={lastName}
          onChangeText={setLastName}
          style={styles.input}
          autoCapitalize="words"
        />
        <TextInput
          placeholder="Phone"
          value={phone}
          onChangeText={setPhone}
          keyboardType="phone-pad"
          style={styles.input}
        />
        <TextInput
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
          style={styles.input}
        />
        <TextInput
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          style={styles.input}
        />

        {/* Caregiver link code — only shown for patients */}
        {isPatient && (
          <View style={styles.linkCodeSection}>
            <Text style={styles.linkCodeLabel}>Caregiver Patient Code</Text>
            <Text style={styles.linkCodeHint}>
              Enter the code your caregiver gave you (e.g. PTK-A1B2).
              This links your account to your caregiver.
            </Text>
            <TextInput
              placeholder="PTK-XXXX"
              value={linkCode}
              onChangeText={setLinkCode}
              autoCapitalize="characters"
              style={styles.input}
            />
          </View>
        )}

        <TouchableOpacity
          onPress={handleCreateAccount}
          style={styles.submitBtn}
        >
          <Text style={styles.submitText}>Create Account</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#fff" },
  container: { padding: 20, paddingBottom: 40 },

  title: { fontSize: 22, fontWeight: "700", marginBottom: 20 },

  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    fontSize: 16,
    backgroundColor: "#fff",
  },

  linkCodeSection: { marginTop: 8, marginBottom: 4 },
  linkCodeLabel: { fontSize: 15, fontWeight: "600", marginBottom: 6 },
  linkCodeHint: {
    fontSize: 13,
    color: "#555",
    lineHeight: 18,
    marginBottom: 10,
  },

  submitBtn: {
    marginTop: 8,
    backgroundColor: "#366a53",
    padding: 14,
    borderRadius: 8,
    alignItems: "center",
  },
  submitText: { color: "#fff", fontWeight: "700", fontSize: 16 },
});
