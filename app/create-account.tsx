import React, { useState } from "react";
import {
  Alert,
  ImageBackground,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  Pressable,
  View
} from "react-native";
import { useRouter } from "expo-router";

import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import { auth } from "../firebaseConfig"; // or your correct path

export default function CreateAccount() {
  const router = useRouter();

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [phone, setPhone] = useState("");

  const handleCreateAccount = async () => {
    if (!firstName || !lastName || !phone || !email || !password) {
      Alert.alert("Missing info", "Please fill all fields.");
      return;
    }

    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email.trim(),
        password
      );

      // ✅ Save caregiver name into Firebase Auth profile (so Home can show it)
      await updateProfile(userCredential.user, {
        displayName: `${firstName.trim()} ${lastName.trim()}`,
      });

      // ✅ Go to Home
      router.replace("/home");
    } catch (error: any) {
      Alert.alert("Sign up failed", error.code || "Unknown error");
      console.log("SIGNUP ERROR:", error);
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, padding: 20 }}>
      <Text style={{ fontSize: 22, fontWeight: "700", marginBottom: 12 }}>
        Create Account
      </Text>

      <TextInput placeholder="First name" value={firstName} onChangeText={setFirstName} style={{ borderWidth: 1, padding: 12, marginBottom: 10 }} />
      <TextInput placeholder="Last name" value={lastName} onChangeText={setLastName} style={{ borderWidth: 1, padding: 12, marginBottom: 10 }} />
      <TextInput placeholder="Phone" value={phone} onChangeText={setPhone} keyboardType="phone-pad" style={{ borderWidth: 1, padding: 12, marginBottom: 10 }} />
      <TextInput placeholder="Email" value={email} onChangeText={setEmail} autoCapitalize="none" keyboardType="email-address" style={{ borderWidth: 1, padding: 12, marginBottom: 10 }} />
      <TextInput placeholder="Password" value={password} onChangeText={setPassword} secureTextEntry style={{ borderWidth: 1, padding: 12, marginBottom: 16 }} />

      <TouchableOpacity onPress={handleCreateAccount} style={{ backgroundColor: "#366a53ff", padding: 14, borderRadius: 8 }}>
        <Text style={{ color: "white", textAlign: "center", fontWeight: "700" }}>
          Create Account
        </Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}