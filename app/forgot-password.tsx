import React, { useState } from "react";
import { View, Text, TextInput, Pressable, Alert } from "react-native";
import { router } from "expo-router";
import { getAuth, sendPasswordResetEmail } from "firebase/auth";

export default function ForgotPassword() {
  const [email, setEmail] = useState("");

  const handleReset = async () => {
    const cleanEmail = email.trim();

    if (!cleanEmail) {
      Alert.alert("Missing email", "Please enter your email address.");
      return;
    }

    try {
      const auth = getAuth(); // uses your initialized firebase app
      await sendPasswordResetEmail(auth, cleanEmail);
      Alert.alert("Sent!", "Check your email for the reset link.");
      router.back(); // go back to login after success
    } catch (err: any) {
      Alert.alert("Reset failed", err?.message ?? "Unknown error");
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 20, gap: 12 }}>
      <Text style={{ fontSize: 26, fontWeight: "700", textAlign: "center" }}>
        Reset Password
      </Text>

      <TextInput
        placeholder="Enter your email"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        keyboardType="email-address"
        style={{
          borderWidth: 1,
          borderColor: "#ccc",
          padding: 12,
          borderRadius: 10,
        }}
      />

      <Pressable
        onPress={handleReset}
        style={{
          backgroundColor: "#3E6F58",
          padding: 14,
          borderRadius: 12,
          alignItems: "center",
        }}
      >
        <Text style={{ color: "white", fontWeight: "700" }}>
          Send reset link
        </Text>
      </Pressable>

      <Pressable onPress={() => router.back()}>
        <Text style={{ textAlign: "center", marginTop: 8 }}>Back to login</Text>
      </Pressable>
    </View>
  );
}