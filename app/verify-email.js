import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { sendEmailVerification, reload } from "firebase/auth";
import { auth } from "../firebaseConfig";
import { getRole } from "./role-store";

export default function VerifyEmail() {
  const router = useRouter();
  const [checking, setChecking] = useState(false);
  const [resending, setResending] = useState(false);

  const userEmail = auth.currentUser?.email ?? "";

  const navigateToDashboard = () => {
    const role = getRole();
    if (role === "patient") {
      router.replace("/patient-home");
    } else if (role === "self") {
      router.replace("/self-home");
    } else {
      router.replace("/home");
    }
  };

  const handleCheckVerification = async () => {
    setChecking(true);
    try {
      // Reload the user object from Firebase so emailVerified is up to date.
      await reload(auth.currentUser);
      if (auth.currentUser?.emailVerified) {
        navigateToDashboard();
      } else {
        Alert.alert(
          "Not yet verified",
          "We haven't received your verification yet. Please check your inbox and tap the link, then try again."
        );
      }
    } catch (error) {
      Alert.alert("Error", "Could not check verification status. Please try again.");
      console.log("VERIFY CHECK ERROR:", error);
    } finally {
      setChecking(false);
    }
  };

  const handleResend = async () => {
    setResending(true);
    try {
      await sendEmailVerification(auth.currentUser);
      Alert.alert("Email sent", `A new verification link has been sent to ${userEmail}.`);
    } catch (error) {
      Alert.alert("Error", "Could not resend the verification email. Please try again shortly.");
      console.log("RESEND ERROR:", error);
    } finally {
      setResending(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>Verify your email</Text>
        <View style={styles.divider} />

        <Text style={styles.body}>
          We sent a verification link to:
        </Text>
        <Text style={styles.email}>{userEmail}</Text>

        <Text style={styles.instructions}>
          Open the email and tap the link to verify your account. Once you've done that, tap the button below to continue.
        </Text>

        <TouchableOpacity
          style={styles.primaryBtn}
          onPress={handleCheckVerification}
          disabled={checking}
          activeOpacity={0.8}
        >
          {checking ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.primaryBtnText}>I've verified my email</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryBtn}
          onPress={handleResend}
          disabled={resending}
          activeOpacity={0.8}
        >
          {resending ? (
            <ActivityIndicator color="#366a53" />
          ) : (
            <Text style={styles.secondaryBtnText}>Resend verification email</Text>
          )}
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#fff" },
  container: { flex: 1, padding: 20, paddingTop: 32 },

  title: { fontSize: 22, fontWeight: "700", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#ccc", marginBottom: 24 },

  body: { fontSize: 16, color: "#333", marginBottom: 6 },
  email: {
    fontSize: 16,
    fontWeight: "700",
    color: "#366a53",
    marginBottom: 20,
  },
  instructions: {
    fontSize: 14,
    color: "#555",
    lineHeight: 22,
    marginBottom: 32,
  },

  primaryBtn: {
    backgroundColor: "#366a53",
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: "center",
    marginBottom: 12,
  },
  primaryBtnText: { color: "#fff", fontSize: 16, fontWeight: "700" },

  secondaryBtn: {
    borderWidth: 1.5,
    borderColor: "#366a53",
    paddingVertical: 13,
    borderRadius: 8,
    alignItems: "center",
  },
  secondaryBtnText: { color: "#366a53", fontSize: 16, fontWeight: "600" },
});
