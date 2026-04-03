import React, { useState } from "react";
import {
  Alert,
  ImageBackground,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  Pressable,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import BG from "../assets/pills/pill8.jpg";

import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
} from "firebase/auth";
import { auth } from "../firebaseConfig";
import { getRole } from "../lib/role-store";

export default function Login() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const navigateToDashboard = () => {
    const role = getRole();
    if (role === "patient") {
      router.replace("/patient-home");
    } else if (role === "self") {
      router.replace("/self-home");
    } else {
      // Default to caregiver
      router.replace("/home");
    }
  };

  // LOGIN
  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert("Missing info", "Please enter both email and password.");
      return;
    }

    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        email.trim(),
        password
      );
      console.log("LOGIN SUCCESS:", userCredential.user.email);
      navigateToDashboard();
    } catch (error: any) {
      console.log("LOGIN ERROR:", error.code, error.message);
      Alert.alert("Login failed", error.code);
    }
  };

  // CREATE ACCOUNT
  const handleSignUp = async () => {
    if (!email || !password) {
      Alert.alert("Missing info", "Please enter both email and password.");
      return;
    }

    try {
      console.log("SIGNUP ATTEMPT:", email.trim());
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email.trim(),
        password
      );
      console.log("SIGNUP SUCCESS:", userCredential.user.email);
      navigateToDashboard();
    } catch (error: any) {
      console.log("SIGNUP ERROR:", error.code, error.message);
      Alert.alert("Sign up failed", error.code);
    }
  };

  return (
    <ImageBackground
      source={BG}
      style={styles.bg}
      imageStyle={{
        width: "100%",
        height: "100%",
        resizeMode: "cover",
        marginTop: 0,
        marginLeft: 0,
        opacity: 0.9,
      }}
    >
      <View style={styles.overlay} />

      <SafeAreaView style={styles.safe}>
        {/* Back button row — same vertical position as dashboard header rows */}
        <View style={styles.backRow}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.replace("/")}>
            <Text style={styles.backBtnText}>← Back</Text>
          </TouchableOpacity>
        </View>

        {/* Centered card */}
        <View style={styles.container}>
          <View style={styles.card}>
            {/* Title */}
            <View style={styles.headerContainer}>
              <Text style={styles.title}>PillTek</Text>
              <Text style={styles.subtitle}>
                Sign in to manage your medications and care.
              </Text>
            </View>

            {/* Email */}
            <View style={styles.fieldGroup}>
              <Text style={styles.label}>Email</Text>
              <TextInput
                style={styles.input}
                placeholder="Enter your email"
                value={email}
                onChangeText={setEmail}
                keyboardType="email-address"
                autoCapitalize="none"
              />
            </View>

            {/* Password */}
            <View style={styles.fieldGroup}>
              <Text style={styles.label}>Password</Text>
              <TextInput
                style={styles.input}
                placeholder="Enter your password"
                secureTextEntry
                value={password}
                onChangeText={setPassword}
              />
            </View>

            {/* Login Button */}
            <TouchableOpacity style={styles.loginButton} onPress={handleLogin}>
              <Text style={styles.loginText}>Login</Text>
            </TouchableOpacity>

            {/* Forgot password */}
            <Pressable
              onPress={() => router.push("/forgot-password")}
              style={{ marginTop: 12 }}
            >
              <Text style={{ textAlign: "center" }}>Forgot password?</Text>
            </Pressable>

            {/* Links */}
            <View style={styles.linksContainer}>
              <TouchableOpacity onPress={() => router.push("/create-account")}>
                <Text style={styles.linkText}>Create account</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </SafeAreaView>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  bg: {
    flex: 1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(255,255,255,0.10)",
  },
  safe: {
    flex: 1,
  },
  // Back button sits at the top in natural flow — same height as dashboard header rows
  backRow: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 4,
  },
  backBtn: {
    paddingVertical: 6,
    paddingHorizontal: 4,
    alignSelf: "flex-start",
  },
  backBtnText: {
    fontSize: 16,
    color: "#366a53ff",
    fontWeight: "600",
  },
  // Remaining space fills and centers the card
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
  },
  card: {
    width: "85%",
    alignItems: "center",
  },
  headerContainer: {
    alignItems: "center",
    marginBottom: 20,
  },
  title: {
    fontSize: 70,
    fontWeight: "bold",
    color: "#366a53ff",
    textAlign: "center",
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 16,
    textAlign: "center",
    color: "#333",
    marginTop: 8,
    marginBottom: 28,
  },
  fieldGroup: {
    width: "100%",
    marginBottom: 25,
  },
  label: {
    fontSize: 14,
    color: "#555",
    textAlign: "center",
    marginBottom: 10,
  },
  input: {
    backgroundColor: "#FFFFFF",
    height: 48,
    borderRadius: 8,
    paddingHorizontal: 14,
    elevation: 2,
  },
  loginButton: {
    marginTop: 20,
    backgroundColor: "#366a53ff",
    paddingVertical: 12,
    width: "60%",
    borderRadius: 8,
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 3,
  },
  loginText: {
    color: "#FFFFFF",
    fontSize: 17,
    fontWeight: "600",
  },
  linksContainer: {
    marginTop: 30,
    alignItems: "center",
  },
  linkText: {
    marginVertical: 6,
    fontSize: 14,
    color: "#daf1dbff",
    fontWeight: "600",
    textAlign: "center",
  },
});
