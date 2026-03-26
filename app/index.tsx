import { Picker } from "@react-native-picker/picker";
import React, { useState } from "react";
import { useRouter } from "expo-router";
import {
  Alert,
  ImageBackground,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  Pressable,
  View,
} from "react-native";

import BG from "../assets/pills/pill9.jpg";

import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
} from "firebase/auth";
import { auth } from "../firebaseConfig";

export default function Index() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("");

  const handleLogin = async () => {
    if (!role) {
      Alert.alert("Missing role", "Please choose a role.");
      return;
    }

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
      router.replace("/home");
    } catch (error: any) {
      Alert.alert("Login failed", error.code);
    }
  };

  const handleSignUp = async () => {
    if (!role) {
      Alert.alert("Missing role", "Please choose a role.");
      return;
    }

    if (!email || !password) {
      Alert.alert("Missing info", "Please enter both email and password.");
      return;
    }

    try {
      await createUserWithEmailAndPassword(
        auth,
        email.trim(),
        password
      );
      router.replace("/create-account");
    } catch (error: any) {
      Alert.alert("Sign up failed", error.code);
    }
  };

  return (
    <ImageBackground
      source={BG}
      style={styles.background}
      resizeMode="contain"   // ✅ SHOW FULL IMAGE
    >
      {/* ✅ LIGHT GREEN BACKGROUND + SOFT OVERLAY */}
      <View style={styles.overlay} />

      <SafeAreaView style={styles.container}>
        <View style={styles.card}>
          <Text style={styles.title}>PillTek</Text>

          {/* ROLE */}
          <View style={styles.fieldGroup}>
            <Text style={styles.label}>Role</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={role}
                onValueChange={(itemValue) => setRole(itemValue)}
                style={styles.picker}
              >
                <Picker.Item label="Choose a role" value="" />
                <Picker.Item label="Caregiver" value="caregiver" />
                <Picker.Item label="Patient" value="patient" />
              </Picker>
            </View>
          </View>

          {/* EMAIL */}
          <View style={styles.fieldGroup}>
            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="Enter your email"
              placeholderTextColor="#666"
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
            />
          </View>

          {/* PASSWORD */}
          <View style={styles.fieldGroup}>
            <Text style={styles.label}>Password</Text>
            <TextInput
              style={styles.input}
              placeholder="Enter your password"
              placeholderTextColor="#666"
              secureTextEntry
              value={password}
              onChangeText={setPassword}
            />
          </View>

          {/* LOGIN */}
          <TouchableOpacity style={styles.loginButton} onPress={handleLogin}>
            <Text style={styles.loginText}>Login</Text>
          </TouchableOpacity>

          <Pressable
            onPress={() => router.push("/forgot-password")}
            style={{ marginTop: 14 }}
          >
            <Text style={styles.forgotText}>Forgot password?</Text>
          </Pressable>

          <View style={styles.linksContainer}>
            <TouchableOpacity onPress={() => router.push("/create-account")}>
              <Text style={styles.linkText}>Create account</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    backgroundColor: "#E8F3ED", // ✅ LIGHT GREEN behind image
    justifyContent: "center",
  },

  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(255, 255, 255, 0.73)", // ✅ soft light tint
  },

  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 28,
    paddingTop: -40,
    zIndex: 1,
  },

  card: {
    width: "85%",
    alignItems: "center",
  },

  title: {
  fontSize: 72,
  fontWeight: "900", // stronger than bold
  color: "#1F4D3A", // darker green
  marginBottom: 20,
  letterSpacing: 1, // sharper look
  },

  fieldGroup: {
    width: "100%",
    marginBottom: 22,
  },

  label: {
    color: "#000",
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 8,
  },

  pickerContainer: {
    backgroundColor: "#FFFFFF",
    borderRadius: 8,
    height: 48,
    justifyContent: "center",
    elevation: 2,
    overflow: "hidden",
  },

  picker: {
    height: 48,
    width: "100%",
    color: "#444",
  },

  input: {
    backgroundColor: "#FFFFFF",
    height: 48,
    borderRadius: 8,
    paddingHorizontal: 14,
    elevation: 2,
  },

  loginButton: {
    marginTop: 18,
    backgroundColor: "#366a53",
    paddingVertical: 12,
    width: "60%",
    borderRadius: 8,
    alignItems: "center",
    elevation: 3,
  },

  loginText: {
    color: "#FFFFFF",
    fontSize: 17,
    fontWeight: "600",
  },

  forgotText: {
  textAlign: "center",
  fontSize: 16,        // bigger
  color: "#1F4D3A",    // darker green (matches your theme)
  fontWeight: "700",   // bold but not too heavy
  },

  linksContainer: {
    marginTop: 26,
    alignItems: "center",
  },

 linkText: {
  fontSize: 16,        // same size as forgotText
  color: "#1F4D3A",    // darker green (same tone)
  fontWeight: "700",   // bold
  },
});