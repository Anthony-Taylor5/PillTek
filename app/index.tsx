import React, { useState } from "react";
import { router } from "expo-router";
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

import { useRouter } from "expo-router";
import BG from "../assets/pills/pill8.jpg";

// 🔐 Firebase imports
import {
  createUserWithEmailAndPassword,
  sendPasswordResetEmail,
  signInWithEmailAndPassword,
} from "firebase/auth";
import { auth } from "../firebaseConfig";

export default function Index() {
  const router = useRouter(); // <-- MUST be inside component

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

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

      router.replace("/home");
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

      router.replace("/(tabs)/patients");
    } catch (error: any) {
      console.log("SIGNUP ERROR:", error.code, error.message);
      Alert.alert("Sign up failed", error.code);
    }
  };

  // FORGOT PASSWORD
    

  return (
    <ImageBackground
      source={BG}
      style={styles.bg}
      imageStyle={{
        width: "100%",
        height: "100%",
        resizeMode: "cover",
        marginTop: 0,   // move image DOWN
        marginLeft: 0,   // move image to the RIGHT
        opacity: 0.9,
      }}
    >
      <View style={styles.overlay} />

      <SafeAreaView style={styles.container}>
        <View style={styles.card}>
          {/* Title */}
          <View style={styles.headerContainer}>
            <Text style={styles.title}>PillTek</Text>
            <Text style={styles.subtitle}>
              Sign in as caregiver to monitor your patients.
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
