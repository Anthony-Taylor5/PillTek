import React from "react";
import {
  ImageBackground,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import BG from "../assets/pills/pill8.jpg";
import { setRole } from "./role-store";

export default function RoleSelect() {
  const router = useRouter();

  const handleRoleSelect = (role: "caregiver" | "patient") => {
    setRole(role);
    router.replace("/login");
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

      <SafeAreaView style={styles.container}>
        <View style={styles.card}>
          {/* Title */}
          <View style={styles.headerContainer}>
            <Text style={styles.title}>PillTek</Text>
            <Text style={styles.subtitle}>
              How will you be using the app?
            </Text>
          </View>

          {/* Caregiver button */}
          <TouchableOpacity
            style={styles.roleButton}
            onPress={() => handleRoleSelect("caregiver")}
          >
            <Text style={styles.roleButtonText}>I am a Caregiver</Text>
            <Text style={styles.roleButtonSub}>
              Monitor and manage your patients
            </Text>
          </TouchableOpacity>

          {/* Patient button */}
          <TouchableOpacity
            style={[styles.roleButton, styles.roleButtonSecondary]}
            onPress={() => handleRoleSelect("patient")}
          >
            <Text style={[styles.roleButtonText, styles.roleButtonTextSecondary]}>
              I am a Patient
            </Text>
            <Text style={[styles.roleButtonSub, styles.roleButtonSubSecondary]}>
              View your medications and schedule
            </Text>
          </TouchableOpacity>
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
    marginBottom: 40,
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
    marginBottom: 10,
  },
  roleButton: {
    marginTop: 16,
    backgroundColor: "#366a53ff",
    paddingVertical: 16,
    paddingHorizontal: 24,
    width: "100%",
    borderRadius: 8,
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 3,
  },
  roleButtonSecondary: {
    backgroundColor: "#FFFFFF",
    borderWidth: 2,
    borderColor: "#366a53ff",
  },
  roleButtonText: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "700",
  },
  roleButtonTextSecondary: {
    color: "#366a53ff",
  },
  roleButtonSub: {
    color: "rgba(255,255,255,0.85)",
    fontSize: 13,
    marginTop: 4,
  },
  roleButtonSubSecondary: {
    color: "#555",
  },
});
