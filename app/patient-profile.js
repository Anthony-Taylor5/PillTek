import React, { useMemo } from "react";
import { View, Text, StyleSheet, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { auth } from "../firebaseConfig";
import { getLinkedCaregiverCode } from "../lib/patient-store";
import { getRole } from "../lib/role-store";

export default function PatientProfile() {
  const user = auth.currentUser;
  const displayName = user?.displayName || user?.email || "Patient";
  const email = user?.email || "—";
  const linkedCode = getLinkedCaregiverCode();
  const role = getRole(); // "patient" | "self"

  const initials = useMemo(() => {
    const parts = displayName.split(" ").filter(Boolean);
    const first = parts[0]?.[0] ?? "P";
    const last = parts.length > 1 ? parts[parts.length - 1]?.[0] : "";
    return (first + last).toUpperCase();
  }, [displayName]);

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>
        {/* Avatar */}
        <View style={styles.avatarWrap}>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>{initials}</Text>
          </View>
          <Text style={styles.name}>{displayName}</Text>
          <Text style={styles.role}>{role === "self" ? "Individual" : "Patient"}</Text>
        </View>

        {/* Info rows */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          <View style={styles.divider} />

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Email</Text>
            <Text style={styles.infoValue}>{email}</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Role</Text>
            <Text style={styles.infoValue}>
              {role === "self" ? "Individual" : "Patient"}
            </Text>
          </View>

          {role === "patient" && (
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Caregiver Code</Text>
              <Text style={styles.infoValue}>
                {linkedCode ?? "Not linked"}
              </Text>
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: { paddingHorizontal: 20, paddingTop: 24, paddingBottom: 40 },

  avatarWrap: { alignItems: "center", marginBottom: 32 },
  avatar: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 1.5,
    borderColor: "#111",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 12,
  },
  avatarText: { fontSize: 26, fontWeight: "700" },
  name: { fontSize: 22, fontWeight: "600", color: "#366a53" },
  role: { fontSize: 14, color: "#555", marginTop: 4 },

  section: { marginTop: 8 },
  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },

  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    minHeight: 52,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: "#d6ebd9",
  },
  infoLabel: { fontSize: 18, color: "#000", flex: 1 },
  infoValue: { fontSize: 14, color: "#555", marginLeft: 12, flexShrink: 0 },
});
