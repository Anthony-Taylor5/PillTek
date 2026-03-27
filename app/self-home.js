import React, { useMemo, useState, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Modal,
  Pressable,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useFocusEffect } from "expo-router";

import { signOut } from "firebase/auth";
import { auth } from "../firebaseConfig";
import DashboardRow from "../components/DashboardRow";
import { getCaregiverPatientMeds } from "./medication-store";

// Self-managed home — mirrors the caregiver home (home.js) exactly in structure
// and interaction style, but manages the user's own medications instead of a
// patient list. "Add Patient" is intentionally absent from the menu.

export default function SelfHome() {
  const router = useRouter();

  const displayName = useMemo(() => {
    const u = auth.currentUser;
    if (u?.displayName && u.displayName.length > 0) return u.displayName;
    if (u?.email) return u.email;
    return "Self";
  }, []);

  // Medications are stored under the user's display name key in the same
  // caregiver-patient med store that add-medication.js writes to.
  const [medications, setMedications] = useState(
    () => getCaregiverPatientMeds(displayName) ?? []
  );

  useFocusEffect(
    useCallback(() => {
      setMedications(getCaregiverPatientMeds(displayName) ?? []);
    }, [displayName])
  );

  const [menuVisible, setMenuVisible] = useState(false);

  const initials = useMemo(() => {
    const parts = displayName.split(" ").filter(Boolean);
    const first = parts[0]?.[0] ?? "S";
    const last = parts.length > 1 ? parts[parts.length - 1]?.[0] : "";
    return (first + last).toUpperCase();
  }, [displayName]);

  const goMenu = (path, params) => {
    setMenuVisible(false);
    router.push(params ? { pathname: path, params } : path);
  };

  const handleLogout = async () => {
    setMenuVisible(false);
    await signOut(auth);
    router.replace("/");
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        {/* Header — identical structure to home.js */}
        <View style={styles.headerRow}>
          <View style={styles.headerLeft}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>{initials}</Text>
            </View>
            <Text style={styles.displayName} numberOfLines={1}>
              {displayName}
            </Text>
          </View>

          <TouchableOpacity
            onPress={() => setMenuVisible(true)}
            hitSlop={10}
            style={styles.menuBtn}
          >
            <Text style={styles.menuIcon}>⋮</Text>
          </TouchableOpacity>
        </View>

        {/* Section label + divider */}
        <Text style={styles.sectionTitle}>medications</Text>
        <View style={styles.divider} />

        {/* Medication list — empty-state hint when nothing added yet */}
        {medications.length === 0 ? (
          <Text style={styles.emptyText}>
            No medications yet. Tap ⋮ → Add Medication to get started.
          </Text>
        ) : (
          <FlatList
            data={medications}
            keyExtractor={(item, index) => item.id ?? String(index)}
            renderItem={({ item }) => (
              <DashboardRow
                label={item.name}
                value={item.time ?? "—"}
              />
            )}
          />
        )}

        {/* 3-dot Menu Modal — same spec as home.js, no "Add Patient" */}
        <Modal
          visible={menuVisible}
          transparent
          animationType="fade"
          onRequestClose={() => setMenuVisible(false)}
        >
          <Pressable style={styles.backdrop} onPress={() => setMenuVisible(false)}>
            <Pressable style={styles.menuCard} onPress={() => {}}>
              <TouchableOpacity
                style={styles.menuItem}
                onPress={() => goMenu("/patient-profile")}
              >
                <Text style={styles.menuText}>Profile</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.menuItem}
                onPress={() =>
                  goMenu("/add-medication", {
                    patientId:   "self",
                    patientName: displayName,
                    patientCode: "",
                  })
                }
              >
                <Text style={styles.menuText}>Add Medication</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.menuItem}
                onPress={() => goMenu("/self-logs")}
              >
                <Text style={styles.menuText}>Logs</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.menuItem}
                onPress={() =>
                  goMenu("/patient-schedule", { selfName: displayName })
                }
              >
                <Text style={styles.menuText}>My Schedule</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.menuItem, { borderBottomWidth: 0 }]}
                onPress={handleLogout}
              >
                <Text style={[styles.menuText, styles.logoutText]}>Log out</Text>
              </TouchableOpacity>
            </Pressable>
          </Pressable>
        </Modal>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: {
    flex: 1,
    backgroundColor: "#e8f5e9",
    paddingHorizontal: 20,
    paddingTop: 20,
  },

  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 4,
  },
  headerLeft: { flexDirection: "row", alignItems: "center", gap: 10, flex: 1 },

  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "#111",
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: { fontWeight: "700", fontSize: 15 },

  displayName: {
    fontSize: 20,
    fontWeight: "600",
    flexShrink: 1,
    color: "#366a53",
  },

  menuBtn: { paddingHorizontal: 8, paddingVertical: 6 },
  menuIcon: { fontSize: 22, fontWeight: "700" },

  sectionTitle: {
    fontSize: 24,
    fontWeight: "600",
    marginTop: 24,
    marginBottom: 10,
  },
  divider: {
    borderBottomWidth: 1,
    borderBottomColor: "#666",
    marginBottom: 0,
  },

  emptyText: {
    marginTop: 32,
    fontSize: 15,
    color: "#888",
    textAlign: "center",
    lineHeight: 22,
  },

  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.15)",
    alignItems: "flex-end",
    justifyContent: "flex-start",
    paddingTop: 60,
    paddingRight: 14,
  },
  menuCard: {
    width: 180,
    backgroundColor: "#fff",
    borderRadius: 10,
    overflow: "hidden",
    elevation: 6,
  },
  menuItem: {
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderBottomWidth: 1,
    borderBottomColor: "#eee",
  },
  menuText: { fontSize: 16 },
  logoutText: { color: "#c0392b" },
});
