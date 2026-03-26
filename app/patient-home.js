import React, { useMemo, useState } from "react";
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
import { useRouter } from "expo-router";

import { signOut } from "firebase/auth";
import { auth } from "../firebaseConfig";
import DashboardRow from "../components/DashboardRow";

export default function PatientHome() {
  const router = useRouter();

  const patientDisplayName = useMemo(() => {
    const u = auth.currentUser;
    if (u?.displayName && u.displayName.length > 0) return u.displayName;
    if (u?.email) return u.email;
    return "Patient";
  }, []);

  const [medications, setMedications] = useState([
    { id: "1", name: "Metformin 500mg", time: "8:00 AM" },
    { id: "2", name: "Lisinopril 10mg", time: "12:00 PM" },
    { id: "3", name: "Atorvastatin 20mg", time: "8:00 PM" },
  ]);

  const [menuVisible, setMenuVisible] = useState(false);

  const initials = useMemo(() => {
    const parts = patientDisplayName.split(" ").filter(Boolean);
    const first = parts[0]?.[0] ?? "P";
    const last = parts.length > 1 ? parts[parts.length - 1]?.[0] : "";
    return (first + last).toUpperCase();
  }, [patientDisplayName]);

  const onMedicationPress = (medication) => {
    router.push({ pathname: "/medication-detail", params: { id: medication.id, name: medication.name, time: medication.time } });
  };

  const goMenu = (path) => {
    setMenuVisible(false);
    router.push(path);
  };

  const handleLogout = async () => {
    setMenuVisible(false);
    await signOut(auth);
    router.replace("/");
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        {/* Header */}
        <View style={styles.headerRow}>
          <View style={styles.headerLeft}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>{initials}</Text>
            </View>
            <Text style={styles.displayName} numberOfLines={1}>
              {patientDisplayName}
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

        {/* Medication list */}
        <FlatList
          data={medications}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <DashboardRow
              label={item.name}
              value={item.time}
              onPress={() => onMedicationPress(item)}
            />
          )}
        />

        {/* 3-dot Menu Modal */}
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
                onPress={() => goMenu("/patient-medications")}
              >
                <Text style={styles.menuText}>My Medications</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.menuItem}
                onPress={() => goMenu("/patient-schedule")}
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
