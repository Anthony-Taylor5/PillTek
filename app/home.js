import React, { useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Modal,
  Pressable,
  SafeAreaView,
} from "react-native";
import { useRouter } from "expo-router";

// Firebase (you already have auth)
import { auth } from "../firebaseConfig";

export default function Home() {
  const router = useRouter();

  // ✅ This is the caregiver who logged in
  const caregiverDisplayName = useMemo(() => {
    const u = auth.currentUser;

    // If you saved displayName in Firebase auth profile:
    if (u?.displayName && u.displayName.length > 0) return u.displayName;

    // Fallback: show email if name not set yet
    if (u?.email) return u.email;

    return "Caregiver";
  }, []);

  // ✅ Patient list (for now hardcoded — later we’ll load from DB)
  const [patients, setPatients] = useState([
    { id: "1", name: "Ahmad" },
    { id: "2", name: "Shahriar" },
    { id: "3", name: "Mina" },
  ]);

  // Menu (3 dots)
  const [menuVisible, setMenuVisible] = useState(false);

  const initials = useMemo(() => {
    // Try to get initials from caregiverDisplayName
    const parts = caregiverDisplayName.split(" ").filter(Boolean);
    const first = parts[0]?.[0] ?? "C";
    const last = parts.length > 1 ? parts[parts.length - 1]?.[0] : "";
    return (first + last).toUpperCase();
  }, [caregiverDisplayName]);

  const onPatientPress = (patient) => {
    // We'll create this route later (example)
    // router.push(`/patient/${patient.id}`);
    console.log("Pressed patient:", patient);
  };

  const goMenu = (path) => {
    setMenuVisible(false);
    router.push(path);
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
            <Text style={styles.caregiverName} numberOfLines={1}>
              {caregiverDisplayName}
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

        {/* Patients label */}
        <Text style={styles.sectionTitle}>patients</Text>
        <View style={styles.divider} />

        {/* Patients list */}
        <FlatList
          data={patients}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.patientRow}
              onPress={() => onPatientPress(item)}
            >
              <Text style={styles.patientName}>{item.name}</Text>
            </TouchableOpacity>
          )}
        />

        {/* 3-dot Menu Modal */}
        <Modal
          visible={menuVisible}
          transparent
          animationType="fade"
          onRequestClose={() => setMenuVisible(false)}
        >
          {/* tap outside to close */}
          <Pressable style={styles.backdrop} onPress={() => setMenuVisible(false)}>
            <Pressable style={styles.menuCard} onPress={() => {}}>
              <TouchableOpacity
                style={styles.menuItem}
                onPress={() => goMenu("/profile")}
              >
                <Text style={styles.menuText}>Profile</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.menuItem}
                onPress={() => goMenu("/add-patient")}
              >
                <Text style={styles.menuText}>Add Patient</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.menuItem, { borderBottomWidth: 0 }]}
                onPress={() => goMenu("/logs")}
              >
                <Text style={styles.menuText}>Logs</Text>
              </TouchableOpacity>
            </Pressable>
          </Pressable>
        </Modal>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#fff" },
  container: 
  { flex: 1, 
    paddingHorizontal: 16, 
    paddingTop: 12,
    backgroundColor: "#e8f5e9" },

  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerLeft: { flexDirection: "row", alignItems: "center", gap: 10, flex: 1 },

  avatar: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "#111",
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: { fontWeight: "700" },

  caregiverName: { 
    fontSize: 20,
    fontWeight: "600",
    flexShrink: 1,
    color: "#366a53",
  } ,

  menuBtn: { paddingHorizontal: 8, paddingVertical: 6 },
  menuIcon: { fontSize: 22, fontWeight: "700" },

  sectionTitle: { marginTop: 18, fontSize: 18, fontWeight: "500" },
  divider: { height: 1, backgroundColor: "#111", marginTop: 10, marginBottom: 6 },

  patientRow: { paddingVertical: 14 },
  patientName: { fontSize: 18 },

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
});