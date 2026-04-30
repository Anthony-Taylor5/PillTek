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
import { useRouter } from "expo-router";

import { useFocusEffect } from "expo-router";
import { signOut } from "firebase/auth";
import { auth } from "../firebaseConfig";
import DashboardRow from "../components/DashboardRow";
import { getPatientMedications } from "../lib/medication-store";
import { fetchPatientByUid, fetchMedications } from "../lib/api";

// Defined outside the component so the reference is stable across renders.
const MOCK_MEDICATIONS = [
  { id: "1", name: "Metformin 500mg",   time: "8:00 AM",  status: "Taken" },
  { id: "2", name: "Lisinopril 10mg",   time: "12:00 PM", status: "Taken" },
  { id: "3", name: "Atorvastatin 20mg", time: "8:00 PM",  status: "Pending" },
];

export default function PatientHome() {
  const router = useRouter();

  const patientDisplayName = useMemo(() => {
    const u = auth.currentUser;
    if (u?.displayName && u.displayName.length > 0) return u.displayName;
    if (u?.email) return u.email;
    return "Patient";
  }, []);

  const [medications, setMedications] = useState(MOCK_MEDICATIONS);
  const [currentDateStr, setCurrentDateStr] = useState(() => new Date().toDateString());

  // Re-read medications each time this screen gains focus.
  // Priority: Supabase (persisted) > session store > MOCK_MEDICATIONS.
  // Resets all statuses to Pending when the calendar date rolls over.
  useFocusEffect(
    useCallback(() => {
      const today = new Date().toDateString();
      const isNewDay = today !== currentDateStr;
      if (isNewDay) setCurrentDateStr(today);

      const uid = auth.currentUser?.uid;
      if (!uid) return;

      fetchPatientByUid(uid)
        .then((patient) => {
          if (!patient) {
            // Not linked to a patient record yet — fall back to session / mock data.
            const stored = getPatientMedications();
            const base   = stored.length > 0 ? stored : MOCK_MEDICATIONS;
            setMedications(isNewDay ? base.map((m) => ({ ...m, status: "Pending" })) : base);
            return;
          }
          return fetchMedications(patient.id).then((dbMeds) => {
            const base = dbMeds.length > 0 ? dbMeds : MOCK_MEDICATIONS;
            setMedications(isNewDay ? base.map((m) => ({ ...m, status: "Pending" })) : base);
          });
        })
        .catch((err) => {
          console.warn('[patient-home] fetch failed:', err);
          const stored = getPatientMedications();
          const base   = stored.length > 0 ? stored : MOCK_MEDICATIONS;
          setMedications(isNewDay ? base.map((m) => ({ ...m, status: "Pending" })) : base);
        });
    }, [currentDateStr])
  );

  // Today's tracking counts — derived from current medications state only.
  const takenCount     = medications.filter((m) => m.status === "Taken").length;
  const missedCount    = medications.filter((m) => m.status === "Missed").length;
  const remainingCount = medications.filter((m) => m.status === "Pending").length;

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

        {/* Today's summary cards */}
        <View style={styles.summaryRow}>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNum}>{takenCount}</Text>
            <Text style={styles.summaryLabel}>Taken</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={[styles.summaryNum, missedCount > 0 && styles.missed]}>{missedCount}</Text>
            <Text style={styles.summaryLabel}>Missed</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryNum}>{remainingCount}</Text>
            <Text style={styles.summaryLabel}>Remaining</Text>
          </View>
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
                onPress={() => goMenu("/patient-schedule")}
              >
                <Text style={styles.menuText}>My Schedule</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.menuItem}
                onPress={() => goMenu("/patient-daily-logs")}
              >
                <Text style={styles.menuText}>My Daily Log</Text>
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

  summaryRow: { flexDirection: "row", gap: 12, marginTop: 20, marginBottom: 16 },
  summaryCard: {
    flex: 1,
    backgroundColor: "#fff",
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: "center",
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
  },
  summaryNum: { fontSize: 28, fontWeight: "700", color: "#366a53" },
  summaryLabel: { fontSize: 12, color: "#555", marginTop: 2 },
  missed: { color: "#c0392b" },

  sectionTitle: {
    fontSize: 24,
    fontWeight: "600",
    marginTop: 0,
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
