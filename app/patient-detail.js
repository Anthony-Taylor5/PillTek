import React, { useState, useCallback } from "react";
import { View, Text, FlatList, StyleSheet, TouchableOpacity, Modal, Pressable, Alert, ActivityIndicator } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useLocalSearchParams, useRouter, useFocusEffect } from "expo-router";
import DashboardRow from "../components/DashboardRow";
import { getCaregiverPatientMeds } from "../lib/medication-store";
import { removePatient } from "../lib/patient-store";
import { fetchMedications } from "../lib/api";

// Mock per-patient medication data for the three pre-seeded patients.
const PATIENT_MEDS = {
  "1": [ // Ahmad
    { id: "m1", name: "Metformin 500mg", time: "8:00 AM", status: "Taken" },
    { id: "m2", name: "Aspirin 81mg",    time: "8:00 PM", status: "Pending" },
  ],
  "2": [ // Shahriar
    { id: "m1", name: "Lisinopril 10mg",   time: "12:00 PM", status: "Taken" },
    { id: "m2", name: "Atorvastatin 20mg", time: "8:00 PM",  status: "Taken" },
  ],
  "3": [ // Mina
    { id: "m1", name: "Metformin 500mg",   time: "8:00 AM",  status: "Missed" },
    { id: "m2", name: "Vitamin D 1000IU",  time: "12:00 PM", status: "Taken" },
    { id: "m3", name: "Atorvastatin 20mg", time: "8:00 PM",  status: "Pending" },
  ],
};

// Merges mock meds (if any) with any medications added through the
// add-medication flow. Store-added meds replace mock entries with the same name.
function buildMedList(id, name) {
  const mock    = PATIENT_MEDS[id] ?? [];
  const stored  = getCaregiverPatientMeds(name) ?? [];
  // Keep mock entries not superseded by a stored entry of the same name
  const filtered = mock.filter((m) => !stored.some((s) => s.name === m.name));
  return [...filtered, ...stored];
}

export default function PatientDetail() {
  const { id, name, patientCode } = useLocalSearchParams();
  const router = useRouter();

  const [meds, setMeds]             = useState(() => buildMedList(id, name));
  const [menuVisible, setMenuVisible] = useState(false);
  const [loading, setLoading]         = useState(false);

  // Refresh the medication list each time this screen gains focus.
  // If the patient id is a UUID (created via Supabase), fetch from DB.
  // Otherwise fall back to the mock / session-store data for pre-seeded patients.
  useFocusEffect(
    useCallback(() => {
      // UUIDs are 36 chars with hyphens; integer IDs ("1","2","3") are not.
      const isUuid = String(id).includes('-');
      if (!isUuid) {
        setMeds(buildMedList(id, name));
        return;
      }

      setLoading(true);
      fetchMedications(id)
        .then((dbMeds) => {
          if (dbMeds.length > 0) {
            setMeds(dbMeds);
          } else {
            // No DB records yet — show session-store meds (e.g., just added this session).
            const sessionMeds = getCaregiverPatientMeds(name) ?? [];
            setMeds(sessionMeds.length > 0 ? sessionMeds : buildMedList(id, name));
          }
        })
        .catch((err) => {
          console.warn('[patient-detail] fetchMedications failed:', err);
          setMeds(buildMedList(id, name));
        })
        .finally(() => setLoading(false));
    }, [id, name])
  );

  const takenCount  = meds.filter((m) => m.status === "Taken").length;
  const missedCount = meds.filter((m) => m.status === "Missed").length;

  const goMenu = (path, params) => {
    setMenuVisible(false);
    router.push(params ? { pathname: path, params } : path);
  };

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <View style={styles.container}>

        {/* Patient identity + 3-dot menu */}
        <View style={styles.headerRow}>
          <View style={styles.patientLeft}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>{String(name)?.[0]?.toUpperCase() ?? "P"}</Text>
            </View>
            <View>
              <Text style={styles.patientName}>{name}</Text>
              {!!patientCode && (
                <Text style={styles.patientCode}>Code: {patientCode}</Text>
              )}
            </View>
          </View>

          <TouchableOpacity
            onPress={() => setMenuVisible(true)}
            hitSlop={10}
            style={styles.menuBtn}
          >
            <Text style={styles.menuIcon}>⋮</Text>
          </TouchableOpacity>
        </View>

        {/* Summary row */}
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
            <Text style={styles.summaryNum}>{meds.length}</Text>
            <Text style={styles.summaryLabel}>Total</Text>
          </View>
        </View>

        {/* Medications */}
        <Text style={styles.sectionTitle}>medications</Text>
        <View style={styles.divider} />
        {loading && <ActivityIndicator color="#366a53" style={{ marginTop: 16 }} />}
        <FlatList
          data={meds}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <DashboardRow
              label={item.name}
              value={`${item.time} · ${item.status}`}
              onPress={() =>
                router.push({
                  pathname: "/med-log",
                  params: {
                    patientId:   id,
                    patientName: name,
                    medId:       item.id,
                    medName:     item.name,
                    medTime:     item.time,
                  },
                })
              }
            />
          )}
        />

        {/* 3-dot menu modal — same style as home.js */}
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
                onPress={() =>
                  goMenu("/add-medication", {
                    patientId:   id,
                    patientName: name,
                    patientCode: patientCode ?? "",
                  })
                }
              >
                <Text style={styles.menuText}>Add Medication</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.menuItem}
                onPress={() =>
                  goMenu("/patient-log-view", {
                    patientName: name,
                  })
                }
              >
                <Text style={styles.menuText}>View Logs</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.menuItem, { borderBottomWidth: 0 }]}
                onPress={() => {
                  setMenuVisible(false);
                  Alert.alert(
                    "Remove Patient",
                    `Are you sure you want to remove ${name}? This cannot be undone.`,
                    [
                      { text: "Cancel", style: "cancel" },
                      {
                        text: "Remove",
                        style: "destructive",
                        onPress: () => {
                          removePatient(id);
                          router.replace("/home");
                        },
                      },
                    ]
                  );
                }}
              >
                <Text style={[styles.menuText, styles.removeText]}>Remove Patient</Text>
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
  container: { flex: 1, paddingHorizontal: 20, paddingTop: 20 },

  // Header row: patient identity on left, ⋮ on right
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 20,
  },
  patientLeft: { flexDirection: "row", alignItems: "center", gap: 14, flex: 1 },
  avatar: {
    width: 52,
    height: 52,
    borderRadius: 26,
    borderWidth: 1.5,
    borderColor: "#111",
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: { fontSize: 20, fontWeight: "700" },
  patientName: { fontSize: 22, fontWeight: "600", color: "#366a53" },
  patientCode: { fontSize: 12, color: "#777", marginTop: 2, letterSpacing: 0.5 },

  menuBtn: { paddingHorizontal: 8, paddingVertical: 6 },
  menuIcon: { fontSize: 22, fontWeight: "700" },

  summaryRow: { flexDirection: "row", gap: 12, marginBottom: 24 },
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

  sectionTitle: { fontSize: 24, fontWeight: "600", marginBottom: 10 },
  divider: { borderBottomWidth: 1, borderBottomColor: "#666", marginBottom: 0 },

  // Modal — same spec as home.js
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
  removeText: { color: "#c0392b" },
});
