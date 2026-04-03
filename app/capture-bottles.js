import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { auth } from "../firebaseConfig";
import {
  setPatientMedications,
  setCaregiverPatientMeds,
  nameToMedObject,
} from "../lib/medication-store";
import { createMedications, fetchPatientByUid } from "../lib/api";

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5000";
const TOTAL_CAPTURES = 24;
const POLL_INTERVAL_MS = 1500;

function sanitize(str) {
  return String(str)
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_]/g, "");
}

function getMedName(med) {
  return typeof med === "object" && med !== null ? (med.name ?? "") : String(med ?? "");
}

export default function CaptureBottles() {
  const router = useRouter();
  const { patientId, patientName, medications: medsParam, returnTo, returnParams, returnMode } = useLocalSearchParams();
  const medications = JSON.parse(medsParam || "[]");

  const [medIndex, setMedIndex]       = useState(0);
  const [sessionId, setSessionId]     = useState(null);
  const [capturesDone, setCapturesDone] = useState(0);
  const [sessionStatus, setSessionStatus] = useState("idle"); // idle | starting | running | done | error
  const pollRef = useRef(null);

  const currentMedName = getMedName(medications[medIndex] ?? "");
  const isLastMed      = medIndex === medications.length - 1;
  const progress       = Math.min(capturesDone / TOTAL_CAPTURES, 1);

  // Stop polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const startCapture = async () => {
    setSessionStatus("starting");
    setCapturesDone(0);
    setSessionId(null);

    const className = `${sanitize(patientName || auth.currentUser?.displayName || "patient")}_${sanitize(currentMedName)}`;

    try {
      const res  = await fetch(`${BACKEND_URL}/start-capture`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ class_name: className }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Failed to start capture");

      setSessionId(data.session_id);
      setSessionStatus("running");

      pollRef.current = setInterval(async () => {
        try {
          const r    = await fetch(`${BACKEND_URL}/capture-status/${data.session_id}`);
          const s    = await r.json();
          setCapturesDone(s.captures_done ?? 0);
          if (s.status === "done" || s.status === "error") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setSessionStatus(s.status);
          }
        } catch {
          // network blip — keep polling
        }
      }, POLL_INTERVAL_MS);
    } catch (err) {
      setSessionStatus("error");
      Alert.alert("Error", err.message ?? "Could not start capture session.");
    }
  };

  const handleNext = () => {
    if (!isLastMed) {
      setMedIndex((i) => i + 1);
      setSessionStatus("idle");
      setCapturesDone(0);
      setSessionId(null);
    } else {
      finishCapture();
    }
  };

  const finishCapture = async () => {
    const medObjects = medications.map((med, i) => {
      if (typeof med === "object" && med !== null) {
        return {
          id:        `setup_${i}`,
          name:      med.name      ?? "",
          dosage:    med.dosage    ?? "—",
          frequency: med.frequency ?? "—",
          times:     med.times     ?? [],
          time:      med.time      ?? "—",
          refill:    med.refill    ?? "—",
          addedAt:   med.addedAt   ?? new Date().toISOString(),
          status:    med.status    ?? "Pending",
        };
      }
      return nameToMedObject(med, i);
    });

    if (returnTo === "/patient-home") {
      setPatientMedications(medObjects);
    } else {
      setCaregiverPatientMeds(patientName, medObjects);
    }

    // Persist medications to Supabase (photos are managed by the Python training pipeline)
    let patientUuid = null;
    const uid = auth.currentUser?.uid ?? null;

    try {
      if (returnTo === "/patient-home" && uid) {
        const rec = await fetchPatientByUid(uid);
        patientUuid = rec?.id ?? null;
      } else if (patientId && String(patientId).includes("-")) {
        patientUuid = patientId;
      }

      if (patientUuid) {
        await createMedications(patientUuid, medObjects);
        console.log("[CaptureBottles] Medications saved to Supabase.");
      } else {
        console.warn("[CaptureBottles] No patient UUID — session store only.");
      }
    } catch (dbErr) {
      console.warn("[CaptureBottles] Supabase persist failed:", dbErr);
    }

    Alert.alert(
      "Setup complete",
      `Bottle capture complete for ${medications.length} medication${medications.length !== 1 ? "s" : ""}.`,
      [{ text: "Done", onPress: () => {
        if (returnMode === "back") {
          router.back();
        } else if (returnParams) {
          router.replace({ pathname: returnTo || "/home", params: JSON.parse(returnParams) });
        } else {
          router.replace(returnTo || "/home");
        }
      }}]
    );
  };

  if (medications.length === 0) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.centered}>
          <Text style={styles.bodyText}>No medications provided.</Text>
          <TouchableOpacity style={styles.btn} onPress={() => router.back()}>
            <Text style={styles.btnText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const captureReady  = sessionStatus === "idle" || sessionStatus === "error";
  const captureActive = sessionStatus === "starting" || sessionStatus === "running";
  const captureDone   = sessionStatus === "done";

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>

        {/* Step indicator */}
        <Text style={styles.stepLabel}>
          Medication {medIndex + 1} of {medications.length}
        </Text>

        {/* Medication name card */}
        <View style={styles.medCard}>
          <Text style={styles.medName} numberOfLines={2}>{currentMedName}</Text>
        </View>

        {/* Instructions */}
        <View style={styles.instructionBox}>
          <Text style={styles.instructionTitle}>Desktop Capture Required</Text>
          <Text style={styles.instructionText}>
            Bottle photos are taken using the ESP32 camera and the desktop capture tool.
            Press <Text style={styles.bold}>Start Capture</Text> below, then complete the
            session on the desktop window that opens.
          </Text>
          <Text style={styles.instructionText}>
            Rotate the bottle every 15° and press <Text style={styles.bold}>SPACE</Text> to
            capture each of the {TOTAL_CAPTURES} required photos.
          </Text>
        </View>

        {/* Progress bar */}
        {(captureActive || captureDone) && (
          <View style={styles.progressSection}>
            <View style={styles.progressTrack}>
              <View style={[styles.progressFill, { width: `${progress * 100}%` }]} />
            </View>
            <Text style={styles.progressText}>
              {captureDone
                ? `All ${TOTAL_CAPTURES} photos captured`
                : `${capturesDone} / ${TOTAL_CAPTURES} photos`}
            </Text>
          </View>
        )}

        {/* Error message */}
        {sessionStatus === "error" && (
          <Text style={styles.errorText}>
            Capture failed. Make sure the backend server is running and try again.
          </Text>
        )}

        {/* Action buttons */}
        <View style={styles.actions}>
          {captureReady && (
            <TouchableOpacity style={styles.btn} onPress={startCapture}>
              <Text style={styles.btnText}>Start Capture</Text>
            </TouchableOpacity>
          )}

          {captureActive && (
            <View style={styles.waitingRow}>
              <ActivityIndicator color="#366a53" size="small" />
              <Text style={styles.waitingText}>Waiting for desktop capture…</Text>
            </View>
          )}

          {captureDone && (
            <TouchableOpacity style={styles.btn} onPress={handleNext}>
              <Text style={styles.btnText}>
                {isLastMed ? "Finish" : "Next Medication"}
              </Text>
            </TouchableOpacity>
          )}
        </View>

      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:      { flex: 1, backgroundColor: "#e8f5e9" },
  container: { flex: 1, padding: 24, alignItems: "center", justifyContent: "center" },

  stepLabel: {
    fontSize: 13,
    color: "#666",
    marginBottom: 12,
    letterSpacing: 0.5,
  },

  medCard: {
    backgroundColor: "#fff",
    borderRadius: 12,
    paddingVertical: 18,
    paddingHorizontal: 24,
    width: "100%",
    alignItems: "center",
    marginBottom: 24,
    shadowColor: "#000",
    shadowOpacity: 0.07,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  medName: {
    fontSize: 22,
    fontWeight: "700",
    color: "#1a1a1a",
    textAlign: "center",
  },

  instructionBox: {
    backgroundColor: "#f0faf4",
    borderRadius: 10,
    padding: 16,
    width: "100%",
    marginBottom: 28,
    borderLeftWidth: 3,
    borderLeftColor: "#366a53",
  },
  instructionTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: "#366a53",
    marginBottom: 8,
  },
  instructionText: {
    fontSize: 13,
    color: "#444",
    lineHeight: 20,
    marginBottom: 6,
  },
  bold: { fontWeight: "700" },

  progressSection: {
    width: "100%",
    marginBottom: 24,
    alignItems: "center",
  },
  progressTrack: {
    width: "100%",
    height: 8,
    backgroundColor: "#d0e8d8",
    borderRadius: 4,
    overflow: "hidden",
    marginBottom: 8,
  },
  progressFill: {
    height: "100%",
    backgroundColor: "#366a53",
    borderRadius: 4,
  },
  progressText: {
    fontSize: 13,
    color: "#555",
  },

  errorText: {
    fontSize: 13,
    color: "#c0392b",
    textAlign: "center",
    marginBottom: 16,
  },

  actions: {
    width: "100%",
    alignItems: "center",
  },
  btn: {
    backgroundColor: "#366a53",
    paddingVertical: 14,
    paddingHorizontal: 40,
    borderRadius: 8,
  },
  btnText: { color: "#fff", fontSize: 16, fontWeight: "600" },

  waitingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  waitingText: {
    fontSize: 14,
    color: "#555",
  },

  centered: { flex: 1, alignItems: "center", justifyContent: "center", padding: 28 },
  bodyText: {
    fontSize: 16,
    color: "#444",
    textAlign: "center",
    lineHeight: 22,
    marginBottom: 24,
  },
});
