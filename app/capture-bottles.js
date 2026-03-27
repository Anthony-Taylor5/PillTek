import React, { useState, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Image,
  ScrollView,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useRouter, useLocalSearchParams } from "expo-router";
import { auth } from "../firebaseConfig";
import {
  setPatientMedications,
  setCaregiverPatientMeds,
  nameToMedObject,
} from "./medication-store";

const PHOTOS_PER_MED = 6;

// Produces a safe filename segment: lowercase, spaces→underscores, strip non-alphanumeric
function sanitize(str) {
  return String(str)
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_]/g, "");
}

// Extracts the display name from a medication entry that may be either a plain
// string (patient self-setup flow) or a full object (caregiver add-patient flow).
function getMedName(med) {
  return typeof med === "object" && med !== null ? (med.name ?? "") : String(med ?? "");
}

export default function CaptureBottles() {
  const router = useRouter();
  const { patientName, medications: medsParam, returnTo, returnParams, returnMode } = useLocalSearchParams();
  const medications = JSON.parse(medsParam || "[]");

  const [permission, requestPermission] = useCameraPermissions();
  const [medIndex, setMedIndex] = useState(0);
  // Keyed by medication name string (not the full object)
  const [capturedPhotos, setCapturedPhotos] = useState({});
  const [capturing, setCapturing] = useState(false);
  const cameraRef = useRef(null);

  const usernameSafe = sanitize(patientName || auth.currentUser?.displayName || "patient");
  const currentMedName = getMedName(medications[medIndex] ?? "");
  const currentMedPhotos = capturedPhotos[currentMedName] ?? [];
  const photoCount = currentMedPhotos.length;
  const allPhotosDone = photoCount >= PHOTOS_PER_MED;
  const isLastMed = medIndex === medications.length - 1;

  const handleTakePhoto = async () => {
    if (!cameraRef.current || allPhotosDone || capturing) return;
    setCapturing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        skipProcessing: true,
      });
      const medSafe = sanitize(currentMedName);
      const index = photoCount + 1;
      // Filename pattern: username_medicationname_# (no extension stored — URI carries that)
      const filename = `${usernameSafe}_${medSafe}_${index}`;
      setCapturedPhotos((prev) => ({
        ...prev,
        [currentMedName]: [...(prev[currentMedName] ?? []), { filename, uri: photo.uri }],
      }));
    } catch {
      Alert.alert("Error", "Could not capture photo. Please try again.");
    } finally {
      setCapturing(false);
    }
  };

  const handleNext = () => {
    if (!isLastMed) {
      setMedIndex((i) => i + 1);
    } else {
      finishCapture();
    }
  };

  const finishCapture = () => {
    // Build medication objects in the shape the rest of the app uses.
    // If the caregiver flow passed full objects (name + dosage + frequency + time + refill),
    // preserve those fields. Otherwise create minimal objects from plain name strings
    // (patient self-setup path via medication-entry.js).
    const medObjects = medications.map((med, i) => {
      if (typeof med === "object" && med !== null) {
        // Full object from caregiver flow — preserve all fields including
        // times array and addedAt so the calendar can use them.
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
      // Plain name string from patient flow
      return nameToMedObject(med, i);
    });

    // Write into the shared store so medication screens update on next focus
    if (returnTo === "/patient-home") {
      // Patient self-setup flow: update the patient's own medication list
      setPatientMedications(medObjects);
    } else {
      // Caregiver add-patient flow: store under the patient's name so
      // patient-detail can look it up when the ID-based mock lookup misses
      setCaregiverPatientMeds(patientName, medObjects);
    }

    // Package all data cleanly for database submission
    const payload = {
      patientName,
      userId: auth.currentUser?.uid ?? null,
      capturedAt: new Date().toISOString(),
      medications: medications.map((med) => {
        const name = getMedName(med);
        return {
          name,
          ...(typeof med === "object" && med !== null
            ? { dosage: med.dosage, frequency: med.frequency, time: med.time, refill: med.refill }
            : {}),
          photos: (capturedPhotos[name] ?? []).map(({ filename, uri }) => ({
            filename,    // e.g. "ahmad_metformin_1"
            localUri: uri, // upload to Firebase Storage / S3 here
          })),
        };
      }),
    };

    // TODO: Upload localUri files to Firebase Storage and save payload to Firestore
    console.log("[CaptureBottles] Payload ready for DB:", JSON.stringify(payload, null, 2));

    Alert.alert(
      "Setup complete",
      `Bottle photos captured for ${medications.length} medication${medications.length !== 1 ? "s" : ""}.`,
      [{ text: "Done", onPress: () => {
        if (returnMode === "back") {
          // add-medication used router.replace to get here, so the stack is
          // [... , patient-detail, capture-bottles]. router.back() pops to patient-detail
          // cleanly without leaving add-medication in the history.
          router.back();
        } else if (returnParams) {
          router.replace({ pathname: returnTo || "/home", params: JSON.parse(returnParams) });
        } else {
          router.replace(returnTo || "/home");
        }
      }}]
    );
  };

  // No medications passed
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

  // Permissions not yet resolved
  if (!permission) return <View style={styles.safe} />;

  // Permission denied
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.safe}>
        <View style={styles.centered}>
          <Text style={styles.bodyText}>
            Camera access is required to photograph your medication bottles.
          </Text>
          <TouchableOpacity style={styles.btn} onPress={requestPermission}>
            <Text style={styles.btnText}>Allow Camera</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.cameraContainer}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back" />

      {/* Top bar — medication name + step counter */}
      <SafeAreaView style={styles.topOverlay} edges={["top"]}>
        <View style={styles.topBar}>
          <Text style={styles.medName} numberOfLines={1}>
            {currentMedName}
          </Text>
          <Text style={styles.stepLabel}>
            Medication {medIndex + 1} of {medications.length}
          </Text>
        </View>
      </SafeAreaView>

      {/* Bottom controls */}
      <View style={styles.bottomOverlay}>
        {/* Thumbnail strip — one slot per required photo */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.thumbnailRow}
        >
          {Array.from({ length: PHOTOS_PER_MED }).map((_, i) => {
            const photo = currentMedPhotos[i];
            return (
              <View
                key={i}
                style={[styles.thumb, photo ? styles.thumbDone : styles.thumbEmpty]}
              >
                {photo ? (
                  <Image source={{ uri: photo.uri }} style={styles.thumbImage} />
                ) : (
                  <Text style={styles.thumbNum}>{i + 1}</Text>
                )}
              </View>
            );
          })}
        </ScrollView>

        <Text style={styles.counterText}>
          {allPhotosDone ? "All 6 photos captured" : `${photoCount} / ${PHOTOS_PER_MED} photos`}
        </Text>

        {!allPhotosDone ? (
          <TouchableOpacity
            style={[styles.shutterBtn, capturing && styles.shutterBtnDisabled]}
            onPress={handleTakePhoto}
            disabled={capturing}
            activeOpacity={0.7}
          >
            <View style={styles.shutterInner} />
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={styles.btn} onPress={handleNext}>
            <Text style={styles.btnText}>
              {isLastMed ? "Finish" : "Next Medication"}
            </Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },

  cameraContainer: { flex: 1, backgroundColor: "#000" },
  camera: { flex: 1 },

  // Top overlay sits above the camera
  topOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
  },
  topBar: {
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 20,
    paddingVertical: 14,
    alignItems: "center",
  },
  medName: {
    fontSize: 20,
    fontWeight: "700",
    color: "#fff",
    textAlign: "center",
  },
  stepLabel: {
    fontSize: 13,
    color: "rgba(255,255,255,0.7)",
    marginTop: 3,
  },

  // Bottom overlay sits below the camera
  bottomOverlay: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0,0,0,0.60)",
    paddingBottom: 44,
    paddingTop: 16,
    alignItems: "center",
  },

  thumbnailRow: {
    paddingHorizontal: 16,
    marginBottom: 12,
  },
  thumb: {
    width: 46,
    height: 46,
    borderRadius: 6,
    marginRight: 8,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1.5,
    overflow: "hidden",
  },
  thumbEmpty: {
    borderColor: "rgba(255,255,255,0.35)",
    backgroundColor: "rgba(255,255,255,0.08)",
  },
  thumbDone: {
    borderColor: "#4caf50",
  },
  thumbImage: { width: "100%", height: "100%" },
  thumbNum: { color: "rgba(255,255,255,0.45)", fontSize: 13 },

  counterText: {
    color: "#fff",
    fontSize: 14,
    opacity: 0.85,
    marginBottom: 20,
  },

  // Shutter button — large circle
  shutterBtn: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: "rgba(255,255,255,0.25)",
    borderWidth: 3,
    borderColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  shutterBtnDisabled: { opacity: 0.35 },
  shutterInner: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: "#fff",
  },

  btn: {
    backgroundColor: "#366a53",
    paddingVertical: 14,
    paddingHorizontal: 36,
    borderRadius: 8,
  },
  btnText: { color: "#fff", fontSize: 16, fontWeight: "600" },

  centered: { flex: 1, alignItems: "center", justifyContent: "center", padding: 28 },
  bodyText: {
    fontSize: 16,
    color: "#444",
    textAlign: "center",
    lineHeight: 22,
    marginBottom: 24,
  },
});
