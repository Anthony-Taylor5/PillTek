import React from "react";
import { View, Text, ScrollView, StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useLocalSearchParams } from "expo-router";

function statusColor(s) {
  if (s === "Taken")  return "#366a53";
  if (s === "Missed") return "#c0392b";
  return "#888";
}

// Full-size bottle photo placeholder — frontend only, no camera/storage
function BottlePhoto({ status }) {
  const hasPhoto = status === "Taken";
  return (
    <View style={[styles.photoBox, !hasPhoto && styles.photoBoxEmpty]}>
      {hasPhoto ? (
        <>
          <Text style={styles.photoIcon}>🧴</Text>
          <Text style={styles.photoLabel}>Pill bottle photo</Text>
        </>
      ) : (
        <>
          <Text style={styles.photoIconGrey}>📷</Text>
          <Text style={styles.photoLabelGrey}>
            {status === "Missed" ? "No photo — dose was missed" : "Photo not yet available"}
          </Text>
        </>
      )}
    </View>
  );
}

export default function LogEntryDetail() {
  const { medName, patientName, date, time, status } = useLocalSearchParams();

  return (
    <SafeAreaView style={styles.safe} edges={["bottom"]}>
      <ScrollView contentContainerStyle={styles.container}>

        {/* Identity */}
        <Text style={styles.medName}>{medName}</Text>
        <Text style={styles.subLabel}>{patientName}</Text>

        {/* Info rows */}
        <Text style={styles.sectionTitle}>details</Text>
        <View style={styles.divider} />

        {[
          { label: "Date",    value: date },
          { label: "Time",    value: time },
          { label: "Patient", value: patientName },
        ].map((row) => (
          <View key={row.label} style={styles.infoRow}>
            <Text style={styles.infoLabel}>{row.label}</Text>
            <Text style={styles.infoValue}>{row.value}</Text>
          </View>
        ))}

        {/* Status row — coloured */}
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Status</Text>
          <Text style={[styles.infoValue, { color: statusColor(status), fontWeight: "600" }]}>
            {status}
          </Text>
        </View>

        {/* Bottle photo section */}
        <Text style={[styles.sectionTitle, { marginTop: 28 }]}>bottle photo</Text>
        <View style={styles.divider} />
        <BottlePhoto status={status} />

        {status === "Taken" && (
          <Text style={styles.captionNote}>
            Photo captured by patient at {time} on {date}.
          </Text>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#e8f5e9" },
  container: { paddingHorizontal: 20, paddingTop: 24, paddingBottom: 40 },

  medName: { fontSize: 22, fontWeight: "700", color: "#000" },
  subLabel: { fontSize: 14, color: "#555", marginTop: 4, marginBottom: 24 },

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

  // Full-size photo placeholder
  photoBox: {
    marginTop: 16,
    width: "100%",
    aspectRatio: 1,
    borderRadius: 14,
    backgroundColor: "#c8e6c9",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "#a5d6a7",
  },
  photoBoxEmpty: {
    backgroundColor: "#eeeeee",
    borderColor: "#bdbdbd",
  },
  photoIcon: { fontSize: 72 },
  photoLabel: { marginTop: 14, fontSize: 15, color: "#366a53", fontWeight: "500" },
  photoIconGrey: { fontSize: 52, opacity: 0.4 },
  photoLabelGrey: {
    marginTop: 14,
    fontSize: 14,
    color: "#888",
    textAlign: "center",
    paddingHorizontal: 24,
  },

  captionNote: {
    marginTop: 12,
    fontSize: 13,
    color: "#666",
    textAlign: "center",
  },
});
