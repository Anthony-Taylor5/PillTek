import React from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";

/**
 * Shared list row for both caregiver (patients list) and patient (medications list).
 * - label: primary left-aligned text
 * - value: optional secondary right-aligned text (e.g. time)
 * - onPress: tap handler
 */
export default function DashboardRow({ label, value, onPress }) {
  return (
    <TouchableOpacity
      style={styles.row}
      onPress={onPress}
      activeOpacity={0.6}
    >
      <Text style={styles.label} numberOfLines={1}>
        {label}
      </Text>
      {value != null ? (
        <Text style={styles.value}>{value}</Text>
      ) : null}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    minHeight: 52,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: "#d6ebd9",
  },
  label: {
    flex: 1,
    fontSize: 18,
    lineHeight: 22,
    color: "#000",
  },
  value: {
    flexShrink: 0,
    marginLeft: 12,
    fontSize: 14,
    lineHeight: 22,
    color: "#666",
  },
});
