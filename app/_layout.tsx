import { Stack } from "expo-router";

export default function Layout() {
  return (
    <Stack>
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="login" options={{ headerShown: false }} />
      <Stack.Screen name="create-account" options={{ title: "Create Account" }} />
      <Stack.Screen name="forgot-password" options={{ title: "Reset Password" }} />
      <Stack.Screen name="home" options={{ headerShown: false }} />
      <Stack.Screen name="patient-home" options={{ headerShown: false }} />

      {/* Caregiver screens */}
      <Stack.Screen name="profile" options={{ title: "Profile" }} />
      <Stack.Screen name="add-patient" options={{ title: "Add Patient" }} />
      <Stack.Screen name="logs" options={{ title: "Activity Logs" }} />
      <Stack.Screen name="patient-detail" options={{ title: "Patient" }} />
      <Stack.Screen name="med-log" options={{ title: "Medication Log" }} />
      <Stack.Screen name="log-entry-detail" options={{ title: "Log Entry" }} />

      {/* Patient screens */}
      <Stack.Screen name="patient-profile" options={{ title: "Profile" }} />
      <Stack.Screen name="patient-medications" options={{ title: "My Medications" }} />
      <Stack.Screen name="patient-schedule" options={{ title: "My Schedule" }} />
      <Stack.Screen name="medication-detail" options={{ title: "Medication" }} />

      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen name="modal" options={{ presentation: "modal" }} />
    </Stack>
  );
}
