import { Stack } from "expo-router";

export default function Layout() {
  return (
    <Stack>
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="login" options={{ headerShown: false }} />
      <Stack.Screen name="create-account" options={{ title: "Create Account" }} />
      <Stack.Screen name="verify-email" options={{ headerShown: false }} />
      <Stack.Screen name="forgot-password" options={{ title: "Reset Password" }} />
      <Stack.Screen name="home" options={{ headerShown: false }} />
      <Stack.Screen name="patient-home" options={{ headerShown: false }} />
      <Stack.Screen name="self-home" options={{ headerShown: false }} />

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
      <Stack.Screen name="patient-daily-logs" options={{ title: "Today's Log" }} />
      <Stack.Screen name="medication-detail" options={{ title: "Medication" }} />
      <Stack.Screen name="medication-entry" options={{ title: "My Medications" }} />

      {/* Caregiver onboarding — medication detail form + add medication */}
      <Stack.Screen name="medication-details-entry" options={{ title: "Medication Details" }} />
      <Stack.Screen name="add-medication" options={{ title: "Add Medication" }} />
      <Stack.Screen name="patient-log-view" options={{ title: "Patient Logs" }} />
      <Stack.Screen name="caregiver-calendar" options={{ title: "Calendar" }} />
      <Stack.Screen name="self-logs" options={{ title: "My Logs" }} />

      {/* Shared onboarding */}
      <Stack.Screen
        name="capture-bottles"
        options={{ title: "Capture Bottles", headerBackVisible: false }}
      />

      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen name="modal" options={{ presentation: "modal" }} />
    </Stack>
  );
}
