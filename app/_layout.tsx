import { Stack } from "expo-router";

export default function Layout() {
  return (
    <Stack>
    <Stack.Screen name="index" options={{ headerShown: false }} />
    <Stack.Screen name="create-account" options={{ title: "Create Account" }} />
    <Stack.Screen name="forgot-password" options={{ title: "Reset Password" }} />
    <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
    <Stack.Screen name="modal" options={{ presentation: "modal" }} />
  </Stack>
  );
}
