// Module-level medication store — same pattern as role-store.js.
// Persists for the lifetime of the JS runtime (one session).
// Replace with Firestore reads/writes when backend is wired up.

// The logged-in patient's own medication list.
// Written by capture-bottles (patient flow), read by patient-home and patient-medications.
let _patientMedications = [];

// Per-patient medication lists keyed by patient name, populated by the caregiver
// add-patient → capture-bottles flow. Read by patient-detail for newly added patients.
let _caregiverPatientMeds = {};

// ── Patient self-setup ──────────────────────────────────────────────────────

export const setPatientMedications = (meds) => {
  _patientMedications = meds;
};

export const getPatientMedications = () => _patientMedications;

// ── Caregiver add-patient flow ──────────────────────────────────────────────

// Appends new medications to a patient's list rather than replacing it, so
// multiple "Add Medication" sessions accumulate correctly.
// If a medication with the same name already exists it is replaced in-place.
export const setCaregiverPatientMeds = (patientName, newMeds) => {
  const existing = _caregiverPatientMeds[patientName] ?? [];
  const filtered = existing.filter(
    (e) => !newMeds.some((m) => m.name === e.name)
  );
  _caregiverPatientMeds[patientName] = [...filtered, ...newMeds];
};

// Returns the stored medication list for a patient by name, or null if not set.
export const getCaregiverPatientMeds = (patientName) =>
  _caregiverPatientMeds[patientName] ?? null;

// ── Shared helper ───────────────────────────────────────────────────────────

// Converts a plain medication name string into the medication object shape
// used throughout the app. Time and status are left as defaults since
// scheduling happens separately.
export const nameToMedObject = (name, index) => ({
  id: `setup_${index}`,
  name,
  time: "—",
  status: "Pending",
});
