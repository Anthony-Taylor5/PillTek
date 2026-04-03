// Temporary store used to pass a completed medication detail object from
// medication-details-entry.js back to add-patient.js via useFocusEffect.
// This avoids needing navigation.navigate-with-callback, which Expo Router
// doesn't support natively.
//
// Pattern:
//   1. add-patient.js pushes to /medication-details-entry with { medName }
//   2. medication-details-entry.js calls setLastCompletedMed(med) then router.back()
//   3. add-patient.js's useFocusEffect calls getLastCompletedMed() on re-focus,
//      adds the med to its list, then calls clearLastCompletedMed()

let _lastCompletedMed = null;

export const setLastCompletedMed = (med) => {
  _lastCompletedMed = med;
};

export const getLastCompletedMed = () => _lastCompletedMed;

export const clearLastCompletedMed = () => {
  _lastCompletedMed = null;
};
