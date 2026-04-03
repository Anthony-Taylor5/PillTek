// Module-level patient store — same pattern as role-store.js.
// Generates unique patient codes for the caregiver↔patient linking system.
// Replace session-level state with Firestore reads/writes when backend is wired up.

const CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

function makePatientCode() {
  let code = "PTK-";
  for (let i = 0; i < 4; i++) {
    code += CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)];
  }
  return code;
}

// Pre-seeded mock patients mirror the IDs used in patient-detail.js / PATIENT_MEDS
let _patients = [
  { id: "1", name: "Ahmad",    dob: "",  phone: "", patientCode: "PTK-A1B2" },
  { id: "2", name: "Shahriar", dob: "",  phone: "", patientCode: "PTK-C3D4" },
  { id: "3", name: "Mina",     dob: "",  phone: "", patientCode: "PTK-E5F6" },
];

let _nextNumericId = 4;

// Caregiver link code that the patient entered during account creation.
// Keyed by Firebase UID when backend is wired; for now just one global value per session.
let _linkedCaregiverCode = null;

// ── Patient list ─────────────────────────────────────────────────────────────

export const getPatients = () => _patients;

// Generates a fresh unique patient code without adding a patient yet.
export const generatePatientCode = () => makePatientCode();

// Registers a new patient and returns the complete patient object.
export const addPatient = ({ name, dob, phone, patientCode }) => {
  const newPatient = {
    id: String(_nextNumericId++),
    name,
    dob: dob || "",
    phone: phone || "",
    patientCode,
  };
  _patients = [..._patients, newPatient];
  return newPatient;
};

// Removes a patient by ID. Returns true if a patient was found and removed.
export const removePatient = (id) => {
  const before = _patients.length;
  _patients = _patients.filter((p) => p.id !== String(id));
  return _patients.length < before;
};

// ── Caregiver link code (patient side) ───────────────────────────────────────

// Called during patient account creation when they enter the caregiver's code.
export const setLinkedCaregiverCode = (code) => {
  _linkedCaregiverCode = code ? code.trim().toUpperCase() : null;
};

// Returns the code the patient linked to, or null if they haven't linked.
export const getLinkedCaregiverCode = () => _linkedCaregiverCode;
