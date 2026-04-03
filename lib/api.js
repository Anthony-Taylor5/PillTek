/**
 * api.js — all Supabase database operations for PillTek.
 *
 * Every function checks whether the Supabase client is configured before
 * making a request.  When it is not (env vars missing), functions return
 * empty arrays / null silently so the app falls back to in-memory stores.
 *
 * Callers still wrap calls in try/catch for actual network / DB errors.
 */

import { supabase } from './supabase';

// ── Profiles ──────────────────────────────────────────────────────────────────

export async function upsertProfile({ firebaseUid, displayName, role, phone, email }) {
  if (!supabase) return;
  const { error } = await supabase
    .from('profiles')
    .upsert(
      { firebase_uid: firebaseUid, display_name: displayName, role, phone, email },
      { onConflict: 'firebase_uid' }
    );
  if (error) throw error;
}

// ── Patients ──────────────────────────────────────────────────────────────────

/**
 * Fetch all patients that belong to a caregiver (or self-managed user).
 * Returns rows shaped like { id, name, patient_code, dob, phone, ... }
 */
export async function fetchPatients(caregiverUid) {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from('patients')
    .select('id, name, patient_code, dob, phone, patient_firebase_uid, beacon_mac, created_at')
    .eq('caregiver_uid', caregiverUid)
    .order('created_at', { ascending: true });
  if (error) throw error;
  return data ?? [];
}

/**
 * Insert a new patient record.
 * Returns the newly created patient row including its UUID id.
 */
export async function createPatient({ caregiverUid, name, dob, phone, patientCode }) {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from('patients')
    .insert({
      caregiver_uid: caregiverUid,
      name,
      dob:          dob   || null,
      phone:        phone || null,
      patient_code: patientCode,
    })
    .select()
    .single();
  if (error) throw error;
  return data;
}

/**
 * Link a patient's Firebase UID to the patient record identified by patient_code.
 * Called during patient account creation when the patient enters the code.
 * Returns the updated patient row, or throws if the code does not exist.
 */
export async function linkPatientByCode({ patientFirebaseUid, patientCode }) {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from('patients')
    .update({ patient_firebase_uid: patientFirebaseUid })
    .eq('patient_code', patientCode.trim().toUpperCase())
    .select()
    .single();
  if (error) throw error;
  return data;
}

/**
 * Look up the patient record linked to a given Firebase UID.
 * Returns null if the patient has not been linked yet.
 */
export async function fetchPatientByUid(firebaseUid) {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from('patients')
    .select('*')
    .eq('patient_firebase_uid', firebaseUid)
    .maybeSingle();
  if (error) throw error;
  return data ?? null;
}

// ── Medications ───────────────────────────────────────────────────────────────

/**
 * Fetch all medications for a patient (by patient UUID).
 */
export async function fetchMedications(patientId) {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from('medications')
    .select('id, name, dosage, frequency, times, time_display, refill, added_at, status')
    .eq('patient_id', patientId)
    .order('added_at', { ascending: true });
  if (error) throw error;
  return (data ?? []).map(normalizeDbMed);
}

/**
 * Bulk-insert medications for a patient.
 * Returns the array of created rows with their UUIDs.
 */
export async function createMedications(patientId, meds) {
  if (!supabase) return [];
  const rows = meds.map((m) => ({
    patient_id:   patientId,
    name:         m.name,
    dosage:       m.dosage    || null,
    frequency:    m.frequency || null,
    times:        m.times     || [],
    time_display: m.time      || null,
    refill:       m.refill    || null,
    added_at:     m.addedAt   || new Date().toISOString(),
    status:       m.status    || 'Pending',
  }));
  const { data, error } = await supabase
    .from('medications')
    .insert(rows)
    .select();
  if (error) throw error;
  return (data ?? []).map(normalizeDbMed);
}

function normalizeDbMed(row) {
  return {
    id:        row.id,
    name:      row.name,
    dosage:    row.dosage       ?? '—',
    frequency: row.frequency    ?? '—',
    times:     row.times        ?? [],
    time:      row.time_display ?? '—',
    refill:    row.refill       ?? '—',
    addedAt:   row.added_at,
    status:    row.status       ?? 'Pending',
  };
}

// ── Medication Logs ───────────────────────────────────────────────────────────

/**
 * Fetch the 30 most-recent log entries for a given medication UUID.
 */
export async function fetchMedLogs(medicationId) {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from('medication_logs')
    .select('id, log_date, scheduled_time, taken_at, status, source')
    .eq('medication_id', medicationId)
    .order('log_date', { ascending: false })
    .limit(30);
  if (error) throw error;
  return data ?? [];
}

/**
 * Fetch all medication log entries for a patient on today's date.
 * Joins with medications so callers get the medication name and time_display.
 */
export async function fetchTodayLogs(patientId) {
  if (!supabase) return [];
  const today = new Date().toISOString().split('T')[0];
  const { data, error } = await supabase
    .from('medication_logs')
    .select('id, log_date, scheduled_time, taken_at, status, source, medications(id, name, time_display)')
    .eq('patient_id', patientId)
    .eq('log_date', today)
    .order('scheduled_time', { ascending: true });
  if (error) throw error;
  return data ?? [];
}

/**
 * Insert or update a log entry for a medication on today's date.
 */
export async function upsertMedLog({
  medicationId,
  patientId,
  scheduledTime,
  status,
  takenAt,
  source,
}) {
  if (!supabase) return null;
  const today = new Date().toISOString().split('T')[0];
  const { data, error } = await supabase
    .from('medication_logs')
    .upsert(
      {
        medication_id:  medicationId,
        patient_id:     patientId,
        log_date:       today,
        scheduled_time: scheduledTime ?? null,
        taken_at:       takenAt       ?? null,
        status:         status        ?? 'Taken',
        source:         source        ?? 'manual',
      },
      { onConflict: 'medication_id,log_date,scheduled_time' }
    )
    .select()
    .single();
  if (error) throw error;
  return data;
}

// ── Bottle Images ─────────────────────────────────────────────────────────────

/**
 * Upload a photo (from a local file:// URI) to the 'bottle-images' Storage
 * bucket and record the path in the bottle_images table.
 */
export async function uploadBottleImage(medicationId, filename, localUri) {
  if (!supabase) return null;
  const response = await fetch(localUri);
  const blob     = await response.blob();

  const storagePath = `${medicationId}/${filename}.jpg`;

  const { error: uploadError } = await supabase.storage
    .from('bottle-images')
    .upload(storagePath, blob, { contentType: 'image/jpeg', upsert: true });
  if (uploadError) throw uploadError;

  const { error: dbError } = await supabase
    .from('bottle_images')
    .insert({ medication_id: medicationId, filename, storage_path: storagePath });
  if (dbError) throw dbError;

  return storagePath;
}

// ── Detection Events ──────────────────────────────────────────────────────────

/**
 * Log a raw detection event from the ESP32 / Python pipeline.
 */
export async function logDetectionEvent({
  eventType,
  bottleClass,
  bottleLabel,
  confidence,
  patientId,
  rawMeta,
}) {
  if (!supabase) return;
  const { error } = await supabase.from('detection_events').insert({
    event_type:   eventType,
    bottle_class: bottleClass  ?? null,
    bottle_label: bottleLabel  ?? null,
    confidence:   confidence   ?? null,
    patient_id:   patientId    ?? null,
    raw_meta:     rawMeta      ?? {},
  });
  if (error) throw error;
}
