-- PillTek Supabase Schema
-- Run this in the Supabase SQL editor (Dashboard → SQL Editor → New query).
-- RLS is disabled for all tables — this is a prototype.  Enable per-table RLS
-- and add firebase_uid-matched policies when moving to production.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── profiles ──────────────────────────────────────────────────────────────────
-- Mirrors Firebase Auth users.  Written on account creation.
CREATE TABLE IF NOT EXISTS profiles (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  firebase_uid  text UNIQUE NOT NULL,
  display_name  text,
  role          text CHECK (role IN ('caregiver', 'patient', 'self')),
  phone         text,
  email         text,
  created_at    timestamptz DEFAULT now()
);

-- ── patients ──────────────────────────────────────────────────────────────────
-- Created by a caregiver (or self-managed user).
-- patient_firebase_uid is null until the patient registers and enters the code.
CREATE TABLE IF NOT EXISTS patients (
  id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  caregiver_uid         text NOT NULL,           -- Firebase UID of the caregiver/self
  patient_firebase_uid  text UNIQUE,             -- Firebase UID of the linked patient account
  name                  text NOT NULL,
  dob                   text,
  phone                 text,
  patient_code          text UNIQUE NOT NULL,    -- e.g. PTK-A1B2
  beacon_mac            text,                    -- BLE beacon MAC address for this patient
  created_at            timestamptz DEFAULT now()
);

-- ── medications ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS medications (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id    uuid NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  name          text NOT NULL,
  dosage        text,
  frequency     text,
  times         jsonb    DEFAULT '[]',   -- e.g. ["8:00 AM","8:00 PM"]
  time_display  text,                   -- backward-compat display string
  refill        text,
  added_at      timestamptz DEFAULT now(),
  status        text DEFAULT 'Pending' CHECK (status IN ('Pending', 'Taken', 'Missed'))
);

-- ── medication_logs ───────────────────────────────────────────────────────────
-- One row per scheduled dose occurrence per day.
-- The detection pipeline inserts rows with source='detection'.
-- Manual confirmation inserts rows with source='manual'.
CREATE TABLE IF NOT EXISTS medication_logs (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  medication_id   uuid NOT NULL REFERENCES medications(id) ON DELETE CASCADE,
  patient_id      uuid NOT NULL REFERENCES patients(id)   ON DELETE CASCADE,
  log_date        date        NOT NULL DEFAULT CURRENT_DATE,
  scheduled_time  text,
  taken_at        timestamptz,
  status          text DEFAULT 'Pending' CHECK (status IN ('Pending', 'Taken', 'Missed')),
  source          text DEFAULT 'manual'  CHECK (source   IN ('manual', 'esp32', 'detection')),
  created_at      timestamptz DEFAULT now(),
  -- Only one log entry per medication per day per scheduled time slot.
  UNIQUE (medication_id, log_date, scheduled_time)
);

-- ── bottle_images ─────────────────────────────────────────────────────────────
-- Photos captured in the capture-bottles flow, stored in Supabase Storage.
CREATE TABLE IF NOT EXISTS bottle_images (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  medication_id uuid NOT NULL REFERENCES medications(id) ON DELETE CASCADE,
  filename      text NOT NULL,
  storage_path  text,          -- path inside the 'bottle-images' bucket
  uploaded_at   timestamptz DEFAULT now()
);

-- ── detection_events ──────────────────────────────────────────────────────────
-- Raw events from the ESP32 + Python detection pipeline.
CREATE TABLE IF NOT EXISTS detection_events (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type    text NOT NULL,   -- 'beacon_near'|'beacon_far'|'bottle_detected'|'hand_bottle_overlap'
  bottle_class  integer,         -- YOLO class id (0-4)
  bottle_label  text,            -- class name string
  confidence    real,
  patient_id    uuid REFERENCES patients(id) ON DELETE SET NULL,
  raw_meta      jsonb    DEFAULT '{}',
  triggered_at  timestamptz DEFAULT now()
);

-- ── Disable RLS on all tables (prototype) ─────────────────────────────────────
ALTER TABLE profiles         DISABLE ROW LEVEL SECURITY;
ALTER TABLE patients         DISABLE ROW LEVEL SECURITY;
ALTER TABLE medications      DISABLE ROW LEVEL SECURITY;
ALTER TABLE medication_logs  DISABLE ROW LEVEL SECURITY;
ALTER TABLE bottle_images    DISABLE ROW LEVEL SECURITY;
ALTER TABLE detection_events DISABLE ROW LEVEL SECURITY;

-- ── Storage bucket ────────────────────────────────────────────────────────────
-- Create this manually in Supabase Dashboard → Storage → New bucket
-- Name: bottle-images   Public: false
-- OR uncomment and run via the service-role connection:
-- INSERT INTO storage.buckets (id, name, public)
--   VALUES ('bottle-images', 'bottle-images', false)
--   ON CONFLICT DO NOTHING;
