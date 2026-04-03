import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL      = process.env.EXPO_PUBLIC_SUPABASE_URL      ?? '';
const SUPABASE_ANON_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY ?? '';

// createClient throws if either arg is empty, so guard here.
// All api.js functions check `supabase !== null` before making requests,
// so the app stays functional (falls back to in-memory stores) until
// .env is configured.
export const supabase = SUPABASE_URL && SUPABASE_ANON_KEY
  ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
  : null;

if (!supabase) {
  console.warn(
    '[Supabase] Client not initialized — EXPO_PUBLIC_SUPABASE_URL or ' +
    'EXPO_PUBLIC_SUPABASE_ANON_KEY is missing.\n' +
    'Copy .env.example → .env and fill in your project values.\n' +
    'The app will run using in-memory stores only until then.'
  );
}
