# Capture Bottles — Live Preview FPS Handoff

## Project at a glance
- **Repo:** `C:\Users\Anthony\Documents\CS490\code`
- **Branch:** `mark-app`
- **App:** PillMotion / PillTek — Expo RN medication tracker
- **Screen under work:** `app/capture-bottles.js` — live preview + capture of medication bottles from an ESP32-CAM
- **Backend:** `backend/server.py` Flask server proxies MJPEG from ESP32 through `/capture-preview/{sessionId}?n={frameNum}`

## Runtime environment (important)
- RN 0.81.5, Expo 54, Expo Go
- `newArchEnabled: true` (Fabric) and `reactCompiler: true` (experimental) — see `app.json`
- Currently tested on Android **emulator** using software GPU: `EGL_emulation app_time_stats: avg=1414ms`. That alone caps throughput hard.

## What was fixed in the last session

| # | Symptom | Root cause | Fix | Status |
|---|---------|-----------|-----|--------|
| 1 | All children of `cameraContainer` invisible despite correct UIAutomator bounds | Fabric clips everything when a View has `borderRadius` + `overflow: "hidden"` | Removed `overflow: "hidden"` from `cameraContainer`; kept `borderRadius` on the inner Image | **Committed** `9828db3` |
| 2 | Red flash between frames | Diagnostic `backgroundColor: "red"` on `cameraFeed` showed during source swaps | Removed the diagnostic | Uncommitted |
| 3 | Black flash between frames | Built-in RN `Image` resets to Fresco `EmptyDrawable` when `source.uri` changes | Swapped to `expo-image` with `recyclingKey="capture-preview"` | **Confirmed fixed by user** — uncommitted |
| 4 | `onLoad` never fires when ESP32 returns byte-identical JPEG → loop stalls | React setState bailout (same data URI) suppresses the render, so no onLoad | Added `lastB64Ref`; if the new base64 equals the last one, skip setState and re-poll immediately | Uncommitted |
| 5 | Fresco cancelled decodes when multiple fetches ran concurrently | `setTimeout(fetchNextFrame, 0)` let fetches race ahead of decodes | Paced the loop on `Image.onLoad`; added 1s `safetyTimerRef` fallback | Uncommitted |

**All changes since `9828db3` are uncommitted.** Good checkpoint: commit the expo-image swap + pacing + dup-detect as one commit before starting FPS work.

## Current state of `app/capture-bottles.js`

- Imports `Image` from `expo-image` (not `react-native`)
- `fetchNextFrame`: fetch → arrayBuffer → chunked `String.fromCharCode` → `btoa` → dup check vs `lastB64Ref` → `setFrameUri(data:image/jpeg;base64,...)`
- `onLoad` schedules the next fetch; `safetyTimerRef` (1s) reschedules if onLoad is missed
- `cameraContainer` has NO `overflow: "hidden"` — there's an explanatory comment in the stylesheet about why. **Do not re-add it.**

## Measured baseline (emulator)
- Black flash: gone (confirmed)
- Decoded frames: ~7 → 8 → 11 over 10s = **~0.4 fps**
- User verdict: "very sluggish"

## Next task: FPS optimization

Ordered simplest → most invasive. Start at #1.

### 1. Drop the JS base64 round-trip — point `expo-image` at the backend URL directly
The biggest remaining per-frame cost is JS-side: `fetch → arrayBuffer → chunked charCode → btoa`. `expo-image` can fetch + decode HTTP URLs natively (off the JS thread) and its `recyclingKey` still holds the previous bitmap.

Sketch:
```js
const [frameNum, setFrameNum] = useState(0);
// ...
<Image
  source={{ uri: `${BACKEND_URL}/capture-preview/${sessionId}?n=${frameNum}` }}
  recyclingKey="capture-preview"
  cachePolicy="none"          // or "memory" — want fresh bytes, not cached
  transition={0}
  onLoad={() => setFrameNum(n => n + 1)}
  onError={...}
/>
```
Caveats:
- Confirm `/capture-preview/{sessionId}` handles `?n=...` as a cache-buster or returns `Cache-Control: no-store`
- Verify `expo-image` won't dedupe requests with different query params (it shouldn't with `cachePolicy="none"`)
- The duplicate-frame detection we added only works on base64 bytes; if we move to URL source, we lose it. That's probably fine — expo-image will just re-decode identical bytes quickly. If it becomes a problem, add an ETag or content hash to the URL and short-circuit when it matches the last.

### 2. Sanity-check on a real device
The emulator's `EGL_emulation avg=1414ms` is a hard cap. Before more backend work, run the current build on a physical Android. If fps jumps to usable, the "optimization" is really "stop benchmarking on software GPU."

### 3. Backend-side tuning (if still slow on device)
- Reduce JPEG quality / resolution on the ESP32 (check `working_server/cam_and_server_code.ino/`)
- Have `/capture-preview` stream MJPEG (`multipart/x-mixed-replace`) and let expo-image handle it — requires verifying expo-image MJPEG support (it historically hasn't; may need a native MJPEG view component)

### 4. Last resort
Write/import a native MJPEG surface view (there are community packages). Significantly more work.

## Useful commands
- `adb logcat | grep -iE "expo|react|image|EGL"` — decode + GPU timing
- `adb exec-out screencap -p > test.png` — pull a screenshot
- `.appium-cli/session.json` shows the active Appium session (localhost:4723, pkg `com.anonymous.pillmotion`)
- `android-appium-claude/appium-cli.js` is the test harness

## User collaboration notes
- User prefers to do Expo Go reloads manually and says "ready" when done — don't try to force-stop the app
- User asks for commits at natural checkpoints — offer one before any risky refactor
- User wants terse, non-narrated replies; if pushing back on an approach, trust the redirection immediately
