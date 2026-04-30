# Headless Mode for `capture_bottles.py`

**Date:** 2026-04-14  
**Status:** Approved

## Problem

Running the Expo app's bottle-capture flow (via `backend/server.py`) also causes an OpenCV desktop window ("Pill Bottle Capture") to appear on the Windows host machine. The window comes from `run_capture()` in `capture_bottles.py` calling `cv2.namedWindow` and `cv2.imshow`. The server + app pipeline is already correct and headless; the standalone script is the only source of the popup.

## Goal

Prevent any OpenCV desktop window from appearing during an app-driven capture session, while preserving the desktop GUI for anyone who deliberately runs `capture_bottles.py` directly.

## Scope

**Only `capture_bottles.py` is modified.** `backend/server.py` and `app/capture-bottles.js` are unchanged.

## Design

### 1. `run_capture()` signature

```python
def run_capture(
    source,
    dataset_base: Path,
    class_name: str = "test_user_bottle_1",
    headless: bool = False,           # NEW
) -> tuple[dict[str, Path], Path, dict]:
```

`headless=False` keeps the default desktop behavior unchanged.

### 2. GUI guards inside the capture loop

Three call sites are guarded by `if not headless`:

| Original call | Headless replacement |
|---|---|
| `cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)` | skipped |
| `cv2.imshow(WIN, canvas)` | skipped |
| `key = cv2.waitKey(1) & 0xFF` | `time.sleep(0.03); key = 255` (no-op key, keeps loop cadence) |

`cv2.setMouseCallback` is also skipped (no window to attach to).

At the end of the loop:
```python
if not headless:
    cv2.destroyAllWindows()
```

### 3. `main()` CLI flag

```
--headless    Run without any OpenCV GUI window (for server / CI use)
```

Passed straight through to `run_capture(headless=args.headless)`.

### 4. No server changes

`server.py` never calls `run_capture()`. Its `MjpegCapture` → `/capture-preview` → Expo `<Image>` pipeline already works correctly. The `headless` parameter is defensive — it protects the standalone CLI path only.

## What does NOT change

- Frame reading, `save_capture`, rotation counting, dataset writing — all untouched.
- The `ThreadedCapture`, `MjpegCapture`, and all server endpoints.
- The React Native `fetchNextFrame` / data-URI display in `app/capture-bottles.js`.

## Success criteria

1. Running `python capture_bottles.py --headless ...` captures all 24 frames with no OpenCV window.
2. Running `python capture_bottles.py` (no flag) still opens the desktop GUI as before.
3. Using the Expo app's capture flow via the server produces no desktop window at any point.
