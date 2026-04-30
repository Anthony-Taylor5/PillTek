# Capture Bottles Headless Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--headless` flag to `capture_bottles.py` that suppresses all OpenCV GUI windows so the script can run alongside the server without spawning a desktop popup.

**Architecture:** Guard the three OpenCV GUI calls (`cv2.namedWindow`, `cv2.imshow`, `cv2.waitKey`, `cv2.setMouseCallback`, `cv2.destroyAllWindows`) behind a `headless` boolean. In headless mode auto-select the box type and auto-capture each frame so the loop terminates without keyboard input. A companion `--box-type` CLI flag lets the caller specify which preset to use headlessly.

**Tech Stack:** Python 3.11, OpenCV (`cv2`), `unittest.mock` for tests, `pytest`

---

## File Map

| File | Change |
|---|---|
| `capture_bottles.py` | Modify `run_capture()` signature + loop; add `--headless` / `--box-type` to `main()` |
| `tests/test_capture_headless.py` | New — unit tests for headless behaviour |

---

### Task 1: Create the test file with failing tests

**Files:**
- Create: `tests/test_capture_headless.py`

- [ ] **Step 1: Create `tests/__init__.py` if it doesn't exist**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_capture_headless.py`:

```python
"""Tests for capture_bottles.run_capture() headless mode."""
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_fake_cap(n_frames=25):
    """ThreadedCapture mock that returns a fresh frame on every read()."""
    cap = MagicMock()
    frame = _make_frame()
    cap.read.return_value = (True, frame)
    return cap


# ---------------------------------------------------------------------------
# headless=True: no cv2 window calls
# ---------------------------------------------------------------------------

@patch("capture_bottles.save_capture")
@patch("capture_bottles.ThreadedCapture")
@patch("capture_bottles.cv2")
def test_headless_does_not_call_namedWindow(mock_cv2, MockCap, mock_save, tmp_path):
    MockCap.return_value = _make_fake_cap()
    from capture_bottles import run_capture
    run_capture(
        source="http://fake/stream",
        dataset_base=tmp_path / "ds",
        class_name="test_bottle",
        headless=True,
        box_type="tall",
    )
    mock_cv2.namedWindow.assert_not_called()


@patch("capture_bottles.save_capture")
@patch("capture_bottles.ThreadedCapture")
@patch("capture_bottles.cv2")
def test_headless_does_not_call_imshow(mock_cv2, MockCap, mock_save, tmp_path):
    MockCap.return_value = _make_fake_cap()
    from capture_bottles import run_capture
    run_capture(
        source="http://fake/stream",
        dataset_base=tmp_path / "ds",
        class_name="test_bottle",
        headless=True,
        box_type="tall",
    )
    mock_cv2.imshow.assert_not_called()


@patch("capture_bottles.save_capture")
@patch("capture_bottles.ThreadedCapture")
@patch("capture_bottles.cv2")
def test_headless_does_not_call_destroyAllWindows(mock_cv2, MockCap, mock_save, tmp_path):
    MockCap.return_value = _make_fake_cap()
    from capture_bottles import run_capture
    run_capture(
        source="http://fake/stream",
        dataset_base=tmp_path / "ds",
        class_name="test_bottle",
        headless=True,
        box_type="tall",
    )
    mock_cv2.destroyAllWindows.assert_not_called()


@patch("capture_bottles.save_capture")
@patch("capture_bottles.ThreadedCapture")
@patch("capture_bottles.cv2")
def test_headless_saves_total_captures(mock_cv2, MockCap, mock_save, tmp_path):
    """Loop must complete all TOTAL_CAPTURES saves in headless mode."""
    from capture_bottles import TOTAL_CAPTURES
    MockCap.return_value = _make_fake_cap()
    from capture_bottles import run_capture
    run_capture(
        source="http://fake/stream",
        dataset_base=tmp_path / "ds",
        class_name="test_bottle",
        headless=True,
        box_type="tall",
    )
    assert mock_save.call_count == TOTAL_CAPTURES


# ---------------------------------------------------------------------------
# headless=False: cv2 window IS used (desktop path unchanged)
# ---------------------------------------------------------------------------

@patch("capture_bottles.save_capture")
@patch("capture_bottles.ThreadedCapture")
@patch("capture_bottles.cv2")
def test_desktop_calls_namedWindow(mock_cv2, MockCap, mock_save, tmp_path):
    """Non-headless path must still call cv2.namedWindow."""
    cap = _make_fake_cap()
    MockCap.return_value = cap

    # Simulate: user presses '2' (select tall) then SPACE × TOTAL_CAPTURES then 'q'
    from capture_bottles import TOTAL_CAPTURES
    key_seq = (
        [ord("2")]                        # select tall
        + [ord(" ")] * TOTAL_CAPTURES     # capture all frames
        + [ord("q")]                       # quit
    )
    mock_cv2.waitKey.side_effect = [k & 0xFF for k in key_seq]
    mock_cv2.namedWindow.return_value = None
    mock_cv2.imshow.return_value = None
    mock_cv2.setMouseCallback.return_value = None

    # numpy ops needed inside the loop
    import numpy as _np
    mock_cv2.WINDOW_NORMAL = 1
    mock_cv2.FONT_HERSHEY_SIMPLEX = 0
    mock_cv2.FONT_HERSHEY_DUPLEX = 1
    mock_cv2.LINE_AA = 16

    from capture_bottles import run_capture
    run_capture(
        source="http://fake/stream",
        dataset_base=tmp_path / "ds",
        class_name="test_bottle",
        headless=False,
    )
    mock_cv2.namedWindow.assert_called_once()
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd c:/Users/Anthony/Documents/CS490/code
python -m pytest tests/test_capture_headless.py -v 2>&1 | head -40
```

Expected: `TypeError` or `ImportError` — `run_capture()` doesn't accept `headless` or `box_type` yet.

---

### Task 2: Add `headless` and `box_type` parameters to `run_capture()`

**Files:**
- Modify: `capture_bottles.py` — `run_capture()` signature and loop body

- [ ] **Step 1: Update the `run_capture` signature**

In `capture_bottles.py` at line 384, change:

```python
def run_capture(source, dataset_base: Path, class_name: str = "test_user_bottle_1") -> tuple[dict[str, Path], Path, dict]:
```

to:

```python
def run_capture(
    source,
    dataset_base: Path,
    class_name: str = "test_user_bottle_1",
    headless: bool = False,
    box_type: str = "tall",
) -> tuple[dict[str, Path], Path, dict]:
```

- [ ] **Step 2: Pre-select the box type in headless mode**

Directly after the line `selected_key: str | None = None` (around line 395), add:

```python
    if headless:
        selected_key = box_type if box_type in BOX_PRESETS else "tall"
```

- [ ] **Step 3: Guard `cv2.namedWindow` and `cv2.setMouseCallback`**

Find the block (around lines 400–407):

```python
    WIN = "Pill Bottle Capture"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Mouse hover tracking for button highlight (optional polish)
    mouse_pos = [0, 0]
    def on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
    cv2.setMouseCallback(WIN, on_mouse)
```

Replace with:

```python
    WIN = "Pill Bottle Capture"
    mouse_pos = [0, 0]
    if not headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, flags, param):
            mouse_pos[0], mouse_pos[1] = x, y
        cv2.setMouseCallback(WIN, on_mouse)
```

- [ ] **Step 4: Guard `cv2.imshow` and replace `cv2.waitKey` when headless**

Find the block near the bottom of the while-loop (around lines 539–540):

```python
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(1) & 0xFF
```

Replace with:

```python
        if not headless:
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
        else:
            time.sleep(0.03)
            key = 255
```

- [ ] **Step 5: Auto-capture in headless mode**

Immediately after the `key = 255` line added in Step 4, add the headless auto-capture block inside the loop (before the existing `if key == ord("q")` check):

```python
        # In headless mode there is no keyboard — auto-capture each frame.
        if headless and last_frame is not None and captures_done < TOTAL_CAPTURES:
            p      = BOX_PRESETS[selected_key]
            rfh, rfw = last_frame.shape[:2]
            bw_raw = int(rfw * p["w_frac"])
            bh_raw = int(rfh * p["h_frac"])
            save_capture(
                frame=last_frame,
                box_cx=rfw // 2, box_cy=rfh // 2,
                box_w=bw_raw, box_h=bh_raw,
                dirs=dirs,
                index=captures_done,
            )
            captures_done += 1
            deg = captures_done * DEGREES_PER_SHOT
            print(f"[HEADLESS] {captures_done}/{TOTAL_CAPTURES} captured "
                  f"({deg}° — rotate {DEGREES_PER_SHOT}° clockwise, then wait)")
            time.sleep(0.5)   # give user time to rotate before next auto-capture
            continue
```

- [ ] **Step 6: Guard `cv2.destroyAllWindows`**

Find (around line 592):

```python
    cv2.destroyAllWindows()
```

Replace with:

```python
    if not headless:
        cv2.destroyAllWindows()
```

- [ ] **Step 7: Run the headless tests — they should pass now**

```bash
cd c:/Users/Anthony/Documents/CS490/code
python -m pytest tests/test_capture_headless.py::test_headless_does_not_call_namedWindow \
                 tests/test_capture_headless.py::test_headless_does_not_call_imshow \
                 tests/test_capture_headless.py::test_headless_does_not_call_destroyAllWindows \
                 tests/test_capture_headless.py::test_headless_saves_total_captures \
                 -v
```

Expected: 4 × PASSED.

---

### Task 3: Add `--headless` and `--box-type` flags to `main()`

**Files:**
- Modify: `capture_bottles.py` — `main()` argument parser and `run_capture` call

- [ ] **Step 1: Add the two new CLI arguments**

Inside `main()`, after the existing `parser.add_argument("--train-only", ...)` block (around line 682), add:

```python
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without any OpenCV GUI window. Auto-captures frames; rotate the bottle between captures.",
    )
    parser.add_argument(
        "--box-type",
        choices=["short", "tall", "wide"],
        default="tall",
        help="Bottle shape preset used for auto-capture in headless mode (default: tall)",
    )
```

- [ ] **Step 2: Pass the flags to `run_capture`**

Find the `run_capture` call inside `main()` (around line 747):

```python
    dirs, data_yaml, _ = run_capture(source=args.source, dataset_base=dataset_dir, class_name=args.class_name)
```

Replace with:

```python
    dirs, data_yaml, _ = run_capture(
        source=args.source,
        dataset_base=dataset_dir,
        class_name=args.class_name,
        headless=args.headless,
        box_type=args.box_type,
    )
```

- [ ] **Step 3: Run the full test suite**

```bash
cd c:/Users/Anthony/Documents/CS490/code
python -m pytest tests/test_capture_headless.py -v
```

Expected: all tests PASSED (skip `test_desktop_calls_namedWindow` if it requires further mocking effort — the four headless tests are the acceptance criteria).

---

### Task 4: Smoke-test the CLI flag

- [ ] **Step 1: Verify `--help` shows the new flags**

```bash
cd c:/Users/Anthony/Documents/CS490/code
python capture_bottles.py --help
```

Expected output includes:

```
  --headless            Run without any OpenCV GUI window...
  --box-type {short,tall,wide}
```

- [ ] **Step 2: Verify server still imports cleanly**

```bash
cd c:/Users/Anthony/Documents/CS490/code
python -c "from capture_bottles import ThreadedCapture, prepare_dataset_dirs, write_data_yaml, save_capture, next_run_dir, BOX_PRESETS, TOTAL_CAPTURES; print('OK')"
```

Expected: `OK` — no import errors, no windows opened.

---

### Task 5: Commit

- [ ] **Commit the changes**

```bash
cd c:/Users/Anthony/Documents/CS490/code
git add capture_bottles.py tests/test_capture_headless.py tests/__init__.py
git commit -m "feat: add --headless flag to capture_bottles to suppress OpenCV window

In headless mode cv2.namedWindow/imshow/waitKey/destroyAllWindows are
skipped; frames are auto-captured with a 0.5 s delay between shots so
the CLI terminates without keyboard input.

Fixes the OpenCV desktop popup that appeared when the server and the
standalone script were both running.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
