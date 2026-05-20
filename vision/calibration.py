"""Mouse-click calibration workflow.

Run with `python main.py --calibrate`. The user is prompted to:
  1. Ctrl+Alt+Space over top-left of board
  2. Ctrl+Alt+Space over bottom-right of board
  3. Ctrl+Alt+Space over center of current piece spawn area
  4. Ctrl+Alt+Space over center of preview piece area
  5. Then auto-tune HSV ranges by hovering a Red, Blue, then Green tile.

Result is written to calibration.json.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from .color import ColorClassifier, RED, BLUE, GREEN

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CALIBRATION_PATH = os.path.join(_PROJECT_ROOT, "calibration.json")


@dataclass
class Calibration:
    board_tl: Tuple[int, int] = (0, 0)
    board_br: Tuple[int, int] = (0, 0)
    current_pair_xy: Tuple[int, int] = (0, 0)
    preview_pair_xy: Tuple[int, int] = (0, 0)
    cols: int = 8
    rows: int = 10
    sample_radius: int = 4
    # Diagonal pair geometry: the two tiles sit at (cx - dx, cy - dy) and
    # (cx + dx, cy + dy) relative to the pair's center point. Left tile is the
    # upper-left one; right tile is the lower-right one.
    pair_dx: int = 22
    pair_dy: int = 18
    drop_y: int = 0           # y row where pieces visually appear at top
    color_ranges: Dict[str, list] = field(default_factory=dict)

    def save(self, path: str = DEFAULT_CALIBRATION_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_CALIBRATION_PATH) -> "Calibration":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON turns tuples into lists; restore.
        for k in ("board_tl", "board_br", "current_pair_xy", "preview_pair_xy"):
            if k in data and isinstance(data[k], list):
                data[k] = tuple(data[k])
        return cls(**data)


def load_calibration(path: str = DEFAULT_CALIBRATION_PATH) -> Optional[Calibration]:
    if not os.path.exists(path):
        return None
    try:
        return Calibration.load(path)
    except Exception:
        return None


def _wait_for_key_and_grab_mouse(prompt: str,
                                 hotkey: str = "ctrl+alt+space"
                                 ) -> Tuple[int, int]:
    import keyboard
    import pyautogui
    print(prompt)
    print(f"  -> Hover the target, then press [{hotkey}]")
    keyboard.wait(hotkey)
    pos = pyautogui.position()
    print(f"  captured: {pos}")
    # Debounce so a single press doesn't fall through to the next wait.
    time.sleep(0.25)
    return (int(pos[0]), int(pos[1]))


def run_calibration(cols: int = 8, rows: int = 10,
                    path: str = DEFAULT_CALIBRATION_PATH) -> Calibration:
    """Interactive calibration. Requires pyautogui + keyboard packages."""
    import pyautogui
    cal = Calibration(cols=cols, rows=rows)
    print("=== ALCHEMIST calibration ===")
    cal.board_tl = _wait_for_key_and_grab_mouse(
        "[1/4] Top-left corner of the playable board.")
    cal.board_br = _wait_for_key_and_grab_mouse(
        "[2/4] Bottom-right corner of the playable board.")
    cal.current_pair_xy = _wait_for_key_and_grab_mouse(
        "[3/4] PARK position for the active pair. The active pair follows\n"
        "      the cursor along the top row, so pick a spot ABOVE the\n"
        "      board where you want it to sit while the bot reads its\n"
        "      colors (e.g. above column 0 of row 0). Hover that exact\n"
        "      point now.")
    cal.preview_pair_xy = _wait_for_key_and_grab_mouse(
        "[4/4] Center of the PREVIEW piece area (the fixed display\n"
        "      showing the NEXT pair on the parchment).")
    cal.drop_y = min(cal.board_tl[1], cal.board_br[1]) - 10

    # Color tuning: click on a known-color tile to sample.
    print("\nColor tuning: hover a RED tile and press Ctrl+Alt+Space."
          " Then BLUE, then GREEN.")
    classifier = ColorClassifier()
    try:
        from .capture import ScreenCapture, Region
        sc = ScreenCapture()
        for color_id, name in [(RED, "RED"), (BLUE, "BLUE"), (GREEN, "GREEN")]:
            xy = _wait_for_key_and_grab_mouse(
                f"  hover the CENTER of a {name} tile.")
            r = 24
            region = Region(xy[0] - r, xy[1] - r, 2 * r + 1, 2 * r + 1)
            patch = sc.grab(region)
            classifier.tune_from_sample(color_id, patch)
            print(f"  {name} tuned -> {classifier.ranges[color_id]}")
    except Exception as e:
        print(f"  (color tuning skipped: {e}; using defaults)")
    cal.color_ranges = classifier.to_dict()
    cal.save(path)
    print(f"\nSaved calibration to {path}")
    return cal
