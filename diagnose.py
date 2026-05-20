"""Dump per-cell HSV statistics from a saved debug_snapshot.png.

Usage (from the project root):

    python diagnose.py

Reads `debug_snapshot.png` and `calibration.json`, samples the centre of
every cell of the configured grid, and prints the median HSV plus the
peak hue of saturated ring pixels. Use the output to hand-tune
calibration.json's color_ranges or to choose a different detection
signal entirely.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np


def load_calibration(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hex_centers(cal: dict):
    """Yield (col, row, x, y) for every cell in the configured grid."""
    cols, rows = cal["cols"], cal["rows"]
    x1, y1 = cal["board_tl"]
    x2, y2 = cal["board_br"]
    w = x2 - x1
    h = y2 - y1
    col_pitch = w / (cols + 1 / 3.0)
    row_pitch = h / (rows + 0.5)
    origin_x = x1 + col_pitch * (2 / 3.0)
    origin_y = y1 + row_pitch * 0.5
    for c in range(cols):
        for r in range(rows):
            x = origin_x + c * col_pitch
            y = origin_y + r * row_pitch + (row_pitch * 0.5 if c % 2 else 0)
            yield c, r, int(round(x)), int(round(y))


def annular_hues(patch_bgr: np.ndarray,
                 inner_frac: float = 0.65,
                 outer_frac: float = 0.98) -> np.ndarray:
    """Return hues of pixels in a ring band around the patch centre."""
    h, w = patch_bgr.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    ys = np.arange(h).reshape(-1, 1)
    xs = np.arange(w).reshape(1, -1)
    d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    outer = min(cy, cx)
    ring = (d >= outer * inner_frac) & (d <= outer * outer_frac)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    return hsv[ring]


def fmt_color(name: str, code: str) -> str:
    return f"\x1b[{code}m{name}\x1b[0m"


def classify_hue_for_human(h: int) -> str:
    if h <= 12 or h >= 168:
        return fmt_color(f"RED  ({h:>3})", "31")
    if 13 <= h <= 30:
        return fmt_color(f"orng ({h:>3})", "33")
    if 31 <= h <= 38:
        return fmt_color(f"ylgn ({h:>3})", "93")
    if 39 <= h <= 85:
        return fmt_color(f"GRN  ({h:>3})", "32")
    if 86 <= h <= 130:
        return fmt_color(f"BLU  ({h:>3})", "34")
    return fmt_color(f"prpl ({h:>3})", "35")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    snap = os.path.join(here, "debug_snapshot.png")
    cal_path = os.path.join(here, "calibration.json")
    if not os.path.isfile(snap):
        print(f"no {snap}; hit 'Save snapshot' in the GUI first")
        return 1
    if not os.path.isfile(cal_path):
        print(f"no {cal_path}")
        return 1

    cal = load_calibration(cal_path)
    full = cv2.imread(snap, cv2.IMREAD_COLOR)
    if full is None:
        print(f"could not read {snap}")
        return 1
    print(f"snapshot:    {full.shape[1]} x {full.shape[0]}")
    print(f"grid:        {cal['cols']} cols x {cal['rows']} rows")
    print(f"board rect:  {cal['board_tl']} -> {cal['board_br']}")
    print()

    r = max(cal.get("sample_radius", 4), 14)

    # Aggregate hue stats across all saturated ring pixels in the board.
    all_hues = []
    print(f"{'col':>3} {'row':>3}  {'medH':>5} {'medS':>5} {'medV':>5}  "
          f"{'#ring':>6}  {'#sat':>5}  peak")
    for c, ry, x, y in hex_centers(cal):
        y0, y1 = max(0, y - r), min(full.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(full.shape[1], x + r + 1)
        patch = full[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        ring_hsv = annular_hues(patch)
        if ring_hsv.size == 0:
            continue
        sat = ring_hsv[:, 1]
        val = ring_hsv[:, 2]
        sat_pixels = ring_hsv[(sat >= 80) & (val >= 50)]
        if sat_pixels.shape[0] == 0:
            continue
        med_h = int(np.median(sat_pixels[:, 0]))
        med_s = int(np.median(sat_pixels[:, 1]))
        med_v = int(np.median(sat_pixels[:, 2]))
        # Peak hue across all hue bins (5-degree).
        hist = np.bincount(sat_pixels[:, 0].astype(np.int32), minlength=180)
        peak = int(np.argmax(hist))
        all_hues.extend(sat_pixels[:, 0].tolist())
        print(f"{c:>3} {ry:>3}  {med_h:>5} {med_s:>5} {med_v:>5}  "
              f"{ring_hsv.shape[0]:>6}  {sat_pixels.shape[0]:>5}  "
              f"{classify_hue_for_human(peak)}")

    print()
    if all_hues:
        all_hues_arr = np.array(all_hues, dtype=np.int32)
        hist = np.bincount(all_hues_arr, minlength=180)
        # Top-5 peaks across the whole board.
        sm = np.convolve(hist, np.ones(5) / 5.0, mode="same")
        top = np.argsort(sm)[-10:][::-1]
        print("global saturated-ring hue histogram peaks:")
        for h in top:
            print(f"  hue {int(h):>3}: count={int(hist[h]):>5}  "
                  f"({classify_hue_for_human(int(h))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
