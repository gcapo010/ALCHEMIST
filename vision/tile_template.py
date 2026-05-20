"""Template-matching tile classifier.

A more robust alternative to HSV color classification: at calibration
time the user captures one reference image per color (red/blue/green
hex tile), saved to templates/tile_red.png etc. At runtime each cell's
patch is compared against every loaded template with normalised cross
correlation; the highest-scoring template wins, and if no template
clears `threshold` the cell is considered EMPTY.

This handles tile icon variation (e.g. red dragon vs. red potion)
because matchTemplate scores the overall pattern -- the colored ring
dominates the correlation even when icons differ. If a single template
per color isn't enough you can add more variants by dropping extra PNGs
into the templates directory: any file matching `tile_<color>*.png` is
treated as another reference for that color.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .color import EMPTY, RED, BLUE, GREEN, COLOR_NAMES

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(_PROJECT_ROOT, "templates")

_NAME_TO_COLOR = {"red": RED, "blue": BLUE, "green": GREEN}


def _template_path(color_name: str, variant: str = "") -> str:
    suffix = f"_{variant}" if variant else ""
    return os.path.join(TEMPLATES_DIR, f"tile_{color_name}{suffix}.png")


class TileTemplateClassifier:
    """Classifies a hex cell patch by best-matching template.

    For each template we precompute its dominant hue (the most common
    hue among saturated ring pixels). At classification time we look at
    how many of the cell's saturated ring pixels fall within ``HUE_PAD``
    of each template's dominant hue and pick the template with the most
    hits. Wood and other muddy backgrounds don't match any tile
    template's hue, so they correctly return EMPTY -- even if some
    wood pixels happen to be saturated enough to pass the gate.
    """

    SAT_FLOOR = 140
    VAL_FLOOR = 70
    HUE_BINS = 36           # 5-degree bins
    HUE_PAD = 12            # +/- this many hue units around a template's hue
    # The colored ring is on the OUTSIDE of each tile; the icon (which can
    # have stray pixels of other colors -- e.g. a red-tipped crab claw on
    # a blue tile) sits in the middle. Drop the inner disc when computing
    # histograms so the icon can't pollute the vote.
    INNER_MASK_FRAC = 0.55
    # Require this many ring pixels matching a template's hue before
    # classifying a cell as that color.
    MIN_MATCH_PIXELS = 20
    # Expected hue zones per color. A template's dominant hue is clamped
    # into its color's zone -- so a green template whose peak landed on a
    # yellow icon highlight (hue ~30) is still treated as green-at-50.
    EXPECTED_HUE_ZONES: Dict[int, List[Tuple[int, int]]] = {
        RED: [(160, 179)],
        BLUE: [(85, 130)],
        GREEN: [(40, 85)],
    }

    def __init__(self, templates_dir: str = TEMPLATES_DIR) -> None:
        self.templates: Dict[int, List[np.ndarray]] = {RED: [], BLUE: [], GREEN: []}
        # Per template: the dominant hue of its saturated ring pixels.
        self.template_hues: Dict[int, List[int]] = {RED: [], BLUE: [], GREEN: []}
        if not os.path.isdir(templates_dir):
            return
        for color_name, color_id in _NAME_TO_COLOR.items():
            for path in sorted(glob.glob(
                    os.path.join(templates_dir, f"tile_{color_name}*.png"))):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                dom = self._dominant_hue_in_zone(img, color_id)
                if dom is None:
                    continue
                self.templates[color_id].append(img)
                self.template_hues[color_id].append(dom)

    def has_templates(self) -> bool:
        return any(len(v) for v in self.templates.values())

    def _ring_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        cy, cx = h / 2.0, w / 2.0
        ys = np.arange(h).reshape(-1, 1)
        xs = np.arange(w).reshape(1, -1)
        d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        outer = min(cy, cx)
        return (d >= outer * self.INNER_MASK_FRAC).astype(np.uint8) * 255

    def _saturated_hues(self, bgr: np.ndarray) -> np.ndarray:
        """Return the array of hues from saturated ring pixels."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ring = self._ring_mask(bgr.shape[:2]) > 0
        sat_ok = (hsv[:, :, 1] >= self.SAT_FLOOR) & (hsv[:, :, 2] >= self.VAL_FLOOR)
        keep = ring & sat_ok
        return hsv[keep][:, 0].astype(np.int32)

    def _dominant_hue_in_zone(self, bgr: np.ndarray,
                              color_id: int) -> Optional[int]:
        """Dominant hue inside the expected zone for this color id, so the
        peak can't land on an icon highlight outside the color's range."""
        hues = self._saturated_hues(bgr)
        if hues.size == 0:
            return None
        hist = np.bincount(hues, minlength=180).astype(np.float32)
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(hist, kernel, mode="same")
        best_h = None
        best_count = -1.0
        for lo_h, hi_h in self.EXPECTED_HUE_ZONES.get(color_id, [(0, 179)]):
            sub = smooth[lo_h:hi_h + 1]
            if sub.size == 0:
                continue
            local = int(np.argmax(sub)) + lo_h
            if smooth[local] > best_count:
                best_count = float(smooth[local])
                best_h = local
        return best_h

    def _count_in_band(self, hues: np.ndarray, centre: int) -> int:
        """Count how many hues fall within HUE_PAD of `centre` (wrap-aware)."""
        if hues.size == 0:
            return 0
        diff = np.abs(hues - centre)
        wrap_diff = np.minimum(diff, 180 - diff)
        return int(np.count_nonzero(wrap_diff <= self.HUE_PAD))

    def classify_patch(self, bgr_patch: np.ndarray) -> int:
        if bgr_patch.size == 0 or not self.has_templates():
            return EMPTY
        hues = self._saturated_hues(bgr_patch)
        if hues.size < self.MIN_MATCH_PIXELS:
            return EMPTY
        best_color = EMPTY
        best_count = self.MIN_MATCH_PIXELS - 1
        for color_id, dom_list in self.template_hues.items():
            for dom in dom_list:
                cnt = self._count_in_band(hues, dom)
                if cnt > best_count:
                    best_count = cnt
                    best_color = color_id
        return best_color

    def classify_patch_verbose(self, bgr_patch: np.ndarray
                               ) -> Tuple[int, int, int]:
        """Like classify_patch but returns (color_id, matching_pixels, total_ring_pixels)."""
        if bgr_patch.size == 0 or not self.has_templates():
            return (EMPTY, 0, 0)
        hues = self._saturated_hues(bgr_patch)
        if hues.size < self.MIN_MATCH_PIXELS:
            return (EMPTY, 0, int(hues.size))
        best_color = EMPTY
        best_count = self.MIN_MATCH_PIXELS - 1
        for color_id, dom_list in self.template_hues.items():
            for dom in dom_list:
                cnt = self._count_in_band(hues, dom)
                if cnt > best_count:
                    best_count = cnt
                    best_color = color_id
        return (best_color, best_count, int(hues.size))


def save_template(color_name: str, bgr_patch: np.ndarray,
                  variant: str = "") -> str:
    """Write a template PNG for the given color."""
    if color_name not in _NAME_TO_COLOR:
        raise ValueError(f"unknown color {color_name!r}")
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    path = _template_path(color_name, variant)
    cv2.imwrite(path, bgr_patch)
    return path
