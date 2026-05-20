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

    Matching uses HUE histogram comparison rather than pixel-by-pixel
    cross-correlation, so red tiles with a chicken icon and red tiles
    with a potion icon are both recognised as red (their hue histograms
    are dominated by the ring color either way). Pixels below a
    saturation/value floor are dropped before histogramming so the
    brown wood background can never tip a comparison.
    """

    SAT_FLOOR = 90
    VAL_FLOOR = 60
    HUE_BINS = 18    # 10-degree-wide bins covering hue 0..179
    # The colored ring is on the OUTSIDE of each tile; the icon (which can
    # have stray pixels of other colors -- e.g. a red-tipped crab claw on
    # a blue tile) sits in the middle. Drop the inner disc when computing
    # histograms so the icon can't pollute the vote.
    INNER_MASK_FRAC = 0.55   # mask radius as fraction of patch half-size

    def __init__(self, templates_dir: str = TEMPLATES_DIR,
                 threshold: float = 0.35) -> None:
        self.threshold = threshold
        self.templates: Dict[int, List[np.ndarray]] = {RED: [], BLUE: [], GREEN: []}
        self.template_hists: Dict[int, List[np.ndarray]] = {RED: [], BLUE: [], GREEN: []}
        if not os.path.isdir(templates_dir):
            return
        for color_name, color_id in _NAME_TO_COLOR.items():
            for path in sorted(glob.glob(
                    os.path.join(templates_dir, f"tile_{color_name}*.png"))):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                self.templates[color_id].append(img)
                self.template_hists[color_id].append(self._hue_hist(img))

    def has_templates(self) -> bool:
        return any(len(v) for v in self.templates.values())

    def _ring_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Boolean mask selecting only the outer ring (icon excluded)."""
        h, w = shape
        cy, cx = h / 2.0, w / 2.0
        ys = np.arange(h).reshape(-1, 1)
        xs = np.arange(w).reshape(1, -1)
        # Distance from centre, normalised so the patch corner = 1.
        d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        outer = min(cy, cx)
        return (d >= outer * self.INNER_MASK_FRAC).astype(np.uint8) * 255

    def _hue_hist(self, bgr: np.ndarray) -> np.ndarray:
        """Hue histogram of saturated ring pixels only (icon masked out)."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ring = self._ring_mask(bgr.shape[:2])
        sat_mask = cv2.inRange(
            hsv,
            np.array((0, self.SAT_FLOOR, self.VAL_FLOOR), dtype=np.uint8),
            np.array((180, 255, 255), dtype=np.uint8))
        mask = cv2.bitwise_and(sat_mask, ring)
        hist = cv2.calcHist([hsv], [0], mask, [self.HUE_BINS], [0, 180])
        s = float(hist.sum())
        if s <= 0:
            return hist.flatten()
        return (hist / s).flatten()

    def classify_patch(self, bgr_patch: np.ndarray) -> int:
        if bgr_patch.size == 0 or not self.has_templates():
            return EMPTY
        hist = self._hue_hist(bgr_patch)
        if hist.sum() <= 0:
            return EMPTY     # no saturated pixels at all -- empty cell
        best_color = EMPTY
        best_score = self.threshold
        for color_id, hists in self.template_hists.items():
            for th in hists:
                if th.sum() <= 0:
                    continue
                # Correlation: 1.0 = identical, 0 = uncorrelated, -1 = opposite.
                score = float(cv2.compareHist(hist.astype(np.float32),
                                              th.astype(np.float32),
                                              cv2.HISTCMP_CORREL))
                if score > best_score:
                    best_score = score
                    best_color = color_id
        return best_color


def save_template(color_name: str, bgr_patch: np.ndarray,
                  variant: str = "") -> str:
    """Write a template PNG for the given color."""
    if color_name not in _NAME_TO_COLOR:
        raise ValueError(f"unknown color {color_name!r}")
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    path = _template_path(color_name, variant)
    cv2.imwrite(path, bgr_patch)
    return path
