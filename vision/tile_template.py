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
    """Classifies a hex cell patch by best-matching template."""

    def __init__(self, templates_dir: str = TEMPLATES_DIR,
                 threshold: float = 0.55) -> None:
        self.threshold = threshold
        self.templates: Dict[int, List[np.ndarray]] = {RED: [], BLUE: [], GREEN: []}
        if not os.path.isdir(templates_dir):
            return
        for color_name, color_id in _NAME_TO_COLOR.items():
            for path in sorted(glob.glob(
                    os.path.join(templates_dir, f"tile_{color_name}*.png"))):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates[color_id].append(img)

    def has_templates(self) -> bool:
        return any(len(v) for v in self.templates.values())

    def classify_patch(self, bgr_patch: np.ndarray) -> int:
        if bgr_patch.size == 0 or not self.has_templates():
            return EMPTY
        best_color = EMPTY
        best_score = self.threshold
        for color_id, tmpls in self.templates.items():
            for t in tmpls:
                if bgr_patch.shape[0] < t.shape[0] or bgr_patch.shape[1] < t.shape[1]:
                    # Patch smaller than template -- resize the template down.
                    th = min(t.shape[0], bgr_patch.shape[0])
                    tw = min(t.shape[1], bgr_patch.shape[1])
                    t_use = cv2.resize(t, (tw, th))
                else:
                    t_use = t
                res = cv2.matchTemplate(bgr_patch, t_use, cv2.TM_CCOEFF_NORMED)
                score = float(res.max())
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
