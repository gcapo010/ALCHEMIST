"""Template matching for UI buttons (Brew Again / OK)."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TemplateMatcher:
    def __init__(self, templates_dir: Optional[str] = None,
                 threshold: float = 0.85) -> None:
        if templates_dir is None:
            templates_dir = os.path.join(_PROJECT_ROOT, "templates")
        self.threshold = threshold
        self.templates: Dict[str, np.ndarray] = {}
        if os.path.isdir(templates_dir):
            for fname in os.listdir(templates_dir):
                if fname.lower().endswith(".png"):
                    name = os.path.splitext(fname)[0]
                    img = cv2.imread(os.path.join(templates_dir, fname),
                                     cv2.IMREAD_COLOR)
                    if img is not None:
                        self.templates[name] = img

    def find(self, frame_bgr: np.ndarray, name: str
             ) -> Optional[Tuple[int, int, float]]:
        tmpl = self.templates.get(name)
        if tmpl is None:
            return None
        if frame_bgr.shape[0] < tmpl.shape[0] or frame_bgr.shape[1] < tmpl.shape[1]:
            return None
        res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < self.threshold:
            return None
        x, y = max_loc
        cx = x + tmpl.shape[1] // 2
        cy = y + tmpl.shape[0] // 2
        return (cx, cy, float(max_val))
