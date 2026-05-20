"""HSV color classification for the 3 game colors.

Only Red / Blue / Green are relevant; tile artwork is ignored. Each color is
defined as one or more HSV ranges (red wraps around the hue circle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

EMPTY = 0
RED = 1
BLUE = 2
GREEN = 3

COLOR_NAMES = {EMPTY: "empty", RED: "red", BLUE: "blue", GREEN: "green"}
COLOR_BGR = {
    EMPTY: (40, 40, 40),
    RED: (0, 0, 220),
    BLUE: (220, 80, 0),
    GREEN: (0, 200, 0),
}


@dataclass
class HSVRange:
    lo: Tuple[int, int, int]
    hi: Tuple[int, int, int]


# Sensible default HSV ranges. Calibration can override these in calibration.json.
DEFAULT_COLOR_RANGES: Dict[int, List[HSVRange]] = {
    RED: [
        HSVRange((0, 80, 60), (12, 255, 255)),
        HSVRange((165, 80, 60), (180, 255, 255)),
    ],
    BLUE: [HSVRange((88, 80, 60), (130, 255, 255))],
    GREEN: [HSVRange((35, 70, 50), (85, 255, 255))],
}


class ColorClassifier:
    """Classifies a small image patch as RED / BLUE / GREEN / EMPTY."""

    def __init__(self, ranges: Optional[Dict[int, List[HSVRange]]] = None,
                 min_fill: float = 0.10) -> None:
        self.ranges: Dict[int, List[HSVRange]] = ranges or {
            k: [HSVRange(r.lo, r.hi) for r in v]
            for k, v in DEFAULT_COLOR_RANGES.items()
        }
        # A patch must have at least this fraction of pixels matching a color
        # mask to be classified as that color; otherwise EMPTY.
        self.min_fill = min_fill

    def classify_patch(self, bgr_patch: np.ndarray) -> int:
        if bgr_patch.size == 0:
            return EMPTY
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        best_color = EMPTY
        best_score = self.min_fill * total
        for color_id, rngs in self.ranges.items():
            mask_total = 0
            for r in rngs:
                m = cv2.inRange(hsv, np.array(r.lo, dtype=np.uint8),
                                np.array(r.hi, dtype=np.uint8))
                mask_total += int(cv2.countNonZero(m))
            if mask_total > best_score:
                best_score = mask_total
                best_color = color_id
        return best_color

    def classify_mean_hsv(self, bgr_patch: np.ndarray) -> int:
        """Faster classifier: take mean HSV of the patch and test ranges."""
        if bgr_patch.size == 0:
            return EMPTY
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        mean = hsv.reshape(-1, 3).mean(axis=0)
        h, s, v = mean
        if s < 50 or v < 50:
            return EMPTY
        for color_id, rngs in self.ranges.items():
            for r in rngs:
                if (r.lo[0] <= h <= r.hi[0] and
                        r.lo[1] <= s <= r.hi[1] and
                        r.lo[2] <= v <= r.hi[2]):
                    return color_id
        return EMPTY

    # --- Auto-tuning helpers ---------------------------------------------

    def tune_from_sample(self, color_id: int, bgr_patch: np.ndarray,
                         h_pad: int = 18, sv_pad: int = 80) -> None:
        """Set the HSV range for one color from a sample patch (mean HSV).

        Hue padding is wide because tile borders have gradients and the
        center icon shifts the local hue; SV floor is low because the
        colored ring is partly desaturated where it meets the icon.
        """
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = hsv.mean(axis=0)
        lo = (max(0, int(h) - h_pad), max(40, int(s) - sv_pad),
              max(40, int(v) - sv_pad))
        hi = (min(180, int(h) + h_pad), 255, 255)
        if color_id == RED and (h < h_pad or h > 180 - h_pad):
            # Wrap red around hue circle.
            self.ranges[RED] = [
                HSVRange((0, lo[1], lo[2]), (h_pad, 255, 255)),
                HSVRange((180 - h_pad, lo[1], lo[2]), (180, 255, 255)),
            ]
        else:
            self.ranges[color_id] = [HSVRange(lo, hi)]

    def to_dict(self) -> dict:
        return {
            str(cid): [{"lo": list(r.lo), "hi": list(r.hi)} for r in rngs]
            for cid, rngs in self.ranges.items()
        }

    @classmethod
    def from_dict(cls, data: dict, min_fill: float = 0.18) -> "ColorClassifier":
        ranges: Dict[int, List[HSVRange]] = {}
        for k, v in data.items():
            ranges[int(k)] = [HSVRange(tuple(r["lo"]), tuple(r["hi"])) for r in v]
        return cls(ranges=ranges, min_fill=min_fill)
