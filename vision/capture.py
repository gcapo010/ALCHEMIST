"""Ultra-fast screen capture wrapper around MSS.

Reuses a single MSS instance and returns BGR numpy frames suitable for OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mss
import numpy as np


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict:
        return {"left": self.left, "top": self.top,
                "width": self.width, "height": self.height}

    @classmethod
    def from_corners(cls, p1: Tuple[int, int], p2: Tuple[int, int]) -> "Region":
        x1, y1 = p1
        x2, y2 = p2
        left, top = min(x1, x2), min(y1, y2)
        return cls(left, top, abs(x2 - x1), abs(y2 - y1))


class ScreenCapture:
    """Singleton-style MSS wrapper. Call grab(region) for a BGR ndarray."""

    def __init__(self) -> None:
        self._sct = mss.mss()

    def grab(self, region: Region) -> np.ndarray:
        raw = self._sct.grab(region.as_mss())
        # MSS returns BGRA; drop alpha and keep BGR for OpenCV.
        arr = np.frombuffer(raw.rgb, dtype=np.uint8)
        arr = arr.reshape(raw.height, raw.width, 3)
        # MSS .rgb is already RGB-ordered; convert to BGR for OpenCV.
        return arr[:, :, ::-1].copy()

    def grab_bgra(self, region: Region) -> np.ndarray:
        raw = self._sct.grab(region.as_mss())
        arr = np.array(raw, dtype=np.uint8)  # H x W x 4 (BGRA)
        return arr
