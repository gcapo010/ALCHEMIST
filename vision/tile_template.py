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

    SAT_FLOOR = 70
    VAL_FLOOR = 50
    HUE_PAD = 18            # +/- this many hue units around a template's hue
    # Sample 6 small probes inside the tile, one at each hex-edge midpoint,
    # placed just INSIDE the colored border so we never bleed into a
    # neighbouring tile's ring. PROBE_FRAC is the fraction of the patch
    # half-width at which the probes sit; PROBE_PATCH is the half-size of
    # each small probe (so the probe is (2*PROBE_PATCH+1)^2 pixels).
    PROBE_FRAC = 0.75
    PROBE_PATCH = 3
    # Require at least this fraction of probes to vote the same color
    # before classifying a cell as that color.
    MIN_VOTES = 3
    # Expected hue zones per color. A template's dominant hue is clamped
    # into its color's zone -- so a green template whose peak landed on a
    # yellow icon highlight (hue ~30) is still treated as green-at-50.
    EXPECTED_HUE_ZONES: Dict[int, List[Tuple[int, int]]] = {
        RED: [(160, 179), (0, 12)],
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

    @staticmethod
    def _probe_offsets(patch_half: float, frac: float) -> List[Tuple[int, int]]:
        """6 (dy, dx) offsets at hex edge-midpoint directions, scaled to
        sit at `frac` of the patch half-width."""
        import math
        r = patch_half * frac
        # Flat-top hexes: edges' perpendicular directions are at
        # 30, 90, 150, 210, 270, 330 degrees from +x.
        offsets = []
        for deg in (30, 90, 150, 210, 270, 330):
            rad = math.radians(deg)
            dx = int(round(r * math.cos(rad)))
            dy = int(round(r * math.sin(rad)))
            offsets.append((dy, dx))
        return offsets

    def _probe_hues(self, bgr: np.ndarray) -> List[Optional[int]]:
        """Median hue at each of the 6 probe positions (None if probe
        pixels are too desaturated to be a tile)."""
        h, w = bgr.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        patch_half = min(cy, cx)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        results: List[Optional[int]] = []
        p = self.PROBE_PATCH
        for dy, dx in self._probe_offsets(patch_half, self.PROBE_FRAC):
            py = int(round(cy + dy))
            px = int(round(cx + dx))
            y0, y1 = max(0, py - p), min(h, py + p + 1)
            x0, x1 = max(0, px - p), min(w, px + p + 1)
            probe = hsv[y0:y1, x0:x1].reshape(-1, 3)
            if probe.size == 0:
                results.append(None)
                continue
            sat_ok = (probe[:, 1] >= self.SAT_FLOOR) & (probe[:, 2] >= self.VAL_FLOOR)
            kept = probe[sat_ok]
            if kept.shape[0] < max(1, probe.shape[0] // 3):
                # Most probe pixels failed the saturation gate -- this
                # probe is on icon or wood, not on the ring.
                results.append(None)
                continue
            results.append(int(np.median(kept[:, 0])))
        return results

    @staticmethod
    def _hue_distance(a: int, b: int) -> int:
        d = abs(a - b)
        return min(d, 180 - d)

    def _dominant_hue_in_zone(self, bgr: np.ndarray,
                              color_id: int) -> Optional[int]:
        """Pick the template's representative hue from its 6 probes,
        constrained to the color's expected zone so an icon highlight
        outside the zone can't be picked as the centre."""
        probes = [h for h in self._probe_hues(bgr) if h is not None]
        if not probes:
            return None
        zones = self.EXPECTED_HUE_ZONES.get(color_id, [(0, 179)])

        def in_zone(h: int) -> bool:
            return any(lo <= h <= hi for lo, hi in zones)

        in_zone_probes = [h for h in probes if in_zone(h)]
        if in_zone_probes:
            return int(np.median(in_zone_probes))
        # Fall back to median of all probes if none are in zone (shouldn't
        # happen for a well-captured template, but be defensive).
        return int(np.median(probes))

    def classify_patch(self, bgr_patch: np.ndarray) -> int:
        if bgr_patch.size == 0 or not self.has_templates():
            return EMPTY
        probes = self._probe_hues(bgr_patch)
        valid = [h for h in probes if h is not None]
        if len(valid) < self.MIN_VOTES:
            return EMPTY
        # For each probe, find the template whose hue is closest (within
        # HUE_PAD). Vote for that color.
        votes: Dict[int, int] = {RED: 0, BLUE: 0, GREEN: 0}
        for h in valid:
            best_color = EMPTY
            best_dist = self.HUE_PAD + 1
            for color_id, dom_list in self.template_hues.items():
                for dom in dom_list:
                    d = self._hue_distance(h, dom)
                    if d < best_dist:
                        best_dist = d
                        best_color = color_id
            if best_color != EMPTY:
                votes[best_color] += 1
        winner = max(votes, key=lambda k: votes[k])
        if votes[winner] < self.MIN_VOTES:
            return EMPTY
        return winner

    def classify_patch_verbose(self, bgr_patch: np.ndarray
                               ) -> Tuple[int, Dict[int, int], List[Optional[int]]]:
        """Like classify_patch but also returns the vote tally and the raw
        probe hues, for diagnostics."""
        votes: Dict[int, int] = {RED: 0, BLUE: 0, GREEN: 0}
        if bgr_patch.size == 0 or not self.has_templates():
            return (EMPTY, votes, [])
        probes = self._probe_hues(bgr_patch)
        valid = [h for h in probes if h is not None]
        if len(valid) < self.MIN_VOTES:
            return (EMPTY, votes, probes)
        for h in valid:
            best_color = EMPTY
            best_dist = self.HUE_PAD + 1
            for color_id, dom_list in self.template_hues.items():
                for dom in dom_list:
                    d = self._hue_distance(h, dom)
                    if d < best_dist:
                        best_dist = d
                        best_color = color_id
            if best_color != EMPTY:
                votes[best_color] += 1
        winner = max(votes, key=lambda k: votes[k])
        if votes[winner] < self.MIN_VOTES:
            return (EMPTY, votes, probes)
        return (winner, votes, probes)


def save_template(color_name: str, bgr_patch: np.ndarray,
                  variant: str = "") -> str:
    """Write a template PNG for the given color."""
    if color_name not in _NAME_TO_COLOR:
        raise ValueError(f"unknown color {color_name!r}")
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    path = _template_path(color_name, variant)
    cv2.imwrite(path, bgr_patch)
    return path
