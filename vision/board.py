"""Map screen pixels into a hex grid of color IDs.

We use an odd-r offset layout (pointy-top hexes, odd rows shifted right by
half a cell). Sampling is done at each hex center using a small square patch
which is fast and robust against icon artwork.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .color import ColorClassifier, EMPTY


@dataclass
class HexGrid:
    """Geometry of the board in screen space."""
    origin_x: float           # x of column 0, row 0 center
    origin_y: float           # y of column 0, row 0 center
    col_spacing: float        # horizontal distance between adjacent column centers
    row_spacing: float        # vertical distance between adjacent row centers
    cols: int
    rows: int
    sample_radius: int = 4    # half-size of the square sampled at each center
    row_offset: float = 0.5   # additional x-shift applied to odd rows

    @classmethod
    def from_rect(cls, top_left: Tuple[int, int], bottom_right: Tuple[int, int],
                  cols: int, rows: int, sample_radius: int = 4) -> "HexGrid":
        x1, y1 = top_left
        x2, y2 = bottom_right
        width = x2 - x1
        height = y2 - y1
        # Account for half-hex shift in odd rows: usable width = (cols + 0.5) cells.
        col_spacing = width / (cols + 0.5) if cols > 0 else 0.0
        row_spacing = height / rows if rows > 0 else 0.0
        origin_x = x1 + col_spacing / 2.0
        origin_y = y1 + row_spacing / 2.0
        return cls(origin_x=origin_x, origin_y=origin_y,
                   col_spacing=col_spacing, row_spacing=row_spacing,
                   cols=cols, rows=rows, sample_radius=sample_radius)

    def cell_center(self, col: int, row: int) -> Tuple[int, int]:
        x = self.origin_x + col * self.col_spacing
        if row % 2 == 1:
            x += self.col_spacing * self.row_offset
        y = self.origin_y + row * self.row_spacing
        return int(round(x)), int(round(y))


class BoardReader:
    """Convert a board-region BGR frame into a (rows, cols) int8 ndarray."""

    def __init__(self, grid: HexGrid, classifier: ColorClassifier,
                 region_origin: Tuple[int, int]) -> None:
        self.grid = grid
        self.classifier = classifier
        # region_origin is (left, top) of the captured region in screen coords;
        # grid coords are absolute screen coords, so subtract to get region-local.
        self.region_origin = region_origin

    def read(self, frame_bgr: np.ndarray) -> np.ndarray:
        rows, cols = self.grid.rows, self.grid.cols
        board = np.zeros((rows, cols), dtype=np.int8)
        r = self.grid.sample_radius
        ox, oy = self.region_origin
        H, W = frame_bgr.shape[:2]
        for ry in range(rows):
            for cx in range(cols):
                sx, sy = self.grid.cell_center(cx, ry)
                px, py = sx - ox, sy - oy
                x0 = max(0, px - r)
                y0 = max(0, py - r)
                x1 = min(W, px + r + 1)
                y1 = min(H, py + r + 1)
                patch = frame_bgr[y0:y1, x0:x1]
                board[ry, cx] = self.classifier.classify_mean_hsv(patch)
        return board

    def cell_screen_xy(self, col: int, row: int) -> Tuple[int, int]:
        return self.grid.cell_center(col, row)
