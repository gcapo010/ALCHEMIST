"""Map screen pixels into a hex grid of color IDs.

Layout: FLAT-TOP hexes in odd-q offset coordinates (odd-numbered columns are
shifted DOWN by half a row). This matches the The Legend of Pirates Online
potion-brewing board.

Sampling is done at each hex center using a small square patch which is fast
and robust against the icon artwork inside each hex.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .color import ColorClassifier, EMPTY


@dataclass
class HexGrid:
    """Geometry of the board in screen space (flat-top, odd-q)."""
    origin_x: float       # x of column 0 center
    origin_y: float       # y of column 0, row 0 center (even-column rows)
    col_spacing: float    # horizontal distance between adjacent column centers
    row_spacing: float    # vertical distance between adjacent rows in the same column
    cols: int
    rows: int
    sample_radius: int = 14
    col_offset: float = 0.5  # odd columns shifted DOWN by this many row_spacings

    @classmethod
    def from_rect(cls, top_left: Tuple[int, int], bottom_right: Tuple[int, int],
                  cols: int, rows: int, sample_radius: int = 14) -> "HexGrid":
        x1, y1 = top_left
        x2, y2 = bottom_right
        width = x2 - x1
        height = y2 - y1
        # Flat-top horizontal pitch = 3/4 * hex_width. Across `cols` columns the
        # board spans (cols - 1) * col_spacing + hex_width = col_spacing * (cols + 1/3).
        col_spacing = width / (cols + 1 / 3.0) if cols > 0 else 0.0
        # Odd columns are offset down by half a row, so usable height is
        # (rows + 0.5) row_spacings.
        row_spacing = height / (rows + 0.5) if rows > 0 else 0.0
        origin_x = x1 + col_spacing * (2.0 / 3.0)
        origin_y = y1 + row_spacing * 0.5
        return cls(origin_x=origin_x, origin_y=origin_y,
                   col_spacing=col_spacing, row_spacing=row_spacing,
                   cols=cols, rows=rows, sample_radius=sample_radius)

    def cell_center(self, col: int, row: int) -> Tuple[int, int]:
        x = self.origin_x + col * self.col_spacing
        y = self.origin_y + row * self.row_spacing
        if col % 2 == 1:
            y += self.row_spacing * self.col_offset
        return int(round(x)), int(round(y))


class BoardReader:
    """Convert a board-region BGR frame into a (rows, cols) int8 ndarray."""

    def __init__(self, grid: HexGrid, classifier: ColorClassifier,
                 region_origin: Tuple[int, int],
                 tile_classifier=None) -> None:
        self.grid = grid
        self.classifier = classifier
        self.region_origin = region_origin
        # Optional template-based classifier. If supplied AND it has
        # templates loaded, it takes precedence over the HSV classifier.
        self.tile_classifier = tile_classifier

    def read(self, frame_bgr: np.ndarray) -> np.ndarray:
        rows, cols = self.grid.rows, self.grid.cols
        board = np.zeros((rows, cols), dtype=np.int8)
        r = self.grid.sample_radius
        ox, oy = self.region_origin
        H, W = frame_bgr.shape[:2]
        use_templates = (self.tile_classifier is not None
                         and self.tile_classifier.has_templates())
        for ry in range(rows):
            for cx in range(cols):
                sx, sy = self.grid.cell_center(cx, ry)
                px, py = sx - ox, sy - oy
                x0 = max(0, px - r)
                y0 = max(0, py - r)
                x1 = min(W, px + r + 1)
                y1 = min(H, py + r + 1)
                patch = frame_bgr[y0:y1, x0:x1]
                if use_templates:
                    board[ry, cx] = self.tile_classifier.classify_patch(patch)
                else:
                    board[ry, cx] = self.classifier.classify_patch(patch)
        return board

    def cell_screen_xy(self, col: int, row: int) -> Tuple[int, int]:
        return self.grid.cell_center(col, row)
