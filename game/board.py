"""Hex board state in odd-r offset coordinates.

The board is stored as an (rows, cols) int8 array. 0 = empty, otherwise a
color id (RED=1 / BLUE=2 / GREEN=3). Row 0 is the TOP of the playfield; pieces
fall toward higher row indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from vision.color import EMPTY, RED, BLUE, GREEN

# 6 hex neighbor offsets for odd-r offset coordinates.
# Different parity rows have different column offsets.
_NEIGHBORS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
_NEIGHBORS_ODD  = [(-1, 0),  (-1, 1), (0, -1), (0, 1), (1, 0),  (1, 1)]


def hex_neighbors(row: int, col: int) -> List[Tuple[int, int]]:
    table = _NEIGHBORS_ODD if (row % 2) else _NEIGHBORS_EVEN
    return [(row + dr, col + dc) for dr, dc in table]


@dataclass
class Board:
    grid: np.ndarray   # (rows, cols) int8

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    def clone(self) -> "Board":
        return Board(self.grid.copy())

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_empty(self, row: int, col: int) -> bool:
        return self.grid[row, col] == EMPTY

    def column_top_row(self, col: int) -> int:
        """First (top-most) row index in `col` that is non-empty, or rows if empty."""
        column = self.grid[:, col]
        nz = np.nonzero(column)[0]
        return int(nz[0]) if nz.size else self.rows

    def column_landing_row(self, col: int) -> int:
        """The row a falling tile would land in (one above the top-most filled)."""
        return self.column_top_row(col) - 1

    def is_top_overflow(self) -> bool:
        """True if any piece occupies row 0 (no room to spawn)."""
        return bool(np.any(self.grid[0, :] != EMPTY))

    def max_stack_height(self) -> int:
        heights = [self.rows - self.column_top_row(c) for c in range(self.cols)]
        return max(heights) if heights else 0
