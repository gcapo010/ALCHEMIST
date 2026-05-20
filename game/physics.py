"""Drop physics, gravity, and chain resolution for the hex board.

Pair model (flat-top odd-q, diagonal pair):
  A piece is two adjacent hex tiles in DIFFERENT columns. With the cursor
  selecting the left column C, the pair occupies columns C and C+1. Each
  tile falls INDEPENDENTLY into its own column to that column's current
  landing row (split physics — one tile can land much earlier than the other
  if the columns have different heights).

  Right-click swaps which color is on the left vs right.

After each chain resolution gravity is re-applied per column, so cascades
form naturally.
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np

from .board import Board
from vision.color import EMPTY


CLEAR_THRESHOLD = 3   # 3 or more connected same-color tiles clear


# --- Gravity ----------------------------------------------------------------

def apply_gravity(board: Board) -> bool:
    """Compact every column toward the bottom. Returns True if anything moved."""
    moved = False
    g = board.grid
    rows, cols = g.shape
    for c in range(cols):
        column = g[:, c]
        nonzero = column[column != EMPTY]
        if nonzero.size == 0:
            continue
        new_col = np.zeros(rows, dtype=g.dtype)
        new_col[rows - nonzero.size:] = nonzero
        if not np.array_equal(new_col, column):
            g[:, c] = new_col
            moved = True
    return moved


# --- Chain resolution -------------------------------------------------------

# Precompute neighbor offsets keyed by COLUMN parity (flat-top odd-q).
# Index [0] = even column, [1] = odd column.
_OFFSETS = (
    np.array([(-1, 0), (1, 0), (-1, 1), (0, 1), (-1, -1), (0, -1)],
             dtype=np.int32),
    np.array([(-1, 0), (1, 0), (0, 1),  (1, 1), (0, -1),  (1, -1)],
             dtype=np.int32),
)


def find_clears(board: Board) -> List[List[Tuple[int, int]]]:
    """Return list of connected-component coordinate lists with size >= CLEAR_THRESHOLD."""
    g = board.grid
    rows, cols = g.shape
    visited = np.zeros_like(g, dtype=bool)
    clears: List[List[Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if visited[r, c] or g[r, c] == EMPTY:
                continue
            color = g[r, c]
            component: List[Tuple[int, int]] = []
            queue = deque([(r, c)])
            visited[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                component.append((rr, cc))
                offsets = _OFFSETS[cc % 2]
                for dr, dc in offsets:
                    nr, nc = rr + int(dr), cc + int(dc)
                    if (0 <= nr < rows and 0 <= nc < cols
                            and not visited[nr, nc] and g[nr, nc] == color):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            if len(component) >= CLEAR_THRESHOLD:
                clears.append(component)
    return clears


def resolve_chains(board: Board, max_chain: int = 16) -> Tuple[int, int]:
    """Apply gravity + clears until stable.

    Returns (cleared_tile_count, chain_depth).
    """
    total_cleared = 0
    chain_depth = 0
    apply_gravity(board)
    for _ in range(max_chain):
        clears = find_clears(board)
        if not clears:
            break
        chain_depth += 1
        for comp in clears:
            for r, c in comp:
                board.grid[r, c] = EMPTY
            total_cleared += len(comp)
        apply_gravity(board)
    return total_cleared, chain_depth


# --- Pair drop --------------------------------------------------------------

def drop_pair(board: Board, left_col: int,
              left_color: int, right_color: int) -> bool:
    """Place a diagonal pair into columns (left_col, left_col + 1).

    Each tile lands independently in its own column's current landing row.
    Returns False if the placement cannot fit (either column is full or
    out of bounds).
    """
    right_col = left_col + 1
    if left_col < 0 or right_col >= board.cols:
        return False
    left_landing = board.column_landing_row(left_col)
    right_landing = board.column_landing_row(right_col)
    if left_landing < 0 or right_landing < 0:
        return False
    board.grid[left_landing, left_col] = left_color
    board.grid[right_landing, right_col] = right_color
    return True
