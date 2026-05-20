"""Drop physics, gravity, and chain resolution for the hex board.

Pair model:
  A piece is two stacked hex tiles in one column (top tile + bottom tile).
  When dropped into column C with bottom color B and top color T, both tiles
  enter column C. The bottom tile lands first; the top tile lands directly
  above it. Either tile can be swapped via right-click before dropping (handled
  upstream by negating the swap flag).

Split behavior (vertical only — pairs are vertical so only one column is used,
which matches an odd-r layout where a pair occupies a single column). After
EACH chain resolution gravity is re-applied per column, so cascades naturally
form.
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

# Precompute neighbor offsets for both row parities to avoid per-cell branching.
_OFFSETS = (
    np.array([(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)], dtype=np.int32),
    np.array([(-1,  0), (-1, 1), (0, -1), (0, 1), (1,  0), (1, 1)], dtype=np.int32),
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
                offsets = _OFFSETS[rr % 2]
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

def drop_pair(board: Board, col: int, bottom_color: int, top_color: int
              ) -> bool:
    """Place a vertical pair into `col`. Returns False if it cannot fit."""
    landing = board.column_landing_row(col)
    if landing < 1:
        return False
    board.grid[landing, col] = bottom_color
    board.grid[landing - 1, col] = top_color
    return True
