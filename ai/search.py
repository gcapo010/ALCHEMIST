"""Shallow placement search.

The pair is diagonal across two adjacent columns. For each (left_column, swap)
placement of the CURRENT pair we simulate the drop, resolve chains, score,
then optionally 1-ply lookahead by simulating the best placement of the
PREVIEW pair on the resulting board. The total search space is at most
2 * (cols-1) * (2 * (cols-1)) -- tiny, so it's well under 50 ms.

Convention: `current_pair = (left_color, right_color)`. `swap=True` flips
left/right.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from game.board import Board
from game.physics import drop_pair, resolve_chains, find_clears
from vision.color import EMPTY
from .heuristic import evaluate


@dataclass
class Move:
    column: int        # LEFT column of the diagonal pair
    swap: bool         # True => swap pair before drop (right-click first)
    score: float


def _simulate(board: Board, left_col: int, left_color: int, right_color: int
              ) -> Optional[Tuple[Board, int, int, int]]:
    """Return (new_board, cleared_tiles, chain_depth, dominant_cleared_color)
    or None if placement is illegal."""
    nb = board.clone()
    if not drop_pair(nb, left_col, left_color, right_color):
        return None
    # Determine the dominant cleared color BEFORE resolution: easiest is to
    # snapshot what would clear on the first pass.
    first_clears = find_clears(nb)
    dominant = EMPTY
    if first_clears:
        from collections import Counter
        ctr: Counter = Counter()
        for comp in first_clears:
            r0, c0 = comp[0]
            ctr[int(nb.grid[r0, c0])] += len(comp)
        dominant = max(ctr.items(), key=lambda kv: kv[1])[0]
    cleared, depth = resolve_chains(nb)
    return nb, cleared, depth, dominant


def choose_move(board: Board, current_pair: Tuple[int, int],
                preview_pair: Optional[Tuple[int, int]] = None,
                lookahead: bool = True) -> Optional[Move]:
    """Pick the best (column, swap) for the current pair.

    `current_pair` is (bottom_color, top_color). `swap=True` flips it.
    """
    best: Optional[Move] = None
    last_left = board.cols - 1   # left column can be 0..cols-2

    options = [(False, current_pair[0], current_pair[1])]
    if current_pair[0] != current_pair[1]:
        options.append((True, current_pair[1], current_pair[0]))

    for swap, l_color, r_color in options:
        for c in range(last_left):
            sim = _simulate(board, c, l_color, r_color)
            if sim is None:
                continue
            nb, cleared, depth, dominant = sim
            score = evaluate(nb, cleared, depth, dominant)

            if lookahead and preview_pair is not None and not nb.is_top_overflow():
                best_look = -1e18
                pv_opts = [(preview_pair[0], preview_pair[1])]
                if preview_pair[0] != preview_pair[1]:
                    pv_opts.append((preview_pair[1], preview_pair[0]))
                for pl, pr in pv_opts:
                    for pc in range(last_left):
                        sim2 = _simulate(nb, pc, pl, pr)
                        if sim2 is None:
                            continue
                        nb2, cl2, dp2, dom2 = sim2
                        s2 = evaluate(nb2, cl2, dp2, dom2)
                        if s2 > best_look:
                            best_look = s2
                if best_look > -1e17:
                    # Discount lookahead so immediate gains still dominate.
                    score += 0.5 * best_look

            if best is None or score > best.score:
                best = Move(column=c, swap=swap, score=score)

    return best
