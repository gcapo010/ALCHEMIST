"""Static board evaluation.

Designed for SPEED, not perfection. Rewards immediate clears (color-weighted)
and cluster formation; penalizes stack height and isolated tiles.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from game.board import Board
from game.physics import _OFFSETS
from vision.color import EMPTY, RED, BLUE, GREEN


COLOR_CLEAR_REWARD = {RED: 250.0, BLUE: 180.0, GREEN: 120.0}
CHAIN_BONUS = 200.0
HEIGHT_PENALTY = 8.0
MAX_HEIGHT_PENALTY = 25.0
ISOLATED_PENALTY = 4.0
CLUSTER_REWARD = 6.0
TOP_OVERFLOW_PENALTY = 10_000.0


def evaluate(board: Board, cleared: int, chain_depth: int,
             dominant_cleared_color: int = EMPTY) -> float:
    score = 0.0
    if cleared:
        # Reward tiles cleared, weighted by color preference.
        weight = COLOR_CLEAR_REWARD.get(dominant_cleared_color, 100.0)
        score += weight * cleared
        score += CHAIN_BONUS * (chain_depth ** 1.4)

    if board.is_top_overflow():
        score -= TOP_OVERFLOW_PENALTY

    g = board.grid
    rows, cols = g.shape

    # Stack heights per column.
    heights = np.zeros(cols, dtype=np.int32)
    for c in range(cols):
        col = g[:, c]
        nz = np.nonzero(col)[0]
        if nz.size:
            heights[c] = rows - int(nz[0])
    score -= HEIGHT_PENALTY * float(heights.sum())
    score -= MAX_HEIGHT_PENALTY * float(heights.max() if heights.size else 0)

    # Cluster vs isolation count: per non-empty cell, count same-color neighbors.
    isolated = 0
    clustered = 0
    for r in range(rows):
        offsets = _OFFSETS[r % 2]
        for c in range(cols):
            color = g[r, c]
            if color == EMPTY:
                continue
            same = 0
            for dr, dc in offsets:
                nr, nc = r + int(dr), c + int(dc)
                if 0 <= nr < rows and 0 <= nc < cols and g[nr, nc] == color:
                    same += 1
            if same == 0:
                isolated += 1
            else:
                clustered += same
    score -= ISOLATED_PENALTY * isolated
    score += CLUSTER_REWARD * clustered

    # Dead-column penalty: empty column surrounded by tall columns is bad.
    for c in range(cols):
        if heights[c] == 0 and 0 < c < cols - 1:
            if heights[c - 1] >= 4 and heights[c + 1] >= 4:
                score -= 30.0

    return score
