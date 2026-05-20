"""Debug overlay: OpenCV window that shows the detected board, chosen move,
current/preview pair, and FPS."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from vision.color import COLOR_BGR, COLOR_NAMES, EMPTY


class DebugOverlay:
    def __init__(self, window_name: str = "ALCHEMIST debug",
                 cell_px: int = 36) -> None:
        self.window_name = window_name
        self.cell_px = cell_px
        self._last_frame_time = time.time()
        self._fps_ema: float = 0.0

    def _tick_fps(self) -> float:
        now = time.time()
        dt = max(1e-6, now - self._last_frame_time)
        self._last_frame_time = now
        inst = 1.0 / dt
        self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst if self._fps_ema else inst
        return self._fps_ema

    def render(self, board: np.ndarray,
               current_pair: Optional[Tuple[int, int]] = None,
               preview_pair: Optional[Tuple[int, int]] = None,
               chosen_col: Optional[int] = None,
               swap: bool = False,
               status: str = "") -> None:
        rows, cols = board.shape
        cell = self.cell_px
        margin = 16
        sidebar = 220
        h = rows * cell + 2 * margin
        w = cols * cell + 2 * margin + sidebar
        img = np.full((h, w, 3), 24, dtype=np.uint8)

        # Hex-ish layout (flat-top odd-q): offset odd columns down half a cell.
        for r in range(rows):
            for c in range(cols):
                color_id = int(board[r, c])
                bgr = COLOR_BGR.get(color_id, (60, 60, 60))
                x = margin + c * cell
                y = margin + r * cell + (cell // 2 if c % 2 else 0)
                cv2.rectangle(img, (x + 2, y + 2),
                              (x + cell - 2, y + cell - 2), bgr, -1)
                cv2.rectangle(img, (x + 2, y + 2),
                              (x + cell - 2, y + cell - 2),
                              (90, 90, 90), 1)

        # Chosen pair highlight (spans the chosen left column + the next one).
        if chosen_col is not None and 0 <= chosen_col < cols - 1:
            x = margin + chosen_col * cell
            cv2.rectangle(img, (x, margin - 6),
                          (x + 2 * cell, margin + rows * cell + cell // 2 + 6),
                          (0, 255, 255), 2)

        # Sidebar info.
        sx = margin + cols * cell + 16
        sy = margin + 20
        fps = self._tick_fps()
        lines = [
            f"FPS: {fps:5.1f}",
            f"swap: {'YES' if swap else 'no'}",
            f"L-col: {chosen_col}",
            f"cur  L/R: {self._pair_str(current_pair)}",
            f"prev L/R: {self._pair_str(preview_pair)}",
            status,
        ]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (sx, sy + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                        cv2.LINE_AA)

        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    @staticmethod
    def _pair_str(pair: Optional[Tuple[int, int]]) -> str:
        if pair is None:
            return "-"
        return f"{COLOR_NAMES.get(pair[0], '?')}/{COLOR_NAMES.get(pair[1], '?')}"

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
