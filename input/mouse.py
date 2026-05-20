"""Mouse control via PyAutoGUI.

The piece follows the cursor at the top of the board. To play:
  1. Move the cursor horizontally to the target column (the piece tracks it).
  2. Right-click in place to swap the pair order (if needed).
  3. Left-click in place to drop the piece.

PyAutoGUI is configured for zero-pause frame-perfect play.
"""
from __future__ import annotations

import time
from typing import Tuple

import pyautogui

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True


class MouseController:
    def __init__(self, click_delay: float = 0.01) -> None:
        self.click_delay = click_delay

    def move_to(self, xy: Tuple[int, int]) -> None:
        pyautogui.moveTo(xy[0], xy[1], duration=0)

    def left_click(self, xy: Tuple[int, int]) -> None:
        pyautogui.click(xy[0], xy[1], button="left")

    def right_click(self, xy: Tuple[int, int]) -> None:
        pyautogui.click(xy[0], xy[1], button="right")

    def play(self, column_x: int, drop_y: int, swap: bool,
             spawn_xy: Tuple[int, int] | None = None) -> None:
        """Execute a placement.

        The cursor's x position is what selects the drop column — the piece
        tracks the cursor. So we move horizontally first, then swap/drop
        in place. `spawn_xy` is unused (kept for API compatibility).
        """
        del spawn_xy  # piece follows cursor; spawn location is irrelevant for input

        pyautogui.moveTo(column_x, drop_y, duration=0)
        if swap:
            pyautogui.click(button="right")
            if self.click_delay:
                time.sleep(self.click_delay)
        pyautogui.click(button="left")
