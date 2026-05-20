"""Mouse control via PyAutoGUI.

Move horizontally over a target column, optionally right-click to swap, then
left-click to drop. PyAutoGUI is configured for zero-pause frame-perfect play.
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
             spawn_xy: Tuple[int, int]) -> None:
        """Execute a placement: optional swap at spawn, then drop in column."""
        if swap:
            self.right_click(spawn_xy)
            if self.click_delay:
                time.sleep(self.click_delay)
        # Move horizontally to the target column at drop_y, then left-click.
        pyautogui.moveTo(column_x, drop_y, duration=0)
        pyautogui.click(button="left")
