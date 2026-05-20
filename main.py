"""ALCHEMIST — hex match-3 automation bot.

Usage:
  python main.py --calibrate    # interactive calibration -> calibration.json
  python main.py                # run the bot
  python main.py --debug        # run with the OpenCV debug overlay
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Tuple

import numpy as np

from vision.capture import ScreenCapture, Region
from vision.color import ColorClassifier, EMPTY, COLOR_NAMES
from vision.board import BoardReader, HexGrid
from vision.calibration import (
    Calibration, load_calibration, run_calibration, DEFAULT_CALIBRATION_PATH,
)
from vision.templates import TemplateMatcher
from game.board import Board
from ai.search import choose_move
from input.mouse import MouseController
from ui.overlay import DebugOverlay


def _sample_pair_at(sc: ScreenCapture, classifier: ColorClassifier,
                    center_xy: Tuple[int, int], pitch: int
                    ) -> Tuple[int, int]:
    """Sample two stacked tiles at center +/- pitch/2. Returns (bottom, top)."""
    cx, cy = center_xy
    r = 6
    top_region = Region(cx - r, cy - pitch // 2 - r, 2 * r + 1, 2 * r + 1)
    bot_region = Region(cx - r, cy + pitch // 2 - r, 2 * r + 1, 2 * r + 1)
    top_patch = sc.grab(top_region)
    bot_patch = sc.grab(bot_region)
    return classifier.classify_mean_hsv(bot_patch), classifier.classify_mean_hsv(top_patch)


def _setup(cal: Calibration):
    sc = ScreenCapture()
    if cal.color_ranges:
        classifier = ColorClassifier.from_dict(cal.color_ranges)
    else:
        classifier = ColorClassifier()

    region = Region.from_corners(cal.board_tl, cal.board_br)
    grid = HexGrid.from_rect(cal.board_tl, cal.board_br,
                             cal.cols, cal.rows, sample_radius=cal.sample_radius)
    reader = BoardReader(grid, classifier, region_origin=(region.left, region.top))
    return sc, classifier, region, grid, reader


def _handle_endgame(sc: ScreenCapture, tm: TemplateMatcher,
                    mouse: MouseController) -> bool:
    """Look for Brew Again / OK buttons full-screen and click if found."""
    import mss
    with mss.mss() as ms:
        mon = ms.monitors[1]   # primary monitor
    full = Region(mon["left"], mon["top"], mon["width"], mon["height"])
    frame = sc.grab(full)
    for name in ("brew_again", "ok"):
        hit = tm.find(frame, name)
        if hit is not None:
            cx, cy, conf = hit
            print(f"[endgame] clicking {name} (conf={conf:.2f})")
            mouse.left_click((full.left + cx, full.top + cy))
            time.sleep(0.6)
            return True
    return False


def run(debug: bool = False) -> int:
    cal = load_calibration()
    if cal is None:
        print(f"No {DEFAULT_CALIBRATION_PATH} found. Run with --calibrate first.")
        return 2

    sc, classifier, region, grid, reader = _setup(cal)
    tm = TemplateMatcher()
    mouse = MouseController()
    overlay = DebugOverlay() if debug else None

    # Global hotkeys (best-effort; keyboard package may need admin).
    import keyboard
    state = {"paused": False, "quit": False, "debug": debug,
             "recalibrate": False}

    def _toggle_pause():
        state["paused"] = not state["paused"]
        print(f"[hotkey] paused = {state['paused']}")

    def _quit():
        state["quit"] = True

    def _toggle_debug():
        state["debug"] = not state["debug"]
        print(f"[hotkey] debug = {state['debug']}")

    def _request_recal():
        state["recalibrate"] = True

    keyboard.add_hotkey("ctrl+alt+p", _toggle_pause)
    keyboard.add_hotkey("ctrl+alt+d", _toggle_debug)
    keyboard.add_hotkey("ctrl+alt+r", _request_recal)
    keyboard.add_hotkey("ctrl+alt+q", _quit)

    print("Bot running. Ctrl+Alt+P pause, Ctrl+Alt+D debug, "
          "Ctrl+Alt+R recalibrate, Ctrl+Alt+Q quit.")
    last_board: Optional[np.ndarray] = None
    empty_frames = 0

    try:
        while not state["quit"]:
            t0 = time.time()
            if state["paused"]:
                time.sleep(0.05)
                continue

            if state["recalibrate"]:
                state["recalibrate"] = False
                cal = run_calibration(cols=cal.cols, rows=cal.rows)
                sc, classifier, region, grid, reader = _setup(cal)

            frame = sc.grab(region)
            board_arr = reader.read(frame)
            board = Board(board_arr.copy())

            current_pair = _sample_pair_at(sc, classifier, cal.current_pair_xy,
                                           cal.pair_tile_pitch)
            preview_pair = _sample_pair_at(sc, classifier, cal.preview_pair_xy,
                                           cal.pair_tile_pitch)

            # If the current pair can't be read, assume end-of-game UI is up.
            if current_pair[0] == EMPTY or current_pair[1] == EMPTY:
                empty_frames += 1
                if empty_frames >= 8:
                    if _handle_endgame(sc, tm, mouse):
                        empty_frames = 0
                        time.sleep(1.0)
                        continue
                time.sleep(0.05)
                continue
            empty_frames = 0

            move = choose_move(board, current_pair,
                               preview_pair if preview_pair[0] != EMPTY else None,
                               lookahead=True)

            if move is None:
                # No legal placement. Wait a beat and re-check (likely overflow).
                time.sleep(0.05)
                continue

            col_x, _ = grid.cell_center(move.column, 0)
            drop_y = cal.drop_y if cal.drop_y else max(1, cal.board_tl[1] - 10)
            mouse.play(col_x, drop_y, move.swap, cal.current_pair_xy)

            if state["debug"] and overlay is not None:
                overlay.render(board_arr, current_pair, preview_pair,
                               chosen_col=move.column, swap=move.swap,
                               status=f"score={move.score:.1f}")

            # Pace loop slightly so we don't double-click during animation.
            elapsed = time.time() - t0
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
    finally:
        if overlay is not None:
            overlay.close()
        keyboard.unhook_all_hotkeys()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="alchemist")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args(argv)

    if args.calibrate:
        run_calibration(cols=args.cols, rows=args.rows)
        return 0
    return run(debug=args.debug)


if __name__ == "__main__":
    sys.exit(main())
