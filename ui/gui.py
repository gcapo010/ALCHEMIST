"""Tkinter control panel for the ALCHEMIST bot.

Runs the detection + decision loop in a background thread and surfaces the
state through a control panel: detected board canvas, current/preview pair,
chosen move, FPS, and a log pane. Buttons for Start / Pause / Probe /
Calibrate / Quit.
"""
from __future__ import annotations

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple

import numpy as np

from vision.capture import ScreenCapture, Region
from vision.color import ColorClassifier, EMPTY, COLOR_NAMES, COLOR_BGR
from vision.board import BoardReader, HexGrid
from vision.calibration import (
    Calibration, load_calibration, run_calibration, DEFAULT_CALIBRATION_PATH,
)
from vision.templates import TemplateMatcher
from game.board import Board
from ai.search import choose_move
from input.mouse import MouseController


# ----- background bot runner ------------------------------------------------

class BotRunner(threading.Thread):
    """Thread running the bot loop. Communicates state via thread-safe attrs."""

    def __init__(self, cal: Calibration, log_q: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.cal = cal
        self.log_q = log_q
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = True   # start paused so the user can confirm setup
        self._probe = True    # start in probe (no-click) by default
        # snapshot of latest state for the GUI to render
        self.state = {
            "board": None,          # np.ndarray (rows, cols) int8
            "current": (EMPTY, EMPTY),
            "preview": (EMPTY, EMPTY),
            "move_col": None,
            "move_swap": False,
            "fps": 0.0,
            "running": False,
            "last_score": 0.0,
        }

        self.sc = ScreenCapture()
        self.classifier = (ColorClassifier.from_dict(cal.color_ranges)
                           if cal.color_ranges else ColorClassifier())
        self.region = Region.from_corners(cal.board_tl, cal.board_br)
        self.grid = HexGrid.from_rect(cal.board_tl, cal.board_br,
                                      cal.cols, cal.rows,
                                      sample_radius=cal.sample_radius)
        self.reader = BoardReader(self.grid, self.classifier,
                                  region_origin=(self.region.left, self.region.top))
        self.tm = TemplateMatcher()
        self.mouse = MouseController()

    # --- control api (called from GUI thread) ---

    def stop(self) -> None:
        self._stop.set()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused
        self._log(f"paused = {paused}")

    def set_probe(self, probe: bool) -> None:
        with self._lock:
            self._probe = probe
        self._log(f"probe (no-click) = {probe}")

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def is_probe(self) -> bool:
        with self._lock:
            return self._probe

    # --- helpers ---

    def _log(self, msg: str) -> None:
        try:
            self.log_q.put_nowait(f"{time.strftime('%H:%M:%S')}  {msg}")
        except queue.Full:
            pass

    def _sample_pair(self, center_xy: Tuple[int, int]) -> Tuple[int, int]:
        dx, dy = self.cal.pair_dx, self.cal.pair_dy
        cx, cy = center_xy
        r = 6
        l = self.sc.grab(Region(cx - dx - r, cy - dy - r, 2 * r + 1, 2 * r + 1))
        rr = self.sc.grab(Region(cx + dx - r, cy + dy - r, 2 * r + 1, 2 * r + 1))
        return (self.classifier.classify_mean_hsv(l),
                self.classifier.classify_mean_hsv(rr))

    def _handle_endgame(self) -> bool:
        import mss
        with mss.mss() as ms:
            mon = ms.monitors[1]
        full = Region(mon["left"], mon["top"], mon["width"], mon["height"])
        frame = self.sc.grab(full)
        for name in ("brew_again", "ok"):
            hit = self.tm.find(frame, name)
            if hit is not None:
                cx, cy, conf = hit
                self._log(f"endgame: click {name} (conf={conf:.2f})")
                self.mouse.left_click((full.left + cx, full.top + cy))
                time.sleep(0.6)
                return True
        return False

    def save_snapshot(self, out_path: str) -> str:
        """Capture a full-screen frame and draw every sample point on it.

        - Cyan rectangle: board capture region (board_tl -> board_br)
        - Yellow dots: per-cell sample centers (labeled col,row)
        - Magenta crosses: current-pair sample points (after parking cursor)
        - Orange crosses: preview-pair sample points
        """
        import cv2
        import os
        import mss

        self.mouse.move_to(self.cal.current_pair_xy)
        time.sleep(0.05)

        with mss.mss() as ms:
            mon = ms.monitors[1]
        full = Region(mon["left"], mon["top"], mon["width"], mon["height"])
        frame = self.sc.grab(full)
        img = frame.copy()
        ox, oy = full.left, full.top

        x1, y1 = self.cal.board_tl
        x2, y2 = self.cal.board_br
        cv2.rectangle(img, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy),
                      (255, 255, 0), 2)

        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                sx, sy = self.grid.cell_center(c, r)
                cv2.circle(img, (sx - ox, sy - oy), 3, (0, 255, 255), -1)
                cv2.putText(img, f"{c},{r}",
                            (sx - ox + 5, sy - oy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 255, 255), 1, cv2.LINE_AA)

        def _mark(center, color, label):
            cx, cy = center
            for px, py in [(cx - self.cal.pair_dx, cy - self.cal.pair_dy),
                           (cx + self.cal.pair_dx, cy + self.cal.pair_dy)]:
                cv2.drawMarker(img, (px - ox, py - oy), color,
                               cv2.MARKER_CROSS, 18, 2)
            cv2.circle(img, (cx - ox, cy - oy), 5, color, 2)
            cv2.putText(img, label, (cx - ox + 8, cy - oy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        _mark(self.cal.current_pair_xy, (255, 0, 255), "CUR")
        _mark(self.cal.preview_pair_xy, (0, 165, 255), "PREV")

        out_path = os.path.abspath(out_path)
        cv2.imwrite(out_path, img)
        return out_path

    # --- main loop ---

    def run(self) -> None:
        self.state["running"] = True
        self._log("bot thread started (paused, probe-on by default)")
        empty_frames = 0
        last_t = time.time()
        fps_ema = 0.0

        while not self._stop.is_set():
            t0 = time.time()
            if self.is_paused():
                time.sleep(0.05)
                continue

            try:
                # Park the cursor so the active pair sits at the sample point.
                self.mouse.move_to(self.cal.current_pair_xy)
                time.sleep(0.02)

                frame = self.sc.grab(self.region)
                board_arr = self.reader.read(frame)
                board = Board(board_arr.copy())

                cur = self._sample_pair(self.cal.current_pair_xy)
                prev = self._sample_pair(self.cal.preview_pair_xy)

                # FPS EMA
                now = time.time()
                dt = max(1e-6, now - last_t)
                last_t = now
                inst = 1.0 / dt
                fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema else inst

                if cur[0] == EMPTY or cur[1] == EMPTY:
                    empty_frames += 1
                    if empty_frames >= 8 and not self.is_probe():
                        self._handle_endgame()
                        empty_frames = 0
                    with self._lock:
                        self.state.update({
                            "board": board_arr, "current": cur, "preview": prev,
                            "move_col": None, "move_swap": False,
                            "fps": fps_ema,
                        })
                    time.sleep(0.05)
                    continue
                empty_frames = 0

                move = choose_move(board, cur,
                                   prev if prev[0] != EMPTY else None,
                                   lookahead=True)
                if move is None:
                    self._log("no legal move (board may be overflowed)")
                    with self._lock:
                        self.state.update({
                            "board": board_arr, "current": cur, "preview": prev,
                            "move_col": None, "move_swap": False,
                            "fps": fps_ema,
                        })
                    time.sleep(0.1)
                    continue

                with self._lock:
                    self.state.update({
                        "board": board_arr, "current": cur, "preview": prev,
                        "move_col": move.column, "move_swap": move.swap,
                        "fps": fps_ema, "last_score": move.score,
                    })

                if self.is_probe():
                    time.sleep(0.25)
                    continue

                left_x, _ = self.grid.cell_center(move.column, 0)
                right_x, _ = self.grid.cell_center(move.column + 1, 0)
                col_x = (left_x + right_x) // 2
                drop_y = (self.cal.drop_y if self.cal.drop_y
                          else max(1, self.cal.board_tl[1] - 10))
                self.mouse.play(col_x, drop_y, move.swap, self.cal.current_pair_xy)

                elapsed = time.time() - t0
                if elapsed < 0.05:
                    time.sleep(0.05 - elapsed)

            except Exception as e:
                self._log(f"ERROR: {e!r}")
                time.sleep(0.5)

        self.state["running"] = False
        self._log("bot thread stopped")


# ----- Tkinter App ----------------------------------------------------------

CANVAS_PADDING = 10


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ALCHEMIST — control panel")
        self.geometry("880x620")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.log_q: queue.Queue = queue.Queue(maxsize=500)
        self.runner: Optional[BotRunner] = None
        self._build_ui()
        self._poll_state()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_start = ttk.Button(top, text="Start (paused)",
                                    command=self._on_start)
        self.btn_start.grid(row=0, column=0, padx=4)
        self.btn_pause = ttk.Button(top, text="Resume", state=tk.DISABLED,
                                    command=self._on_pause_toggle)
        self.btn_pause.grid(row=0, column=1, padx=4)
        self.probe_var = tk.BooleanVar(value=True)
        self.chk_probe = ttk.Checkbutton(top, text="Probe (no-click)",
                                         variable=self.probe_var,
                                         command=self._on_probe_toggle)
        self.chk_probe.grid(row=0, column=2, padx=8)
        self.btn_cal = ttk.Button(top, text="Recalibrate",
                                  command=self._on_calibrate)
        self.btn_cal.grid(row=0, column=3, padx=4)
        self.btn_snap = ttk.Button(top, text="Save snapshot",
                                   command=self._on_snapshot)
        self.btn_snap.grid(row=0, column=4, padx=4)
        self.btn_quit = ttk.Button(top, text="Quit", command=self._on_close)
        self.btn_quit.grid(row=0, column=5, padx=4)

        mid = ttk.Frame(self, padding=(10, 0))
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Status panel (left)
        status = ttk.LabelFrame(mid, text="Status", padding=8)
        status.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.lbl_running = ttk.Label(status, text="bot: stopped")
        self.lbl_running.pack(anchor=tk.W)
        self.lbl_fps = ttk.Label(status, text="FPS: 0.0")
        self.lbl_fps.pack(anchor=tk.W)
        self.lbl_cur = ttk.Label(status, text="current L/R: -")
        self.lbl_cur.pack(anchor=tk.W, pady=(8, 0))
        self.lbl_prev = ttk.Label(status, text="preview L/R: -")
        self.lbl_prev.pack(anchor=tk.W)
        self.lbl_move = ttk.Label(status, text="move: -")
        self.lbl_move.pack(anchor=tk.W, pady=(8, 0))
        self.lbl_score = ttk.Label(status, text="score: -")
        self.lbl_score.pack(anchor=tk.W)

        ttk.Separator(status, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(status, text="Hotkeys:").pack(anchor=tk.W)
        ttk.Label(status, text="(GUI controls only)").pack(anchor=tk.W)

        # Board canvas (center)
        canv_frame = ttk.LabelFrame(mid, text="Detected board", padding=4)
        canv_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canv_frame, bg="#181818",
                                width=320, height=380, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Log pane (bottom)
        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False,
                       padx=10, pady=(8, 10))
        self.log = tk.Text(log_frame, height=8, bg="#0c0c0c", fg="#d8d8d8",
                          font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state=tk.DISABLED)

    # --- button handlers ---

    def _on_start(self) -> None:
        if self.runner is not None and self.runner.is_alive():
            return
        cal = load_calibration()
        if cal is None:
            self._append_log("ERROR: calibration.json not found. "
                             "Click 'Recalibrate' first.")
            return
        self.runner = BotRunner(cal, self.log_q)
        self.runner.set_probe(self.probe_var.get())
        self.runner.start()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL, text="Resume")
        self._append_log("bot started (paused). Click 'Resume' to play.")

    def _on_pause_toggle(self) -> None:
        if self.runner is None:
            return
        if self.runner.is_paused():
            self.runner.set_paused(False)
            self.btn_pause.config(text="Pause")
        else:
            self.runner.set_paused(True)
            self.btn_pause.config(text="Resume")

    def _on_probe_toggle(self) -> None:
        if self.runner is not None:
            self.runner.set_probe(self.probe_var.get())

    def _on_calibrate(self) -> None:
        if self.runner is not None and self.runner.is_alive():
            self._append_log("stop the bot before recalibrating.")
            return
        self._append_log("starting calibration in a console window — "
                         "follow the prompts (Ctrl+Alt+Space).")
        self.update_idletasks()
        try:
            run_calibration()
            self._append_log("calibration saved.")
        except Exception as e:
            self._append_log(f"calibration failed: {e!r}")

    def _on_snapshot(self) -> None:
        # Build a transient runner if none exists, so the snapshot can be
        # taken before pressing Start.
        runner = self.runner
        temp = False
        if runner is None:
            cal = load_calibration()
            if cal is None:
                self._append_log("ERROR: no calibration.json; can't snapshot.")
                return
            runner = BotRunner(cal, self.log_q)
            temp = True
        try:
            import os
            out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "debug_snapshot.png")
            path = runner.save_snapshot(out)
            self._append_log(f"snapshot saved: {path}")
        except Exception as e:
            self._append_log(f"snapshot failed: {e!r}")
        if temp:
            runner.stop()

    def _on_close(self) -> None:
        if self.runner is not None:
            self.runner.stop()
        self.destroy()

    # --- periodic UI update ---

    def _poll_state(self) -> None:
        # Drain log queue.
        try:
            while True:
                line = self.log_q.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass

        if self.runner is not None:
            with self.runner._lock:
                st = dict(self.runner.state)
                board = st["board"]
            self.lbl_running.config(
                text=f"bot: {'running' if st['running'] else 'stopped'}"
                     f" {'(paused)' if self.runner.is_paused() else ''}")
            self.lbl_fps.config(text=f"FPS: {st['fps']:.1f}")
            cur = st["current"]
            prev = st["preview"]
            self.lbl_cur.config(text=f"current L/R: "
                f"{COLOR_NAMES.get(cur[0],'?')} / {COLOR_NAMES.get(cur[1],'?')}")
            self.lbl_prev.config(text=f"preview L/R: "
                f"{COLOR_NAMES.get(prev[0],'?')} / {COLOR_NAMES.get(prev[1],'?')}")
            mc = st["move_col"]
            self.lbl_move.config(
                text=f"move: col={mc} swap={'YES' if st['move_swap'] else 'no'}"
                     if mc is not None else "move: -")
            self.lbl_score.config(text=f"score: {st['last_score']:.1f}")
            self._draw_board(board, mc)

        self.after(120, self._poll_state)

    def _append_log(self, line: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        # Cap the log buffer.
        if int(self.log.index("end-1c").split(".")[0]) > 400:
            self.log.delete("1.0", "200.0")
        self.log.configure(state=tk.DISABLED)

    def _draw_board(self, board: Optional[np.ndarray],
                    chosen_col: Optional[int]) -> None:
        self.canvas.delete("all")
        if board is None:
            return
        rows, cols = board.shape
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        # Flat-top hex sized so the whole board fits the canvas.
        # board span: width = (cols + 1/3) * col_pitch ; col_pitch = 3/2 * size
        # height span: (rows + 0.5) * row_pitch ; row_pitch = sqrt(3) * size
        max_size_w = (cw - 2 * CANVAS_PADDING) / ((cols + 1 / 3.0) * 1.5)
        max_size_h = (ch - 2 * CANVAS_PADDING) / ((rows + 0.5) * math.sqrt(3))
        size = max(6.0, min(max_size_w, max_size_h))
        col_pitch = 1.5 * size
        row_pitch = math.sqrt(3) * size
        ox = CANVAS_PADDING + size
        oy = CANVAS_PADDING + row_pitch / 2

        for r in range(rows):
            for c in range(cols):
                color_id = int(board[r, c])
                bgr = COLOR_BGR.get(color_id, (60, 60, 60))
                # convert BGR -> hex
                fill = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
                cx = ox + c * col_pitch
                cy = oy + r * row_pitch + (row_pitch / 2 if c % 2 else 0)
                pts = []
                for i in range(6):
                    ang = math.radians(60 * i)
                    pts.extend([cx + size * math.cos(ang),
                                cy + size * math.sin(ang)])
                outline = "#333333"
                if chosen_col is not None and c in (chosen_col, chosen_col + 1):
                    outline = "#ffd400"
                self.canvas.create_polygon(pts, fill=fill, outline=outline,
                                            width=2 if outline == "#ffd400" else 1)


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
