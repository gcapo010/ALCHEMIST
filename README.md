# ALCHEMIST — Hex Match-3 Automation Bot

A computer-vision-only Windows bot for the **potion-brewing minigame in The
Legend of Pirates Online** (TLOPO) — or any visually similar 3-color hex
match-3 with a diagonal pair. No memory reading, no OCR — just MSS screen
capture, OpenCV HSV segmentation, a fast hex simulator, and PyAutoGUI mouse
input.

Board layout: **flat-top hexes, odd-q offset, 7 columns × 8 rows**. The
falling piece is **two adjacent hexes on the diagonal** (upper-left tile +
lower-right tile), with split physics — each tile falls independently into
its own column.

## Modules

```
vision/   capture, HSV color segmentation, board grid sampling, calibration, templates
game/     hex board model, gravity, chain resolution, cloning
ai/       move generation, evaluation heuristic, beam search
input/    mouse control (move, swap, drop)
ui/       debug overlay window + FPS counter
main.py   game loop + hotkeys
```

## Install

```bash
pip install -r requirements.txt
```

`keyboard` and global mouse hooks usually require Administrator on Windows.

## Calibration

Run once before first play:

```bash
python main.py --calibrate
```

You will be prompted (each press of `Ctrl+Alt+Space` confirms the next click):

1. Top-left of the playable board
2. Bottom-right of the playable board
3. Center of the **current** piece spawn area (top row)
4. Center of the **preview** piece area
5. The bot will then sample colors. Place a Red, Blue, and Green tile on the
   board (or use the in-game tutorial state) and follow the prompts to click
   each color so HSV ranges are auto-tuned.

Calibration writes `calibration.json` in the project root. Edit by hand if
needed.

### Board geometry

The board geometry (columns, rows, hex pitch) is inferred from the rectangle
plus the `--cols` and `--rows` flags (defaults 7 columns × 8 rows, flat-top
hexes in odd-q offset layout). Override in `calibration.json` as needed.

The diagonal-pair geometry uses `pair_dx` / `pair_dy` (pixel offsets from the
pair's center to each of its two tiles). If pair color detection looks wrong
in the overlay, tune those two numbers in `calibration.json`.

## Run

```bash
python main.py
```

Hotkeys (global):

| Key            | Action |
|----------------|--------|
| Ctrl+Alt+P     | Pause / resume |
| Ctrl+Alt+D     | Toggle debug overlay |
| Ctrl+Alt+R     | Recalibrate (mouse points) |
| Ctrl+Alt+Q     | Quit |
| Ctrl+Alt+Space | (during `--calibrate`) capture the current mouse position / color sample |

## Debug overlay

`--debug` (or Ctrl+Alt+D at runtime) opens an OpenCV window showing:

- detected per-cell colors
- chosen drop column + swap state
- current/preview pair
- FPS

## Architecture notes

- **Vision** samples the *center* of each hex (HSV majority vote in a small
  patch) — cheap and robust against icon artwork because only color matters.
- **Simulator** uses an odd-r offset grid with 6-direction neighbor table.
  Gravity drops each column independently (split-pair behavior falls out of
  this naturally: each of the two tiles is dropped into its own column after
  the placement is decided). Chains resolve via flood-fill connected
  components.
- **Search** is a single-ply beam over (column, swap) placements for the
  *current* pair, plus a 1-ply lookahead simulating the *preview* pair landing
  in its best column on the resulting board. No deep tree — total move budget
  ~50 ms.
- **Heuristic** rewards immediate clears (weighted Red > Blue > Green),
  chain depth, low max-stack height, color clustering; penalizes isolated
  tiles and dead columns.

## End-of-game

Place button templates in `templates/brew_again.png` and `templates/ok.png`.
The main loop matches them with `cv2.matchTemplate` (TM_CCOEFF_NORMED ≥ 0.85)
and clicks the center to restart automatically.

If templates are missing the bot will skip restart handling and just log.

## Safety

Slam the mouse into the screen corner to trigger PyAutoGUI's failsafe abort,
or press Ctrl+Alt+Q.
