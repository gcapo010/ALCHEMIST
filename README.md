# ALCHEMIST — Hex Match-3 Automation Bot

A computer-vision-only Windows bot for a hex-based 3-color match-3 alchemy
game. No memory reading, no OCR — just MSS screen capture, OpenCV HSV
segmentation, a fast hex simulator, and PyAutoGUI mouse input.

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

You will be prompted (each press of `F8` confirms the next click):

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
plus the `--cols` and `--rows` flags (defaults 8 columns × 12 rows, pointy-top
hexes in odd-r offset layout). Override in `calibration.json` as needed.

## Run

```bash
python main.py
```

Hotkeys (global):

| Key  | Action |
|------|--------|
| F7   | Pause / resume |
| F8   | Toggle debug overlay |
| F9   | Recalibrate (mouse points) |
| F12  | Quit |

## Debug overlay

`--debug` (or F8 at runtime) opens an OpenCV window showing:

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
or press F12.
