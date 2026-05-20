# Templates

Place button screenshots here as PNGs:

- `brew_again.png` — the "Brew Again" button from the end-game window
- `ok.png` — the confirmation popup OK button

The matcher uses `cv2.matchTemplate` with `TM_CCOEFF_NORMED >= 0.85`. Capture
the button as tightly cropped as possible from a normal-resolution screenshot
of the running game.
