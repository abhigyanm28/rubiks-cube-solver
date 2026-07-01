# Rubik's Cube Solver with OpenCV and Kociemba

A real-time Rubik's Cube solver built with Python and OpenCV that:

1. Scans a physical cube using your webcam
2. Classifies sticker colors with HSV/LAB-based logic
3. Computes a solution using the Kociemba two-phase algorithm
4. Guides the user with visual move overlays and a live state viewer

## Features

- Live webcam scanning of all 6 cube faces
- Kociemba-based optimal move sequence generation
- Mirrored-camera handling fallback
- Live viewer window using TCP socket communication
- Step-by-step move guidance overlays
- Manual correction commands for scanned stickers

## Tech Stack

- Python 3
- OpenCV
- NumPy
- kociemba
- socket
- pickle

## Project Structure

```text
Rubiks-Cube-Solver/
|-- Main.py        # Scanner, solver, and overlay logic
|-- State.py       # Live cube state viewer
|-- resources/     # Sticker images and move arrows
|-- README.md
```

## Installation

```bash
pip install opencv-python numpy kociemba
```

## Usage

1. Start the viewer in one terminal:

```bash
python State.py
```

2. Start the scanner/solver in another terminal:

```bash
python Main.py
```

3. Scan the cube faces using keys:

- U, R, F, D, L, B -> capture that face
- ESC -> finish scan

4. Follow solve guidance:

- SPACE -> advance to next move
- ESC -> exit

## Troubleshooting

- Viewer is blank: start `State.py` before `Main.py`
- No cube updates: check localhost port `9999`
- Color mismatch: tune thresholds in `classify_hue()` in `Main.py`

## License

MIT License
