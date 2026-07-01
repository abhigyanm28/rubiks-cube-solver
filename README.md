# Rubik's Cube Solver

Rubik's Cube Solver is a Python project for scanning a physical cube, classifying the sticker colors, and generating a solve sequence. It also includes a lightweight cube-state viewer and a browser demo for presentation purposes.

## What It Does

- Captures cube faces with a webcam
- Classifies colors using OpenCV-based image analysis
- Builds a cube state and solves it with Kociemba
- Shows the cube state in a separate viewer window
- Includes a simple HTML demo for showcasing the project

## Project Files

```text
Rubiks-Cube-Solver/
|-- scanner_solver.py          # Webcam capture, color classification, and solver logic
|-- cube_viewer.py             # Live cube state viewer
|-- rubiks_cube_web_solver.html # Browser-based demo page
|-- resources/                 # Sticker images and move icons
|-- README.md
```

## Requirements

- Python 3
- OpenCV
- NumPy
- kociemba

Install the Python dependencies with:

```bash
pip install opencv-python numpy kociemba
```

## How To Run

1. Start the cube viewer:

```bash
python cube_viewer.py
```

2. Start the scanner and solver in another terminal:

```bash
python scanner_solver.py
```

3. Scan each face of the cube when prompted.
4. Follow the displayed solve moves until the cube is complete.

## Notes

- The project is designed for a local webcam and a physical cube.
- If the viewer opens blank, start `cube_viewer.py` before the scanner.
- The browser demo is separate from the Python solver and can be used for presentation or experimentation.

## License

MIT License