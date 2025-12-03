# Discrete Breather Init (DBI)

Visual DNVM / discrete-breather selection and **numeric** displacement export for LAMMPS.

## What it does
- 2D visualization of LAMMPS data files (Atoms section only)
- Rectangle and Horizontal/Vertical line selection (with spacing & offset)
- Circle mask with dimming (limit region of interest)
- Multiple color groups per selection, normalized directions
- Undo/Redo, action log
- Project Save/Load (session state)
- Localizing function **A/cosh(βR)** with live plot & sliders
- Numeric LAMMPS exporter with vector bucketing + single temporary group (avoids 32-group limit)

## Quick start (Windows)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m dbi examples\simple_square.data
```

## License
MIT
