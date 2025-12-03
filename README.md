<<<<<<< HEAD
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
=======
# DBinit
A helper to write Discrete Breather initial displacements for LAMMPS simulations of a 2d data file.

---

## Installation

DBinit is a small Python / PyQt application. The easiest way to run it from source is:

```bash
git clone https://github.com/amiralitgh/dbinit.git
cd dbinit

python -m venv .venv
source .venv/bin/activate   # on Windows: ".venv\Scripts\activate.bat"

# install dependencies
pip install -r requirements.txt

# run the app
python -m dbi
>>>>>>> 59270780c85ed7169e4c418651d97781dca91635
