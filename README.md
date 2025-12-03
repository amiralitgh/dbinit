# dBinit
A helper to write Discrete Breather initial displacements for LAMMPS simulations of a 2d data file.

---

## What it does (roughly)

DBI is a 2D viewer + selector for LAMMPS-style data files:

- **Loads a LAMMPS data file** and draws atoms in a 2D plot.
- Lets you create **groups of atoms** with different colors.
- Provides several **selection tools**:
  - Click selection
  - Rectangle selection
  - Horizontal / vertical line selection with _“keep 1 of N”_ pattern
  - Optional **circle mask** to dim atoms outside a given radius
- Per-group **displacement direction**:
  - Set direction by components (ux, uy, uz)
  - Or by angle in the XY plane
- **Localizing parameters** (“breather” parameters):
  - Amplitude `A`
  - `β` (beta)
  - Center `(x0, y0)` and related UI helpers
  - Optional “apply localizing function” toggle
- **Project files** (`.bpj`):
  - Save / load group assignments, colors, view, circle mask, etc.
- **Export helpers**:
  - `Export LAMMPS Displacement Command` – writes a small block you can include in your LAMMPS input to apply the displacements.
  - `Export IDs…` – dump selected atom IDs to a text file (all groups or just the active group).

Again: this is just a helper, not a full workflow manager.

---

## Installation

### 1. From source (Python)

You’ll need:

- Python 3.9+ (3.10+ recommended)
- A C-capable environment is **not** strictly necessary; everything is pure Python + wheels.
- On most platforms, `PyQt6` and `pyqtgraph` install via `pip` without extra steps.

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/your-user/discrete-breather-init.git
cd discrete-breather-init

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
