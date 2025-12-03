# DBinit
A helper to write Discrete Breather initial displacements for LAMMPS simulations of a 2d data file.

## Installation

```bash
git clone https://github.com/amiralitgh/dbinit.git
cd dbinit

python -m venv .venv
source .venv/bin/activate   # on Windows: ".venv\Scripts\activate.bat"

# install dependencies
pip install -r requirements.txt

# run the app
python -m dbi
