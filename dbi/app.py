from pathlib import Path
from PyQt6 import QtWidgets, QtGui
from .ui.main_window import MainWindow
from .core.project_io import load_project
import sys

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Discrete Breather Init")
    app.setOrganizationName("DBI")
    app.setWindowIcon(QtGui.QIcon(str(Path(__file__).resolve().parent.parent / "assets" / "dbi_icon.svg")))

    w = MainWindow()
    w.setWindowIcon(app.windowIcon())

    if len(sys.argv) > 1:
        maybe_path = Path(sys.argv[1]).resolve()
        if maybe_path.exists():
            suffix = maybe_path.suffix.lower()
            if suffix in {".bpj", ".json"}:
                # open as project
                state, ui = load_project(maybe_path)
                w._apply_loaded_state(state, ui)
                w.current_project_path = maybe_path
            else:
                # open as LAMMPS data file
                w.load_datafile(str(maybe_path))

    w.show()
    sys.exit(app.exec())
