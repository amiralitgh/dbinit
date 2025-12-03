from __future__ import annotations
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

class LocalizeDialog(QtWidgets.QDialog):
    values_applied = QtCore.pyqtSignal(float, float, float, float)

    def __init__(self, A: float, beta: float, x0: float, y0: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Localizing Function (A/cosh(βR))")
        self.resize(600, 420)

        self.A = QtWidgets.QDoubleSpinBox(); self.A.setRange(0.0, 1e6); self.A.setDecimals(6); self.A.setValue(A)
        self.beta = QtWidgets.QDoubleSpinBox(); self.beta.setRange(0.0, 1e6); self.beta.setDecimals(6); self.beta.setValue(beta)
        self.x0 = QtWidgets.QDoubleSpinBox(); self.x0.setRange(-1e9, 1e9); self.x0.setDecimals(6); self.x0.setValue(x0)
        self.y0 = QtWidgets.QDoubleSpinBox(); self.y0.setRange(-1e9, 1e9); self.y0.setDecimals(6); self.y0.setValue(y0)

        form = QtWidgets.QFormLayout()
        form.addRow("A:", self.A)
        form.addRow("β:", self.beta)
        form.addRow("x0:", self.x0)
        form.addRow("y0:", self.y0)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setBackground("#0f0f0f")
        self.curve = self.plot.plot([], [], pen=pg.mkPen(200,200,255,200, width=2))
        self._update_plot()

        for w in (self.A, self.beta, self.x0, self.y0):
            w.valueChanged.connect(self._update_plot)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel |
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply_emit)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.plot, 1)
        layout.addWidget(btns)

    def _update_plot(self):
        A = self.A.value(); beta = self.beta.value()
        R = np.linspace(0.0, 10.0, 400)
        y = np.where(beta == 0.0, A*np.ones_like(R), A / np.cosh(beta*R))
        self.curve.setData(R, y)

    def _apply_emit(self):
        self.values_applied.emit(self.A.value(), self.beta.value(), self.x0.value(), self.y0.value())

    def accept(self):
        self._apply_emit()
        super().accept()
