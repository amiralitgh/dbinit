from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import math
import numpy as np

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from ..core.lammps_parser import read_lammps_data
from ..core.model import DataSet, ProjectState, BreatherParams, Edit
from ..core.exporter import export_lammps_block, export_selected_ids
from ..core.color_utils import next_color, tuple_to_qcolor
from ..core.project_io import save_project, load_project
from .localize_dialog import LocalizeDialog

UNASSIGNED_RGBA = (160,160,160,255)
pg.setConfigOptions(antialias=True)


def _lighter(c: QtGui.QColor, f: float = 1.25) -> QtGui.QColor:
    h, s, v, a = c.getHsv()
    v = min(255, int(v * f))
    return QtGui.QColor.fromHsv(h, s, v, a)


def _make_shaded_brush(base: Tuple[int, int, int, int]) -> QtGui.QBrush:
    c = QtGui.QColor(*base)
    light = _lighter(c, 1.35)
    grad = QtGui.QRadialGradient(QtCore.QPointF(0.35, 0.35), 0.9)
    grad.setCoordinateMode(QtGui.QGradient.CoordinateMode.ObjectBoundingMode)
    grad.setColorAt(0.0, light)
    grad.setColorAt(0.55, c)
    grad.setColorAt(1.0, QtGui.QColor(c.red() // 2, c.green() // 2, c.blue() // 2, c.alpha()))
    return QtGui.QBrush(grad)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Discrete Breather Init (DBI)")
        self.resize(1320, 880)

        self.dataset: Optional[DataSet] = None
        self.state: Optional[ProjectState] = None
        self.active_group_id: int = 0
        self.group_counter = 0

        self.circle_enabled = False
        self.circle_center: Optional[Tuple[float, float]] = None
        self.circle_radius: float = 0.0
        self.circle_item = None

        self.mode = "select"  # 'select', 'rect', 'hline', 'vline'
        self.dragging_rect = False
        self.rect_origin_scene: Optional[QtCore.QPointF] = None
        self.rect_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self.line_item: Optional[QtWidgets.QGraphicsLineItem] = None
        self.line_pen = pg.mkPen((255, 255, 0, 180), width=1, style=QtCore.Qt.PenStyle.DashLine)

        self.last_line_orientation: Optional[str] = None  # 'h' or 'v'
        self.last_line_indices: Optional[np.ndarray] = None

        self.base_diam_data: float = 0.0
        self.atom_select_radius: float = 0.0

        self.current_project_path: Optional[Path] = None
        self.is_dirty: bool = False

        self.use_shading = True
        self._brush_cache: Dict[Tuple[int, int, int, int], QtGui.QBrush] = {}

        self._build_menu()
        self._build_quick_toolbar()
        self._build_toolbar_tools()
        self._build_canvas()
        self._build_docks()
        self._update_enabled(False)

    # ---------- UI helpers ----------
    def _with_label(self, text, widget):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(8, 0, 8, 0)
        l.addWidget(QtWidgets.QLabel(text))
        l.addWidget(widget)
        l.addStretch(1)
        return w

    # ---------- Menus & toolbars ----------
    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        self.act_open_data = QtGui.QAction("Open data…", self)
        self.act_open_data.triggered.connect(self.on_open_data)
        self.act_open_project = QtGui.QAction("Open project…", self)
        self.act_open_project.triggered.connect(self.on_open_project)
        self.act_save_project = QtGui.QAction("Save project", self)
        self.act_save_project.triggered.connect(self.on_save_project)
        self.act_save_as = QtGui.QAction("Save project as…", self)
        self.act_save_as.triggered.connect(lambda: self.on_save_project(save_as=True))
        self.act_export = QtGui.QAction("Export LAMMPS block…", self)
        self.act_export.triggered.connect(self.on_export)
        act_quit = QtGui.QAction("Quit", self)
        act_quit.triggered.connect(self.close)

        for a in (
            self.act_open_data,
            self.act_open_project,
            self.act_save_project,
            self.act_save_as,
            self.act_export,
        ):
            file_menu.addAction(a)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        edit_menu = menubar.addMenu("&Edit")
        self.act_undo = QtGui.QAction("Undo", self)
        self.act_undo.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.on_undo)
        self.act_redo = QtGui.QAction("Redo", self)
        self.act_redo.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        self.act_redo.triggered.connect(self.on_redo)
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)

        view = menubar.addMenu("&View")
        self.point_scale_spin = QtWidgets.QSpinBox()
        self.point_scale_spin.setRange(10, 400)
        self.point_scale_spin.setValue(100)
        self.point_scale_spin.valueChanged.connect(self._on_point_scale_changed)
        act_pointscale = QtWidgets.QWidgetAction(self)
        act_pointscale.setDefaultWidget(self._with_label("Point scale (%)", self.point_scale_spin))
        view.addAction(act_pointscale)

    def _build_quick_toolbar(self):
        tb = QtWidgets.QToolBar("Quick", self)
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(22, 22))
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)

        style = self.style()
        ico_open = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
        ico_save = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton)
        ico_undo = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack)
        ico_redo = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward)
        ico_export = style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DriveFDIcon)

        a_undo = QtGui.QAction(ico_undo, "Undo", self)
        a_undo.triggered.connect(self.on_undo)
        a_undo.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        a_redo = QtGui.QAction(ico_redo, "Redo", self)
        a_redo.triggered.connect(self.on_redo)
        a_redo.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        a_open = QtGui.QAction(ico_open, "Open", self)
        a_open.triggered.connect(self.on_open_project)
        a_save = QtGui.QAction(ico_save, "Save", self)
        a_save.triggered.connect(self.on_save_project)
        a_export = QtGui.QAction(ico_export, "Export", self)
        a_export.triggered.connect(self.on_export)

        tb.addAction(a_undo)
        tb.addAction(a_redo)
        tb.addSeparator()
        tb.addAction(a_open)
        tb.addAction(a_save)
        tb.addSeparator()
        tb.addAction(a_export)

        self.tb_undo = a_undo
        self.tb_redo = a_redo
        self.tb_open = a_open
        self.tb_save = a_save
        self.tb_export = a_export

    def _build_toolbar_tools(self):
        tb = self.addToolBar("Tools")
        tb.setMovable(False)
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)

        self.btn_select = QtGui.QAction("Click Select (S)", self)
        self.btn_select.setCheckable(True)
        self.btn_select.setShortcut("S")
        self.btn_rect = QtGui.QAction("Rect Select", self)
        self.btn_rect.setCheckable(True)
        self.btn_hline = QtGui.QAction("Horizontal Line (H)", self)
        self.btn_hline.setCheckable(True)
        self.btn_hline.setShortcut("H")
        self.btn_vline = QtGui.QAction("Vertical Line (V)", self)
        self.btn_vline.setCheckable(True)
        self.btn_vline.setShortcut("V")

        for a in (self.btn_select, self.btn_rect, self.btn_hline, self.btn_vline):
            group.addAction(a)
            tb.addAction(a)

        # Line selection controls (only visible in H/V line mode)
        self.line_spacing = QtWidgets.QSpinBox()
        self.line_spacing.setRange(1, 9999)
        self.line_spacing.setValue(3)
        self.line_offset = QtWidgets.QSpinBox()
        self.line_offset.setRange(0, 9998)
        self.line_offset.setValue(0)

        line_widget = QtWidgets.QWidget()
        lw = QtWidgets.QHBoxLayout(line_widget)
        lw.setContentsMargins(6, 0, 6, 0)
        lw.addWidget(QtWidgets.QLabel("Keep 1 of N:"))
        lw.addWidget(self.line_spacing)
        lw.addWidget(QtWidgets.QLabel("Offset:"))
        lw.addWidget(self.line_offset)

        self.line_widget_action = QtWidgets.QWidgetAction(self)
        self.line_widget_action.setDefaultWidget(line_widget)
        tb.addAction(self.line_widget_action)
        self.line_widget_action.setVisible(False)

        self.btn_select.toggled.connect(lambda on: self._set_mode("select" if on else None))
        self.btn_rect.toggled.connect(lambda on: self._set_mode("rect" if on else None))
        self.btn_hline.toggled.connect(lambda on: self._set_mode("hline" if on else None))
        self.btn_vline.toggled.connect(lambda on: self._set_mode("vline" if on else None))

        tb.addSeparator()
        self.btn_circle = QtGui.QAction("Circle Mask", self)
        self.btn_circle.setCheckable(True)
        self.btn_circle.toggled.connect(self.on_toggle_circle)
        tb.addAction(self.btn_circle)

        tb.addSeparator()
        self.btn_localize = QtGui.QAction("Localizing…", self)
        self.btn_localize.triggered.connect(self.on_open_localizer)
        tb.addAction(self.btn_localize)

    def _build_canvas(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setAspectLocked(True)
        self.viewbox = self.plot.getPlotItem().getViewBox()
        self.box_item: Optional[QtWidgets.QGraphicsRectItem] = None

        self.scatter = pg.ScatterPlotItem(
            size=1.0,
            pxMode=False,
            symbol="o",
            brush=pg.mkBrush(*UNASSIGNED_RGBA),
            pen=pg.mkPen(40, 40, 40, 130, width=0.5),
        )
        self.plot.addItem(self.scatter)
        v.addWidget(self.plot)

        bottom = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bottom)
        h.setContentsMargins(6, 4, 6, 4)

        self.coord_label = QtWidgets.QLabel(" x: —    y: — ")
        self.coord_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.coord_label.setStyleSheet(
            "QLabel { background:#111; color:#ddd; padding: 4px 8px; "
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; }"
        )
        h.addWidget(self.coord_label, 1)

        v.addWidget(bottom)

        self.plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)

    def _build_docks(self):
        # Single right-side dock with scrollable content
        self.tools_dock = QtWidgets.QDockWidget("Tools", self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.tools_dock)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.tools_dock.setWidget(scroll)

        container = QtWidgets.QWidget()
        scroll.setWidget(container)

        container.setMinimumWidth(0)
        container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        gl = QtWidgets.QVBoxLayout(container)
        gl.setContentsMargins(8, 8, 8, 8)
        gl.setSpacing(8)

        # 1) Selection tools
        self._build_selection_panel(gl)

        # 2) Groups & Localizing
        self.group_list = QtWidgets.QListWidget()
        self.group_list.currentRowChanged.connect(self.on_group_selection_changed)

        btns = QtWidgets.QHBoxLayout()
        self.btn_add_group = QtWidgets.QPushButton("Add Group")
        self.btn_add_group.clicked.connect(self.on_add_group)
        self.btn_remove_group = QtWidgets.QPushButton("Remove")
        self.btn_remove_group.clicked.connect(self.on_remove_group)
        btns.addWidget(self.btn_add_group)
        btns.addWidget(self.btn_remove_group)

        color_row = QtWidgets.QHBoxLayout()
        self.btn_color = QtWidgets.QPushButton("Color…")
        self.btn_color.clicked.connect(self.on_change_color)
        self.chk_shading = QtWidgets.QCheckBox("Shaded spheres")
        self.chk_shading.setChecked(True)
        self.chk_shading.toggled.connect(self.on_toggle_shading)
        color_row.addWidget(self.btn_color)
        color_row.addWidget(self.chk_shading)
        color_row.addStretch(1)

        dir_box = QtWidgets.QGroupBox("Displacement Vector Direction")
        dir_box.setToolTip("Per-group displacement direction (unit vector).")
        form = QtWidgets.QFormLayout(dir_box)
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(-360, 360)
        self.angle_spin.setSuffix("°")
        self.angle_spin.valueChanged.connect(self.on_angle_changed)
        self.ux = QtWidgets.QDoubleSpinBox(); self.ux.setRange(-1e3, 1e3); self.ux.setDecimals(6)
        self.uy = QtWidgets.QDoubleSpinBox(); self.uy.setRange(-1e3, 1e3); self.uy.setDecimals(6)
        self.uz = QtWidgets.QDoubleSpinBox(); self.uz.setRange(-1e3, 1e3); self.uz.setDecimals(6); self.uz.setValue(0.0)
        for w in (self.ux, self.uy, self.uz):
            w.valueChanged.connect(self.on_components_changed)
        form.addRow("Angle in XY:", self.angle_spin)
        form.addRow("ux:", self.ux)
        form.addRow("uy:", self.uy)
        form.addRow("uz:", self.uz)

        b_box = QtWidgets.QGroupBox("Localizing Parameters")
        b_box.setToolTip("Localizing function center and parameters used to compute displacements.")
        bform = QtWidgets.QFormLayout(b_box)

        self.A = QtWidgets.QDoubleSpinBox(); self.A.setRange(0.0, 1e6); self.A.setDecimals(6); self.A.setValue(0.05)
        self.beta = QtWidgets.QDoubleSpinBox(); self.beta.setRange(0.0, 1e6); self.beta.setDecimals(6); self.beta.setValue(1.0)
        self.x0 = QtWidgets.QDoubleSpinBox(); self.x0.setRange(-1e9, 1e9); self.x0.setDecimals(6)
        self.y0 = QtWidgets.QDoubleSpinBox(); self.y0.setRange(-1e9, 1e9); self.y0.setDecimals(6)

        self.chk_apply_local = QtWidgets.QCheckBox("Apply localizing function (A/cosh(βR))")
        self.chk_apply_local.setChecked(True)
        self.chk_apply_local.toggled.connect(self.on_toggle_apply_local)
        self.chk_preserve = QtWidgets.QCheckBox("Preserve base selection (additive tools)")
        self.chk_preserve.setChecked(True)
        self.chk_preserve.toggled.connect(self.on_toggle_preserve)

        self.btn_localize_inpanel = QtWidgets.QPushButton("Parameters…")
        self.btn_localize_inpanel.clicked.connect(self.on_open_localizer)

        center_row = QtWidgets.QHBoxLayout()
        center_row.addWidget(QtWidgets.QLabel("Center (x0,y0):"))
        center_row.addWidget(self.x0)
        center_row.addWidget(self.y0)
        self.btn_pick_center = QtWidgets.QPushButton("Pick on canvas…")
        self.btn_pick_center.setToolTip("Pick the breather center by clicking on the canvas.")
        self.btn_pick_center.clicked.connect(self.on_pick_center_dialog)
        center_row.addWidget(self.btn_pick_center)

        # Circle mask radius inside Localizing Parameters
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 1e12)
        self.radius_spin.setDecimals(6)
        self.radius_spin.setValue(0.0)
        self.radius_spin.valueChanged.connect(self.on_radius_changed)
        bform.addRow("Circle mask radius:", self.radius_spin)
        radius_hint = QtWidgets.QLabel(
            "Toggle Circle Mask from the toolbar; atoms outside this radius are dimmed."
        )
        radius_hint.setWordWrap(True)
        bform.addRow(radius_hint)

        bform.addRow("A:", self.A)
        bform.addRow("β:", self.beta)
        bform.addRow(center_row)
        bform.addRow("", self.chk_apply_local)
        bform.addRow("", self.chk_preserve)
        bform.addRow("", self.btn_localize_inpanel)

        gl.addWidget(QtWidgets.QLabel("Groups (Active selected):"))
        gl.addWidget(self.group_list, 1)
        gl.addLayout(btns)
        gl.addLayout(color_row)
        gl.addWidget(dir_box)
        gl.addWidget(b_box)

        export_row = QtWidgets.QHBoxLayout()
        self.btn_export_panel = QtWidgets.QPushButton("Export LAMMPS Displacement Command")
        self.btn_export_panel.clicked.connect(self.on_export)
        self.btn_export_ids = QtWidgets.QPushButton("Export IDs…")
        self.btn_export_ids.clicked.connect(self.on_export_ids)
        export_row.addWidget(self.btn_export_panel)
        export_row.addWidget(self.btn_export_ids)
        gl.addLayout(export_row)

        gl.addStretch(1)

    def _build_selection_panel(self, parent_layout: QtWidgets.QVBoxLayout):
        sel_box = QtWidgets.QGroupBox("Selection Tools")
        L = QtWidgets.QFormLayout(sel_box)
        L.setContentsMargins(8, 8, 8, 8)

        self.atom_radius_spin = QtWidgets.QDoubleSpinBox()
        self.atom_radius_spin.setRange(0.0, 1e9)
        self.atom_radius_spin.setDecimals(6)
        self.atom_radius_spin.setValue(0.0)
        self.atom_radius_spin.valueChanged.connect(self.on_atom_radius_changed)
        L.addRow("Atom radius:", self.atom_radius_spin)

        self.btn_rule = QtWidgets.QPushButton("Selection Rule…")
        self.btn_rule.setToolTip(
            "Apply a structured selection (rows/columns) to the whole dataset, "
            "based on the last horizontal or vertical line selection."
        )
        self.btn_rule.clicked.connect(self.on_rule_clicked)
        L.addRow(self.btn_rule)

        parent_layout.addWidget(sel_box)

    # ---------- Enable/disable ----------
    def _update_enabled(self, enabled: bool):
        self.act_open_data.setEnabled(True)
        self.act_open_project.setEnabled(True)
        if hasattr(self, "tb_open"):
            self.tb_open.setEnabled(True)

        for a in (
            self.act_save_project,
            self.act_save_as,
            self.act_export,
            self.act_undo,
            self.act_redo,
            self.tb_save,
            self.tb_export,
            self.tb_undo,
            self.tb_redo,
            self.btn_localize,
        ):
            a.setEnabled(enabled)

        widgets = [
            self.group_list,
            self.btn_add_group,
            self.btn_remove_group,
            self.btn_color,
            self.chk_shading,
            self.angle_spin,
            self.ux,
            self.uy,
            self.uz,
            self.A,
            self.beta,
            self.x0,
            self.y0,
            self.btn_localize_inpanel,
            self.btn_pick_center,
            self.point_scale_spin,
            self.btn_select,
            self.btn_rect,
            self.btn_hline,
            self.btn_vline,
            self.atom_radius_spin,
            self.line_spacing,
            self.line_offset,
            self.radius_spin,
            self.chk_apply_local,
            self.chk_preserve,
            self.btn_rule,
            self.btn_export_panel,
            self.btn_export_ids,
        ]
        for w in widgets:
            if isinstance(w, QtGui.QAction):
                w.setEnabled(enabled)
            elif w is not None:
                w.setEnabled(enabled)

        self._refresh_undo_redo_actions()

    # ---------- Data load/save ----------
    def on_open_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open LAMMPS data file", "", "LAMMPS data (*.data *.dat *.*)"
        )
        if path:
            self.load_datafile(path)

    def on_open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", "", "DBI Project (*.bpj *.json)"
        )
        if not path:
            return
        state, ui = load_project(Path(path))
        self._apply_loaded_state(state, ui)
        self.current_project_path = Path(path)
        self.is_dirty = False
        self._log(f"Opened project: {path}")

    def on_save_project(self, save_as: bool = False):
        if self.state is None:
            return
        if save_as or self.current_project_path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save project", "", "DBI Project (*.bpj)"
            )
            if not path:
                return
            self.current_project_path = Path(path)
        ui = self._gather_ui_state()
        save_project(self.state, self.current_project_path, ui_state=ui)
        self.is_dirty = False
        self._log(f"Saved project: {self.current_project_path}")

    def _gather_ui_state(self) -> Dict[str, Any]:
        vr = self.viewbox.viewRect()
        return {
            "point_scale": int(self.point_scale_spin.value()),
            "circle_enabled": bool(self.circle_enabled),
            "circle_center": list(self.circle_center) if self.circle_center else None,
            "circle_radius": float(self.circle_radius),
            "active_group_id": int(self.active_group_id),
            "view": {
                "x": [vr.left(), vr.right()],
                "y": [vr.top(), vr.bottom()],
            },
            "use_shading": bool(self.use_shading),
            "atom_radius": float(self.atom_radius_spin.value()),
            "apply_localizing": bool(self.state.apply_localizing if self.state else True),
            "preserve_base": bool(
                self.state.preserve_base_selection if self.state else True
            ),
        }

    def _apply_loaded_state(self, state: ProjectState, ui: Dict[str, Any]):
        self.state = state
        self.dataset = state.data
        self.group_counter = max(state.groups.keys(), default=0)
        self.group_list.clear()
        for gid in sorted(state.groups.keys()):
            self._add_group_item(state.groups[gid])

        active_gid = int(ui.get("active_group_id", next(iter(state.groups.keys()), 0)))
        if active_gid in state.groups:
            for row in range(self.group_list.count()):
                item = self.group_list.item(row)
                if int(item.data(QtCore.Qt.ItemDataRole.UserRole)) == active_gid:
                    self.group_list.setCurrentRow(row)
                    break
        elif self.group_list.count() > 0:
            self.group_list.setCurrentRow(0)

        self.A.setValue(state.breather.A)
        self.beta.setValue(state.breather.beta)
        self.x0.setValue(state.breather.x0)
        self.y0.setValue(state.breather.y0)

        self.circle_enabled = bool(ui.get("circle_enabled", False))
        self.btn_circle.setChecked(self.circle_enabled)
        cc = ui.get("circle_center")
        self.circle_center = (float(cc[0]), float(cc[1])) if cc else None
        self.circle_radius = float(ui.get("circle_radius", 0.0))
        self.radius_spin.setValue(self.circle_radius)

        ps = int(ui.get("point_scale", 100))
        self.point_scale_spin.blockSignals(True)
        self.point_scale_spin.setValue(ps)
        self.point_scale_spin.blockSignals(False)
        self.use_shading = bool(ui.get("use_shading", True))
        self.chk_shading.setChecked(self.use_shading)
        self.atom_radius_spin.setValue(float(ui.get("atom_radius", 0.0)))

        self.state.apply_localizing = bool(ui.get("apply_localizing", True))
        self.chk_apply_local.setChecked(self.state.apply_localizing)
        self.state.preserve_base_selection = bool(ui.get("preserve_base", True))
        self.chk_preserve.setChecked(self.state.preserve_base_selection)

        self._populate_scatter()
        self._draw_cell_outline()
        view = ui.get("view")
        if view and "x" in view and "y" in view:
            self.plot.setXRange(view["x"][0], view["x"][1], padding=0.0)
            self.plot.setYRange(view["y"][0], view["y"][1], padding=0.0)
        else:
            self._fit_view()

        self.btn_select.setChecked(True)
        self._update_enabled(True)
        self.update_scatter_colors()
        self._update_circle_item()

    def load_datafile(self, path: str):
        data = read_lammps_data(path)
        self.dataset = data
        self.state = ProjectState(data)
        self.state.breather = BreatherParams(
            A=self.A.value(),
            beta=self.beta.value(),
            x0=0.5 * (data.box.xlo + data.box.xhi),
            y0=0.5 * (data.box.ylo + data.box.yhi),
        )
        self.x0.setValue(self.state.breather.x0)
        self.y0.setValue(self.state.breather.y0)
        self.state.apply_localizing = self.chk_apply_local.isChecked()
        self.state.preserve_base_selection = self.chk_preserve.isChecked()

        self._init_groups()
        self._populate_scatter()
        self._fit_view()
        self._draw_cell_outline()
        self._log(f"Loaded data file: {path} with {data.ids.size} atoms.")
        self._update_enabled(True)
        self.btn_select.setChecked(True)
        self.current_project_path = None
        self.is_dirty = False

    # ---------- Groups ----------
    def _init_groups(self):
        self.group_list.clear()
        self.state.clear_groups()
        self.active_group_id = 0
        self.on_add_group()

    def on_add_group(self):
        self.group_counter += 1
        gid = self.group_counter
        color = next_color(gid - 1)
        g = self.state.add_group(
            gid,
            f"Group {gid}",
            (color.red(), color.green(), color.blue(), 255),
        )
        self._add_group_item(g)
        self.group_list.setCurrentRow(self.group_list.count() - 1)
        self._log(f"Created group {gid}.")
        self.is_dirty = True

    def _add_group_item(self, g):
        item = QtWidgets.QListWidgetItem(g.name)
        pix = QtGui.QPixmap(16, 16)
        pix.fill(tuple_to_qcolor(g.color))
        item.setIcon(QtGui.QIcon(pix))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, g.gid)
        self.group_list.addItem(item)

    def on_remove_group(self):
        item = self.group_list.currentItem()
        if not item:
            return
        gid = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        self.state.remove_group(gid)
        self.group_list.takeItem(self.group_list.row(item))
        self._log(f"Removed group {gid} (atoms reverted to unassigned).")
        self.update_scatter_colors()
        self._refresh_undo_redo_actions()
        self.is_dirty = True

    def on_group_selection_changed(self, row: int):
        if row < 0:
            self.active_group_id = 0
            return
        item = self.group_list.item(row)
        gid = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        self.active_group_id = gid
        g = self.state.groups[gid]
        self.ux.blockSignals(True)
        self.uy.blockSignals(True)
        self.uz.blockSignals(True)
        self.angle_spin.blockSignals(True)
        self.ux.setValue(g.direction[0])
        self.uy.setValue(g.direction[1])
        self.uz.setValue(g.direction[2])
        self.angle_spin.setValue(math.degrees(math.atan2(g.direction[1], g.direction[0])))
        self.ux.blockSignals(False)
        self.uy.blockSignals(False)
        self.uz.blockSignals(False)
        self.angle_spin.blockSignals(False)

    def on_change_color(self):
        item = self.group_list.currentItem()
        if not item:
            return
        gid = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        g = self.state.groups[gid]
        col = QtWidgets.QColorDialog.getColor(
            tuple_to_qcolor(g.color), self, "Pick group color"
        )
        if col.isValid():
            g.color = (col.red(), col.green(), col.blue(), 255)
            pix = QtGui.QPixmap(16, 16)
            pix.fill(col)
            item.setIcon(QtGui.QIcon(pix))
            self.update_scatter_colors()
            self._log(f"Changed color of group {gid}.")
            self.is_dirty = True

    def on_angle_changed(self, ang_deg: float):
        ux = math.cos(math.radians(ang_deg))
        uy = math.sin(math.radians(ang_deg))
        self.ux.blockSignals(True)
        self.uy.blockSignals(True)
        self.ux.setValue(ux)
        self.uy.setValue(uy)
        self.ux.blockSignals(False)
        self.uy.blockSignals(False)
        self.on_components_changed()

    def on_components_changed(self):
        item = self.group_list.currentItem()
        if not item or not self.state:
            return
        gid = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
        g = self.state.groups[gid]
        v = np.array(
            [self.ux.value(), self.uy.value(), self.uz.value()], dtype=float
        )
        n = np.linalg.norm(v)
        v = np.array([1.0, 0.0, 0.0]) if n == 0 else v / n
        g.direction = (float(v[0]), float(v[1]), float(v[2]))
        self._log(
            f"Direction of group {gid} = ({g.direction[0]:.6f}, {g.direction[1]:.6f}, {g.direction[2]:.6f})"
        )
        self.is_dirty = True

    def on_atom_radius_changed(self):
        """Apply atom radius both to hit-testing and on-screen size."""
        if self.dataset is None:
            return
        r = float(self.atom_radius_spin.value())
        self.atom_select_radius = r

        # Map radius -> point diameter (simple: diameter ≈ 2r)
        if r <= 0.0:
            # fall back to automatic estimate
            self.base_diam_data = self._estimate_point_diameter()
        else:
            new_diam = 2.0 * r
            # keep point scale behavior consistent
            scale = max(1e-6, self.point_scale_spin.value() / 100.0)
            self.base_diam_data = new_diam / scale

        # update scatter item size
        self.scatter.setSize(self._current_diam())
        self.is_dirty = True


    # ---------- Visuals ----------
    def _estimate_point_diameter(self) -> float:
        b = self.dataset.box
        area = (b.xhi - b.xlo) * (b.yhi - b.ylo)
        n = max(1, self.dataset.ids.size)
        return 0.55 * math.sqrt(area / n)

    def _current_diam(self) -> float:
        return self.base_diam_data * (self.point_scale_spin.value() / 100.0)

    def _populate_scatter(self):
        if not self.dataset:
            return
        x = self.dataset.x
        y = self.dataset.y
        self.base_diam_data = self._estimate_point_diameter()
        if self.atom_select_radius == 0.0:
            self.atom_select_radius = 0.5 * self.base_diam_data
            self.atom_radius_spin.setValue(self.atom_select_radius)
        self.scatter.setData(
            x=x,
            y=y,
            size=self._current_diam(),
            pxMode=False,
            symbol="o",
            pen=pg.mkPen(40, 40, 40, 130, width=0.5),
        )
        self.update_scatter_colors()

    def on_toggle_shading(self, checked: bool):
        self.use_shading = checked
        self.update_scatter_colors()

    def update_scatter_colors(self):
        if not self.state:
            return
        n = self.state.assignment.size
        if self.use_shading:
            base_brushes = []
            for i in range(n):
                gid = self.state.assignment[i]
                key = UNASSIGNED_RGBA if gid == 0 else self.state.groups[gid].color
                b = self._brush_cache.get(key)
                if b is None:
                    b = _make_shaded_brush(key)
                    self._brush_cache[key] = b
                base_brushes.append(b)
            if self.circle_enabled and self.circle_center and self.circle_radius > 0.0:
                x0, y0 = self.circle_center
                R2 = self.circle_radius * self.circle_radius
                dx = self.dataset.x - x0
                dy = self.dataset.y - y0
                inside = (dx * dx + dy * dy) <= R2
                brushes = []
                for i in range(n):
                    if inside[i]:
                        brushes.append(base_brushes[i])
                    else:
                        gid = self.state.assignment[i]
                        col = (
                            UNASSIGNED_RGBA
                            if gid == 0
                            else self.state.groups[gid].color
                        )
                        dim_col = (
                            max(0, int(col[0] * 0.4)),
                            max(0, int(col[1] * 0.4)),
                            max(0, int(col[2] * 0.4)),
                            255,
                        )
                        brushes.append(QtGui.QBrush(QtGui.QColor(*dim_col)))
                self.scatter.setBrush(brushes)
            else:
                self.scatter.setBrush(base_brushes)
        else:
            rgba = self.state.colors_rgba_for_all(UNASSIGNED_RGBA)
            if self.circle_enabled and self.circle_center and self.circle_radius > 0.0:
                x0, y0 = self.circle_center
                R2 = self.circle_radius * self.circle_radius
                dx = self.dataset.x - x0
                dy = self.dataset.y - y0
                inside = (dx * dx + dy * dy) <= R2
                out_rgba = rgba.copy()
                out_rgba[~inside, :3] = (out_rgba[~inside, :3] * 0.4).astype(np.ubyte)
                self.scatter.setBrush([
                    pg.mkBrush(*out_rgba[i]) for i in range(n)
                ])
            else:
                self.scatter.setBrush([
                    pg.mkBrush(*rgba[i]) for i in range(n)
                ])

    def _on_point_scale_changed(self, _):
        if self.dataset is not None:
            self.scatter.setSize(self._current_diam())
            self.is_dirty = True

    def _fit_view(self):
        b = self.dataset.box
        self.plot.setXRange(b.xlo, b.xhi, padding=0.03)
        self.plot.setYRange(b.ylo, b.yhi, padding=0.03)

    def _draw_cell_outline(self):
        if self.box_item is not None:
            self.plot.removeItem(self.box_item)
            self.box_item = None
        b = self.dataset.box
        rect = QtCore.QRectF(b.xlo, b.ylo, b.xhi - b.xlo, b.yhi - b.ylo)
        self.box_item = QtWidgets.QGraphicsRectItem(rect)
        self.box_item.setPen(pg.mkPen(220, 220, 220, 160, width=1))
        self.plot.addItem(self.box_item)

    # ---------- Modes & mouse ----------
    def _set_mode(self, m: Optional[str]):
        if not m:
            return
        if m != "rect":
            self._clear_rect_overlay()
        if m not in ("hline", "vline"):
            self._remove_line_overlay()
        self.mode = m
        if m in ("hline", "vline"):
            self._ensure_line_overlay()
        if hasattr(self, "line_widget_action"):
            self.line_widget_action.setVisible(m in ("hline", "vline"))
        self._log(f"Mode: {m}")

    def _clear_rect_overlay(self):
        if self.rect_item is not None:
            self.plot.removeItem(self.rect_item)
            self.rect_item = None
        self.dragging_rect = False
        self.rect_origin_scene = None

    def _ensure_line_overlay(self):
        if self.line_item is None:
            self.line_item = QtWidgets.QGraphicsLineItem()
            self.line_item.setPen(self.line_pen)
            self.plot.addItem(self.line_item)

    def _remove_line_overlay(self):
        if self.line_item is not None:
            self.plot.removeItem(self.line_item)
            self.line_item = None

    def on_mouse_moved(self, scene_pos):
        pt = self.viewbox.mapSceneToView(scene_pos)
        self.coord_label.setText(f"  x: {pt.x():.6f}    y: {pt.y():.6f}  ")
        if self.mode == "rect" and self.dragging_rect and self.rect_origin_scene is not None:
            self._update_rect_item(self.rect_origin_scene, scene_pos)
        if self.dataset and self.mode in ("hline", "vline") and self.line_item is not None:
            b = self.dataset.box
            if self.mode == "hline":
                y = pt.y()
                self.line_item.setLine(b.xlo, y, b.xhi, y)
            else:
                x = pt.x()
                self.line_item.setLine(x, b.ylo, x, b.yhi)

    def on_mouse_clicked(self, ev):
        if not self.dataset:
            return
        btn = ev.button()
        pt = self.viewbox.mapSceneToView(ev.scenePos())
        x, y = pt.x(), pt.y()

        if self.mode == "rect":
            if btn == QtCore.Qt.MouseButton.LeftButton:
                if not self.dragging_rect:
                    self.dragging_rect = True
                    self.rect_origin_scene = ev.scenePos()
                    self._update_rect_item(self.rect_origin_scene, ev.scenePos())
                else:
                    rect = self.rect_item.rect()
                    idx = np.nonzero(
                        (self.dataset.x >= rect.left())
                        & (self.dataset.x <= rect.right())
                        & (self.dataset.y >= rect.top())
                        & (self.dataset.y <= rect.bottom())
                    )[0]
                    idx = self._apply_circle_constraint(idx)
                    if idx.size:
                        self._assign_indices(
                            idx,
                            mode="add"
                            if self.state.preserve_base_selection
                            else "toggle",
                        )
                    self._clear_rect_overlay()
                return
            elif btn == QtCore.Qt.MouseButton.RightButton:
                self._clear_rect_overlay()
                return

        if self.mode in ("hline", "vline"):
            if btn == QtCore.Qt.MouseButton.LeftButton:
                R = float(self.atom_radius_spin.value())
                if R <= 0.0:
                    if self.base_diam_data == 0.0:
                        self.base_diam_data = self._estimate_point_diameter()
                    R = 0.5 * self.base_diam_data
                    self.atom_radius_spin.setValue(R)
                N = max(1, int(self.line_spacing.value()))
                off = int(self.line_offset.value()) % N

                if self.mode == "hline":
                    dy = np.abs(self.dataset.y - y)
                    idx = np.nonzero(dy <= R)[0]
                    idx = self._apply_circle_constraint(idx)
                    if idx.size == 0:
                        self._log("Horizontal line: 0 atoms hit.")
                        return
                    order = np.argsort(self.dataset.x[idx])
                    idx_sorted = idx[order]
                    keep = np.arange(idx_sorted.size) % N == off
                    hits = idx_sorted[keep]
                    self.last_line_orientation = "h"
                    self.last_line_indices = hits.copy()
                    self._assign_indices(
                        hits,
                        mode="add"
                        if self.state.preserve_base_selection
                        else "toggle",
                    )
                    self._log(
                        f"H-line @ y={y:.6f}: hit {idx.size}, kept {hits.size} (N={N}, off={off})."
                    )
                else:
                    dx = np.abs(self.dataset.x - x)
                    idx = np.nonzero(dx <= R)[0]
                    idx = self._apply_circle_constraint(idx)
                    if idx.size == 0:
                        self._log("Vertical line: 0 atoms hit.")
                        return
                    order = np.argsort(self.dataset.y[idx])
                    idx_sorted = idx[order]
                    keep = np.arange(idx_sorted.size) % N == off
                    hits = idx_sorted[keep]
                    self.last_line_orientation = "v"
                    self.last_line_indices = hits.copy()
                    self._assign_indices(
                        hits,
                        mode="add"
                        if self.state.preserve_base_selection
                        else "toggle",
                    )
                    self._log(
                        f"V-line @ x={x:.6f}: hit {idx.size}, kept {hits.size} (N={N}, off={off})."
                    )
                return

        if self.mode == "select" and btn == QtCore.Qt.MouseButton.LeftButton:
            idx = self._nearest_atom_index(x, y)
            if idx is None:
                return
            if self.circle_enabled and self.circle_center and self.circle_radius > 0.0:
                cx, cy = self.circle_center
                dx = self.dataset.x[idx] - cx
                dy = self.dataset.y[idx] - cy
                if dx * dx + dy * dy > self.circle_radius * self.circle_radius:
                    return
            if self.state.preserve_base_selection:
                self._assign_indices(
                    np.array([idx], dtype=int),
                    mode="add" if self.state.assignment[idx] == 0 else "remove",
                )
            else:
                self._assign_indices(np.array([idx], dtype=int), mode="toggle")

    def _update_rect_item(self, scene_p1, scene_p2):
        v1 = self.viewbox.mapSceneToView(scene_p1)
        v2 = self.viewbox.mapSceneToView(scene_p2)
        left, right = sorted([v1.x(), v2.x()])
        bottom, top = sorted([v1.y(), v2.y()])
        rect = QtCore.QRectF(left, bottom, right - left, top - bottom)
        if self.rect_item is None:
            pen = pg.mkPen(
                (255, 255, 0, 160), width=1, style=QtCore.Qt.PenStyle.DashLine
            )
            self.rect_item = QtWidgets.QGraphicsRectItem(rect)
            self.rect_item.setPen(pen)
            self.plot.addItem(self.rect_item)
        else:
            self.rect_item.setRect(rect)

    # ---------- Helpers ----------
    def _view_px_to_data_scale(self) -> Tuple[float, float]:
        r = self.viewbox.viewRect()
        w = max(1, self.plot.width())
        h = max(1, self.plot.height())
        return (r.width() / w, r.height() / h) if (r.width() > 0 and r.height() > 0) else (1.0, 1.0)

    def _nearest_atom_index(self, x: float, y: float, pixel_tol: int = 10) -> Optional[int]:
        if not self.dataset:
            return None
        sx, sy = self._view_px_to_data_scale()
        tol2 = (pixel_tol * sx) ** 2 + (pixel_tol * sy) ** 2
        dx = self.dataset.x - x
        dy = self.dataset.y - y
        dist2 = dx * dx + dy * dy
        idx = int(np.argmin(dist2))
        return idx if dist2[idx] <= tol2 else None

    def _apply_circle_constraint(self, indices: np.ndarray) -> np.ndarray:
        if not (self.circle_enabled and self.circle_center and self.circle_radius > 0.0):
            return indices
        x0, y0 = self.circle_center
        R = self.circle_radius
        dx = self.dataset.x[indices] - x0
        dy = self.dataset.y[indices] - y0
        return indices[(dx * dx + dy * dy) <= R * R]

    def _assign_indices(self, indices: np.ndarray, mode: str = "add"):
        if not self.state:
            return
        gid = self.active_group_id
        indices = np.unique(indices)
        curr = self.state.assignment[indices]

        if mode == "remove":
            new = np.zeros_like(curr)
            desc = f"remove {len(indices)}"
        elif mode == "toggle":
            if gid == 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No active group",
                    "Please select or create a group first.",
                )
                return
            new = np.where(curr == gid, 0, gid)
            desc = f"toggle {len(indices)} to group {gid}"
        elif mode == "add":
            if gid == 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No active group",
                    "Please select or create a group first.",
                )
                return
            new = np.where(curr == gid, curr, gid)
            desc = f"add {len(indices)} to group {gid}"
        else:
            return

        before = curr.copy()
        after = new.copy()
        self.state.assignment[indices] = after
        self.state.undo_stack.append(
            Edit(indices=indices, before=before, after=after, description=desc)
        )
        if len(self.state.undo_stack) > self.state.max_history:
            self.state.undo_stack.pop(0)
        self.state.redo_stack.clear()

        ids = [int(self.dataset.ids[i]) for i in indices]
        self._log(f"{desc}; e.g. {ids[:10]}{'...' if len(ids) > 10 else ''}")
        self.update_scatter_colors()
        self._refresh_undo_redo_actions()
        self.is_dirty = True

    # ---------- Localizer ----------
    def on_open_localizer(self):
        if not self.state:
            return
        dlg = LocalizeDialog(
            self.A.value(),
            self.beta.value(),
            self.x0.value(),
            self.y0.value(),
            self,
        )

        def _apply(A, beta, x0, y0):
            self.A.setValue(A)
            self.beta.setValue(beta)
            self.x0.setValue(x0)
            self.y0.setValue(y0)
            self.state.breather.x0 = x0
            self.state.breather.y0 = y0
            self._log(
                f"Localizer applied: A={A:.6g} β={beta:.6g} x0={x0:.6g} y0={y0:.6g}"
            )

        dlg.values_applied.connect(_apply)
        dlg.exec()

    def on_toggle_apply_local(self, checked: bool):
        if self.state:
            self.state.apply_localizing = checked
        self._log(f"Apply localizing: {'ON' if checked else 'OFF'}")

    def on_toggle_preserve(self, checked: bool):
        if self.state:
            self.state.preserve_base_selection = checked
        self._log(f"Preserve base selection: {'ON' if checked else 'OFF'}")

    # ---------- Circle mask + center ----------
    def on_pick_center_dialog(self):
        if not self.dataset:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Set Breather Center (x0, y0)")
        form = QtWidgets.QFormLayout(dlg)
        layout = QtWidgets.QVBoxLayout(dlg)
        info = QtWidgets.QLabel(
            "The breather center (x0, y0) is the point where the localized mode is centered.\n"
            "You can enter coordinates manually or click 'Pick on canvas' to choose visually."
            "Choose how to set the breather center (x0, y0). You can evaluate an expression, "
            "enter coordinates manually, or pick directly on the canvas."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        sx0 = QtWidgets.QDoubleSpinBox()
        sx0.setRange(-1e12, 1e12)
        sx0.setDecimals(6)
        sx0.setValue(self.x0.value())
        sy0 = QtWidgets.QDoubleSpinBox()
        sy0.setRange(-1e12, 1e12)
        sy0.setDecimals(6)
        sy0.setValue(self.y0.value())

        tabs = QtWidgets.QTabWidget()

        # --- Expression tab ---
        expr_widget = QtWidgets.QWidget()
        expr_layout = QtWidgets.QVBoxLayout(expr_widget)
        expr_hint = QtWidgets.QLabel(
            "Use simple vector math; pos(id) returns the (x, y) coordinates of an atom ID.\n"
            "Examples: pos(12), (pos(10) + pos(20)) / 2"
        )
        expr_hint.setWordWrap(True)
        expr_input = QtWidgets.QLineEdit()
        expr_input.setPlaceholderText("(pos(10) + pos(20)) / 2")
        expr_result = QtWidgets.QLabel("")
        eval_btn = QtWidgets.QPushButton("Evaluate expression")

        id_to_index = {int(i): idx for idx, i in enumerate(self.dataset.ids)}

        def _eval_expr():
            expr = expr_input.text().strip()
            if not expr:
                QtWidgets.QMessageBox.warning(
                    dlg,
                    "No expression",
                    "Enter an expression such as pos(12) or (pos(1)+pos(2))/2.",
                )
                return

            def pos(atom_id):
                try:
                    aid = int(atom_id)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid atom id: {atom_id}")
                if aid not in id_to_index:
                    raise ValueError(f"Atom id {aid} not found in dataset")
                idx = id_to_index[aid]
                return np.array([self.dataset.x[idx], self.dataset.y[idx]], dtype=float)

            try:
                res = eval(expr, {"__builtins__": {}}, {"pos": pos, "np": np})
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(
                    dlg,
                    "Expression error",
                    f"Could not evaluate expression:\n{exc}",
                )
                return

            try:
                vec = np.asarray(res, dtype=float).reshape(-1)
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(
                    dlg,
                    "Expression error",
                    f"Result is not numeric:\n{exc}",
                )
                return

            if vec.size < 2:
                QtWidgets.QMessageBox.critical(
                    dlg,
                    "Expression error",
                    "Result must have at least two components (x and y).",
                )
                return

            sx0.setValue(float(vec[0]))
            sy0.setValue(float(vec[1]))
            expr_result.setText(f"Result: x0={vec[0]:.6f}, y0={vec[1]:.6f}")

        eval_btn.clicked.connect(_eval_expr)

        expr_layout.addWidget(expr_hint)
        expr_layout.addWidget(expr_input)
        expr_layout.addWidget(eval_btn)
        expr_layout.addWidget(expr_result)
        expr_layout.addStretch(1)
        tabs.addTab(expr_widget, "Expression")

        # --- Manual tab ---
        manual_widget = QtWidgets.QWidget()
        manual_form = QtWidgets.QFormLayout(manual_widget)
        manual_form.addRow("x0:", sx0)
        manual_form.addRow("y0:", sy0)
        tabs.addTab(manual_widget, "Manual")

        # --- Pick on canvas tab ---
        pick_widget = QtWidgets.QWidget()
        pick_layout = QtWidgets.QVBoxLayout(pick_widget)
        pick_btn = QtWidgets.QPushButton("Pick on canvas")
        picked_label = QtWidgets.QLabel("")
        pick_layout.addWidget(QtWidgets.QLabel("Click to pick the center directly on the canvas."))
        pick_layout.addWidget(pick_btn)
        pick_layout.addWidget(picked_label)
        pick_layout.addStretch(1)
        tabs.addTab(pick_widget, "Pick on canvas")

        form.addRow(info)
        form.addRow("x0:", sx0)
        form.addRow("y0:", sy0)
        form.addRow(pick_btn)
        form.addRow(picked_label)
        layout.addWidget(tabs)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(btns)
        layout.addWidget(btns)

        def accept_vals():
            self.x0.setValue(sx0.value())
            self.y0.setValue(sy0.value())
            if self.state:
                self.state.breather.x0 = sx0.value()
                self.state.breather.y0 = sy0.value()
            self.circle_center = (sx0.value(), sy0.value())
            self._update_circle_item()
            self.update_scatter_colors()
            dlg.accept()

        btns.accepted.connect(accept_vals)
        btns.rejected.connect(dlg.reject)

        def begin_pick():
            picked_label.setText("Click on the canvas to set the center…")
            self.plot.scene().sigMouseClicked.disconnect(self.on_mouse_clicked)

            def _capture(ev):
                pt = self.viewbox.mapSceneToView(ev.scenePos())
                sx0.setValue(pt.x())
                sy0.setValue(pt.y())
                picked_label.setText(
                    f"Picked: x0={pt.x():.6f}, y0={pt.y():.6f}"
                )
                self.plot.scene().sigMouseClicked.disconnect(_capture)
                self.plot.scene().sigMouseClicked.connect(self.on_mouse_clicked)

            self.plot.scene().sigMouseClicked.connect(_capture)

        pick_btn.clicked.connect(begin_pick)

        # Make dialog non-modal and keep it alive
        dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dlg.show()
        self._pick_center_dialog = dlg




    def on_toggle_circle(self, checked: bool):
        self.circle_enabled = checked
        self._update_circle_item()
        self.update_scatter_colors()
        self._log(f"Circle Mask {'ENABLED' if self.circle_enabled else 'DISABLED'}")
        self.is_dirty = True

    def on_radius_changed(self, r: float):
        self.circle_radius = float(r)
        self._update_circle_item()
        self.update_scatter_colors()
        self.is_dirty = True

    def _update_circle_item(self):
        if self.circle_item is not None:
            self.plot.removeItem(self.circle_item)
            self.circle_item = None
        if self.circle_enabled and self.circle_center and self.circle_radius > 0.0:
            x0, y0 = self.circle_center
            R = self.circle_radius
            rect = QtCore.QRectF(x0 - R, y0 - R, 2 * R, 2 * R)
            pen = pg.mkPen((50, 200, 255, 160), width=2)
            self.circle_item = QtWidgets.QGraphicsEllipseItem(rect)
            self.circle_item.setPen(pen)
            self.plot.addItem(self.circle_item)

    # ---------- Rule tool ----------
    def on_rule_clicked(self):
        if self.last_line_indices is None or self.last_line_orientation not in ("h", "v"):
            QtWidgets.QMessageBox.warning(
                self,
                "Selection Rule",
                "Select a horizontal or vertical line first.",
            )
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Selection Rule")
        form = QtWidgets.QFormLayout(dlg)
        row_stride = QtWidgets.QSpinBox(); row_stride.setRange(1, 9999); row_stride.setValue(max(1, self.line_spacing.value()))
        row_off = QtWidgets.QSpinBox(); row_off.setRange(0, 9998); row_off.setValue(int(self.line_offset.value()) % max(1, self.line_spacing.value()))
        col_stride = QtWidgets.QSpinBox(); col_stride.setRange(1, 9999); col_stride.setValue(5)
        col_off = QtWidgets.QSpinBox(); col_off.setRange(0, 9998); col_off.setValue(0)
        form.addRow("Along line: keep 1 of N", row_stride)
        form.addRow("Along line: offset (0..N-1)", row_off)
        form.addRow("Perpendicular: keep 1 of M", col_stride)
        form.addRow("Perpendicular: offset (0..M-1)", col_off)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        N, Off, M, Off2 = (
            int(row_stride.value()),
            int(row_off.value()),
            int(col_stride.value()),
            int(col_off.value()),
        )

        line_idx = self.last_line_indices
        if self.last_line_orientation == "h":
            order = np.argsort(self.dataset.x[line_idx])
        else:
            order = np.argsort(self.dataset.y[line_idx])
        line_idx_sorted = line_idx[order]
        anchors = line_idx_sorted[
            np.arange(line_idx_sorted.size) % max(1, N) == (Off % max(1, N))
        ]
        if anchors.size == 0:
            self._log("Rule: no anchors after row stride/offset.")
            return

        R = float(self.atom_radius_spin.value())
        if R <= 0.0:
            if self.base_diam_data == 0.0:
                self.base_diam_data = self._estimate_point_diameter()
            R = 0.5 * self.base_diam_data
            self.atom_radius_spin.setValue(R)

        final = []
        if self.last_line_orientation == "h":
            for i in anchors:
                xi = self.dataset.x[i]
                dx = np.abs(self.dataset.x - xi)
                idx = np.nonzero(dx <= R)[0]
                idx = self._apply_circle_constraint(idx)
                order = np.argsort(self.dataset.y[idx])
                idx_sorted = idx[order]
                keep = np.arange(idx_sorted.size) % max(1, M) == (Off2 % max(1, M))
                final.extend(idx_sorted[keep].tolist())
        else:
            for i in anchors:
                yi = self.dataset.y[i]
                dy = np.abs(self.dataset.y - yi)
                idx = np.nonzero(dy <= R)[0]
                idx = self._apply_circle_constraint(idx)
                order = np.argsort(self.dataset.x[idx])
                idx_sorted = idx[order]
                keep = np.arange(idx_sorted.size) % max(1, M) == (Off2 % max(1, M))
                final.extend(idx_sorted[keep].tolist())

        if not final:
            self._log("Rule: 0 atoms after perpendicular propagation.")
            return
        self._assign_indices(np.array(sorted(set(final)), dtype=int), mode="add")

    # ---------- Undo/Redo ----------
    def on_undo(self):
        if not self.state:
            return
        e = self.state.undo()
        if e:
            self._log(f"Undo: {e.description}")
            self.update_scatter_colors()
            self.is_dirty = True
        self._refresh_undo_redo_actions()

    def on_redo(self):
        if not self.state:
            return
        e = self.state.redo()
        if e:
            self._log(f"Redo: {e.description}")
            self.update_scatter_colors()
            self.is_dirty = True
        self._refresh_undo_redo_actions()

    def _refresh_undo_redo_actions(self):
        en_u = self.state.can_undo() if self.state else False
        en_r = self.state.can_redo() if self.state else False
        self.act_undo.setEnabled(en_u)
        self.act_redo.setEnabled(en_r)
        self.tb_undo.setEnabled(en_u)
        self.tb_redo.setEnabled(en_r)

    # ---------- Export ----------
    def on_export(self):
        if not self.state:
            return
        self.state.breather.A = self.A.value()
        self.state.breather.beta = self.beta.value()
        self.state.breather.x0 = self.x0.value()
        self.state.breather.y0 = self.y0.value()
        self.state.apply_localizing = self.chk_apply_local.isChecked()
        outdir_str = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output directory"
        )
        if not outdir_str:
            return
        p = export_lammps_block(self.state, Path(outdir_str))
        self._log(f"Exported LAMMPS block: {p}")
        QtWidgets.QMessageBox.information(self, "Export", f"Exported to {p}")

    def on_export_ids(self):
        if not self.state:
            return
        if self.active_group_id == 0 and np.count_nonzero(self.state.assignment != 0) == 0:
            QtWidgets.QMessageBox.information(
                self,
                "Export IDs",
                "No selected atoms to export.",
            )
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export Selected IDs")
        form = QtWidgets.QFormLayout(dlg)
        scope_all = QtWidgets.QRadioButton("All assigned atoms")
        scope_active = QtWidgets.QRadioButton(
            f"Active group only (g{self.active_group_id})"
        )
        if self.active_group_id != 0:
            scope_active.setChecked(True)
        else:
            scope_all.setChecked(True)
            scope_active.setEnabled(False)
        form.addRow(scope_all)
        form.addRow(scope_active)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        scope = "active" if scope_active.isChecked() else "all"
        outdir_str = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output directory for IDs"
        )
        if not outdir_str:
            return
        p = export_selected_ids(
            self.state, Path(outdir_str), scope=scope, active_gid=self.active_group_id
        )
        self._log(f"Exported IDs: {p}")
        QtWidgets.QMessageBox.information(self, "Export IDs", f"Exported to {p}")

    # ---------- Logging ----------
    def _log(self, msg: str):
        print(msg)
