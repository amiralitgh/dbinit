from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class Box:
    xlo: float; xhi: float
    ylo: float; yhi: float
    zlo: float; zhi: float

@dataclass
class DataSet:
    ids: np.ndarray  # int
    x: np.ndarray    # float
    y: np.ndarray    # float
    z: np.ndarray    # float
    box: Box

@dataclass
class Group:
    gid: int
    name: str
    color: Tuple[int,int,int,255] = (200,200,200,255)
    direction: Tuple[float,float,float] = (1.0, 0.0, 0.0)

@dataclass
class BreatherParams:
    A: float = 0.05
    beta: float = 1.0
    x0: float = 0.0
    y0: float = 0.0

@dataclass
class Edit:
    indices: np.ndarray
    before: np.ndarray
    after: np.ndarray
    description: str

@dataclass
class ProjectState:
    data: DataSet
    assignment: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    groups: Dict[int, Group] = field(default_factory=dict)
    undo_stack: list[Edit] = field(default_factory=list)
    redo_stack: list[Edit] = field(default_factory=list)
    max_history: int = 100
    breather: BreatherParams = field(default_factory=BreatherParams)
    apply_localizing: bool = True
    preserve_base_selection: bool = True

    def __post_init__(self):
        if self.assignment.size == 0:
            self.assignment = np.zeros(self.data.ids.shape[0], dtype=np.int32)

    def clear_groups(self):
        self.groups.clear()
        self.assignment[...] = 0
        self.undo_stack.clear()
        self.redo_stack.clear()

    def add_group(self, gid: int, name: str, color=(200,200,200,255)) -> Group:
        g = Group(gid=gid, name=name, color=color)
        self.groups[gid] = g
        return g

    def remove_group(self, gid: int):
        if gid in self.groups:
            del self.groups[gid]
            self.assignment[self.assignment == gid] = 0

    def colors_rgba_for_all(self, rgba_unassigned=(160,160,160,255)) -> np.ndarray:
        out = np.empty((self.assignment.size, 4), dtype=np.ubyte)
        for i, gid in enumerate(self.assignment):
            if gid == 0 or gid not in self.groups:
                out[i] = rgba_unassigned
            else:
                out[i] = np.array(self.groups[gid].color, dtype=np.ubyte)
        return out

    # --- history ---
    def can_undo(self) -> bool: return len(self.undo_stack) > 0
    def can_redo(self) -> bool: return len(self.redo_stack) > 0

    def undo(self) -> Optional[Edit]:
        if not self.undo_stack: return None
        e = self.undo_stack.pop()
        self.assignment[e.indices] = e.before
        self.redo_stack.append(e)
        return e

    def redo(self) -> Optional[Edit]:
        if not self.redo_stack: return None
        e = self.redo_stack.pop()
        self.assignment[e.indices] = e.after
        self.undo_stack.append(e)
        return e
