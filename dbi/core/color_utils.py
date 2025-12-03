from __future__ import annotations
from typing import Tuple
from PyQt6 import QtGui

PALETTE = [
    (230, 25, 75, 255),   # red
    (60, 180, 75, 255),   # green
    (255, 225, 25, 255),  # yellow
    (0, 130, 200, 255),   # blue
    (245, 130, 48, 255),  # orange
    (145, 30, 180, 255),  # purple
    (70, 240, 240, 255),  # cyan
    (240, 50, 230, 255),  # magenta
    (210, 245, 60, 255),  # lime
    (250, 190, 190, 255), # pink
]

def next_color(i: int) -> QtGui.QColor:
    r,g,b,a = PALETTE[i % len(PALETTE)]
    return QtGui.QColor(r,g,b,a)

def tuple_to_qcolor(rgba: Tuple[int,int,int,int]) -> QtGui.QColor:
    return QtGui.QColor(*rgba)
