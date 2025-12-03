from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from .model import DataSet, Box

HEADER_KEYS = {
    "atoms", "bonds", "angles", "dihedrals", "impropers",
    "velocities", "masses",
    "pair coeffs", "bond coeffs", "angle coeffs", "dihedral coeffs", "improper coeffs",
    "ellipsoids", "lines", "triangles",
    "atom types", "bond types", "angle types", "dihedral types", "improper types",
    "groups", "fixes"
}

def _parse_box(lines: List[str]) -> Box:
    xlo = xhi = ylo = yhi = zlo = zhi = None
    for raw in lines:
        s0 = raw.split("#", 1)[0].strip()
        if not s0: continue
        toks = s0.split()
        if len(toks) < 4: continue
        label = " ".join(toks[-2:]).lower()
        try:
            a, b = float(toks[0]), float(toks[1])
        except ValueError:
            continue
        if label == "xlo xhi": xlo, xhi = a, b
        if label == "ylo yhi": ylo, yhi = a, b
        if label == "zlo zhi": zlo, zhi = a, b
    if any(v is None for v in (xlo, xhi, ylo, yhi)):
        raise ValueError("Failed to parse x/y box bounds.")
    if zlo is None or zhi is None:
        zlo, zhi = 0.0, 0.0
    return Box(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi)

def _find_section(lines: List[str], key: str) -> Optional[int]:
    key = key.lower()
    for i, raw in enumerate(lines):
        left = raw.split("#", 1)[0].strip().lower()
        if left.startswith(key):
            return i
    return None

def _iter_atoms_block(lines: List[str], start_idx: int) -> List[str]:
    out: List[str] = []
    i = start_idx + 1
    n = len(lines)
    while i < n and not lines[i].strip(): i += 1
    while i < n:
        s = lines[i].strip()
        if s:
            bare = s.split("#", 1)[0].strip().lower()
            fw = bare.split()[0] if bare else ""
            if fw in HEADER_KEYS and fw != "atoms":
                break
        out.append(lines[i]); i += 1
    return out

def _parse_atom_line(line: str) -> Tuple[int, float, float, float]:
    s = line.split("#", 1)[0].strip()
    toks = s.split()
    if len(toks) < 4:
        raise ValueError(f"Too few columns: {line!r}")
    try:
        x, y, z = float(toks[-3]), float(toks[-2]), float(toks[-1])
        atom_id = int(toks[0])
    except ValueError as e:
        raise ValueError(f"Parse error in atom line: {line!r}") from e
    return atom_id, x, y, z

def read_lammps_data(path: str | Path) -> DataSet:
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    box = _parse_box(lines)
    atoms_idx = _find_section(lines, "Atoms")
    if atoms_idx is None:
        raise ValueError("Could not find 'Atoms' section.")
    block = _iter_atoms_block(lines, atoms_idx)

    ids, xs, ys, zs = [], [], [], []
    for raw in block:
        s = raw.strip()
        if not s or s.startswith("#"): continue
        try:
            a_id, x, y, z = _parse_atom_line(s)
        except ValueError:
            continue
        ids.append(a_id); xs.append(x); ys.append(y); zs.append(z)

    if not ids:
        raise ValueError("Found 'Atoms' section but parsed 0 atoms.")

    ids = np.asarray(ids, dtype=int)
    xs  = np.asarray(xs, dtype=float)
    ys  = np.asarray(ys, dtype=float)
    zs  = np.asarray(zs, dtype=float)

    order = np.argsort(ids, kind="mergesort")
    return DataSet(ids=ids[order], x=xs[order], y=ys[order], z=zs[order], box=box)
