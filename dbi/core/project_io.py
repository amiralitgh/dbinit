from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
from .model import ProjectState, DataSet, Box, BreatherParams

def save_project(state: ProjectState, path: Path, ui_state: Dict[str, Any]) -> None:
    path = Path(path)
    data = {
        "data": {
            "ids": state.data.ids.tolist(),
            "x": state.data.x.tolist(),
            "y": state.data.y.tolist(),
            "z": state.data.z.tolist(),
            "box": {
                "xlo": state.data.box.xlo, "xhi": state.data.box.xhi,
                "ylo": state.data.box.ylo, "yhi": state.data.box.yhi,
                "zlo": state.data.box.zlo, "zhi": state.data.box.zhi
            }
        },
        "assignment": state.assignment.tolist(),
        "groups": [
            {
                "gid": int(g.gid), "name": g.name, "color": list(g.color),
                "direction": list(g.direction)
            } for g in state.groups.values()
        ],
        "breather": {
            "A": state.breather.A, "beta": state.breather.beta,
            "x0": state.breather.x0, "y0": state.breather.y0
        },
        "apply_localizing": state.apply_localizing,
        "preserve_base_selection": state.preserve_base_selection,
        "ui": ui_state,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_project(path: Path) -> tuple[ProjectState, Dict[str, Any]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    box = Box(**d["data"]["box"])
    ds = DataSet(
        ids=np.asarray(d["data"]["ids"], dtype=int),
        x=np.asarray(d["data"]["x"], dtype=float),
        y=np.asarray(d["data"]["y"], dtype=float),
        z=np.asarray(d["data"]["z"], dtype=float),
        box=box
    )
    st = ProjectState(data=ds)
    st.assignment = np.asarray(d["assignment"], dtype=np.int32)
    st.groups.clear()
    for g in d["groups"]:
        gid = int(g["gid"])
        st.add_group(gid, g["name"], tuple(g["color"])).direction = tuple(float(x) for x in g["direction"])
    b = d["breather"]
    st.breather = BreatherParams(A=b["A"], beta=b["beta"], x0=b["x0"], y0=b["y0"])
    st.apply_localizing = bool(d.get("apply_localizing", True))
    st.preserve_base_selection = bool(d.get("preserve_base_selection", True))
    return st, d.get("ui", {})

