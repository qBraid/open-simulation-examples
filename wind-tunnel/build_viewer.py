"""Inline the solver results into viewer.html (one self-contained file).

Usage: python build_viewer.py
Reads results/validation.json, results/ahmed.json, results/stamp.json,
results/cylinder_frames.npz and results/ahmed_viewer.npz (the npz files come from
cylinder2d/frames.py and ahmed/extract.py and are not committed).
"""
import base64
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def quant(a, lo, hi):
    """Map to uint8 1..255 over [lo, hi]; NaN -> 0 (masked)."""
    q = np.clip((a - lo) / (hi - lo), 0, 1) * 254 + 1
    return np.where(np.isnan(a), 0, q).astype(np.uint8)


val = json.loads((R / "validation.json").read_text())
car = json.loads((R / "ahmed.json").read_text())
stamp = json.loads((R / "stamp.json").read_text())

cz = np.load(R / "cylinder_frames.npz")
w, t = cz["w"], cz["t"]
St = next(m["St"] for m in val["meshes"] if m["frames"])
dt = float(np.median(np.diff(t)))
nf = min(len(t), int(round(2 / St / dt)))  # two shedding periods -> near-seamless loop
w, t = w[-nf:], t[-nf:]

az = np.load(R / "ahmed_viewer.npz")
su, sc = az["slice_u"], az["slice_cp"]
data = dict(
    validation=val,
    ahmed_forces=dict(Cd_mean=car["Cd_mean"], Cd_reference=car["Cd_reference"], cells=car["cells"]),
    stamp=stamp["text"],
    limits=stamp["limits"],
    cyl=dict(frames=b64(quant(w, -4, 4)), nx=int(w.shape[2]), ny=int(w.shape[1]), nf=int(nf),
             t=[round(float(x), 3) for x in t], x=[float(v) for v in cz["x"]], y=[float(v) for v in cz["y"]]),
    ahmed=dict(verts=b64(az["verts"]), norms=b64(az["norms"]), tris=b64(az["tris"]), cp=b64(az["cp"]),
               line_pts=b64(az["line_pts"]), line_spd=b64(az["line_spd"]), line_offs=b64(az["line_offs"]),
               slice_u=b64(quant(su, 0, 1.4)), slice_cp=b64(quant(sc, -1.2, 1.0)),
               slice_nx=int(su.shape[1]), slice_nz=int(su.shape[0]),
               slice_extent=[float(v) for v in az["slice_extent"]], cells=car["cells"]),
)
html = (HERE / "viewer_template.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
out = HERE / "viewer.html"
out.write_text(html)
print(f"{out} {out.stat().st_size / 1e6:.2f} MB, {nf} cylinder frames")
