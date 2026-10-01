"""Inline the solver results into viewer.html (one self-contained file, gzip+base64 arrays).

Usage: python build_viewer.py [payload_dir]
Reads results/ahmed_v2.json, results/validation_v2.json, results/stamp.json and, from payload_dir
(default results/payload, not committed):
  a<slant>_surface.npz       from ahmed/extract_v2.py --surface   (best mesh of each slant)
  a<slant>_<level>.png       from ahmed/extract_v2.py --slice-png (each mesh level that ran)
  a<slant>_<level>_slice.npz (extent of that slice)
  cylinder_frames.npz        from cylinder2d/frames_v2.py
"""
import base64
import gzip
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"
PAY = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "payload"


def spec(a, d, q=None):
    a = np.ascontiguousarray(a)
    s = {"d": d, "b": base64.b64encode(gzip.compress(a.tobytes(), 9)).decode()}
    if q is not None:
        s["q"] = [float(v) for v in q]
    return s


def q_u2(v, lo, hi):
    return np.round((np.clip(v, lo, hi) - lo) / (hi - lo) * 65535).astype(np.uint16)


def q_u1(v, lo, hi):
    return np.round((np.clip(v, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def mesh(z, pre, colour, crange):
    v = z[f"{pre}verts"]
    lo, hi = v.min(0) - 1e-4, v.max(0) + 1e-4
    vq = np.stack([q_u2(v[:, k], lo[k], hi[k]) for k in range(3)], 1)
    t = z[f"{pre}tris"]
    td = (t.astype(np.uint16), "u2") if v.shape[0] < 65536 else (t.astype(np.uint32), "u4")
    n = np.clip(np.round(z[f"{pre}norms"] * 127), -127, 127).astype(np.int8)
    return {"verts": spec(vq, "u2", [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]), "tris": spec(td[0], td[1]),
            "norms": spec(n, "i1", [-1, 1]), colour[0]: spec(q_u1(z[colour[1]], *crange), "u1", crange)}


ah = json.loads((R / "ahmed_v2.json").read_text())
val = json.loads((R / "validation_v2.json").read_text())
stamp = json.loads((R / "stamp.json").read_text())

slants = {}
for key, S in ah["slants"].items():
    z = np.load(PAY / f"a{key}_surface.npz")
    o = {"exp": S["exp"], "exp_src": S["exp_src"], "best": S["best"],
         "levels": {lv: {k: L[k] for k in ("cells", "Cd", "Cd_std", "iters")} for lv, L in S["levels"].items()},
         "surf": mesh(z, "s_", ("cp", "s_cp"), (-1.6, 1.0))}
    for lev in ("2", "8"):
        if f"q{lev}_verts" in z.files:
            o["iso" + lev] = mesh(z, f"q{lev}_", ("wx", f"q{lev}_wx"), (-65, 65))
    g = z["grid_u"].astype(np.float32) * 1.5 / 127
    o["grid"] = {"u": spec(np.clip(np.round(g / 1.5 * 127), -127, 127).astype(np.int8), "i1", [-1.5, 1.5]),
                 "dims": [int(v) for v in z["grid_dims"]], "box": [float(v) for v in z["grid_box"]]}
    o["slices"] = {}
    for lv in S["levels"]:
        png = PAY / f"a{key}_{lv}.png"
        if png.exists():
            ext = np.load(PAY / f"a{key}_{lv}_slice.npz")["slice_extent"]
            o["slices"][lv] = {"png": base64.b64encode(png.read_bytes()).decode(), "extent": [float(v) for v in ext]}
    slants[key] = o

cz = np.load(PAY / "cylinder_frames.npz")
w, u, t, F = cz["w"], cz["u"], cz["t"], cz["force"]
best = val["best"]
dt = float(np.median(np.diff(t)))
nf = min(len(t), int(round(2 / best["St"] / dt)))      # two shedding periods: a near-seamless loop
w, u, t = w[-nf:], u[-nf:], t[-nf:]
wq = np.where(np.isnan(w), 0, np.clip(np.round((np.clip(w, -4, 4) + 4) / 8 * 254) + 1, 1, 255)).astype(np.uint8)
uq = np.clip(np.round(u / 1.6 * 127), -127, 127).astype(np.int8)
cyl = {"nf": int(nf), "nx": int(w.shape[2]), "ny": int(w.shape[1]), "unx": int(u.shape[2]), "uny": int(u.shape[1]),
       "t": [round(float(x), 3) for x in t], "x": [float(v) for v in cz["x"]], "y": [float(v) for v in cz["y"]],
       "w": spec(wq, "u1"), "u": spec(uq, "i1", [-1.6, 1.6]),
       "force": {"t": [round(float(x), 3) for x in F[:, 0]], "cd": [round(float(x), 4) for x in F[:, 1]],
                 "cl": [round(float(x), 4) for x in F[:, 2]]},
       "runs": val["runs"], "ref": val["reference"], "best": best}

data = {"bench": {"definition": ah["bar"]}, "ahmed": {"slants": slants, "note": ah["note"]}, "cyl": cyl,
        "stamp": stamp["text"], "limits": stamp["limits"]}
html = (HERE / "viewer_template.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
out = HERE / "viewer.html"
out.write_text(html)
print(f"{out} {out.stat().st_size / 1e6:.2f} MB; slants {list(slants)}; {nf} cylinder frames")
