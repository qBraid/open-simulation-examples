"""Collect results into the compact payload the viewer inlines, and export the printable STL.

Reads results/{benchmarks,validation,bracket}.json, results/*.npz and reference/out.
Writes results/viewer_data.json, results/bracket.stl and results/stl.json.
"""
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RES, REF = ROOT / "results", ROOT / "reference" / "out"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def u8(x):
    return np.clip(np.rint(np.asarray(x, np.float32) * 255), 0, 255).astype(np.uint8)


def viridis_lut():
    from matplotlib import colormaps
    return (colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).round().astype(int).tolist()


def designs_2d(bench):
    out = []
    for name, b in bench.items():
        if not name.startswith("top88"):
            continue
        d = np.load(RES / f"py_{name}.npz")
        ref = np.loadtxt(REF / (("ref_" + name) + ".txt"), delimiter=",")
        item = dict(name=name, nelx=int(d["x"].shape[1]), nely=int(d["x"].shape[0]),
                    py=b64(u8(d["x"])), ref=b64(u8(ref)), hist=d["hist"][:, 1].round(5).tolist(),
                    ref_hist=b["reference_history"])
        if len(d["snap_iters"]) > 2:  # playback snapshots
            item["snap_iters"] = d["snap_iters"].tolist()
            item["snaps"] = b64(u8(d["snaps"]))
        out.append(item)
    return out


def top3d_payload(bench):
    d = np.load(RES / "py_top3d_60x20x4.npz")
    ref = np.loadtxt(REF / "ref_top3d_60x20x4.txt", delimiter=",").reshape(d["x"].shape, order="F")
    # snapshots are element vectors in top3d order (k, i, j); convert each to [j, i, k]
    nely, nelx, nelz = d["x"].shape
    snaps = d["snaps"].reshape(len(d["snaps"]), nelz, nelx, nely).transpose(0, 3, 2, 1)
    keep = sorted(set(list(range(0, len(snaps), 2)) + [len(snaps) - 1]))
    return dict(nelx=nelx, nely=nely, nelz=nelz, py=b64(u8(d["x"])), ref=b64(u8(ref)),
                snap_iters=d["snap_iters"][keep].tolist(), snaps=b64(u8(snaps[keep])),
                hist=d["hist"][:, 1].round(4).tolist(), ref_hist=bench["top3d_60x20x4"]["reference_history"])


def bracket_payload():
    import os

    import trimesh
    from skimage.measure import marching_cubes

    tag = "" if (RES / "bracket.json").exists() else "_2mm"   # prefer the 1.5 mm run when present
    summ = json.loads((RES / f"bracket{tag}.json").read_text())
    os.environ["BRACKET_H_MM"] = str(summ.get("brick_mm", 1.5))
    import bracket as B
    d = np.load(RES / f"bracket{tag}.npz")
    NX, NY, NZ, h = B.NX, B.NY, B.NZ, B.H_MM

    def to_xyz(v):  # element vector (k, i, j) -> array [i, y_up, k]
        a = v.reshape(NZ, NX, NY).transpose(1, 2, 0)        # [i, j, k]
        return a[:, ::-1, :]                                 # j (rows from top) -> y up

    x = to_xyz(d["x"])
    vm = to_xyz(d["vm"])
    pad = np.pad(x, 1)
    verts, faces, _, _ = marching_cubes(pad, level=0.5, spacing=(h, h, h))
    verts = verts - h + h / 2                                # padding and cell-centre offsets
    mesh = trimesh.Trimesh(verts, faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=12)
    mesh.fix_normals()
    # export the printable part
    mesh.export(RES / f"bracket{tag}.stl")
    stl = dict(file=f"results/bracket{tag}.stl", triangles=int(len(mesh.faces)), watertight=bool(mesh.is_watertight),
               volume_mm3=round(float(mesh.volume), 1), mass_g_ti64=round(float(mesh.volume) * 4.43e-3, 1),
               bbox_mm=np.round(mesh.extents, 2).tolist())
    (RES / "stl.json").write_text(json.dumps(stl, indent=1))
    # vertex stress: densest of the 8 cells around each vertex, then its element stress
    v = np.asarray(mesh.vertices)
    base = np.floor(v / h - 0.5).astype(int)
    best_x = np.full(len(v), -1.0); best_vm = np.zeros(len(v))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                ii = np.clip(base[:, 0] + di, 0, NX - 1); jj = np.clip(base[:, 1] + dj, 0, NY - 1); kk = np.clip(base[:, 2] + dk, 0, NZ - 1)
                xs, vs = x[ii, jj, kk], vm[ii, jj, kk]
                upd = xs > best_x
                best_x[upd] = xs[upd]; best_vm[upd] = vs[upd]
    vmax = float(summ["vm_p99_MPa"]) * 1.05
    q = np.clip(np.round(v / h * 256), 0, 65535).astype(np.uint16)  # 1/256 element precision
    idx = np.asarray(mesh.faces, np.uint32)
    # hot spot (for the tour): highest-stress body element
    i_hot = int(np.argmax(np.where(best_x > 0.5, best_vm, 0)))
    snaps = d["snaps"].astype(np.float32)
    keep = sorted(set(list(range(0, len(snaps), 2)) + [len(snaps) - 1]))
    snap_xyz = np.stack([to_xyz(s) for s in snaps[keep]])
    hist = d["hist"]
    pads = [[0.0, (NY - pj) * h, pk * h] for pj, pk in B.PADS]
    lug = [B.LUG_C[0] * h, (NY - B.LUG_C[1]) * h]
    return dict(
        nx=NX, ny=NY, nz=NZ, h=h, summary=summ, stl=stl,
        verts=b64(q), vscale=h / 256, faces=b64(idx), nverts=int(len(v)), nfaces=int(len(idx)),
        vm=b64(u8(np.clip(best_vm / vmax, 0, 1))), vm_max=vmax, hot=np.round(v[i_hot], 2).tolist(),
        snap_iters=d["snap_iters"][keep].tolist(), snaps=b64(u8(snap_xyz)),
        hist_c=hist[:, 1].round(5).tolist(), hist_beta=hist[:, 4].tolist(),
        pads=pads, lug=lug, lug_r_in=B.LUG_R_IN * h, lug_r_out=B.LUG_R_OUT * h,
        loads=[dict(name=k, f=[v_[0], v_[1], 0.0]) for k, v_ in B.LOADS_N.items()],
    )


def main():
    bench = json.loads((RES / "benchmarks.json").read_text())
    val = json.loads((RES / "validation.json").read_text())
    data = dict(benchmarks={k: {kk: vv for kk, vv in v.items() if kk != "reference_history"} for k, v in bench.items()},
                designs2d=designs_2d(bench), top3d=top3d_payload(bench), validation=val,
                bracket=bracket_payload(), viridis=viridis_lut(),
                stamp=json.loads((RES / "stamp.json").read_text()) if (RES / "stamp.json").exists() else {})
    s = json.dumps(data, separators=(",", ":"))
    (RES / "viewer_data.json").write_text(s)
    print(f"viewer_data.json {len(s) / 1e6:.2f} MB; STL", json.loads((RES / "stl.json").read_text()))


if __name__ == "__main__":
    main()
