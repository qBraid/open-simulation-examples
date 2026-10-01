"""Run post3d.participation on a Palace electrostatic output directory."""
import json, sys, time
import numpy as np
import post3d

def load_C(post_dir):
    rows = [l.split(",") for l in open(f"{post_dir}/terminal-C.csv").read().strip().splitlines()[1:]]
    return np.array([[float(x) for x in r[1:]] for r in rows])

def load_Vinc(post_dir):
    rows = [l.split(",") for l in open(f"{post_dir}/terminal-V.csv").read().strip().splitlines()[1:]]
    return np.array([float(r[1]) for r in rows])

def main(layout, post_dir, stable_path, x0, **kw):
    C = load_C(post_dir)
    Vinc = load_Vinc(post_dir)
    names = [n for n in layout["conductors"] if n != layout.get("ground")]
    grids = [post3d.read_pvtu(f"{post_dir}/paraview/electrostatic/Cycle{k+1:06d}/data.pvtu") for k in range(len(names))]
    st = {float(k): v for k, v in json.load(open(stable_path)).items()}
    t0 = time.time()
    from vtk.util.numpy_support import vtk_to_numpy
    for g, v in zip(grids, Vinc):
        a = g.GetPointData().GetArray("E")
        vtk_to_numpy(a)[:] /= v   # in-place: fields per 1 V terminal excitation
        a.Modified()
    r = post3d.participation(layout, grids, C, x0, st, **kw)
    r["V_inc"] = Vinc.tolist()
    r["C_F"] = C.tolist(); r["post_seconds"] = time.time() - t0
    return r


def fieldmap(layout, post_dir, window, spacing=2.0, z=0.05, out=None):
    """|E| of the mode just above the chip surface on a regular grid (for the viewer)."""
    import post3d
    from vtk.util.numpy_support import vtk_to_numpy
    C = load_C(post_dir); Vinc = load_Vinc(post_dir)
    names = [n for n in layout["conductors"] if n != layout.get("ground")]
    grids = [post3d.read_pvtu(f"{post_dir}/paraview/electrostatic/Cycle{k+1:06d}/data.pvtu") for k in range(len(names))]
    for g, v in zip(grids, Vinc):
        a = g.GetPointData().GetArray("E"); vtk_to_numpy(a)[:] /= v; a.Modified()
    mode = layout["mode"]
    if layout["mode_type"] == "floating_pair":
        V = np.linalg.solve(C, np.array([mode.get(n, 0.0) for n in names]))
    else:
        V = np.array([mode.get(n, 0.0) for n in names])
    xs = np.arange(window[0] + spacing / 2, window[2], spacing)
    ys = np.arange(window[1] + spacing / 2, window[3], spacing)
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
    E = post3d.Prober(grids).E(P, V)
    U = 0.5 * V @ C @ V
    np.savez_compressed(out, x=xs, y=ys, Emag=np.linalg.norm(E, axis=1).reshape(X.shape).astype(np.float32),
                        U=U, V=V)
