"""Drag from saved prediction meshes, plus decimated surfaces for the three.js viewer.
For each inference_<model>_*_<case>.vtp: drag D = -sum((p n_x + tau_x) A) over cells, n the outward normal.
DrivAerML stores kinematic pressure (p/rho) and wall shear stress with OpenFOAM's sign (the force of the wall on
the fluid), hence the + on tau. With U = 38.889 m/s, q = U^2/2 = 756.18 m^2/s^2, and this reproduces the published
high-fidelity Cd of every DrivAerML car (checked per car: q_check below). Cd = D / (q * A), with A = aRef for
DrivAerML and the rasterised frontal area of the mesh for the production cars.
usage: postproc.py <out_dir> <data_root> <result.json> <viewer_dir>
"""
import glob, json, os, re, sys, numpy as np, pyvista as pv
from scipy.spatial import cKDTree
OUT, ROOT, RES, VDIR = sys.argv[1:5]
os.makedirs(VDIR, exist_ok=True)

def frontal_area(m, h=0.005):
    # project the surface onto the y-z plane at h resolution; occupied pixels after closing small gaps
    from scipy import ndimage
    pts = m.points[:, 1:3]
    f = m.faces.reshape(-1, 4)[:, 1:]
    tri = pts[f]; r = np.random.default_rng(0).random((len(f), 6, 2))
    r = np.where(r.sum(-1, keepdims=True) > 1, 1 - r, r)
    s = (tri[:, None, 0] + r[..., :1] * (tri[:, None, 1] - tri[:, None, 0]) + r[..., 1:] * (tri[:, None, 2] - tri[:, None, 0])).reshape(-1, 2)
    s = np.vstack([s, pts]); lo = s.min(0); ij = ((s - lo) / h).astype(int)
    g = np.zeros(ij.max(0) + 1, bool); g[ij[:, 0], ij[:, 1]] = True
    g = ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=2))
    return float(g.sum() * h * h)

def field(m, names):
    for n in names:
        if n in m.cell_data: return np.asarray(m.cell_data[n])
    return None

rows = []
import time
FILES = [OUT] if os.path.isfile(OUT) else sorted(glob.glob(f"{OUT}/**/inference_*.vtp", recursive=True))
for f in FILES:
    if time.time() - os.path.getmtime(f) < 90: continue      # still being written
    done = f"{VDIR}/{os.path.basename(f)}.json"
    if os.path.exists(done): rows.append(json.load(open(done))); continue
    case = re.search(r"(run_\d+)\.vtp$", f).group(1); model = re.search(r"inference_([a-z]+)", f).group(1)
    m = pv.read(f).triangulate()
    m = m.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True, consistent_normals=True)
    n = np.asarray(m.cell_data["Normals"]); A = m.compute_cell_sizes(length=False, volume=False).cell_data["Area"]
    c = m.cell_centers().points; c = c - c.mean(0)
    if np.sum((c * n).sum(1) * A) < 0: n = -n   # outward normals: enclosed volume (1/3) sum(c.n)A must be positive
    p_pr = field(m, ["pressure_pred"]); t_pr = field(m, ["wall_shear_stress_pred"])
    p_gt = field(m, ["pressure_true", "pMeanTrim"]); t_gt = field(m, ["wall_shear_stress_true", "wallShearStressMeanTrim"])
    Fx = lambda p, t: float(-np.sum((p * n[:, 0] + t[:, 0]) * A))     # drag, positive downstream
    rid = int(case.split("_")[1]); fan = not os.path.exists(f"{ROOT}/{case}/force_mom_{rid}.csv")   # no CFD truth
    r = dict(case=case, model=model, fan=fan, cells=int(m.n_cells), Fx_pred=Fx(p_pr, t_pr), nan_frac=float(np.isnan(p_pr).mean()))
    if not fan:
        Fg = Fx(p_gt, t_gt); cd = float(open(f"{ROOT}/{case}/force_mom_{rid}.csv").read().split("\n")[1].split(",")[0])
        aref = float(open(f"{ROOT}/{case}/geo_ref_{rid}.csv").read().split("\n")[1].split(",")[1])
        r.update(Fx_true=Fg, Cd_hifi=cd, aRef=aref, q_check=Fg / (cd * aref))
    r["A_frontal"] = frontal_area(m)
    rows.append(r); json.dump(r, open(done, "w"))
    # viewer surface: decimate, carry Cp at points
    # quadric decimation keeps silhouette and creases (decimate_pro without topology preservation tore holes);
    # 120k triangles is enough for mirrors, wheel arches and the underbody to read in the viewer
    d = m.extract_surface().triangulate().decimate(1 - min(1, 120000 / m.n_cells), volume_preservation=True).clean()
    tree = cKDTree(m.cell_centers().points); _, idx = tree.query(d.points)
    np.savez_compressed(f"{VDIR}/{model}_{case}.npz", pts=d.points.astype(np.float32), tri=d.faces.reshape(-1, 4)[:, 1:].astype(np.int32),
                        p_pred=p_pr[idx].astype(np.float32), p_true=(p_gt[idx] if not fan else np.zeros(len(idx))).astype(np.float32))
    print(r, flush=True)
Q = 0.5 * 38.889 ** 2
for r in rows:
    r["Cd_pred"] = r["Fx_pred"] / (Q * r.get("aRef", r["A_frontal"]))   # NaN if any cell is NaN: a partial prediction has no drag
    if not r["fan"]: r["Cd_true_check"] = r["Fx_true"] / (Q * r["aRef"])
json.dump(dict(q=Q, rows=rows), open(RES, "w"), indent=1)
print("q", Q, [round(r["q_check"], 1) for r in rows if not r["fan"]])
