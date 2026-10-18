"""Place fan-made production-car meshes in the DrivAerML frame so the DrivAerML-trained models can run on them.
DrivAerML frame: x streamwise (front at x = -0.924 m), y lateral (symmetric), z up, ground at z = -0.3176 m.
Each mesh is scaled to the real car's length, rotated so length -> x, height -> z, front -> -x,
refined so no edge exceeds 3 cm (the models sample points from the surface), and written as
run_<id>/drivaer_<id>.stl plus a boundary_<id>.vtp with zero 'truth' fields (there is no ground truth).
usage: prep_fan.py <src_dir> <out_root>
"""
import json, sys, numpy as np, trimesh, pyvista as pv
SRC, OUT = sys.argv[1:3]
# real overall length (m, approximate from manufacturer specs) and front-direction override if the heuristic fails
CARS = {"modelx": (9001, 5.036), "taycan": (9002, 4.963), "ioniq5": (9003, 4.635), "i8": (9004, 4.689),
        "golf": (9005, 4.255), "e92": (9006, 4.615), "bmw_e30": (9007, 4.345)}
FLIP = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
X0, Z0 = -0.924, -0.3176
meta = {}
for name, (rid, L) in CARS.items():
    m = trimesh.load(f"{SRC}/{name}.stl", force="mesh")
    m.merge_vertices()
    ext = m.extents
    ax = np.argsort(ext)            # [height, width, length]
    R = np.zeros((3, 3)); R[0, ax[2]] = 1; R[1, ax[1]] = 1; R[2, ax[0]] = 1
    if np.linalg.det(R) < 0: R[1] *= -1
    v = (m.vertices - m.bounds.mean(0)) @ R.T
    # up sign: the roof is narrower than the floor plan; the top 10% of height has less spread in y
    top, bot = v[v[:, 2] > np.percentile(v[:, 2], 90)], v[v[:, 2] < np.percentile(v[:, 2], 10)]
    if np.ptp(top[:, 1]) > np.ptp(bot[:, 1]): v[:, 2] *= -1; v[:, 1] *= -1
    # front sign: the hood (front 20%) sits lower than the boot/roof line at the rear 20%
    fr, rr = v[v[:, 0] < np.percentile(v[:, 0], 15)], v[v[:, 0] > np.percentile(v[:, 0], 85)]
    if fr[:, 2].max() > rr[:, 2].max(): v[:, 0] *= -1; v[:, 1] *= -1
    if FLIP.get(name): v[:, 0] *= -1; v[:, 1] *= -1
    v *= L / np.ptp(v[:, 0])
    v[:, 0] += X0 - v[:, 0].min(); v[:, 1] -= v[:, 1].mean(); v[:, 2] += Z0 - v[:, 2].min()
    mm = trimesh.Trimesh(v, m.faces, process=True)
    mm.fix_normals()
    vv, ff = trimesh.remesh.subdivide_to_size(mm.vertices, mm.faces, max_edge=0.03)
    mm = trimesh.Trimesh(vv, ff, process=True)
    d = f"{OUT}/run_{rid}"; import os; os.makedirs(d, exist_ok=True)
    mm.export(f"{d}/drivaer_{rid}.stl")
    pd = pv.PolyData(mm.vertices, np.hstack([np.full((len(mm.faces), 1), 3), mm.faces]).ravel())
    n = pd.n_cells
    for k, w in (("CpMeanTrim", 1), ("pMeanTrim", 1), ("pPrime2MeanTrim", 1), ("wallShearStressMeanTrim", 3)):
        pd.cell_data[k] = np.zeros((n, w), np.float32).squeeze()
    pd.save(f"{d}/boundary_{rid}.vtp")
    meta[name] = dict(run=rid, length_m=L, faces_src=int(len(m.faces)), faces=int(n), extents=np.ptp(mm.vertices, 0).round(3).tolist(),
                      watertight=bool(mm.is_watertight))
    print(name, meta[name])
json.dump(meta, open(f"{OUT}/fan_meta.json", "w"), indent=1)
