"""Shrink-wrap a car mesh into one clean, watertight outer skin with outward normals.

Fan-made meshes have zero-area faces (GeoTransolver turns one into NaN over the whole car), interior
shells (seats, engines, doubled panels) and inconsistent winding (the models read surface normals, so an
inside-out car gets suction where the stagnation pressure should be). This rebuilds the surface:
sample the input densely, mark voxels within r of it, flood-fill the outside air from the domain corner,
erode the closed solid back by r (distance field level set), marching cubes, Taubin smoothing, keep the
largest component, drop degenerate faces, orient outward. Gaps narrower than ~2r (panel seams, grille
slots) close; everything inside disappears. The car is then put back on the DrivAerML frame
(front at x = -0.924 m, ground at z = -0.3176 m, centred in y).

usage: wrap_mesh.py <in.stl> <out_dir> <run_id> [--h 0.008] [--r 0.024] [--rot x180|y180] [--truth boundary.vtp] [--keep-frame]
  --truth: a DrivAerML boundary file; its fields are copied to the wrapped surface by nearest cell
           (used for the control case, so the benchmark can score the wrapped DrivAer against CFD)
"""
import argparse, json, os, numpy as np, trimesh, pyvista as pv
from scipy import ndimage
from skimage.measure import marching_cubes

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("rid", type=int)
ap.add_argument("--h", type=float, default=0.008); ap.add_argument("--r", type=float, default=0.024)
ap.add_argument("--truth"); ap.add_argument("--keep-frame", action="store_true")
ap.add_argument("--rot", choices=["x180", "y180"], help="turn the car over: x180 = upside down, y180 = upside down and back to front")
a = ap.parse_args()
X0, Z0 = -0.924, -0.3176

m = trimesh.load(a.src, force="mesh")
if a.rot: m.vertices = m.vertices * ({"x180": [1, -1, -1], "y180": [-1, 1, -1]}[a.rot])   # proper rotations, winding kept
area0 = float(m.area)
n = int(4 * m.area / (a.h / 2) ** 2)                      # ~4 samples per (h/2)^2 cell: no holes in the voxel shell
pts = np.vstack([trimesh.sample.sample_surface(m, n, seed=0)[0], m.vertices])
pad = a.r + 4 * a.h
lo = pts.min(0) - pad; shape = np.ceil((pts.max(0) + pad - lo) / a.h).astype(int) + 1
shell = np.zeros(shape, bool); ij = np.round((pts - lo) / a.h).astype(int); shell[tuple(ij.T)] = True
near = ndimage.distance_transform_edt(~shell) * a.h <= a.r          # thick shell: closes gaps < 2r
lab, _ = ndimage.label(~near); outside = lab == lab[0, 0, 0]
f = ndimage.distance_transform_edt(~outside).astype(np.float32) * a.h  # depth inside the closed solid
f = ndimage.gaussian_filter(f, 1.0)                                    # soften the voxel staircase
v, fc, _, _ = marching_cubes(f, level=a.r, spacing=(a.h,) * 3, allow_degenerate=False)  # erode back by r: the outer skin
w = trimesh.Trimesh(v + lo, fc, process=False)
w = max(w.split(only_watertight=False), key=lambda c: c.area)
trimesh.smoothing.filter_taubin(w, lamb=0.5, nu=-0.53, iterations=10)
w.fix_normals()
if w.volume < 0: w.invert()
if not a.keep_frame:
    w.vertices[:, 0] += X0 - w.vertices[:, 0].min(); w.vertices[:, 1] -= (w.vertices[:, 1].max() + w.vertices[:, 1].min()) / 2
    w.vertices[:, 2] += Z0 - w.vertices[:, 2].min()
A = w.area_faces
assert A.min() > 0 and np.isfinite(w.face_normals).all(), "degenerate faces left"

d = f"{a.out}/run_{a.rid}"; os.makedirs(d, exist_ok=True)
w.export(f"{d}/drivaer_{a.rid}.stl")
# DrivAerML convention, which the models were trained on: the STL is wound outward, the boundary .vtp INWARD
# (OpenFOAM patch normals point out of the fluid, into the car). An outward-wound .vtp reads as an inside-out car.
pd = pv.PolyData(w.vertices, np.hstack([np.full((len(w.faces), 1), 3), w.faces[:, ::-1]]).ravel())
names = (("CpMeanTrim", 1), ("pMeanTrim", 1), ("pPrime2MeanTrim", 1), ("wallShearStressMeanTrim", 3))
if a.truth:
    from scipy.spatial import cKDTree
    t = pv.read(a.truth); _, idx = cKDTree(t.cell_centers().points).query(pd.cell_centers().points)
    for k, _ in names: pd.cell_data[k] = np.asarray(t.cell_data[k])[idx]
else:
    for k, wd in names: pd.cell_data[k] = np.zeros((pd.n_cells, wd), np.float32).squeeze()
pd.save(f"{d}/boundary_{a.rid}.vtp")
meta = dict(run=a.rid, src=os.path.basename(a.src), h=a.h, r=a.r, faces=int(len(w.faces)), area_in=round(area0, 2),
            area_out=round(float(w.area), 2), volume=round(float(w.volume), 3), watertight=bool(w.is_watertight),
            min_face_area=float(A.min()), extents=np.ptp(w.vertices, 0).round(3).tolist())
json.dump(meta, open(f"{d}/wrap_{a.rid}.json", "w"), indent=1)
print(json.dumps(meta))
