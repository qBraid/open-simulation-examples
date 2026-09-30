"""Extract a compact viewer payload from the converged Ahmed case.

Usage: python extract.py <case_dir> <out.npz>
Surface: the body patch with Cp = p / (0.5 U^2), mirrored to a full body and decimated.
Streamlines: seeded on a rake upstream of the body, mirrored, resampled.
Slice: |U|/U and Cp on the symmetry plane (y = 0), sampled on a regular grid.
"""
import sys
import numpy as np
import pyvista as pv

case, out = sys.argv[1], sys.argv[2]
UINF = 40.0
Q = 0.5 * UINF**2
open(f"{case}/case.foam", "a").close()
r = pv.OpenFOAMReader(f"{case}/case.foam")
r.set_active_time_value(r.time_values[-1])
r.enable_all_patch_arrays()
data = r.read()
vol = data["internalMesh"]
body = data["boundary"]["ahmed"]

# --- surface, Cp on points, mirrored, decimated
surf = body.extract_surface().triangulate().cell_data_to_point_data()
surf["Cp"] = surf["p"] / Q
full = surf.merge(surf.reflect((0, 1, 0), point=(0, 0, 0))).clean(tolerance=1e-6)
full = full.extract_surface().triangulate()
target = 30000
if full.n_cells > target:
    full = full.decimate_pro(1 - target / full.n_cells, preserve_topology=True, feature_angle=30)
full = full.compute_normals(auto_orient_normals=True, split_vertices=False)
tris = full.faces.reshape(-1, 4)[:, 1:].astype(np.uint32)

# --- streamlines from an upstream rake (half domain), then mirrored
vol_pt = vol.cell_data_to_point_data()
ys = np.linspace(0.01, 0.33, 12)
zs = np.linspace(0.07, 0.46, 14)
seeds = pv.PolyData(np.array([(-0.25, y, z) for y in ys for z in zs]))
sl = vol_pt.streamlines_from_source(seeds, vectors="U", integration_direction="forward",
                                    max_length=4.0, initial_step_length=0.2, max_steps=4000)
lines, speeds = [], []
for i in range(sl.n_cells):
    c = sl.get_cell(i)
    ids = c.point_ids
    if len(ids) < 4:
        continue
    pts = sl.points[ids]
    spd = np.linalg.norm(sl["U"][ids], axis=1) / UINF
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 0.5:
        continue
    n = int(min(90, max(12, s[-1] / 0.04)))
    si = np.linspace(0, s[-1], n)
    p = np.stack([np.interp(si, s, pts[:, k]) for k in range(3)], 1)
    v = np.interp(si, s, spd)
    for sign in (1, -1):
        q = p.copy(); q[:, 1] *= sign
        lines.append(q.astype(np.float32)); speeds.append(v.astype(np.float32))
offs = np.cumsum([0] + [len(l) for l in lines]).astype(np.uint32)

# --- symmetry-plane slice on a grid
NX, NZ = 300, 80
g = pv.ImageData(dimensions=(NX, 1, NZ), spacing=(3.6 / (NX - 1), 1, 0.8 / (NZ - 1)), origin=(-0.6, 0.0005, 0.0))
sm = g.sample(vol_pt)
mask = sm["vtkValidPointMask"].reshape(NZ, NX) > 0
umag = (np.linalg.norm(sm["U"], axis=1) / UINF).reshape(NZ, NX)
cp = (sm["p"] / Q).reshape(NZ, NX)

np.savez_compressed(out, verts=full.points.astype(np.float32), norms=full.point_data["Normals"].astype(np.float32),
                    tris=tris, cp=full["Cp"].astype(np.float32),
                    line_pts=np.concatenate(lines), line_spd=np.concatenate(speeds), line_offs=offs,
                    slice_u=np.where(mask, umag, np.nan).astype(np.float32),
                    slice_cp=np.where(mask, cp, np.nan).astype(np.float32),
                    slice_extent=np.array([-0.6, 3.0, 0.0, 0.8], np.float32))
print(f"surface {full.n_points} pts / {len(tris)} tris; {len(lines)} streamlines, "
      f"{offs[-1]} pts; Cp range {full['Cp'].min():.2f}..{full['Cp'].max():.2f}")
