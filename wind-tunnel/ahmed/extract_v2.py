"""Extract the v2 viewer payload from a converged Ahmed case.

Usage: python extract_v2.py <case_dir> <out.npz> [--surface] [--slice-png out.png]
  --surface      body Cp surface, Q-criterion vortex isosurface, and a velocity grid for particles
  --slice-png    the symmetry-plane mesh (real cell edges) coloured by |U|/U, as a PNG texture

The half model is mirrored to a full body. Streamwise vorticity flips sign under the
mirror (it is a pseudo-vector), so the two C-pillar vortices show opposite colours.
"""
import sys
import numpy as np
import pyvista as pv

UINF, H = 40.0, 0.288
QDYN = 0.5 * UINF**2
# Particle grid (half domain, y >= 0) around the body. Ground clearance 0.05, roof at 0.338.
GX, GY, GZ = (-0.45, 2.75, 128), (0.0, 0.62, 40), (0.0, 0.66, 42)

case, out = sys.argv[1], sys.argv[2]
want_surface = "--surface" in sys.argv
png = sys.argv[sys.argv.index("--slice-png") + 1] if "--slice-png" in sys.argv else None

open(f"{case}/case.foam", "a").close()
r = pv.OpenFOAMReader(f"{case}/case.foam")
r.set_active_time_value(r.time_values[-1])
r.enable_all_patch_arrays()
data = r.read()
payload = {}

if png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    sym = data["boundary"]["symmetry"]
    x0, x1, z0, z1 = -0.35, 2.05, 0.0, 0.62
    polys, vals = [], []
    U = sym.cell_data["U"]
    for i in range(sym.n_cells):
        pts = sym.get_cell(i).points
        if pts[:, 0].max() < x0 or pts[:, 0].min() > x1 or pts[:, 2].min() > z1:
            continue
        polys.append(pts[:, [0, 2]])
        vals.append(np.linalg.norm(U[i]) / UINF)
    W_IN = 24.0
    fig = plt.figure(figsize=(W_IN, W_IN * (z1 - z0) / (x1 - x0)), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(z0, z1); ax.axis("off")
    pc = PolyCollection(polys, array=np.array(vals), cmap="viridis", clim=(0, 1.3),
                        edgecolors=(1, 1, 1, 0.28), linewidths=0.25)
    ax.add_collection(pc)
    fig.savefig(png, transparent=True)
    payload["slice_cells_in_view"] = np.int32(len(polys))
    payload["slice_extent"] = np.array([x0, x1, z0, z1], np.float32)
    print(f"slice: {len(polys)} cells in view -> {png}")

if want_surface:
    vol = data["internalMesh"]
    body = data["boundary"]["ahmed"]
    # --- body surface with Cp, mirrored and decimated
    surf = body.extract_surface().triangulate().cell_data_to_point_data()
    surf["Cp"] = surf["p"] / QDYN
    full = surf.merge(surf.reflect((0, 1, 0), point=(0, 0, 0))).clean(tolerance=1e-6).extract_surface().triangulate()
    if full.n_cells > 48000:
        full = full.decimate_pro(1 - 48000 / full.n_cells, preserve_topology=True, feature_angle=25)
    full = full.compute_normals(auto_orient_normals=True, split_vertices=True, feature_angle=40)
    payload.update(s_verts=full.points.astype(np.float32), s_norms=full.point_data["Normals"].astype(np.float32),
                   s_tris=full.faces.reshape(-1, 4)[:, 1:].astype(np.uint32), s_cp=full["Cp"].astype(np.float32))

    # --- Q-criterion and streamwise vorticity on points, restricted to the wake/near-body region
    roi = vol.clip_box((-0.3, 2.6, 0.0, 0.6, 0.0, 0.7), invert=False)
    roi = roi.cell_data_to_point_data()
    d = roi.compute_derivative(scalars="U", gradient="G", vorticity="vort", qcriterion="Q")
    qn = d["Q"] / (UINF / H) ** 2                # normalised Q* = Q (H/U)^2
    d["Qn"] = qn
    d["wx"] = d["vort"][:, 0] * H / UINF        # normalised streamwise vorticity
    iso = {}
    for lev in (2.0, 8.0):
        c = d.contour([lev], scalars="Qn").extract_surface().triangulate()
        c = c.connectivity("largest") if c.n_cells and lev > 50 else c
        if c.n_cells > 26000:
            c = c.decimate_pro(1 - 26000 / c.n_cells, preserve_topology=False)
        if c.n_cells == 0:
            continue
        c2 = c.reflect((0, 1, 0), point=(0, 0, 0))
        c2["wx"] = -c2["wx"]
        m = c.merge(c2).compute_normals(auto_orient_normals=False, split_vertices=False, consistent_normals=True)
        iso[lev] = m
        payload[f"q{int(lev)}_verts"] = m.points.astype(np.float32)
        payload[f"q{int(lev)}_norms"] = m.point_data["Normals"].astype(np.float32)
        payload[f"q{int(lev)}_tris"] = m.faces.reshape(-1, 4)[:, 1:].astype(np.uint32)
        payload[f"q{int(lev)}_wx"] = m["wx"].astype(np.float32)
        print(f"Q*={lev}: {m.n_points} pts / {m.n_cells} tris, wx {m['wx'].min():.1f}..{m['wx'].max():.1f}")

    # --- velocity grid (half domain) for in-browser particle advection
    g = pv.ImageData(dimensions=(GX[2], GY[2], GZ[2]),
                     spacing=((GX[1] - GX[0]) / (GX[2] - 1), (GY[1] - GY[0]) / (GY[2] - 1), (GZ[1] - GZ[0]) / (GZ[2] - 1)),
                     origin=(GX[0], GY[0], GZ[0]))
    vpt = vol.cell_data_to_point_data()
    s = g.sample(vpt)
    u = s["U"] / UINF
    u[s["vtkValidPointMask"] == 0] = 0.0
    payload["grid_u"] = np.clip(np.round(u * 127 / 1.5), -127, 127).astype(np.int8)   # x fastest, then y, then z
    payload["grid_dims"] = np.array([GX[2], GY[2], GZ[2]], np.int32)
    payload["grid_box"] = np.array([GX[0], GX[1], GY[0], GY[1], GZ[0], GZ[1]], np.float32)
    payload["grid_valid"] = (s["vtkValidPointMask"] > 0).astype(np.uint8)
    print(f"surface {full.n_points} pts / {full.n_cells} tris, Cp {full['Cp'].min():.2f}..{full['Cp'].max():.2f}; "
          f"grid {GX[2]}x{GY[2]}x{GZ[2]}")

np.savez_compressed(out, **payload)
print("wrote", out)
