"""Global half of the two-step surface-participation method.

Reads Palace electrostatic terminal solutions, builds the qubit-mode field as a linear
combination of terminal fields, and computes the metal-air (MA), metal-substrate (MS)
and substrate-air (SA) interface participation ratios:

  * interior (farther than x0 from any metal edge): integrate the coarse 3D field
    sampled just above / below the zero-thickness metal sheets and on bare substrate;
  * edge band (within x0 of an edge): extract the edge-field strength K(y) from the
    surface charge in the band x0/4 < u < x0 and multiply K^2 by the universal
    finite-thickness integrals S_i(g_local) from edge2d.py.

Conventions follow Palace / Wang et al.: thickness t, permittivity eps_i for every lossy
layer; p_i = (eps0 t / 2) * int (weights) |E|^2 dA / U.
"""
import json

import numpy as np
import vtk
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

EPS0 = 8.8541878128e-12


def read_pvtu(path):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


class Prober:
    def __init__(self, grids):
        self.grids = grids  # one per terminal
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(grids[0])
        self.locator.BuildLocator()

    def E(self, pts, weights):
        """pts (N,3) in um -> mode field (N,3) in V/m, combination sum_k w_k E_k."""
        poly = vtk.vtkPolyData()
        vp = vtk.vtkPoints()
        vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float64), deep=True))
        poly.SetPoints(vp)
        out = np.zeros((len(pts), 3))
        for g, w in zip(self.grids, weights):
            if w == 0:
                continue
            pf = vtk.vtkProbeFilter()
            pf.SetInputData(poly)
            pf.SetSourceData(g)
            pf.SetCellLocatorPrototype(self.locator)
            pf.ComputeToleranceOff()   # otherwise points near the sheet snap to the other side's cell
            pf.SetTolerance(1e-9)
            pf.Update()
            arr = vtk_to_numpy(pf.GetOutput().GetPointData().GetArray("E"))
            out += w * arr
        return out


def raster(region, spacing, near=None, x0=None, fine=None, near_width=None):
    """Points covering a shapely region; finer spacing within near_width of `near` edges."""
    minx, miny, maxx, maxy = region.bounds
    xs = np.arange(minx + spacing / 2, maxx, spacing)
    ys = np.arange(miny + spacing / 2, maxy, spacing)
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel()]
    import shapely
    inside = shapely.contains_xy(region, P[:, 0], P[:, 1])
    pts, area = [P[inside]], [np.full(inside.sum(), spacing**2)]
    if near is not None and fine is not None:
        band = region.intersection(near.buffer(near_width))
        # remove coarse points inside the band and replace with fine ones
        inb = shapely.contains_xy(band, pts[0][:, 0], pts[0][:, 1])
        pts[0], area[0] = pts[0][~inb], area[0][~inb]
        minx, miny, maxx, maxy = band.bounds
        xs = np.arange(minx + fine / 2, maxx, fine)
        ys = np.arange(miny + fine / 2, maxy, fine)
        X, Y = np.meshgrid(xs, ys)
        Q = np.c_[X.ravel(), Y.ravel()]
        ok = shapely.contains_xy(band, Q[:, 0], Q[:, 1])
        pts.append(Q[ok])
        area.append(np.full(ok.sum(), fine**2))
    return np.vstack(pts), np.concatenate(area)


def edge_samples(geom, ds, sym_x=None):
    """Points along all polygon boundaries with inward normals (into the metal)."""
    out = []
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for poly in polys:
        rings = [poly.exterior] + list(poly.interiors)
        for ring in rings:
            c = np.asarray(ring.coords)
            for a, b in zip(c[:-1], c[1:]):
                L = np.linalg.norm(b - a)
                if L < 1e-9:
                    continue
                if sym_x is not None and abs(a[0] - sym_x) < 1e-6 and abs(b[0] - sym_x) < 1e-6:
                    continue  # cut line on the symmetry plane, not a physical edge
                n = max(1, int(np.ceil(L / ds)))
                tvec = (b - a) / L
                nrm = np.array([-tvec[1], tvec[0]])
                mid = (a + b) / 2
                if not poly.buffer(1e-6).contains(Point(*(mid + 1e-3 * nrm))):
                    nrm = -nrm
                for k in range(n):
                    p = a + (k + 0.5) / n * (b - a)
                    out.append((p[0], p[1], nrm[0], nrm[1], L / n))
    return np.array(out)


def local_gap(points, normals, all_metal_lines, own_geom, cap=500.0):
    """Distance from an edge, going outward, to the nearest other metal edge."""
    g = np.full(len(points), cap)
    for i, (p, n) in enumerate(zip(points, normals)):
        ray = LineString([p - 1e-3 * n, p - cap * n])
        hit = ray.intersection(all_metal_lines)
        if not hit.is_empty:
            d = Point(*p).distance(hit)
            if d > 1e-2:
                g[i] = min(cap, d)
    return g


def participation(layout, grids, C, x0, s_table, t=0.003, eps_i=10.0, delta=0.02,
                  fine=0.5, coarse=8.0, near_width=40.0, ds=1.0, window_pad=600.0, window=None):
    eps_s = layout["eps_sub"]
    names = [n for n in layout["conductors"] if n != layout.get("ground")]
    # terminal voltages for the mode
    mode = layout["mode"]
    if layout["mode_type"] == "floating_pair":
        q = np.array([mode.get(n, 0.0) for n in names])
        V = np.linalg.solve(C, q)
    else:  # driven island(s): voltages given directly, others grounded
        V = np.array([mode.get(n, 0.0) for n in names])
    U = 0.5 * V @ C @ V  # J, for these terminal voltages
    prober = Prober(grids)
    metal_all = unary_union(list(layout["conductors"].values()))
    lines = metal_all.boundary
    res = {"U_J": U, "V": V.tolist(), "x0_um": x0, "t_nm": t * 1e3, "eps_i": eps_i}
    c = 0.5 * EPS0 * t * 1e-6 / U  # p = c * int |E|^2 dA, dA in um^2 -> m^2 (1e-12), t um->m
    c *= 1e-12
    # region of interest: metal + bare substrate within window_pad of metal, clipped to chip
    x0c, y0c, x1c, y1c = layout["chip"]
    chip = Polygon([(x0c, y0c), (x1c, y0c), (x1c, y1c), (x0c, y1c)])
    if window is not None:  # restrict to the region where the mode field lives
        roi = Polygon([(window[0], window[1]), (window[2], window[1]), (window[2], window[3]),
                       (window[0], window[3])]).intersection(chip)
    else:
        roi = metal_all.buffer(window_pad).intersection(chip)
    sym_x = x0c if layout.get("half_x") else None
    res["roi_area_um2"] = roi.area
    per = {}
    # --- interiors
    for name, geom in layout["conductors"].items():
        inner = geom.buffer(-x0)
        if inner.is_empty:
            per[name] = {"MS_int": 0.0, "MA_int": 0.0}
            continue
        reg = inner.intersection(roi)
        P, A = raster(reg, coarse, near=reg.boundary, fine=fine, near_width=near_width)
        top = prober.E(np.c_[P, np.full(len(P), +delta)], V)
        bot = prober.E(np.c_[P, np.full(len(P), -delta)], V)
        I_MA = np.sum(A * top[:, 2] ** 2) / eps_i                  # (eps_air^2 / eps_MA) |E_n,air|^2
        I_MS = np.sum(A * bot[:, 2] ** 2) * eps_s**2 / eps_i       # (eps_s^2 / eps_MS) |E_n,sub|^2
        per[name] = {"MS_int": c * I_MS, "MA_int": c * I_MA, "n_pts": int(len(P))}
    # --- bare substrate interior (farther than x0 from metal)
    bare = roi.difference(metal_all.buffer(x0))
    P, A = raster(bare, coarse, near=metal_all.boundary, fine=fine, near_width=near_width)
    top = prober.E(np.c_[P, np.full(len(P), +delta)], V)
    I_SA = np.sum(A * (eps_i * (top[:, 0] ** 2 + top[:, 1] ** 2) + top[:, 2] ** 2 / eps_i))
    res["SA_int"] = c * I_SA
    # --- edges
    gs = np.array(sorted(s_table))
    def S(kind, g):
        vals = np.array([s_table[k][kind] for k in gs])
        return np.interp(np.log(np.clip(g, gs[0], gs[-1])), np.log(gs), vals)
    us = np.geomspace(x0 / 4, x0, 10)
    for name, geom in layout["conductors"].items():
        if name == layout.get("ground"):
            # ground plane edges: only those within the region of interest
            pass
        E = edge_samples(geom, ds, sym_x=sym_x)
        keep = np.array([roi.contains(Point(x, y)) for x, y in E[:, :2]])
        E = E[keep]
        pts, nrm, dl = E[:, :2], E[:, 2:4], E[:, 4]
        # band charge
        sig = np.zeros(len(pts))
        for k in range(len(us) - 1):
            um = 0.5 * (us[k] + us[k + 1])
            du = us[k + 1] - us[k]
            q = pts + um * nrm
            top = prober.E(np.c_[q, np.full(len(q), +delta)], V)[:, 2]
            bot = prober.E(np.c_[q, np.full(len(q), -delta)], V)[:, 2]
            sig += (np.abs(top) + eps_s * np.abs(bot)) * du * 1e-6  # (V/m)*m = V
        K = sig / ((1 + eps_s) * np.sqrt(x0 * 1e-6))  # V / m^0.5
        g = local_gap(pts, nrm, lines, geom)
        # S_i are dimensionless per K^2 per unit length (um-based in 2D -> consistent)
        # p_edge = (eps0 t / 2U) * sum K^2 [V^2/m] * S * dl[m]
        f = 0.5 * EPS0 * t * 1e-6 / U
        for kind in ("MS", "MA", "SA"):
            per[name][f"{kind}_edge"] = float(f * np.sum(K**2 * S(kind, g) * dl * 1e-6))
        per[name]["edge_len_um"] = float(dl.sum())
        per[name]["K_rms"] = float(np.sqrt(np.mean(K**2)))
        per[name]["g_median_um"] = float(np.median(g))
    res["per_conductor"] = per
    tot = {k: 0.0 for k in ("MS", "MA", "SA")}
    for name, d in per.items():
        tot["MS"] += d["MS_int"] + d.get("MS_edge", 0)
        tot["MA"] += d["MA_int"] + d.get("MA_edge", 0)
        tot["SA"] += d.get("SA_edge", 0)
    tot["SA"] += res["SA_int"]
    res["p"] = tot
    return res
