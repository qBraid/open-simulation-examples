"""Fine 2D cross-section of a thin-film metal edge: the "local" half of the two-step
surface-participation method (Wang et al., APL 107, 162601 (2015), Supplement A).

Geometry: two films of thickness h on a substrate, separated by a slot of width g
(right film at 1 V, left at 0 V). Near each film edge the field of a zero-thickness
sheet follows |E| = K / sqrt(u) (u = distance from the edge), whatever the global
geometry. Finite thickness regularises the singularity. This script computes, per unit
edge length and per K^2, the surface-energy integrals inside the band u < x0:

    S_MS = (eps_s^2/eps_MS) * int_0^x0 |E_n,sub|^2 du                     (film underside)
    S_MA = (1/eps_MA) * [int_0^x0 |E_n,air|^2 du + int_0^h |E_n|^2 dz]     (top + sidewall)
    S_SA = int_-x0^0 (eps_SA |E_t|^2 + |E_n,air|^2/eps_SA) du               (bare substrate)

K is defined exactly as the global 3D post-processing defines it: from the total
surface charge on the film in the band x0/4 < u < x0,
    Q_band / eps0 = (1 + eps_s) * K * sqrt(x0),
which is the zero-thickness asymptote, valid for h << x0 << g. Units: micrometres.
The surface participation of a 3D design is then
    p_i(edges) = (t / (2 U/eps0)) * sum_edges int K(y)^2 S_i dy.
"""
import json
import sys

import gmsh
import numpy as np
from skfem import Basis, ElementTriP2, MeshTri, asm, condense, solve
from skfem.models.poisson import laplace


def build_mesh(g, h, X, size_corner, size_far):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    occ = gmsh.model.occ
    air = occ.addRectangle(-X, 0, 0, 2 * X, X)
    sub = occ.addRectangle(-X, -X, 0, 2 * X, X)
    left = occ.addRectangle(-X, 0, 0, X - g / 2, h)
    right = occ.addRectangle(g / 2, 0, 0, X - g / 2, h)
    out, _ = occ.cut([(2, air)], [(2, left), (2, right)])
    occ.fragment(out, [(2, sub)])
    occ.synchronize()
    surfs = gmsh.model.getEntities(2)
    sub_tags, air_tags = [], []
    for _, s in surfs:
        com = occ.getCenterOfMass(2, s)
        (sub_tags if com[1] < 0 else air_tags).append(s)
    gmsh.model.addPhysicalGroup(2, sub_tags, 1)
    gmsh.model.addPhysicalGroup(2, air_tags, 2)
    corners = []
    for x in (-g / 2, g / 2):
        for y in (0.0, h):
            corners.append(occ.addPoint(x, y, 0))
    occ.synchronize()
    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "PointsList", corners)
    f2 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f2, "InField", f1)
    gmsh.model.mesh.field.setNumber(f2, "SizeMin", size_corner)
    gmsh.model.mesh.field.setNumber(f2, "SizeMax", size_far)
    gmsh.model.mesh.field.setNumber(f2, "DistMin", size_corner * 2)
    gmsh.model.mesh.field.setNumber(f2, "DistMax", X)
    gmsh.model.mesh.field.setNumber(f2, "Sigmoid", 0)
    # geometric grading: mesh size grows ~linearly with distance (keeps element count small)
    f3 = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f3, "F", f"Min({size_far}, {size_corner} + 0.12*F{f1})")
    f4 = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f4, "FieldsList", [f2, f3])
    gmsh.model.mesh.field.setAsBackgroundMesh(f4)
    for k in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{k}", 0)
    gmsh.model.mesh.generate(2)
    nodes, coords, _ = gmsh.model.mesh.getNodes()
    xy = coords.reshape(-1, 3)[:, :2]
    idx = {int(t): i for i, t in enumerate(nodes)}
    tris, region = [], []
    for phys, tags in ((1, sub_tags), (2, air_tags)):
        for s in tags:
            et, _, en = gmsh.model.mesh.getElements(2, s)
            for typ, conn in zip(et, en):
                if typ == 2:
                    c = np.array([idx[int(n)] for n in conn]).reshape(-1, 3)
                    tris.append(c)
                    region.append(np.full(len(c), phys))
    gmsh.finalize()
    tris = np.vstack(tris)
    region = np.concatenate(region)
    used = np.unique(tris)
    remap = -np.ones(len(xy), int)
    remap[used] = np.arange(len(used))
    return xy[used].T, remap[tris].T, region


def run(g=30.0, h=0.1, eps_s=10.34, x0=3.0, eps_i=10.0, t=0.003, X=None, corner_div=40):
    X = X or max(40 * g, 40 * x0, 400.0)
    size_corner = min(h, t) / corner_div
    size_far = X / 15
    pts, tri, region = build_mesh(g, h, X, size_corner, size_far)
    m = MeshTri(pts, tri)
    eps_el = np.where(region == 1, eps_s, 1.0)
    e = ElementTriP2()
    basis = Basis(m, e)
    from skfem import BilinearForm
    from skfem.helpers import dot, grad


    @BilinearForm
    def a(u, v, w):
        return w["eps"] * dot(grad(u), grad(v))

    eps_field = np.repeat(eps_el[:, None], basis.X.shape[-1], axis=1)
    A = a.assemble(basis, eps=eps_field)
    # Dirichlet on film boundaries: nodes on conductor surfaces
    x, y = basis.doflocs
    tol = size_corner * 1e-3
    on_right = (x >= g / 2 - tol) & (y >= -tol) & (y <= h + tol)
    on_left = (x <= -g / 2 + tol) & (y >= -tol) & (y <= h + tol)
    # restrict to dofs actually on the film surface (interior of film is not meshed)
    D = np.where(on_right | on_left)[0]
    u = basis.zeros()
    u[on_right] = 1.0
    phi = solve(*condense(A, x=u, D=D))

    def probe(px, py):
        P = basis.probes(np.vstack([px, py]))
        return P @ phi

    d = 1e-5  # um, central difference well inside one P2 element

    # log-spaced sample points in u from ~h/2000 to beyond x0 (for the K band)
    us = np.geomspace(t / 2, x0, 600)
    xr = g / 2 + us
    # Fields are evaluated on the midline of each t-thick interface layer (offset t/2 from
    # the bare surface), as in Wang et al. Fig. S2: this is the physical regularisation of the
    # corner singularity, which otherwise depends on sub-nm mesh detail.
    o = t / 2
    def Ey(px, py):
        return -(probe(px, py + d / 2) - probe(px, py - d / 2)) / d
    def Ex(px, py):
        return -(probe(px + d / 2, py) - probe(px - d / 2, py)) / d
    one = np.ones_like(us)
    Ey_sub = Ey(xr, -o * one)            # MS layer (just below the film, in the substrate)
    Ey_top = Ey(xr, (h + o) * one)        # MA layer on the film top
    zs = np.linspace(o, h, 200)
    Ex_side = Ex((g / 2 - o) * np.ones_like(zs), zs)   # MA layer on the sidewall facing the slot
    ug = np.geomspace(o, min(x0, g / 2), 600)
    xg = g / 2 - ug
    Ex_gap = Ex(xg, o * np.ones_like(xg))  # SA layer on the bare substrate (air side of the surface)
    Ey_gap = Ey(xg, o * np.ones_like(xg))
    def logint(f, s):
        return np.trapezoid(f * s, np.log(s))

    # surface charge density / eps0 on the film: top (air) + bottom (substrate)
    sig = np.abs(Ey_top) + eps_s * np.abs(Ey_sub)
    band = (us >= x0 / 4) & (us <= x0)
    Qband = logint(sig[band], us[band])
    K = Qband / ((1 + eps_s) * np.sqrt(x0))  # same definition as the 3D post-processing
    I_MS = (eps_s**2 / eps_i) * logint(Ey_sub**2, us)
    I_MA = (1 / eps_i) * (logint(Ey_top**2, us) + np.trapezoid(Ex_side**2, zs))
    I_SA = eps_i * logint(Ex_gap**2, ug) + (1 / eps_i) * logint(Ey_gap**2, ug)
    # zero-thickness analytic check: K0 = 1/(pi sqrt(g)) for a slot between half-planes
    K0 = 1 / (np.pi * np.sqrt(g))
    out = dict(g=g, h=h, eps_s=eps_s, x0=x0, eps_i=eps_i, t=t, ndof=int(basis.N),
               K=K, K_analytic_zero_thickness=K0,
               S_MS=I_MS / K**2, S_MA=I_MA / K**2, S_SA=I_SA / K**2)
    return out


if __name__ == "__main__":
    args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(run(**args)))
