"""FE validation: tip deflection of a clamped 3D cantilever against Timoshenko beam theory.

Beam: L = 10, square section 1 x 1, E = 1, nu = 0.3, total tip shear P = 1.
Two discretisations, each refined three to four times:
  * H8: the trilinear brick used by the topology optimizer (same element stiffness, same grid code)
  * P2: quadratic tetrahedra in scikit-fem, as an independent higher-order check
Writes results/validation.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from topopt import Grid3D, lk_h8, solve_spd

L, B, Hh, E, NU, P = 10.0, 1.0, 1.0, 1.0, 0.3, 1.0
I = B * Hh**3 / 12
A = B * Hh
G = E / (2 * (1 + NU))
KAPPA = 10 * (1 + NU) / (12 + 11 * NU)  # Cowper's shear coefficient, rectangle
D_EB = P * L**3 / (3 * E * I)
D_TIM = D_EB + P * L / (KAPPA * G * A)


def h8_tip_deflection(n):
    """n elements through the depth; element size h = 1/n."""
    g = Grid3D(10 * n, n, n)
    KE, _ = lk_h8(NU)
    h = 1.0 / n
    sK = np.tile(KE.ravel() * h, g.nele)  # brick of side h: K = h * K_unit (E = 1)
    K = sp.coo_matrix((sK, (g.iK, g.jK)), shape=(g.ndof, g.ndof)).tocsc()
    K = (K + K.T) / 2
    F = np.zeros(g.ndof)
    # consistent load: each tip face (n x n faces) gets P/n^2, shared equally by its 4 nodes
    for j in range(n):
        for k in range(n):
            for jj, kk in ((j, k), (j + 1, k), (j, k + 1), (j + 1, k + 1)):
                F[3 * g.node_id(10 * n, jj, kk) + 1] -= P / n**2 / 4
    fixed = np.array([3 * g.node_id(0, j, k) + d for j in range(n + 1) for k in range(n + 1) for d in range(3)])
    free = np.setdiff1d(np.arange(g.ndof), fixed)
    U = np.zeros(g.ndof)
    U[free] = solve_spd(K[free][:, free], F[free])
    tip = [3 * g.node_id(10 * n, j, k) + 1 for j in range(n + 1) for k in range(n + 1)]
    w = np.zeros((n + 1, n + 1))  # trapezoid weights over the tip face
    w += 1; w[0, :] *= 0.5; w[-1, :] *= 0.5; w[:, 0] *= 0.5; w[:, -1] *= 0.5
    return float(-(U[tip] * w.ravel()).sum() / w.sum()), g.ndof


def p2_tip_deflection(n):
    import skfem
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    m = skfem.MeshTet.init_tensor(np.linspace(0, L, 10 * n + 1), np.linspace(0, Hh, n + 1), np.linspace(0, B, n + 1))
    e = skfem.ElementVector(skfem.ElementTetP2())
    ib = skfem.Basis(m, e, intorder=4)
    K = skfem.asm(linear_elasticity(*lame_parameters(E, NU)), ib)
    tip_facets = m.facets_satisfying(lambda x: np.isclose(x[0], L))
    fb = skfem.FacetBasis(m, e, facets=tip_facets, intorder=4)

    @skfem.LinearForm
    def traction(v, w):
        return -P / (B * Hh) * v.value[1]

    f = skfem.asm(traction, fb)
    D = ib.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    u = skfem.solve(*skfem.condense(K, f, D=D))

    @skfem.Functional
    def uy(w):
        return w["u"].value[1]

    defl = -skfem.asm(uy, fb, u=fb.interpolate(u)) / (B * Hh)
    return float(defl), int(ib.N)


def main():
    out = {"beam": dict(L=L, b=B, h=Hh, E=E, nu=NU, P=P),
           "analytic": {"euler_bernoulli": D_EB, "timoshenko": D_TIM, "kappa": KAPPA}, "h8": [], "p2": []}
    for n in [int(v) for v in __import__("os").environ.get("H8_LEVELS", "1,2,4,8,16").split(",")]:
        t = time.time(); d, nd = h8_tip_deflection(n)
        out["h8"].append(dict(n=n, h=1 / n, ndof=nd, deflection=d, err_pct=100 * (d - D_TIM) / D_TIM, seconds=round(time.time() - t, 2)))
        print("H8", out["h8"][-1], flush=True)
    for n in (1, 2, 3, 4):
        t = time.time(); d, nd = p2_tip_deflection(n)
        out["p2"].append(dict(n=n, h=1 / n, ndof=nd, deflection=d, err_pct=100 * (d - D_TIM) / D_TIM, seconds=round(time.time() - t, 2)))
        print("P2", out["p2"][-1], flush=True)
    # Richardson extrapolation of the H8 sequence (asymptotic rate from the last three meshes)
    d = [r["deflection"] for r in out["h8"]][-3:]
    rate = np.log2(abs(d[1] - d[0]) / abs(d[2] - d[1]))
    ext = d[2] + (d[2] - d[1]) / (2**rate - 1)
    out["h8_richardson"] = dict(observed_order=float(rate), extrapolated=float(ext), err_pct=float(100 * (ext - D_TIM) / D_TIM))
    # 3D continuum value of this exact problem: fit d(h) = d_inf - C h^p through the last three P2 meshes
    from scipy.optimize import brentq
    (h1, d1), (h2, d2), (h3, d3) = [(r["h"], r["deflection"]) for r in out["p2"][-3:]]
    p = brentq(lambda p: (d2 - d1) / (d3 - d2) - (h1**p - h2**p) / (h2**p - h3**p), 0.5, 8.0)
    C = (d2 - d1) / (h1**p - h2**p)
    d_inf = d3 + C * h3**p
    out["continuum_3d"] = dict(source="P2 tetrahedra, extrapolated", observed_order=float(p), deflection=float(d_inf),
                               vs_timoshenko_pct=float(100 * (d_inf - D_TIM) / D_TIM),
                               vs_euler_bernoulli_pct=float(100 * (d_inf - D_EB) / D_EB))
    for r in out["h8"]:
        r["err_vs_continuum_pct"] = 100 * (r["deflection"] - d_inf) / d_inf
    out["h8_richardson"]["err_vs_continuum_pct"] = float(100 * (ext - d_inf) / d_inf)
    print(out["analytic"], out["h8_richardson"], out["continuum_3d"])
    Path("results").mkdir(exist_ok=True)
    Path("results/validation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
