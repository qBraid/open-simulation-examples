"""SIMP topology optimization: faithful Python ports of top88 (2D) and top3d (3D),
plus a general 3D solver with passive regions and multiple load cases.

top88: Andreassen, Clausen, Schevenels, Lazarov, Sigmund, Struct Multidisc Optim 43 (2011) 1-16.
top3d: Liu and Tovar, Struct Multidisc Optim 50 (2014) 1175-1196.

The ports keep the reference algorithm step for step (same element stiffness,
same filter, same optimality-criteria bisection, same stopping rule); only the
linear solve differs (CHOLMOD instead of MATLAB's backslash), so results agree
with the reference codes to solver round-off.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

try:
    from sksparse.cholmod import cholesky as _cholesky
except Exception:  # pragma: no cover - fallback when CHOLMOD is unavailable
    _cholesky = None


def solve_spd(K, f):
    """Solve K u = f for a symmetric positive definite sparse K."""
    if _cholesky is not None:
        return _cholesky(K.tocsc())(f)
    from scipy.sparse.linalg import spsolve
    return spsolve(K.tocsc(), f)


# ---------------------------------------------------------------- 2D (top88)
def lk_q4(nu=0.3):
    """top88 bilinear Q4 plane-stress element stiffness (E=1, unit square)."""
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE = 1 / (1 - nu**2) / 24 * (np.block([[A11, A12], [A12.T, A11]])
                                 + nu * np.block([[B11, B12], [B12.T, B11]]))
    return KE


def _filter_2d(nelx, nely, rmin):
    r = int(np.ceil(rmin)) - 1
    rows, cols, vals = [], [], []
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            for i2 in range(max(i1 - r, 0), min(i1 + r + 1, nelx)):
                for j2 in range(max(j1 - r, 0), min(j1 + r + 1, nely)):
                    w = max(0.0, rmin - np.hypot(i1 - i2, j1 - j2))
                    rows.append(e1); cols.append(i2 * nely + j2); vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(nelx * nely, nelx * nely))
    return H, np.asarray(H.sum(1)).ravel()


def top88(nelx, nely, volfrac, penal, rmin, ft, bc="mbb", maxloop=10000, snapshot_every=0, log=None):
    """Port of top88.m. bc='mbb' (half MBB beam, as published) or 'cantilever'
    (Sigmund 2001, Sec. 5: left edge clamped, unit load down at the lower-right corner).
    Element/node numbering is column-major with rows from the top, exactly as in MATLAB.
    Returns dict with final xPhys (nely x nelx), compliance history and snapshots."""
    E0, Emin, nu = 1.0, 1e-9, 0.3
    KE = lk_q4(nu)
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(1 + nelx, 1 + nely).T  # MATLAB reshape
    edofVec = (2 * nodenrs[:-1, :-1] + 1).T.reshape(-1)  # column-major vectorization
    edofMat = edofVec[:, None] + np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    edofMat -= 1  # 0-based
    iK = np.kron(edofMat, np.ones((8, 1), dtype=int)).ravel()
    jK = np.kron(edofMat, np.ones((1, 8), dtype=int)).ravel()
    ndof = 2 * (nely + 1) * (nelx + 1)
    F = np.zeros(ndof)
    if bc == "mbb":
        F[1] = -1.0                                            # MATLAB dof 2
        fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), [ndof - 1])
    elif bc == "cantilever":
        F[ndof - 1] = -1.0                                     # dof 2*(nely+1)*(nelx+1)
        fixed = np.arange(0, 2 * (nely + 1))
    else:
        raise ValueError(bc)
    free = np.setdiff1d(np.arange(ndof), fixed)
    H, Hs = _filter_2d(nelx, nely, rmin)

    # element e in MATLAB column-major order: e = i*nely + j  (x stored as nely x nelx)
    x = np.full(nelx * nely, volfrac)
    xPhys = x.copy()
    hist, snaps = [], []
    loop, change = 0, 1.0
    U = np.zeros(ndof)
    KEf = KE.ravel()
    t0 = time.time()
    while change > 0.01 and loop < maxloop:
        loop += 1
        sK = (KEf[:, None] * (Emin + xPhys**penal * (E0 - Emin))[None, :]).T.ravel()
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2
        U[:] = 0.0
        U[free] = solve_spd(K[free][:, free], F[free])
        Ue = U[edofMat]
        ce = np.einsum("ij,jk,ik->i", Ue, KE, Ue)
        c = float(((Emin + xPhys**penal * (E0 - Emin)) * ce).sum())
        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones_like(x)
        if ft == 1:
            dc = H @ (x * dc) / Hs / np.maximum(1e-3, x)
        elif ft == 2:
            dc = H @ (dc / Hs)
            dv = H @ (dv / Hs)
        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            xPhys = xnew if ft == 1 else (H @ xnew) / Hs
            if xPhys.sum() > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = xnew
        hist.append((loop, c, float(xPhys.mean()), change))
        if snapshot_every and (loop == 1 or loop % snapshot_every == 0):
            snaps.append((loop, xPhys.copy()))
        if log:
            log(f" It.:{loop:5d} Obj.:{c:11.4f} Vol.:{xPhys.mean():7.3f} ch.:{change:7.3f}")
    snaps.append((loop, xPhys.copy()))
    to2d = lambda v: v.reshape(nelx, nely).T  # back to MATLAB's nely x nelx layout
    return {"x": to2d(xPhys), "hist": hist, "snaps": [(i, to2d(s)) for i, s in snaps],
            "c": hist[-1][1], "iters": loop, "seconds": time.time() - t0}


# ---------------------------------------------------------------- 3D (top3d)
def lk_h8(nu=0.3, E=1.0):
    """Trilinear 8-node brick stiffness for a unit cube by 2x2x2 Gauss quadrature
    (exact for this element). Node order matches top3d's edofMat:
    (-,-,-) (+,-,-) (+,+,-) (-,+,-) then the same four at +z; dofs (ux,uy,uz) per node."""
    xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6)); D[:3, :3] = lam; D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    g = 1 / np.sqrt(3)
    KE = np.zeros((24, 24))
    for a in (-g, g):
        for b in (-g, g):
            for cc in (-g, g):
                B = _bmat_h8(xi, a, b, cc)
                KE += B.T @ D @ B * 0.125  # detJ = (1/2)^3 for a unit cube
    return KE, D


def _bmat_h8(xi, a, b, c):
    dN = np.empty((8, 3))
    for n in range(8):
        sx, sy, sz = xi[n]
        dN[n] = [sx * (1 + sy * b) * (1 + sz * c), sy * (1 + sx * a) * (1 + sz * c), sz * (1 + sx * a) * (1 + sy * b)]
    dN *= 0.125 * 2.0  # d/dxi -> d/dx for unit cube (dx/dxi = 1/2)
    B = np.zeros((6, 24))
    for n in range(8):
        dx, dy, dz = dN[n]
        B[0, 3 * n] = dx; B[1, 3 * n + 1] = dy; B[2, 3 * n + 2] = dz
        B[3, 3 * n] = dy; B[3, 3 * n + 1] = dx
        B[4, 3 * n + 1] = dz; B[4, 3 * n + 2] = dy
        B[5, 3 * n] = dz; B[5, 3 * n + 2] = dx
    return B


def h8_center_B():
    xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    return _bmat_h8(xi, 0.0, 0.0, 0.0)


class Grid3D:
    """Structured hex grid in top3d numbering (nely rows from the top, then x, then z)."""

    def __init__(self, nelx, nely, nelz):
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        self.nele = nelx * nely * nelz
        self.nnode = (nelx + 1) * (nely + 1) * (nelz + 1)
        self.ndof = 3 * self.nnode
        nodegrd = np.arange(1, (nely + 1) * (nelx + 1) + 1).reshape(nelx + 1, nely + 1).T
        nodeids = nodegrd[:-1, :-1].T.reshape(-1)            # column-major
        nodeidz = np.arange(0, nelz) * (nely + 1) * (nelx + 1)
        nodeids = (nodeids[:, None] + nodeidz[None, :]).T.reshape(-1)  # MATLAB repmat + (:)
        edofVec = 3 * nodeids + 1
        off2 = np.array([0, 1, 2, 3 * nely + 3, 3 * nely + 4, 3 * nely + 5, 3 * nely + 0, 3 * nely + 1, 3 * nely + 2, -3, -2, -1])
        off = np.concatenate([off2, 3 * (nely + 1) * (nelx + 1) + off2])
        self.edofMat = edofVec[:, None] + off[None, :] - 1     # 0-based
        self.iK = np.kron(self.edofMat, np.ones((24, 1), dtype=np.int64)).ravel()
        self.jK = np.kron(self.edofMat, np.ones((1, 24), dtype=np.int64)).ravel()

    def node_id(self, i, j, k):
        """0-based node id at x-index i (0..nelx), row j (0 = top .. nely), layer k."""
        return k * (self.nelx + 1) * (self.nely + 1) + i * (self.nely + 1) + j

    def elem_index(self, i, j, k):
        """0-based element index for element column i, row j (0 = top), layer k."""
        return k * self.nelx * self.nely + i * self.nely + j

    def to_array(self, v):
        """Element vector -> array [nely, nelx, nelz] (MATLAB layout)."""
        return v.reshape(self.nelz, self.nelx, self.nely).transpose(2, 1, 0)

    def filter(self, rmin):
        nelx, nely, nelz = self.nelx, self.nely, self.nelz
        r = int(np.ceil(rmin)) - 1
        k1, i1, j1 = np.meshgrid(np.arange(nelz), np.arange(nelx), np.arange(nely), indexing="ij")
        k1, i1, j1 = k1.ravel(), i1.ravel(), j1.ravel()
        e1 = k1 * nelx * nely + i1 * nely + j1
        rows, cols, vals = [], [], []
        for dk in range(-r, r + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    k2, i2, j2 = k1 + dk, i1 + di, j1 + dj
                    ok = (k2 >= 0) & (k2 < nelz) & (i2 >= 0) & (i2 < nelx) & (j2 >= 0) & (j2 < nely)
                    w = np.maximum(0.0, rmin - np.sqrt(di * di + dj * dj + dk * dk))
                    if w <= 0:
                        continue
                    rows.append(e1[ok]); cols.append((k2 * nelx * nely + i2 * nely + j2)[ok])
                    vals.append(np.full(ok.sum(), w))
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.nele, self.nele))
        return H, np.asarray(H.sum(1)).ravel()


def simp3d(grid, F, fixed, volfrac, penal, rmin, maxloop=200, tolx=0.01, passive_solid=None,
           passive_void=None, nu=0.3, snapshot_every=0, log=None, beta_schedule=None):
    """3D SIMP with density filter and OC update (top3d algorithm).
    F: (ndof,) or (ndof, nload) load matrix; compliance is summed over load cases.
    passive_solid / passive_void: boolean element masks kept at 1 / 0.
    beta_schedule: optional Heaviside projection continuation, e.g. [(1,1),(60,2),(90,4),(120,8)]
    (iteration -> beta); None reproduces top3d exactly."""
    E0, Emin = 1.0, 1e-9
    KE, _ = lk_h8(nu)
    KEf = KE.ravel()
    F = F.reshape(grid.ndof, -1)
    free = np.setdiff1d(np.arange(grid.ndof), fixed)
    H, Hs = grid.filter(rmin)
    n = grid.nele
    ps = np.zeros(n, bool) if passive_solid is None else passive_solid
    pv = np.zeros(n, bool) if passive_void is None else passive_void
    active = ~(ps | pv)
    target = volfrac * n
    x = np.full(n, volfrac); x[ps] = 1.0; x[pv] = 0.0
    beta, eta = 1.0, 0.5

    def project(xt):
        if beta_schedule is None:
            return xt, np.ones_like(xt)
        th = np.tanh(beta * eta)
        xp = (th + np.tanh(beta * (xt - eta))) / (th + np.tanh(beta * (1 - eta)))
        dx = beta * (1 - np.tanh(beta * (xt - eta)) ** 2) / (th + np.tanh(beta * (1 - eta)))
        return xp, dx

    def phys(xd):
        xt = (H @ xd) / Hs
        xp, dx = project(xt)
        xp = xp.copy(); xp[ps] = 1.0; xp[pv] = 0.0
        return xp, dx

    xPhys, dproj = phys(x)
    U = np.zeros_like(F)
    hist, snaps = [], []
    loop, change = 0, 1.0
    t0 = time.time()
    while loop < maxloop and (change > tolx or (beta_schedule and beta < beta_schedule[-1][1])):
        loop += 1
        if beta_schedule:
            beta = max(b for it, b in beta_schedule if loop >= it)
            xPhys, dproj = phys(x)
        sK = (KEf[:, None] * (Emin + xPhys**penal * (E0 - Emin))[None, :]).T.ravel()
        K = sp.coo_matrix((sK, (grid.iK, grid.jK)), shape=(grid.ndof, grid.ndof)).tocsc()
        K = (K + K.T) / 2
        Kff = K[free][:, free]
        if _cholesky is not None:
            fac = _cholesky(Kff.tocsc())
            U[free] = fac(F[free])
        else:
            for l in range(F.shape[1]):
                U[free, l] = solve_spd(Kff, F[free, l])
        ce = np.zeros(n)
        for l in range(F.shape[1]):
            Ue = U[grid.edofMat, l]
            ce += np.einsum("ij,jk,ik->i", Ue, KE, Ue)
        c = float(((Emin + xPhys**penal * (E0 - Emin)) * ce).sum())
        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones(n)
        dc = H @ (dc * dproj / Hs)
        dv = H @ (dv * dproj / Hs)
        dc[~active] = 0.0; dv[~active] = 1.0
        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(np.maximum(-dc / dv / lmid, 0))))))
            xnew[ps] = 1.0; xnew[pv] = 0.0
            xPhys, _ = phys(xnew)
            if xPhys.sum() > target:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = xnew
        xPhys, dproj = phys(x)
        hist.append((loop, c, float(xPhys.mean()), change, beta))
        if snapshot_every and (loop == 1 or loop % snapshot_every == 0):
            snaps.append((loop, xPhys.astype(np.float32).copy()))
        if log:
            log(f" It.:{loop:5d} Obj.:{c:11.4f} Vol.:{xPhys.mean():7.3f} ch.:{change:7.3f} beta:{beta:g}")
    snaps.append((loop, xPhys.astype(np.float32).copy()))
    return {"xPhys": xPhys, "U": U, "hist": hist, "snaps": snaps, "c": hist[-1][1], "iters": loop,
            "seconds": time.time() - t0, "KE": KE}


def top3d(nelx, nely, nelz, volfrac, penal, rmin, snapshot_every=0, log=None):
    """Port of top3d.m with its default cantilever: left face clamped, unit downward
    line load along the lower-right edge (all z)."""
    g = Grid3D(nelx, nely, nelz)
    F = np.zeros(g.ndof)
    for k in range(nelz + 1):
        F[3 * g.node_id(nelx, nely, k) + 1] = -1.0           # y-dof, bottom row (j = nely)
    fixed_nodes = [g.node_id(0, j, k) for j in range(nely + 1) for k in range(nelz + 1)]
    fixed = np.sort(np.concatenate([3 * np.array(fixed_nodes) + d for d in range(3)]))
    r = simp3d(g, F, fixed, volfrac, penal, rmin, maxloop=200, tolx=0.01, snapshot_every=snapshot_every, log=log)
    r["grid"] = g
    r["x"] = g.to_array(r["xPhys"])
    return r


def von_mises(grid, U, xPhys, penal=3.0, nu=0.3, E=1.0, relax=True):
    """Element-centroid von Mises stress for each load case (max over load cases).
    With relax=True the SIMP-consistent stress x^0.5 * sigma_solid is used for display,
    which is the common convention to avoid singular stresses in void."""
    _, D = lk_h8(nu, E)
    B = h8_center_B()
    vm = np.zeros(grid.nele)
    U = U.reshape(grid.ndof, -1)
    for l in range(U.shape[1]):
        s = (D @ B @ U[grid.edofMat, l].T).T  # (nele, 6) solid-material stress
        v = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                    + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
        vm = np.maximum(vm, v)
    return vm * (np.sqrt(xPhys) if relax else 1.0)
