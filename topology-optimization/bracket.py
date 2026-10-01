"""Industrial-flavour 3D design: a lightweight titanium mounting bracket.

Design space 96 x 48 x 36 mm (1.5 mm bricks, 64 x 32 x 24 elements):
  * two bolted mounting pads on the back face (clamped, kept solid)
  * a pin lug at the front (solid ring kept, pin hole kept empty)
  * three load cases on the pin hole, minimised together (sum of compliances):
      LC1 vertical 8.0 kN, LC2 axial pull 6.0 kN, LC3 45-degree 8.5 kN
Material Ti-6Al-4V (E = 113.8 GPa, nu = 0.342, yield 880 MPa), 12 % of the design volume.

Outputs results/bracket.npz (density snapshots, final field, stresses) and results/bracket.json;
the printable surface goes to results/bracket.stl.
"""
import json
import time
from pathlib import Path

import numpy as np

from topopt import Grid3D, simp3d, von_mises

NX, NY, NZ = 64, 32, 24
H_MM = 1.5
E_MPA, NU, YIELD = 113.8e3, 0.342, 880.0
VOLFRAC, PENAL, RMIN = 0.12, 3.0, 2.0
LOADS_N = {"LC1 vertical 8.0 kN": (0.0, -8000.0), "LC2 axial 6.0 kN": (6000.0, 0.0),
           "LC3 45 deg 8.5 kN": (8500 * np.cos(np.pi / 4), -8500 * np.cos(np.pi / 4))}
PADS = [(8.0, 12.0), (24.0, 12.0)]          # (row j, layer k) centres on the x = 0 face
LUG_C, LUG_R_OUT, LUG_R_IN = (56.0, 16.0), 7.0, 3.5   # (x i, row j), radii in elements


def build():
    g = Grid3D(NX, NY, NZ)
    k, i, j = np.meshgrid(np.arange(NZ), np.arange(NX), np.arange(NY), indexing="ij")
    k, i, j = k.ravel(), i.ravel(), j.ravel()          # element index order: k, i, j
    cx, cy, cz = i + 0.5, j + 0.5, k + 0.5
    r_lug = np.hypot(cx - LUG_C[0], cy - LUG_C[1])
    lug_z = (cz > 4) & (cz < NZ - 4)                   # lug is 16 elements (24 mm) thick
    passive_void = (r_lug < LUG_R_IN)
    passive_solid = (r_lug >= LUG_R_IN) & (r_lug < LUG_R_OUT - 1.5) & lug_z
    for pj, pk in PADS:                                 # 3 mm thick mounting pads
        passive_solid |= (cx < 2) & (np.hypot(cy - pj, cz - pk) < 6.0)
    # clamp: nodes on the x = 0 face inside the bolt circles
    fixed_nodes = [g.node_id(0, jj, kk) for jj in range(NY + 1) for kk in range(NZ + 1)
                   if min(np.hypot(jj - pj, kk - pk) for pj, pk in PADS) <= 5.0]
    fixed = np.sort(np.concatenate([3 * np.array(fixed_nodes) + d for d in range(3)]))
    # load: spread over the pin-hole nodes in the lower (bearing) half for each direction
    hole_nodes = []
    for ii in range(NX + 1):
        for jj in range(NY + 1):
            if abs(np.hypot(ii - LUG_C[0], jj - LUG_C[1]) - LUG_R_IN) < 0.75:
                hole_nodes += [(ii, jj, kk) for kk in range(5, NZ - 4)]
    F = np.zeros((g.ndof, len(LOADS_N)))
    for l, (fx, fy) in enumerate(LOADS_N.values()):
        # bearing side: nodes whose outward direction opposes the load
        sel = [(ii, jj, kk) for ii, jj, kk in hole_nodes
               if ((ii - LUG_C[0]) * fx + (LUG_C[1] - jj) * fy) > 0]
        for ii, jj, kk in sel:
            n = g.node_id(ii, jj, kk)
            F[3 * n, l] += fx / len(sel)
            F[3 * n + 1, l] += fy / len(sel)
    return g, F, fixed, passive_solid, passive_void


def main():
    t = time.time()
    g, F, fixed, ps, pv = build()
    # unit model: brick side 1, E = 1, forces in kN-scaled units; results rescaled below
    Fu = F / 1000.0
    res = simp3d(g, Fu, fixed, VOLFRAC, PENAL, RMIN, maxloop=160, tolx=0.005, nu=NU,
                 passive_solid=ps, passive_void=pv, snapshot_every=4,
                 beta_schedule=[(1, 1), (50, 2), (80, 4), (105, 8), (125, 16)],
                 log=lambda s: print(s, flush=True))
    x = res["xPhys"]
    # stresses for each load case on the final design (solid-material stress, MPa)
    vm_unit = von_mises(g, res["U"], x, nu=NU, relax=False)      # per 1 kN, unit brick
    vm_mpa = vm_unit * 1000.0 / H_MM**2                           # N / mm^2
    solid = x > 0.5
    # exclude the clamped pad layer and the loaded hole ring (contact/BC singularities)
    k, i, j = np.meshgrid(np.arange(NZ), np.arange(NX), np.arange(NY), indexing="ij")
    i, j = i.ravel(), j.ravel()
    near_bc = (i < 2) | (np.abs(np.hypot(i + 0.5 - LUG_C[0], j + 0.5 - LUG_C[1]) - LUG_R_IN) < 1.5)
    body = solid & ~near_bc
    mass_g = float(x.sum() * H_MM**3 * 4.43e-3)                  # Ti-6Al-4V 4.43 g/cm^3
    compl = []
    for l in range(F.shape[1]):
        u_mm = res["U"][:, l] * 1.0 / (E_MPA / 1000.0 * H_MM)     # unit model -> mm (F in kN)
        compl.append(float(F[:, l] @ (u_mm)))                     # N*mm
    summary = dict(
        design_space_mm=[NX * H_MM, NY * H_MM, NZ * H_MM], elements=g.nele, dofs=g.ndof, volfrac=VOLFRAC,
        iterations=res["iters"], seconds=round(time.time() - t, 1), mass_g=round(mass_g, 1),
        full_block_mass_g=round(g.nele * H_MM**3 * 4.43e-3, 1),
        grey_fraction=float(((x > 0.1) & (x < 0.9)).mean()),
        compliance_Nmm={k_: round(c, 2) for k_, c in zip(LOADS_N, compl)},
        vm_p99_MPa=round(float(np.percentile(vm_mpa[body], 99)), 1),
        vm_max_body_MPa=round(float(vm_mpa[body].max()), 1),
        vm_max_all_MPa=round(float(vm_mpa[solid].max()), 1),
        safety_factor_p99=round(YIELD / float(np.percentile(vm_mpa[body], 99)), 2),
        tip_disp_mm={k_: round(float(np.abs(res["U"][:, l]).max() / (E_MPA / 1000.0 * H_MM)), 4) for l, k_ in enumerate(LOADS_N)},
        loads_N={k_: list(v) for k_, v in LOADS_N.items()}, material="Ti-6Al-4V (E 113.8 GPa, nu 0.342, yield 880 MPa)",
    )
    print(json.dumps(summary, indent=1))
    Path("results").mkdir(exist_ok=True)
    np.savez_compressed("results/bracket.npz", x=x.astype(np.float32), vm=vm_mpa.astype(np.float32),
                        snap_iters=np.array([s[0] for s in res["snaps"]]),
                        snaps=np.stack([s[1] for s in res["snaps"]]).astype(np.float16),
                        hist=np.array(res["hist"]), passive_solid=ps, passive_void=pv,
                        F=F.astype(np.float32), fixed=fixed)
    Path("results/bracket.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
