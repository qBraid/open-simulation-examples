"""DFT anchor: PBE/def2-TZVP geometries with PySCF (the DFT level MACE-MP learned from, here
in a molecular Gaussian basis) against MACE geometries and experiment for small molecules.

Experimental equilibrium geometries: NIST CCCBDB (r_e where available).
usage: dft_anchor.py <out.json>   (CPU; GPU4PySCF is a drop-in for larger systems)
"""
import sys, json, os
import numpy as np
from ase.build import molecule
from ase.optimize import BFGS
from mace.calculators import mace_mp
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

EXP = {"H2O": {"r(O-H)": 0.9578, "angle(H-O-H)": 104.48}, "NH3": {"r(N-H)": 1.012, "angle(H-N-H)": 106.7},
       "CH4": {"r(C-H)": 1.087}, "CO2": {"r(C-O)": 1.162}, "N2": {"r(N-N)": 1.098}, "CO": {"r(C-O)": 1.128}}


def measure(name, pos):
    if name in ("N2", "CO"):
        return {list(EXP[name])[0]: float(np.linalg.norm(pos[0] - pos[1]))}
    c = pos[0]; h = pos[1:]
    r = float(np.mean([np.linalg.norm(x - c) for x in h]))
    out = {list(EXP[name])[0]: r}
    if len(EXP[name]) > 1:
        v1, v2 = h[0] - c, h[1] - c
        out[list(EXP[name])[1]] = float(np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2))))
    return out


calcs = {m: mace_mp(model={"mp0": "medium", "mpa0": "medium-mpa-0"}[m], device="cpu", default_dtype="float64") for m in ("mp0", "mpa0")}
res = {}
for name in EXP:
    at = molecule(name)
    if name == "CH4" or name == "CO2":
        pass
    # PySCF PBE/def2-TZVP
    mol = gto.M(atom=[(s, p) for s, p in zip(at.get_chemical_symbols(), at.get_positions())], basis="def2-tzvp", verbose=0)
    mf = dft.RKS(mol); mf.xc = "pbe"
    mol_eq = optimize(mf, maxsteps=50)
    pbe = measure(name, mol_eq.atom_coords(unit="Angstrom"))
    row = {"exp": EXP[name], "pbe_def2tzvp": pbe}
    for m, calc in calcs.items():
        a = at.copy(); a.center(vacuum=8.0); a.calc = calc
        BFGS(a, logfile=None).run(fmax=1e-3, steps=300)
        row[m] = measure(name, a.get_positions())
    res[name] = row
    print(name, json.dumps(row), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
