"""Silicon phonon dispersion with a MACE model via phonopy, compared to neutron data.

Reference: inelastic neutron scattering, G. Nilsson and G. Nelin, Phys. Rev. B 6, 3777 (1972)
(high-symmetry frequencies in THz). Model lattice constant from its own EOS minimum.
Also writes Gamma and X eigenvectors for the viewer's phonon animation.
usage: phonons_si.py <mp0|mpa0> <out.json>
"""
import os, sys, json
import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from mace.calculators import mace_mp

model, out = sys.argv[1], sys.argv[2]
calc = mace_mp(model={"mp0": "medium", "mpa0": "medium-mpa-0", "mp0b3": "medium-0b3"}[model], device=os.environ.get("DEVICE", "cuda"), default_dtype="float64")
si = bulk("Si", "diamond", a=5.43); si.calc = calc
BFGS(FrechetCellFilter(si), logfile=None).run(fmax=1e-4)
a = float(np.linalg.norm(si.cell[0]) * np.sqrt(2))
ph = Phonopy(PhonopyAtoms(symbols=si.get_chemical_symbols(), cell=si.cell.array, scaled_positions=si.get_scaled_positions()),
             supercell_matrix=np.diag([4, 4, 4]),
             primitive_matrix=None)
ph.generate_displacements(distance=0.01)
forces = []
for sc in ph.supercells_with_displacements:
    from ase import Atoms
    at = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True); at.calc = calc
    forces.append(at.get_forces())
ph.forces = forces
ph.produce_force_constants()
# reciprocal coordinates of the primitive fcc cell
pts = {"G": [0, 0, 0], "X": [0.5, 0, 0.5], "L": [0.5, 0.5, 0.5], "W": [0.5, 0.25, 0.75], "K": [0.375, 0.375, 0.75]}
freqs = {k: sorted(ph.get_frequencies(v).tolist()) for k, v in pts.items()}
path = [["G", "X"], ["X", "W"], ["W", "K"], ["K", "G"], ["G", "L"]]
bands = []
for p, q in path:
    seg = [np.array(pts[p]) + (np.array(pts[q]) - np.array(pts[p])) * t for t in np.linspace(0, 1, 41)]
    bands.append(dict(path=f"{p}-{q}", freqs=[ph.get_frequencies(k).tolist() for k in seg]))
# Animated modes: phonopy's own modulations on a 3x3x3 conventional supercell (216 atoms),
# so the inter-atomic phases are exactly phonopy's convention.
dim = (np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * 3).tolist()
modes = {}
for label, q, band in [("G-LTO", pts["G"], 5), ("X-TA", pts["X"], 0), ("X-LA", pts["X"], 2), ("X-TO", pts["X"], 5),
                       ("L-TA", pts["L"], 0), ("L-TO", pts["L"], 5)]:
    ph.run_modulations(dim, [[q, band, 1.0, 0.0]])
    mods, sc = ph.get_modulations_and_supercell()
    m = np.array(mods[0])
    modes[label] = dict(freq=float(ph.get_frequencies(q)[band]), re=np.real(m).round(5).tolist(), im=np.imag(m).round(5).tolist())
modes["supercell_positions"] = np.array(sc.positions).round(4).tolist()
modes["supercell_cell"] = np.array(sc.cell).round(4).tolist()
neutron = {"G_LTO": 15.53, "X_TA": 4.49, "X_LA": 12.32, "X_TO": 13.90, "L_TA": 3.43, "L_LA": 11.35, "L_TO": 14.68}
model_hs = {"G_LTO": freqs["G"][-1], "X_TA": freqs["X"][0], "X_LA": freqs["X"][2], "X_TO": freqs["X"][-1],
            "L_TA": freqs["L"][0], "L_LA": freqs["L"][2], "L_TO": freqs["L"][-1]}
err = {k: (model_hs[k] - neutron[k]) / neutron[k] * 100 for k in neutron}
res = dict(model=model, a0=a, high_symmetry_model=model_hs, neutron=neutron, rel_err_pct=err,
           mare_pct=float(np.mean(np.abs(list(err.values())))), bands=bands, modes=modes,
           cell=si.cell.array.tolist(), positions=si.get_positions().tolist())
json.dump(res, open(out, "w"))
print(json.dumps({k: v for k, v in res.items() if k in ("a0", "high_symmetry_model", "rel_err_pct", "mare_pct")}, indent=1))
