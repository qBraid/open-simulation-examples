"""Simulate the transmon-resonator Hamiltonian derived from the Palace run.

Inputs: the JSON written by derive_hamiltonian.py.
Outputs (JSON): truncation convergence, a Rabi trace with leakage to |2>,
a Ramsey trace, and the dispersive readout lineshapes.

Pure numpy/scipy (Lindblad master equation by direct superoperator
exponentiation), so it runs anywhere the Palace environment does.

Usage:  python simulate_dynamics.py hamiltonian.json out.json
"""

import json
import sys

import numpy as np
from scipy.linalg import expm

from derive_hamiltonian import numerical


def lindblad_propagator(h, c_ops, dt):
    """Superoperator for rho' = -i[H,rho] + sum D[c]rho, column-stacked vec(rho)."""
    d = h.shape[0]
    eye = np.eye(d)
    L = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for c in c_ops:
        cdc = c.conj().T @ c
        L += np.kron(c.conj(), c) - 0.5 * (np.kron(eye, cdc) + np.kron(cdc.T, eye))
    return expm(L * dt)


def evolve(h, c_ops, rho0, t_max, n):
    dt = t_max / (n - 1)
    prop = lindblad_propagator(h, c_ops, dt)
    v = rho0.reshape(-1, order="F")
    out = []
    for _ in range(n):
        out.append(v.reshape(rho0.shape, order="F"))
        v = prop @ v
    return np.linspace(0, t_max, n), out


def main(ham_json, out_json):
    H = json.load(open(ham_json))
    inp, num, der = H["inputs"], H["numerical"], H["derived"]
    f, p, ej = np.array(inp["mode_freqs_linear_GHz"]), np.array(inp["junction_participation"]), inp["E_J_GHz"]
    iq, ir = inp["qubit_mode_index"], inp["resonator_mode_index"]

    # 1. truncation convergence of the numerical diagonalization
    conv = []
    for nq, nr in [(4, 4), (6, 6), (8, 10), (12, 14)]:
        r = numerical(f, p, ej, nq=nq, nr=nr, iq=iq, ir=ir)
        conv.append({"nq": nq, "nr": nr, "f01_GHz": r["f01_GHz"],
                     "alpha_MHz": r["anharmonicity_GHz"] * 1e3, "chi_kHz": r["chi_full_GHz"] * 1e6})

    alpha = num["anharmonicity_GHz"]  # GHz, negative
    t1_ns = der["qubit_T1_bound_us"] * 1e3
    two_pi = 2 * np.pi

    # 2. Rabi: 3-level Duffing transmon, rotating frame at f01, resonant drive (GHz * ns = cycles)
    b = np.diag(np.sqrt([1.0, 2.0]), 1)
    rabi_mhz = 25.0
    h = two_pi * (alpha * np.diag([0, 0, 1.0]) + 0.5 * rabi_mhz * 1e-3 * (b + b.T))
    c_ops = [np.sqrt(1 / t1_ns) * b]
    rho0 = np.diag([1.0 + 0j, 0, 0])
    t, rhos = evolve(h, c_ops, rho0, 200.0, 401)
    rabi = {"t_ns": t.tolist(), "P1": [float(r[1, 1].real) for r in rhos],
            "P2": [float(r[2, 2].real) for r in rhos], "rabi_MHz": rabi_mhz,
            "max_leakage_P2": float(max(r[2, 2].real for r in rhos))}

    # 3. Ramsey: after a perfect pi/2, free evolution with detuning; T2 = 2*T1 (no pure dephasing modelled)
    det_mhz = 5.0
    h2 = two_pi * det_mhz * 1e-3 * np.diag([0, 1.0])
    sm = np.array([[0, 1.0], [0, 0]])
    plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    tr, rr = evolve(h2, [np.sqrt(1 / t1_ns) * sm], plus, 2000.0, 801)
    # second pi/2 then measure |1>: P1 = 1/2 (1 + Re rho01 * 2) in this frame
    ramsey = {"t_ns": tr.tolist(), "P1": [float(0.5 + r[0, 1].real) for r in rr],
              "detuning_MHz": det_mhz, "T2_assumed_us": 2 * der["qubit_T1_bound_us"]}

    # 4. dispersive readout: hanger |S21| for qubit in |g> and |e>
    fr_g, fr_e = num["f_res_ground_GHz"], num["f_res_excited_GHz"]
    qt = inp["mode_Q_total"][ir]
    qc = inp["Q_ext_ports"][ir]
    span = 4 * max(abs(fr_e - fr_g), fr_g / qt)
    fx = np.linspace(min(fr_g, fr_e) - span, max(fr_g, fr_e) + span, 400)

    def s21(fr):
        return np.abs(1 - (qt / qc) / (1 + 2j * qt * (fx - fr) / fr))

    readout = {"f_GHz": fx.tolist(), "S21_g": s21(fr_g).tolist(), "S21_e": s21(fr_e).tolist(),
               "separation_kHz": (fr_e - fr_g) * 1e6, "linewidth_kHz": fr_g / qt * 1e6}

    res = {"convergence": conv, "rabi": rabi, "ramsey": ramsey, "readout": readout}
    json.dump(res, open(out_json, "w"))
    c = conv[-1]
    print(f"converged: f01 {c['f01_GHz']:.5f} GHz, alpha {c['alpha_MHz']:.2f} MHz, chi {c['chi_kHz']:.1f} kHz")
    print(f"Rabi {rabi_mhz} MHz: max leakage to |2> = {rabi['max_leakage_P2']:.4f}")
    print(f"readout: line separation {readout['separation_kHz']:.0f} kHz vs linewidth {readout['linewidth_kHz']:.0f} kHz")


if __name__ == "__main__":
    main(*sys.argv[1:3])
