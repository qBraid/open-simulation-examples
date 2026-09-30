"""From Palace eigenmode output to a transmon-resonator Hamiltonian.

Energy-participation-ratio (EPR) quantization, following Minev et al.,
npj Quantum Inf. 7, 131 (2021):

  * Palace solves the *linearized* circuit: the Josephson junction is a lumped
    inductor L_J (config: LumpedPort index 3, "L"). Each eigenmode m has a
    frequency f_m and a junction energy participation p_m (port-EPR.csv).
  * The junction's zero-point phase in mode m is  phi_m^2 = p_m h f_m / (2 E_J),
    with E_J = (Phi_0 / 2pi)^2 / L_J.
  * The full Hamiltonian is  H = sum_m h f_m a_m^+ a_m  - E_J [cos(phi) - 1 + phi^2/2],
    phi = sum_m phi_m (a_m + a_m^+).

Two estimates are reported:
  1. first order (analytic):  chi_mn = p_m p_n f_m f_n / (4 E_J/h),  alpha_m = chi_mm / 2
  2. numerical diagonalization of the truncated Hamiltonian, which also gives
     the Lamb-shifted qubit frequency f_01 and the dispersive shift from the
     dressed spectrum.

Usage:  python derive_hamiltonian.py <postpro_dir> <palace_config.json> [out.json]
"""

import csv
import json
import sys

import numpy as np

PHI0 = 2.067833848e-15  # Wb
H = 6.62607015e-34  # J s


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    return [dict(zip(head, (float(x) for x in r))) for r in rows[1:] if r]


def load(postpro, config):
    eig = read_csv(f"{postpro}/eig.csv")
    epr = read_csv(f"{postpro}/port-EPR.csv")
    portq = read_csv(f"{postpro}/port-Q.csv")
    cfg = json.load(open(config))
    jj = next(p for p in cfg["Boundaries"]["LumpedPort"] if "L" in p)
    f = np.array([r["Re{f} (GHz)"] for r in eig])
    q = np.array([r["Q"] for r in eig])
    pkey = next(k for k in epr[0] if k.startswith("p["))
    p = np.array([r[pkey] for r in epr])
    qext = np.array(
        [1.0 / sum(1.0 / v for k, v in r.items() if k.startswith("Q_ext")) for r in portq]
    )
    return f, q, p, qext, jj["L"], jj.get("C", 0.0)


def first_order(f, p, ej_ghz):
    chi = np.outer(p * f, p * f) / (4 * ej_ghz)  # GHz; full cross-Kerr
    alpha = np.diag(chi) / 2
    return chi, alpha


def ladder(n):
    return np.diag(np.sqrt(np.arange(1, n)), 1)


def numerical(f, p, ej_ghz, nq=12, nr=14, iq=0, ir=1):
    """Diagonalize H/h (GHz) for the qubit-like mode iq and resonator-like mode ir."""
    from scipy.linalg import cosm, eigh

    aq, ar = ladder(nq), ladder(nr)
    Iq, Ir = np.eye(nq), np.eye(nr)
    A = np.kron(aq, Ir)
    B = np.kron(Iq, ar)
    phi_q = np.sqrt(p[iq] * f[iq] / (2 * ej_ghz))
    phi_r = np.sqrt(p[ir] * f[ir] / (2 * ej_ghz))
    phi = phi_q * (A + A.T) + phi_r * (B + B.T)
    h0 = f[iq] * A.T @ A + f[ir] * B.T @ B
    hnl = -ej_ghz * (cosm(phi) - np.eye(nq * nr) + phi @ phi / 2)
    ham = h0 + hnl
    ev, vec = eigh(ham)

    def dressed(nqx, nrx):
        bare = np.zeros(nq * nr)
        bare[nqx * nr + nrx] = 1
        k = np.argmax(np.abs(vec.T @ bare) ** 2)
        return ev[k]

    e = {s: dressed(*s) for s in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]}
    f01 = e[(1, 0)] - e[(0, 0)]
    f12 = e[(2, 0)] - e[(1, 0)]
    fr_g = e[(0, 1)] - e[(0, 0)]
    fr_e = e[(1, 1)] - e[(1, 0)]
    levels = sorted(
        (float(e_ - ev[0]), f"|{a},{b}>")
        for (a, b), e_ in {(a, b): dressed(a, b) for a in range(3) for b in range(3)}.items()
    )
    return {
        "f01_GHz": f01,
        "anharmonicity_GHz": f12 - f01,
        "f_res_ground_GHz": fr_g,
        "f_res_excited_GHz": fr_e,
        "chi_full_GHz": fr_e - fr_g,  # resonator shift when the qubit is excited (= 2*chi in chi*sigma_z convention)
        "phi_zpf_qubit": phi_q,
        "phi_zpf_res": phi_r,
        "levels_GHz": levels,
        "truncation": {"qubit": nq, "resonator": nr},
    }


def main(postpro, config, out=None):
    f, q, p, qext, lj, cj = load(postpro, config)
    ej_j = (PHI0 / (2 * np.pi)) ** 2 / lj
    ej_ghz = ej_j / H / 1e9
    iq, ir = int(np.argmax(p)), int(np.argmin(p))  # qubit mode carries the junction energy
    chi, alpha = first_order(f, p, ej_ghz)
    num = numerical(f, p, ej_ghz, iq=iq, ir=ir)
    ec_ghz = -num["anharmonicity_GHz"]  # transmon: alpha ~ -E_C
    res = {
        "inputs": {
            "L_J_H": lj,
            "C_J_F": cj,
            "E_J_GHz": ej_ghz,
            "mode_freqs_linear_GHz": f.tolist(),
            "mode_Q_total": q.tolist(),
            "junction_participation": p.tolist(),
            "Q_ext_ports": qext.tolist(),
            "qubit_mode_index": iq,
            "resonator_mode_index": ir,
        },
        "first_order": {
            "qubit_anharmonicity_GHz": -alpha[iq],
            "resonator_self_kerr_GHz": -alpha[ir],
            "cross_kerr_chi_GHz": -chi[iq, ir],
            "qubit_f01_GHz_approx": f[iq] - alpha[iq] - chi[iq, ir] / 2,
        },
        "numerical": num,
        "derived": {
            "EJ_over_EC": ej_ghz / ec_ghz,
            "qubit_T1_bound_us": q[iq] / (2 * np.pi * f[iq] * 1e9) * 1e6,
            "qubit_T1_purcell_bound_us": qext[iq] / (2 * np.pi * f[iq] * 1e9) * 1e6,
            "resonator_kappa_MHz": f[ir] * 1e3 / qext[ir],
            "resonator_kappa_total_MHz": f[ir] * 1e3 / q[ir],
            "detuning_GHz": f[ir] - num["f01_GHz"],
        },
    }
    d = res["derived"]
    d["chi_over_kappa"] = abs(num["chi_full_GHz"]) * 1e3 / d["resonator_kappa_MHz"]
    txt = json.dumps(res, indent=2, default=float)
    if out:
        open(out, "w").write(txt + "\n")
    return res


if __name__ == "__main__":
    r = main(*sys.argv[1:4])
    n, d, fo = r["numerical"], r["derived"], r["first_order"]
    print(f"E_J/h = {r['inputs']['E_J_GHz']:.3f} GHz,  E_J/E_C = {d['EJ_over_EC']:.1f}")
    print(f"qubit f01 = {n['f01_GHz']:.4f} GHz, alpha = {n['anharmonicity_GHz']*1e3:.1f} MHz "
          f"(first order {fo['qubit_anharmonicity_GHz']*1e3:.1f} MHz)")
    print(f"resonator = {n['f_res_ground_GHz']:.4f} GHz, dispersive shift = {n['chi_full_GHz']*1e6:.1f} kHz "
          f"(first order {fo['cross_kerr_chi_GHz']*1e6:.1f} kHz)")
    print(f"kappa/2pi = {d['resonator_kappa_MHz']*1e3:.0f} kHz, chi/kappa = {d['chi_over_kappa']:.2f}, "
          f"T1 bound = {d['qubit_T1_bound_us']:.2f} us (Purcell only {d['qubit_T1_purcell_bound_us']:.0f} us)")
