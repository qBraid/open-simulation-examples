"""From capacitance matrix + surface participation to Hamiltonian, readout and T1 budget.

Lumped-oscillator (LOM) route: E_C from the island's total capacitance, E_J chosen to hit
the target f01, coupling g from the claw capacitance, a lambda/4 readout resonator, and
a dispersive shift from exact diagonalisation of transmon (charge basis) x resonator.

Loss models (all participations in the t = 3 nm, eps = 10 convention):
  * wang2015  - Al on sapphire, lift-off (Wang et al., APL 107, 162601 (2015)):
                tan d_MS + 1.2 tan d_SA + 0.1 tan d_MA = 2.6e-3, i.e. 1/Q = p_MS * 2.6e-3.
  * ganjam2024 - Ta on annealed sapphire (Ganjam et al., Nat. Commun. 15, 3687 (2024),
                arXiv:2308.15539): surface loss factor 3.4e-4 on p_MS + p_SA + p_MA,
                bulk sapphire 2.6e-8 on the substrate energy fraction.
Participation pipelines differ by O(1) factors between groups; `k_cal` rescales ours to
Wang's Design A so that loss tangents fitted with Wang's pipeline can be reused.
"""
import json

import numpy as np

h = 6.62607015e-34
e = 1.602176634e-19
hbar = h / (2 * np.pi)
Phi0 = h / (2 * e)
RK = h / e**2

LOSS = {
    "wang2015": {"label": "Al on sapphire, lift-off (Wang 2015)", "surface": ("MS", 2.6e-3), "bulk": 0.0},
    "ganjam2024": {"label": "Ta on annealed sapphire (Ganjam 2024)", "surface": ("ALL", 3.4e-4), "bulk": 2.6e-8},
}


def transmon_levels(EC, EJ, ng=0.0, ncut=20, nlev=6):
    n = np.arange(-ncut, ncut + 1)
    H = np.diag(4 * EC * (n - ng) ** 2) - 0.5 * EJ * (np.eye(len(n), k=1) + np.eye(len(n), k=-1))
    w, v = np.linalg.eigh(H)
    return w[:nlev] - w[0], v[:, :nlev], n


def solve_EJ(EC, f01_target):
    lo, hi = 1.0, 200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = transmon_levels(EC, mid)[0][1]
        lo, hi = (mid, hi) if f < f01_target else (lo, mid)
    return 0.5 * (lo + hi)


def dispersive(EC, EJ, fr, g_coef, nq=6, nr=10):
    """Exact chi: H = H_tr + fr a+a + g_coef * n (a + a+), all in GHz."""
    E, V, n = transmon_levels(EC, EJ, nlev=nq)
    nmat = V.T @ np.diag(n) @ V
    a = np.diag(np.sqrt(np.arange(1, nr)), 1)
    Iq, Ir = np.eye(nq), np.eye(nr)
    H = np.kron(np.diag(E), Ir) + fr * np.kron(Iq, a.T @ a) + g_coef * np.kron(nmat, a + a.T)
    w, vecs = np.linalg.eigh(H)
    # identify dressed states by maximum overlap with bare |q, r>
    def dressed(q, r):
        idx = q * nr + r
        return w[np.argmax(np.abs(vecs[idx, :]))]
    E00, E01, E10, E11 = dressed(0, 0), dressed(0, 1), dressed(1, 0), dressed(1, 1)
    fr_g, fr_e = E01 - E00, E11 - E10
    g01 = g_coef * abs(nmat[0, 1])
    return {"f01_dressed": E10 - E00, "fr_dressed_g": fr_g, "pull_2chi": fr_e - fr_g,
            "chi": 0.5 * (fr_e - fr_g), "g01": g01, "n01": abs(nmat[0, 1])}


def design(C_half, p, p_sub, f01_target=4.8, fr=7.0, Zr=50.0, Z0=50.0, CJ=2.0e-15,
           kappa_rule="optimal", QF=30.0, k_cal=1.0, leads_wang=None, half=True):
    m = 2.0 if half else 1.0
    C = np.asarray(C_half) * m
    CS = C[0, 0] + CJ
    Cg = -C[0, 1]
    EC = e**2 / (2 * CS) / h / 1e9  # GHz
    EJ = solve_EJ(EC, f01_target)
    E, _, _ = transmon_levels(EC, EJ)
    f01, alpha = E[1], E[2] - 2 * E[1]
    LJ = (Phi0 / (2 * np.pi)) ** 2 / (EJ * 1e9 * h)
    Ic = 2 * np.pi * EJ * 1e9 * h / Phi0
    Rn = (np.pi * 180e-6 / 2) / Ic  # Ambegaokar-Baratoff, Al gap 180 ueV
    wr = 2 * np.pi * fr * 1e9
    Cr = np.pi / (4 * wr * Zr)
    beta = Cg / CS
    Vrms = np.sqrt(hbar * wr / (2 * Cr))
    g_coef = 2 * e * beta * Vrms / h / 1e9  # GHz per unit charge-number matrix element
    disp = dispersive(EC, EJ, fr, g_coef)
    chi = abs(disp["chi"])
    kappa = 2 * chi  # GHz; kappa = 2 chi maximises readout SNR (chi/kappa = 0.5)
    Qext = fr / kappa
    Ck = np.sqrt(np.pi / (2 * Zr * Z0 * wr**2 * Qext))
    g = disp["g01"]
    Delta = disp["f01_dressed"] - fr
    gamma_purcell = 2 * np.pi * kappa * 1e9 * (g / Delta) ** 2
    gamma_purcell_f = gamma_purcell * (f01 / fr) * (fr / (2 * QF * abs(Delta)))
    wq = 2 * np.pi * f01 * 1e9
    budgets = {}
    for key, mdl in LOSS.items():
        for cal in ("raw", "calibrated"):
            kk = k_cal if cal == "calibrated" else 1.0
            pp = {k: v * kk for k, v in p.items()}
            if leads_wang:
                for k, v in leads_wang.items():
                    pp[k] = pp.get(k, 0) + (v if cal == "calibrated" else v / k_cal)
            kind, tan = mdl["surface"]
            ps = pp["MS"] if kind == "MS" else pp["MS"] + pp["SA"] + pp["MA"]
            g_surf = wq * ps * tan
            g_bulk = wq * p_sub * mdl["bulk"]
            tot = g_surf + g_bulk + gamma_purcell_f
            budgets[f"{key}_{cal}"] = {
                "label": mdl["label"], "p_used": ps, "T1_surface_us": 1e6 / g_surf,
                "T1_bulk_us": (1e6 / g_bulk) if g_bulk > 0 else None,
                "T1_purcell_filtered_us": 1e6 / gamma_purcell_f,
                "T1_total_us": 1e6 / tot}
    return {
        "C_sigma_fF": CS * 1e15, "C_g_fF": Cg * 1e15, "C_J_fF": CJ * 1e15,
        "EC_GHz": EC, "EJ_GHz": EJ, "EJ_over_EC": EJ / EC, "f01_GHz": f01, "alpha_MHz": alpha * 1e3,
        "levels_GHz": [float(x) for x in E[:4]],
        "LJ_nH": LJ * 1e9, "Ic_nA": Ic * 1e9, "Rn_kOhm_AB": Rn / 1e3,
        "fr_GHz": fr, "Zr_Ohm": Zr, "g_MHz": g * 1e3, "Delta_GHz": Delta,
        "chi_MHz": chi * 1e3, "pull_2chi_MHz": abs(disp["pull_2chi"]) * 1e3,
        "kappa_MHz": kappa * 1e3, "chi_over_kappa": chi / kappa, "Q_ext": Qext, "C_kappa_fF": Ck * 1e15,
        "T1_purcell_unfiltered_us": 1e6 / gamma_purcell, "T1_purcell_filtered_us": 1e6 / gamma_purcell_f,
        "Q_filter": QF, "p": p, "p_sub": p_sub, "k_cal": k_cal, "T1_budget": budgets}


def read_domain_fraction(path):
    rows = [l.split(",") for l in open(path).read().strip().splitlines()]
    hdr = [c.strip() for c in rows[0]]
    r = [float(x) for x in rows[1]]  # excitation 1 = island
    return r[hdr.index("p_elec[1]")]


if __name__ == "__main__":
    import sys
    print(json.dumps(design(**json.loads(sys.argv[1])), indent=1))
