"""Full-chip 3D eigenmode (Palace + EPR) vs the capacitance (LOM) design, and the final
T1 budget built from eigenmode quantities + the surface participation of the final island.

usage: python eigen/summarize.py <final_iter>        (writes results/v2/eigen/summary.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results", "v2")
sys.path.insert(0, os.path.join(HERE, "..", "loss"))
import design_v2 as D  # noqa: E402

K_CAL = json.load(open(f"{R}/validation_wang2015.json"))["k_cal"]
LEADS = {"MS": 0.17e-4, "SA": 0.20e-4, "MA": 0.02e-4}
GEOM = {"iter1": {"W": 240, "L": 190, "G": 150, "resonator_um": 4000},
        "iter2": {"W": 240, "L": 146, "G": 150, "resonator_um": 3550},
        "iter3": {"W": 240, "L": 146, "G": 150, "resonator_um": 3550}}


def eig(it):
    h = json.load(open(f"{R}/eigen/{it}/ham.json"))
    n, d, i = h["numerical"], h["derived"], h["inputs"]
    rows = [l.split(",") for l in open(f"{R}/eigen/{it}/postpro/domain-E.csv").read().strip().splitlines()]
    hdr = [c.strip() for c in rows[0]]
    p_sub = float(rows[1][hdr.index("p_elec[1]")])
    return {"geometry": GEOM[it], "L_J_nH": i["L_J_H"] * 1e9, "E_J_GHz": i["E_J_GHz"],
            "f_linear_GHz": i["mode_freqs_linear_GHz"], "junction_participation": i["junction_participation"],
            "f01_GHz": n["f01_GHz"], "alpha_MHz": n["anharmonicity_GHz"] * 1e3,
            "alpha_first_order_MHz": h["first_order"]["qubit_anharmonicity_GHz"] * 1e3,
            "fr_GHz": n["f_res_ground_GHz"], "pull_2chi_MHz": abs(n["chi_full_GHz"]) * 1e3,
            "chi_MHz": abs(n["chi_full_GHz"]) * 1e3 / 2, "kappa_MHz": d["resonator_kappa_MHz"],
            "chi_over_kappa": abs(n["chi_full_GHz"]) / 2 / (d["resonator_kappa_MHz"] * 1e-3),
            "EJ_over_EC": d["EJ_over_EC"], "Delta_GHz": d["detuning_GHz"],
            "T1_purcell_unfiltered_us": d["qubit_T1_purcell_bound_us"], "p_sub_qubit_mode": p_sub}


def t1_budget(e, p, QF=30.0):
    wq = 2 * np.pi * e["f01_GHz"] * 1e9
    g_p = 1e6 / e["T1_purcell_unfiltered_us"]
    g_pf = g_p * e["f01_GHz"] / (2 * QF * abs(e["Delta_GHz"]))
    out = {}
    for key, mdl in D.LOSS.items():
        for cal in ("raw", "calibrated"):
            kk = K_CAL if cal == "calibrated" else 1.0
            pp = {k: v * kk + (LEADS[k] if cal == "calibrated" else LEADS[k] / K_CAL) for k, v in p.items()}
            kind, tan = mdl["surface"]
            ps = pp["MS"] if kind == "MS" else pp["MS"] + pp["SA"] + pp["MA"]
            gs, gb = wq * ps * tan, wq * e["p_sub_qubit_mode"] * mdl["bulk"]
            out[f"{key}_{cal}"] = {"label": mdl["label"], "p_used": ps, "T1_surface_us": 1e6 / gs,
                                   "T1_bulk_us": (1e6 / gb) if gb else None,
                                   "T1_purcell_filtered_us": 1e6 / g_pf, "T1_total_us": 1e6 / (gs + gb + g_pf)}
    return out


def main(final_it):
    lom = json.load(open(f"{R}/design_final.json"))
    its = {it: eig(it) for it in GEOM if os.path.exists(f"{R}/eigen/{it}/ham.json")}
    fin = its[final_it]
    p = json.load(open(f"{R}/final_participation.json"))["p"]
    fin["T1_budget"] = t1_budget(fin, p)
    cmp = {"lom_W240_L190": {k: lom[k] for k in ("f01_GHz", "alpha_MHz", "fr_GHz", "pull_2chi_MHz", "chi_MHz",
                                                  "EJ_over_EC", "LJ_nH", "kappa_MHz", "chi_over_kappa")},
           "eigen_iter1_same_geometry": {k: its["iter1"][k] for k in ("f01_GHz", "alpha_MHz", "fr_GHz", "pull_2chi_MHz",
                                                                     "chi_MHz", "EJ_over_EC", "L_J_nH", "kappa_MHz", "chi_over_kappa")}}
    out = {"final_iteration": final_it, "iterations": its, "lom_vs_eigen": cmp, "final": fin,
           "notes": ["Eigenmode: Palace v0.18.1, DeviceLayout.jl SingleTransmon full chip (island, claw, lambda/4 meander, "
                     "feedline, 50-ohm lumped ports), anisotropic sapphire eps=(9.3, 9.3, 11.5), order 2, junction as lumped L_J || C_J=2 fF.",
                     "Hamiltonian by energy-participation quantization (numerical diagonalization, 12x14 levels).",
                     "chi here is half the dispersive pull (chi = |2chi|/2), the same convention as the LOM design.",
                     "T1 budget: surface participation from the electrostatic two-step model of the final island; "
                     "substrate fraction and Purcell (port Q_ext) from the eigenmode; Q=30 Purcell filter assumed."]}
    json.dump(out, open(f"{R}/eigen/summary.json", "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    o = main(sys.argv[1] if len(sys.argv) > 1 else "iter3")
    for k, v in o["lom_vs_eigen"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    f = o["final"]
    print("final", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items() if k not in ("T1_budget",)})
    for k, v in f["T1_budget"].items():
        print(k, round(v["T1_surface_us"]), v["T1_bulk_us"] and round(v["T1_bulk_us"]), round(v["T1_purcell_filtered_us"]), round(v["T1_total_us"]))
