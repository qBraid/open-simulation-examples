"""Rank the geometry sweep: rescale each (W, G) island to the target C_sigma and compare
surface participation and predicted T1 (Ta on sapphire, Ganjam 2024 loss model)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import design_v2 as D, run_post

def load(d):
    meta = json.load(open(f"{d}/meta.json"))
    C = run_post.load_C(f"{d}/post")
    p = json.load(open(f"{d}/participation.json"))["p"]
    psub = D.read_domain_fraction(f"{d}/post/domain-E.csv")
    return meta["kw"], C, p, psub

def main(root, CS_target=67e-15, CJ=2e-15, footprint_max=600.0):
    rows = []
    for d in sorted(glob.glob(f"{root}/sw_*")):
        if not os.path.exists(f"{d}/participation.json"):
            continue
        kw, C, p, psub = load(d)
        W, G, L = kw["W"], kw["G"], kw["L"]
        CS_full = 2 * C[0, 0]
        L_t = L * (CS_target - CJ) / CS_full  # p is ~independent of L for a long island
        psum = p["MS"] + p["SA"] + p["MA"]
        wq = 2 * np.pi * 4.8e9
        T1_ganjam = 1e6 / (wq * (psum * 3.4e-4 + psub * 2.6e-8))
        rows.append(dict(W=W, G=G, L_at_sweep=L, C_sigma_fF_at_L=CS_full * 1e15, L_target=L_t,
                         footprint_x=W + 2 * G, footprint_y=L_t + G + 12, p_MS=p["MS"], p_SA=p["SA"],
                         p_MA=p["MA"], p_sum=psum, p_sub=psub, T1_surface_bulk_ganjam_raw_us=T1_ganjam,
                         fits=bool(max(W + 2 * G, L_t + G + 12) <= footprint_max)))
    rows.sort(key=lambda r: r["p_sum"])
    return rows

if __name__ == "__main__":
    rows = main(sys.argv[1])
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
    for r in rows:
        print(f"W={r['W']:5.0f} G={r['G']:5.0f} L*={r['L_target']:6.0f}  p_sum={r['p_sum']:.3e}  T1(Ta,raw)={r['T1_surface_bulk_ganjam_raw_us']:6.0f} us  fits={r['fits']}")
