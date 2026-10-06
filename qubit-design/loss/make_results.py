"""Compose results/v2/*.json (designs, sweep, validation, stamp) from the run directories."""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import design_v2 as D, run_post, analyze_sweep

RUN = os.environ.get("QD_RUNS", "/tmp/runs/qubit-design")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "v2")
K_CAL = 0.83 / 1.643          # Wang Table S1 Design A pads p_MS / ours (validation run)
LEADS = {"MS": 0.17e-4, "SA": 0.20e-4, "MA": 0.02e-4}  # Wang 2015 Design A leads (1-10 um + far), their pipeline

def summarize(d, f01=4.8, fr=7.0, CJ=2e-15):
    meta = json.load(open(f"{d}/meta.json"))
    C = run_post.load_C(f"{d}/post"); p = json.load(open(f"{d}/participation.json"))["p"]
    psub = D.read_domain_fraction(f"{d}/post/domain-E.csv")
    ham = D.design(C, p, psub, f01_target=f01, fr=fr, CJ=CJ, k_cal=K_CAL, leads_wang=LEADS)
    b = ham["T1_budget"]
    return meta, ham, {"p_sum": p["MS"] + p["SA"] + p["MA"], "T1_raw": b["ganjam2024_raw"]["T1_total_us"],
                       "T1_cal": b["ganjam2024_calibrated"]["T1_total_us"]}

def main(final_tag):
    os.makedirs(OUT, exist_ok=True)
    designs = []
    # v1 geometry, same Ta/sapphire stack and the same targets, for an apples-to-apples comparison
    meta, ham, s = summarize(f"{RUN}/v1")
    shutil.copy(f"{RUN}/v1/participation.json", f"{OUT}/v1_participation.json")
    designs.append({"name": "v1 (Palace example)", "kind": "v1", "kw": {"W": 24.0, "L": 620.0, "G": 30.0},
                    "participation": "v1_participation.json", "ham": ham, "summary": s})
    rows = analyze_sweep.main(RUN)
    json.dump(rows, open(f"{OUT}/sweep.json", "w"), indent=1)
    sweep_dirs = sorted([r for r in rows], key=lambda r: -r["p_sum"])
    for r in sweep_dirs:
        tag = f"sw_W{int(r['W'])}_G{int(r['G'])}"
        pj = json.load(open(f"{RUN}/{tag}/participation.json"))
        fn = f"sweep_{tag}_participation.json"
        json.dump(pj, open(f"{OUT}/{fn}", "w"))
        wq = 2 * np.pi * 4.8e9
        T1r = 1e6 / (wq * (r["p_sum"] * 3.4e-4 + r["p_sub"] * 2.6e-8))
        T1c = 1e6 / (wq * (r["p_sum"] * K_CAL * 3.4e-4 + r["p_sub"] * 2.6e-8))
        designs.append({"name": f"sweep W{int(r['W'])} G{int(r['G'])}", "kind": "sweep",
                        "kw": {"W": r["W"], "L": r["L_target"], "G": r["G"]}, "participation": fn,
                        "summary": {"p_sum": r["p_sum"], "T1_raw": T1r, "T1_cal": T1c}})
    if final_tag:
        meta, ham, s = summarize(f"{RUN}/{final_tag}")
        shutil.copy(f"{RUN}/{final_tag}/participation.json", f"{OUT}/final_participation.json")
        designs.append({"name": "final", "kind": "final", "kw": {k: meta["kw"][k] for k in ("W", "L", "G")},
                        "kw_full": meta["kw"], "participation": "final_participation.json", "ham": ham, "summary": s})
        json.dump(ham, open(f"{OUT}/design_final.json", "w"), indent=1, default=float)
    json.dump(designs[0]["ham"], open(f"{OUT}/design_v1_Ta.json", "w"), indent=1, default=float)
    json.dump(designs, open(f"{OUT}/designs.json", "w"), indent=1, default=float)
    print(len(designs), "designs")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
