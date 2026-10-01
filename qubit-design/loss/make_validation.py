"""Validation summary for the surface-loss pipeline (writes results/v2/validation_wang2015.json)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "v2")

def main(extra_rows=None):
    A = json.load(open(f"{OUT}/wangA_participation_x4.json"))
    A8 = json.load(open(f"{OUT}/wangA_participation_x8.json"))
    st = {float(k): v for k, v in json.load(open(f"{OUT}/edge2d_sapphire_h100_x4.json")).items()}
    pads_ms = A["p"]["MS"]
    pads_ms_x8 = A8["p"]["MS"]
    wang = {"pads_MS": 0.83e-4, "total_MS": 0.99e-4, "leads_MS": 0.17e-4, "T1_meas": [75, 66, 95]}
    w = 2 * np.pi * 6e9; g0 = 3e3  # Wang: qubits at ~6 GHz, residual 3 +- 1 ms^-1
    T1_pred_raw = 1e6 / (w * (pads_ms + wang["leads_MS"]) * 2.6e-3 + g0)
    rows = [
        {"label": "2D edge model: 4x finer corner mesh", "ours": "0.03 % change", "ref": "< 1 %", "verdict": "match"},
        {"label": "2D edge strength K vs analytic thin slot (g=500 um)", "ours": f"{st[500.0]['K']/st[500.0]['K_analytic_zero_thickness']:.3f}", "ref": "1 (h->0)", "verdict": "match"},
        {"label": "3D field follows K/sqrt(u), 1.4-10 um from edges", "ours": "flat to +-2 %", "ref": "theory: flat", "verdict": "match"},
        {"label": "Edge band x0 = 4 vs 8 um (same 3D mesh)", "ours": f"{(pads_ms_x8/pads_ms-1)*100:+.0f} %", "ref": "0 % ideal", "verdict": "close"},
        {"label": "Wang 2015 Design A, pad p_MS (Table S1)", "ours": f"{pads_ms*1e4:.2f}e-4", "ref": f"{wang['pads_MS']*1e4:.2f}e-4", "verdict": f"{pads_ms/wang['pads_MS']:.1f}x high"},
        {"label": "Design A T1 from our p (Wang loss model)", "ours": f"{T1_pred_raw:.0f} us", "ref": "66-95 us measured", "verdict": "conservative"},
        {"label": "v1 Palace eigenmode vs Palace reference (8 qty)", "ours": "8/8, f to 9e-10", "ref": "regression data", "verdict": "match"},
        {"label": "v1 E_C: electrostatic LOM vs eigenmode EPR", "ours": "226 MHz", "ref": "213 MHz", "verdict": "6 % high"},
    ] + (extra_rows or [])
    note = ("Wang et al., APL 107, 162601 (2015), Table S1: Design A (two 250x500 um Al pads on sapphire, 3D cavity), "
            "measured T1 = 75+-6, 66+-7, 95+-8 us; their fitted surface loss tan d_MS + 1.2 tan d_SA + 0.1 tan d_MA = 2.6e-3 "
            "(t = 3 nm, eps = 10) and a residual decay of 3 ms^-1. Our participation for the same pads is "
            f"{pads_ms/wang['pads_MS']:.1f}x theirs, so our raw T1 predictions are conservative; the 'calibrated' "
            f"numbers rescale our participations by k = {wang['pads_MS']/pads_ms:.3f} to Wang's pipeline. Their pad-to-pad gap "
            "and film thickness are not stated exactly (we used a 65 um gap with 25 um stubs and a 100 nm film).")
    out = {"rows": rows, "note": note, "k_cal": wang["pads_MS"] / pads_ms, "T1_pred_raw_us": T1_pred_raw,
           "pads_MS_ours": pads_ms, "pads_MS_wang": wang["pads_MS"]}
    json.dump(out, open(f"{OUT}/validation_wang2015.json", "w"), indent=1)
    return out

if __name__ == "__main__":
    extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(main(extra), indent=1)[:1500])
