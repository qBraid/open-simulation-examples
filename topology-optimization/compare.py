"""Compare the Python ports against the published reference codes (run in Octave).

Reads results/py_*.npz and reference/out + logs/ref_*.log, writes results/benchmarks.json.
Agreement metrics per case: final compliance (relative difference), iteration count,
and layout difference |x_py - x_ref| (max, mean, share of elements differing by > 0.1).
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RES = ROOT / "results"
LOGS = ROOT / "logs"
REF = ROOT / "reference" / "out"

CASES = [  # (name, log file, order of the run inside that log, reference design file)
    ("top88_mbb_60x20_ft1", "ref_mbb.log", 0, "ref_top88_mbb_60x20_ft1.txt"),
    ("top88_mbb_150x50_ft1", "ref_mbb.log", 1, "ref_top88_mbb_150x50_ft1.txt"),
    ("top88_mbb_300x100_ft1", "ref_mbb.log", 2, "ref_top88_mbb_300x100_ft1.txt"),
    ("top88_mbb_60x20_ft2", "ref_mbb.log", 3, "ref_top88_mbb_60x20_ft2.txt"),
    ("top88_mbb_150x50_ft2", "ref_mbb.log", 4, "ref_top88_mbb_150x50_ft2.txt"),
    ("top88_mbb_300x100_ft2", "ref_mbb.log", 5, "ref_top88_mbb_300x100_ft2.txt"),
    ("top88_cant_32x20_ft1", "ref_cant.log", 0, "ref_top88_cant_32x20_ft1.txt"),
    ("top88_cant_32x20_ft2", "ref_cant.log", 1, "ref_top88_cant_32x20_ft2.txt"),
    ("top3d_60x20x4", "ref_top3d.log", 0, "ref_top3d_60x20x4.txt"),
    ("top3d_60x20x4_200it", "ref_top3d.log", 0, "ref_top3d_60x20x4.txt"),
]
IT = re.compile(r"It\.:\s*(\d+)\s+Obj\.:\s*([-\d.eE+]+)\s+Vol\.:\s*([\d.]+)\s+ch\.:\s*([\d.]+)")


def ref_histories(logfile):
    runs, cur = [], None
    for line in (LOGS / logfile).read_text().splitlines():
        m = IT.search(line)
        if not m:
            continue
        it, c, v, ch = int(m[1]), float(m[2]), float(m[3]), float(m[4])
        if it == 1:
            cur = []; runs.append(cur)
        cur.append((it, c, v, ch))
    return runs


def main():
    py_runs = json.loads((RES / "py_runs.json").read_text())
    out = {}
    hist_cache = {}
    for name, log, order, ref_file in CASES:
        if name not in py_runs or not (LOGS / log).exists():
            print("skip", name); continue
        hist_cache.setdefault(log, ref_histories(log))
        if order >= len(hist_cache[log]):
            print("skip (reference run not started)", name); continue
        rh = np.array(hist_cache[log][order])
        if not (REF / ref_file).exists():
            # reference stopped by the compute cap before converging: compare the iterates both completed
            ph = np.load(RES / f"py_{name}.npz")["hist"][:, 1]
            m = min(len(ph), len(rh))
            rel = np.abs(ph[:m] - rh[:m, 1]) / rh[:m, 1]
            out[name] = dict(
                params={k: v for k, v in py_runs[name].items() if k not in ("compliance", "iterations", "seconds")},
                status="partial: reference stopped by the compute cap",
                compliance_python=py_runs[name]["compliance"], iterations_python=py_runs[name]["iterations"],
                compared_iterations=int(m), compliance_reference_at_last=float(rh[m - 1, 1]),
                compliance_python_at_same_iter=float(ph[m - 1]), max_iterwise_rel_diff=float(rel.max()),
                rel_diff_pct=float(100 * (ph[m - 1] - rh[m - 1, 1]) / rh[m - 1, 1]),
                iterations_reference=int(rh[-1, 0]), layout_max_abs_diff=None,
                layout_mean_abs_diff=None, layout_share_diff_gt_0p1=None,
                seconds_python=py_runs[name]["seconds"], reference_history=rh[:, 1].round(6).tolist(),
                iterwise_rel_diff=rel.tolist(), identical_until_iter=int(np.argmax(rel > 1e-6)) if (rel > 1e-6).any() else int(m))
            print(f"{name:24s} PARTIAL: {m} common iterations, max iterwise rel diff {rel.max():.2e}")
            continue
        d = np.load(RES / f"py_{name}.npz")
        x_py = d["x"]
        x_ref = np.loadtxt(REF / ref_file, delimiter=",")
        if name.startswith("top3d"):
            x_ref = x_ref.reshape(x_py.shape, order="F")
        dx = np.abs(x_py - x_ref)
        c_py, c_ref = py_runs[name]["compliance"], float(rh[-1, 1])
        out[name] = dict(
            status="complete",
            params={k: v for k, v in py_runs[name].items() if k not in ("compliance", "iterations", "seconds")},
            compliance_python=c_py, compliance_reference=c_ref,
            rel_diff_pct=100 * (c_py - c_ref) / c_ref,
            iterations_python=py_runs[name]["iterations"], iterations_reference=int(rh[-1, 0]),
            layout_max_abs_diff=float(dx.max()), layout_mean_abs_diff=float(dx.mean()),
            layout_share_diff_gt_0p1=float((dx > 0.1).mean()),
            seconds_python=py_runs[name]["seconds"],
            reference_history=rh[:, 1].round(6).tolist(),
        )
        ph = np.load(RES / f"py_{name}.npz")["hist"][:, 1]
        m = min(len(ph), len(rh))
        rel = np.abs(ph[:m] - rh[:m, 1]) / rh[:m, 1]
        out[name]["iterwise_rel_diff"] = rel.tolist()
        out[name]["identical_until_iter"] = int(np.argmax(rel > 1e-6)) if (rel > 1e-6).any() else m
        print(f"{name:24s} c_py={c_py:11.4f} c_ref={c_ref:11.4f} diff={out[name]['rel_diff_pct']:+.2e}% "
              f"it={out[name]['iterations_python']}/{out[name]['iterations_reference']} max|dx|={dx.max():.2e}")
    (RES / "benchmarks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
