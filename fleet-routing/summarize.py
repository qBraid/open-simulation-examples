"""Summarise the CVRPLIB X benchmark against the published bar.

The bar ("top-10% for this field"): the published state of the art on the X set is
HGS-CVRP at a 0.11% mean gap to BKS and PyVRP at 0.22% (Tmax = 2.4 n s, mean of
10 seeds; Wouda, Lan & Kool 2024). Academic CVRP heuristics over the last decade
mostly sit between ~0.1% and ~1% on this set; generic routing libraries are
typically several percent away. We call the bar REACHED when the open-source race
is within the published PyVRP gap at the published budget, CLOSE when it is within
~2x of it, NOT REACHED otherwise.

    python summarize.py      # reads results/xbench*, writes results/xbench_summary.json
"""

from __future__ import annotations

import glob
import json
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    xb = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, "xbench", "*.json")))]
    cu = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(R, "xbench_cuopt", "*.json"))}
    full = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, "xbench_full", "*.json")))]
    f30 = {json.load(open(p))["instance"]: json.load(open(p)) for p in glob.glob(os.path.join(R, "xbench_full30", "*.json"))}
    n = len(xb)
    race_cpu = mean([b["gap_pct"] for b in xb])
    def best_gap(b):  # the full race: CPU racers plus the GPU racer
        g = [b["gap_pct"]]
        c = cu.get(b["instance"])
        if c and c.get("gap_pct") is not None:
            g.append(c["gap_pct"])
        return min(x for x in g if x is not None)
    race = mean([best_gap(b) for b in xb])
    cu_wins = sum(1 for b in xb if b["instance"] in cu and cu[b["instance"]].get("gap_pct") is not None
                  and cu[b["instance"]]["gap_pct"] < b["gap_pct"])
    groups = {}
    for lab, lo, hi in (("100-299", 0, 300), ("300-599", 300, 600), ("600-1000", 600, 10**6)):
        sel = [b for b in xb if lo <= b["n"] < hi]
        groups[lab] = {"instances": len(sel), "race": mean([best_gap(b) for b in sel]),
                       "pyvrp": mean([b["solvers"]["pyvrp"]["gap_pct"] for b in sel]),
                       "cuopt": mean([cu[b["instance"]]["gap_pct"] for b in sel if b["instance"] in cu])}
    pyvrp = mean([b["solvers"].get("pyvrp", {}).get("gap_pct") for b in xb])
    ortools_vals = [b["solvers"].get("ortools", {}).get("gap_pct") for b in xb]
    ortools = mean(ortools_vals)
    ortools_fail = sum(1 for v in ortools_vals if v is None)
    cu_vals = [cu[b["instance"]].get("gap_pct") for b in xb if b["instance"] in cu]
    cuopt = mean(cu_vals)
    cu_mode = next((c.get("mode") for c in cu.values() if c.get("mode")), None)
    at_bks = sum(1 for b in xb if best_gap(b) <= 1e-9)
    within_half = sum(1 for b in xb if best_gap(b) <= 0.5)
    full_mean = mean([f["gap_pct"] for f in full])
    # Same instances at 10% budget, for a like-for-like budget effect
    small = {f["instance"] for f in full}
    small10 = mean([b["gap_pct"] for b in xb if b["instance"] in small])

    # Full published budget (2.4 n s) on the same 30 instances, PyVRP, one pinned core each.
    F = None
    if f30:
        rows = [f30[b["instance"]] for b in xb if b["instance"] in f30]
        def fgap(r):
            return r["solvers"]["pyvrp"]["gap_pct"]
        def frace(r):  # race with the cuOpt runs (they had only 10% of the time: a conservative race)
            g = [fgap(r)]
            c = cu.get(r["instance"])
            if c and c.get("gap_pct") is not None:
                g.append(c["gap_pct"])
            return min(g)
        fg = {}
        for lab, lo, hi in (("100-299", 0, 300), ("300-599", 300, 600), ("600-1000", 600, 10**6)):
            sel = [r for r in rows if lo <= r["n"] < hi]
            fg[lab] = {"instances": len(sel), "pyvrp": mean([fgap(r) for r in sel]), "race": mean([frace(r) for r in sel])}
        F = {"instances": len(rows), "pyvrp_mean_gap_pct": mean([fgap(r) for r in rows]),
             "race_mean_gap_pct": mean([frace(r) for r in rows]),
             "cuopt_wins": sum(1 for r in rows if frace(r) < fgap(r)),
             "at_bks": sum(1 for r in rows if frace(r) <= 1e-9),
             "within_0_5pct": sum(1 for r in rows if frace(r) <= 0.5), "by_size": fg,
             "host": rows[0]["stamp"]["host"].split(" (core")[0] if rows else None}
    # Verdict: on the full-budget stratified 30 if available, else on the 10% run.
    head = F["pyvrp_mean_gap_pct"] if F and F["instances"] >= 30 else race
    head_race = F["race_mean_gap_pct"] if F and F["instances"] >= 30 else race
    if head_race is not None and head_race <= 0.22:
        verdict = "reached"
    elif head_race is not None and head_race <= 0.44:
        verdict = "close"
    else:
        verdict = "not reached"
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}%"
    note = (
        f"{n} of the 100 X instances (every ~3rd by size, n = 100 to 1000), one seed, at <b>10% of the published "
        f"time budget</b> (0.24·n s instead of 2.4·n s), because the shared CPU allows ~5 cores across ten projects. "
        f"The full race (PyVRP and OR-Tools on CPU, cuOpt on one L4 GPU) gives a mean gap to best-known of "
        f"<b>{fmt(race)}</b> (CPU racers alone {fmt(race_cpu)}; cuOpt wins {cu_wins} of {n} instances); "
        f"{at_bks} instances hit the best-known solution exactly and {within_half} are within 0.5%. "
        f"OR-Tools alone averages {fmt(ortools)}"
        + (f" and found no feasible plan within budget on {ortools_fail} tight-capacity instance(s)" if ortools_fail else "")
        + ". "
        + (f"NVIDIA cuOpt on one L4 GPU, same budget, averages {fmt(cuopt)} over {len(cu_vals)} instances "
           f"(objective setting chosen on X-n101 only: {cu_mode}; cuOpt minimises fleet size before distance by default). "
           if cu_vals else "")
        + (f"At the <b>full published budget</b> on the {len(full)} smallest instances PyVRP averages {fmt(full_mean)} "
           f"(the same instances at 10% budget: {fmt(small10)}), against the published 0.22% (PyVRP) and 0.11% (HGS-CVRP) "
           f"over all 100 instances and 10 seeds. " if full else "")
        + "Verdict rule: reached = within the published PyVRP gap; close = within about 2x of it or matching it at full "
          "budget on a subset; otherwise not reached."
    )
    short = (f"{n} CVRPLIB X instances at 10% of the published time: the open-source race (CPU + GPU) averages {fmt(race)} "
             f"from best-known (published state of the art at full time: 0.11–0.22%).")
    if F and F["instances"] >= 30:
        note = (f"<b>Full published budget</b> (2.4·n s per instance, the protocol behind the published numbers) on "
                f"{F['instances']} stratified X instances (n 100–1000), PyVRP single-thread, <b>one seed</b>, one pinned core "
                f"each: mean gap to best-known <b>{fmt(F['pyvrp_mean_gap_pct'])}</b>. Racing in the earlier cuOpt (L4) runs, "
                f"which had only 10% of that time, gives <b>{fmt(F['race_mean_gap_pct'])}</b> (cuOpt still wins "
                f"{F['cuopt_wins']} instances). {F['at_bks']} instances hit best-known exactly; {F['within_0_5pct']} of "
                f"{F['instances']} are within 0.5%. Published: PyVRP 0.22%, HGS-CVRP 0.11% (all 100 instances, mean of 10 seeds). "
                f"A 30-instance stratified subset and one seed is a fair but not identical comparison: the published mean is over "
                f"all 100 and averages out seed luck. At 10% of the budget the same race averaged {fmt(race)}. "
                f"Verdict rule: reached = race mean ≤ 0.22%; close = ≤ 0.44%; otherwise not reached.")
        short = (f"{F['instances']} CVRPLIB X instances at the full published time: PyVRP {fmt(F['pyvrp_mean_gap_pct'])}, "
                 f"race with cuOpt {fmt(F['race_mean_gap_pct'])} from best-known (published 0.11–0.22%).")
    out = {
        "instances": n, "race_mean_gap_pct": race, "race_cpu_only_mean_gap_pct": race_cpu, "cuopt_wins": cu_wins,
        "by_size": groups, "pyvrp_mean_gap_pct": pyvrp, "ortools_mean_gap_pct": ortools,
        "ortools_no_solution": ortools_fail, "cuopt_mean_gap_pct": cuopt, "cuopt_instances": len(cu_vals),
        "cuopt_mode": cu_mode, "at_bks": at_bks, "within_0_5pct": within_half,
        "full_budget": {"instances": len(full), "pyvrp_mean_gap_pct": full_mean, "same_at_10pct": small10},
        "published": {"PyVRP": 0.22, "HGS-CVRP": 0.11, "protocol": "Tmax = 2.4 n s, 10 seeds, all 100 X instances"},
        "full30": F,
        "verdict": verdict.upper(), "note": note, "short": short,
        "stamp": f"{date.today().isoformat()} · CPU solvers 1 thread each (21 instances on the qBraid subscription pod, "
                 f"9 largest on the shared qBraid gpu-l4 instance after a pod restart) · cuOpt on the shared L4 · "
                 f"0.1 × 2.4·n s per solver; full-budget check 2.4·n s, PyVRP, 6 smallest instances",
    }
    with open(os.path.join(R, "xbench_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("note",)}, indent=1))


if __name__ == "__main__":
    main()
