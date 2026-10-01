"""AC optimal power flow on PGLib-OPF cases, checked against the published baseline.

For each case:
  * race three exact AC formulations (polar PSV, rectangular RSV, current-voltage RIV) with Ipopt,
  * compare the best objective with the PGLib v23.07 baseline (PowerModels.jl + Ipopt),
  * solve the conic SOC relaxation (convex, so Ipopt's optimum is a valid lower bound) and report
    the certified optimality gap of the AC solution,
  * save the AC solution (voltages, dispatch, branch flows and loading) for the viewer.

usage: python acopf_pglib.py <pglib_dir> <out_dir> case118_ieee [case1354_pegase ...]
"""
import json, math, re, sys, time
from pathlib import Path

import pyomo.environ as pe
from egret.parsers.matpower_parser import create_ModelData
from egret.models.acopf import (solve_acopf, create_psv_acopf_model,
                                create_rsv_acopf_model, create_riv_acopf_model)
from egret.models.ac_relaxations import create_soc_relaxation

FORMULATIONS = {"polar": create_psv_acopf_model,
                "rectangular": create_rsv_acopf_model,
                "current-voltage": create_riv_acopf_model}
IPOPT = {"tol": 1e-8, "max_iter": 3000, "linear_solver": "mumps", "print_level": 0}


def baseline(pglib_dir):
    """PGLib BASELINE.md, all operating conditions: case (e.g. case118_ieee__api) -> AC objective ($/h)."""
    ref = {}
    for line in (Path(pglib_dir) / "BASELINE.md").read_text().splitlines():
        if line.startswith("| pglib_opf_"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            num = lambda x: float(x) if x.replace(".", "").replace("e", "").replace("+", "").replace("-", "").isdigit() else None
            ac = num(cells[4])           # the DC column is "inf." for the SAD cases: parse fields independently
            if ac is not None:
                ref[cells[0].replace("pglib_opf_", "")] = {"ac": ac, "dc": num(cells[3]), "soc_gap_pct": num(cells[6])}
    return ref


def solve_ac(md, name):
    t0 = time.time()
    md_out, m, res = solve_acopf(md.clone(), "ipopt", solver_tee=False, options=IPOPT,
                                 acopf_model_generator=FORMULATIONS[name],
                                 return_model=True, return_results=True)
    term = str(res.solver.termination_condition)
    return {"formulation": name, "objective": md_out.data["system"]["total_cost"], "status": term,
            "seconds": round(time.time() - t0, 2)}, md_out


def solve_soc(md):
    t0 = time.time()
    m, _ = create_soc_relaxation(md.clone(), use_linear_relaxation=False)
    res = pe.SolverFactory("ipopt").solve(m, options=IPOPT)
    return {"objective": pe.value(m.obj), "status": str(res.solver.termination_condition),
            "seconds": round(time.time() - t0, 2)}


def extract(md):
    """Compact solution for the viewer."""
    buses = dict(md.elements("bus"))
    gens = dict(md.elements("generator"))
    branches = dict(md.elements("branch"))
    loads = dict(md.elements("load"))
    out = {"bus": {}, "gen": [], "branch": [], "load": {}}
    for b, d in buses.items():
        out["bus"][b] = {"vm": round(d["vm"], 5), "va": round(d["va"], 4),
                         "vmin": d["v_min"], "vmax": d["v_max"], "kv": d.get("base_kv")}
    for g, d in gens.items():
        out["gen"].append({"bus": d["bus"], "pg": round(d["pg"], 4), "qg": round(d["qg"], 4),
                           "pmax": d["p_max"], "pmin": d["p_min"]})
    for k, d in branches.items():
        if d.get("pf") is None or not d.get("in_service", True):
            continue                      # out-of-service branch
        rate = d.get("rating_long_term") or 0.0
        s = max(math.hypot(d["pf"], d["qf"]), math.hypot(d["pt"], d["qt"]))
        out["branch"].append({"f": d["from_bus"], "t": d["to_bus"], "pf": round(d["pf"], 4),
                              "qf": round(d["qf"], 4), "rate": rate,
                              "loading": round(s / rate, 4) if rate else None})
    for k, d in loads.items():
        out["load"][d["bus"]] = round(out["load"].get(d["bus"], 0.0) + d["p_load"], 4)
    return out


def backfill(pglib, out):
    """Recompute reference fields of stored results with the current baseline parser (no re-solve)."""
    ref = baseline(pglib)
    for f in Path(out).glob("case*.json"):
        if f.name.endswith("_solution.json"):
            continue
        rec = json.loads(f.read_text()); rec["reference"] = ref.get(rec["case"])
        if rec.get("best") and rec["reference"]:
            rec["rel_diff_vs_reference"] = (rec["best"]["objective"] - rec["reference"]["ac"]) / rec["reference"]["ac"]
        f.write_text(json.dumps(rec, indent=1))


def main():
    if sys.argv[3:4] == ["--backfill"]:
        return backfill(sys.argv[1], sys.argv[2])
    pglib, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ref = baseline(pglib)
    for case in sys.argv[3:]:
        sub = "api/" if case.endswith("__api") else "sad/" if case.endswith("__sad") else ""
        md = create_ModelData(str(pglib / f"{sub}pglib_opf_{case}.m"))
        runs, best, best_md = [], None, None
        for name in FORMULATIONS:
            try:
                r, md_out = solve_ac(md, name)
            except Exception as e:  # a formulation that fails to converge is a valid race result
                r, md_out = {"formulation": name, "status": f"error: {type(e).__name__}: {e}"[:200]}, None
            runs.append(r)
            print(case, r, flush=True)
            if md_out is not None and r["status"] == "optimal" and (best is None or r["objective"] < best["objective"]):
                best, best_md = r, md_out
        soc = solve_soc(md)
        rec = {"case": case, "buses": len(dict(md.elements("bus"))), "branches": len(dict(md.elements("branch"))),
               "race": runs, "best": best, "soc": soc, "reference": ref.get(case)}
        if best and rec["reference"]:
            rec["rel_diff_vs_reference"] = (best["objective"] - rec["reference"]["ac"]) / rec["reference"]["ac"]
        if best and soc["status"] == "optimal":
            rec["certified_gap_pct"] = 100 * (best["objective"] - soc["objective"]) / best["objective"]
        print(case, "SUMMARY", {k: rec.get(k) for k in ("rel_diff_vs_reference", "certified_gap_pct")}, flush=True)
        (out / f"{case}.json").write_text(json.dumps(rec, indent=1))
        if best_md is not None:
            try:
                (out / f"{case}_solution.json").write_text(json.dumps(extract(best_md)))
            except Exception as e:
                print(case, "solution export failed:", e, flush=True)


if __name__ == "__main__":
    main()
