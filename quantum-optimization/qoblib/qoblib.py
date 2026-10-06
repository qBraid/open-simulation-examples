#!/usr/bin/env python3
"""QOBLIB on qBraid: fetch, solve, check and package benchmark submissions.

QOBLIB (Quantum Optimization Benchmarking Library, Koch et al., 2025;
https://github.com/ZIB-AOPT/QOBLIB) is ten hard optimization problem classes with
reference solutions, solution checkers and a public leaderboard
(https://zib-aopt.github.io/QOBLIB/). Code is Apache-2.0; instances are CC BY 4.0
and are fetched at run time, never copied into this repository.

This script covers three classes end to end:

    01-marketsplit     feasibility: find x in {0,1}^n with Ax = b
    07-independentset  maximum independent set (maximize |I|)
    09-routing         capacitated vehicle routing, 20 customers, 4 vehicles

Commands:

    python qoblib.py fetch  [--classes 01-marketsplit 07-independentset 09-routing]
    python qoblib.py list   [--class 07-independentset]
    python qoblib.py solve  <class> <instance> [--time 60] [--solver module:function]
    python qoblib.py check  <class> <instance> <solution-file>
    python qoblib.py submit <class> <instance> <solution-file> --name "Jane Doe"
                            --affiliation "Org" --reference URL [--paradigm ...]

A solver is any function ``solve(instance, time_limit) -> solution``. For quantum
or hybrid work, ``load_qubo(cls, instance)`` returns the library's own published
QUBO (the ``models/binary_unconstrained/qs_files`` model), so a sampler only has
to return a bit vector; ``qubo_energy`` scores it with the same convention.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import lzma
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_URL = "https://github.com/ZIB-AOPT/QOBLIB"
HOME = Path(os.environ.get("QOBLIB_HOME", Path.cwd() / "QOBLIB"))
SUPPORTED = ("01-marketsplit", "07-independentset", "09-routing")
EXT = {"01-marketsplit": ".dat", "07-independentset": ".gph", "09-routing": ".vrp"}
SENSE = {"01-marketsplit": "min", "07-independentset": "max", "09-routing": "min"}
CHECKER = {"01-marketsplit": "check_marketsplit", "07-independentset": "check_stableset",
           "09-routing": "check_cvrp"}


# --------------------------------------------------------------------------- fetch

def fetch(classes=SUPPORTED, home: Path = HOME) -> Path:
    """Sparse, blob-filtered clone: only `misc/` and the requested classes are downloaded."""
    if not (home / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        REPO_URL, str(home)], check=True)
    subprocess.run(["git", "-C", str(home), "sparse-checkout", "set", "misc", *classes], check=True)
    return home


def class_dir(cls: str) -> Path:
    d = HOME / cls
    if not d.is_dir():
        sys.exit(f"{d} not found: run `python qoblib.py fetch --classes {cls}` first "
                 f"(or set QOBLIB_HOME to an existing clone).")
    return d


def instances(cls: str) -> list[str]:
    return sorted(p.name[: -len(EXT[cls])] for p in (class_dir(cls) / "instances").glob("*" + EXT[cls]))


def best_known(cls: str) -> dict[str, tuple[str, str]]:
    """Instance -> (best-known value, status) from the auto-generated table in solutions/README.md."""
    text = (class_dir(cls) / "solutions" / "README.md").read_text()
    table = {}
    for m in re.finditer(r"^\| (\S+) \| *([^|]+?) *\| *([^|]+?) *\|", text, re.M):
        if m.group(1) not in ("Instance", ":-------"):
            table[m.group(1)] = (m.group(2), m.group(3))
    return table


# ------------------------------------------------------------------------- loaders

def _data_lines(path: Path):
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield line


@dataclass
class Instance:
    cls: str
    name: str
    path: Path
    data: dict = field(default_factory=dict)


def load(cls: str, name: str) -> Instance:
    path = class_dir(cls) / "instances" / (name + EXT[cls])
    inst = Instance(cls, name, path)
    if cls == "01-marketsplit":
        rows = [list(map(int, re.split(r"[,\s]+", l))) for l in _data_lines(path)]
        m, n = rows[0]
        inst.data = {"A": [r[:n] for r in rows[1 : m + 1]], "b": [r[n] for r in rows[1 : m + 1]], "n": n}
    elif cls == "07-independentset":
        n, edges = 0, []
        for line in path.read_text().splitlines():
            t = line.split()
            if t and t[0] == "p":
                n = int(t[2])
            elif t and t[0] == "e":
                u, v = int(t[1]), int(t[2])
                if u != v:
                    edges.append((min(u, v), max(u, v)))
        inst.data = {"n": n, "edges": sorted(set(edges))}
    elif cls == "09-routing":
        import vrplib
        inst.data = vrplib.read_instance(str(path))
        inst.data["vehicles"] = int(re.search(r"-k(\d+)-", name).group(1))
    else:
        raise ValueError(f"{cls} is not wired up in this kit yet (supported: {', '.join(SUPPORTED)})")
    return inst


def load_qubo(cls: str, name: str):
    """The library's published QUBO: returns (offset, {(i, j): q}) with 1-based i <= j.

    Energy convention, verified against the reference solutions:
        E(x) = offset + sum_i q_ii x_i + 2 * sum_{i<j} q_ij x_i x_j
    Market split: E = 0 exactly for feasible x. Independent set: E = -|I| for an
    independent set (the published model is the negated maximisation).
    """
    path = class_dir(cls) / "models" / "binary_unconstrained" / "qs_files" / f"{name}.qs.xz"
    offset, q = 0.0, {}
    with lzma.open(path, "rt") as fh:
        header_seen = False
        for line in fh:
            if line.startswith("#"):
                m = re.search(r"ObjectiveOffset\s+(-?[\d.eE+]+)", line)
                if m:
                    offset = float(m.group(1))
                continue
            t = line.split()
            if not t:
                continue
            if not header_seen:          # "<vars> <nonzeros>"
                header_seen = True
                continue
            i, j, c = int(t[0]), int(t[1]), float(t[2])
            q[(min(i, j), max(i, j))] = q.get((min(i, j), max(i, j)), 0.0) + c
    return offset, q


def qubo_energy(offset: float, q: dict, x: list[int]) -> float:
    """x is a 0/1 list indexed from 0 for variable 1."""
    e = offset
    for (i, j), c in q.items():
        if x[i - 1] and x[j - 1]:
            e += c if i == j else 2 * c
    return e


# ------------------------------------------------------- objective and feasibility

def evaluate(inst: Instance, sol) -> tuple[bool, float]:
    """(feasible, objective) in the class's own convention, independent of the official checker."""
    d = inst.data
    if inst.cls == "01-marketsplit":
        res = [abs(bi - sum(a * xi for a, xi in zip(row, sol))) for row, bi in zip(d["A"], d["b"])]
        return all(r == 0 for r in res), float(sum(res))   # total |Ax - b|; 0 iff feasible
    if inst.cls == "07-independentset":
        s = set(sol)
        ok = all(not (u in s and v in s) for u, v in d["edges"]) and all(1 <= v <= d["n"] for v in s)
        return ok, float(len(s))
    if inst.cls == "09-routing":
        xy, dem, cap = d["node_coord"], d["demand"], d["capacity"]
        seen, cost, ok = [], 0, True
        for r in sol:
            if not r:
                continue
            path = [0, *r, 0]
            cost += sum(round(math.dist(xy[a], xy[b])) for a, b in zip(path, path[1:]))
            ok &= sum(dem[c] for c in r) <= cap
            seen += r
        ok &= sorted(seen) == list(range(1, len(xy))) and len([r for r in sol if r]) <= d["vehicles"]
        return ok, float(cost)
    raise ValueError(inst.cls)


def write_solution(inst: Instance, sol, path: Path) -> Path:
    if inst.cls == "01-marketsplit":
        path.write_text("".join(str(int(v)) for v in sol) + "\n")
    elif inst.cls == "07-independentset":
        path.write_text(f"# Objective value = {len(sol)}\n" + "".join(f"{v}\n" for v in sorted(sol)))
    elif inst.cls == "09-routing":
        _, cost = evaluate(inst, sol)
        lines = [f"Route #{k}: {' '.join(map(str, r))}" for k, r in enumerate([r for r in sol if r], 1)]
        path.write_text("\n".join(lines) + f"\nCost {int(cost)}\n")
    return path


def read_solution(inst: Instance, path: Path):
    text = path.read_text()
    if inst.cls == "01-marketsplit":
        bits = re.sub(r"[^01]", "", "".join(l for l in text.splitlines() if not l.startswith("#")))
        return [int(c) for c in bits]
    if inst.cls == "07-independentset":
        return [int(l) for l in text.split("\n") if l.strip() and not l.startswith("#")]
    if inst.cls == "09-routing":
        return [list(map(int, l.split(":", 1)[1].split())) for l in text.splitlines() if l.startswith("Route")]
    raise ValueError(inst.cls)


def best_of_samples(inst: Instance, bitstrings) -> tuple[object, float, bool]:
    """Turn raw sampler output (QPU or simulator) into the best solution QOBLIB will accept.

    `bitstrings` is any iterable of 0/1 sequences over the QUBO variables (variable 1
    first). Independent-set samples are repaired greedily (drop a vertex from every
    violated edge), which is standard post-processing and must be reported as such
    in the submission's Workflow field. Market-split samples are not repaired: a
    sample either satisfies Ax = b or it does not.
    """
    best, best_obj, best_feas = None, None, False
    for bits in bitstrings:
        bits = [int(b) for b in bits]
        if inst.cls == "07-independentset":
            chosen = {i + 1 for i, b in enumerate(bits) if b}
            for u, v in inst.data["edges"]:
                if u in chosen and v in chosen:
                    chosen.discard(v)
            sol = sorted(chosen)
        elif inst.cls == "01-marketsplit":
            sol = bits
        else:
            raise ValueError("sample post-processing is wired for 01-marketsplit and 07-independentset")
        feas, obj = evaluate(inst, sol)
        better = (best is None or (feas and not best_feas) or (feas == best_feas and
                  (obj > best_obj if SENSE[inst.cls] == "max" else obj < best_obj)))
        if better:
            best, best_obj, best_feas = sol, obj, feas
    return best, best_obj, best_feas


# ------------------------------------------------------------- classical baselines

@dataclass
class Result:
    solution: object
    objective: float
    bound: float | None          # proven bound, if the solver gives one
    feasible: bool
    runtime: float
    history: list = field(default_factory=list)   # [(seconds, incumbent)]
    solver: str = ""


def _highs(time_limit: float):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("threads", 1)
    return h


def solve_highs(inst: Instance, time_limit: float = 60.0) -> Result:
    """Exact MIP with HiGHS (market split as min sum|Ax-b|, independent set as max |I|)."""
    import highspy
    import numpy as np
    d, t0 = inst.data, time.time()
    h = _highs(time_limit)
    inf = highspy.kHighsInf
    if inst.cls == "01-marketsplit":
        n, m = d["n"], len(d["b"])
        # x (n binaries), then s+ and s- (m each, continuous >= 0); minimize total slack.
        cost = [0.0] * n + [1.0] * (2 * m)
        lower, upper = [0.0] * (n + 2 * m), [1.0] * n + [inf] * (2 * m)
        h.addVars(n + 2 * m, np.array(lower), np.array(upper))
        h.changeColsCost(n + 2 * m, np.arange(n + 2 * m, dtype=np.int32), np.array(cost))
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                                np.array([highspy.HighsVarType.kInteger] * n))
        for i, (row, bi) in enumerate(zip(d["A"], d["b"])):
            idx = [j for j in range(n) if row[j]] + [n + i, n + m + i]
            val = [float(row[j]) for j in range(n) if row[j]] + [1.0, -1.0]
            h.addRow(float(bi), float(bi), len(idx), np.array(idx, dtype=np.int32), np.array(val))
    elif inst.cls == "07-independentset":
        n = d["n"]
        h.addVars(n, np.zeros(n), np.ones(n))
        h.changeColsCost(n, np.arange(n, dtype=np.int32), -np.ones(n))      # maximize |I|
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                                np.array([highspy.HighsVarType.kInteger] * n))
        for u, v in d["edges"]:
            h.addRow(-inf, 1.0, 2, np.array([u - 1, v - 1], dtype=np.int32), np.array([1.0, 1.0]))
    else:
        raise ValueError("HiGHS baseline covers 01-marketsplit and 07-independentset; "
                         "use solve_pyvrp for 09-routing")
    h.run()
    info, rt = h.getInfo(), time.time() - t0
    if info.primal_solution_status != 2:                  # no feasible point found
        return Result(None, math.inf, None, False, rt, solver="HiGHS")
    x = list(h.getSolution().col_value)
    if inst.cls == "01-marketsplit":
        sol = [int(round(v)) for v in x[: d["n"]]]
        bound = max(0.0, info.mip_dual_bound)
    else:
        sol = [i + 1 for i in range(d["n"]) if x[i] > 0.5]
        bound = math.floor(-info.mip_dual_bound + 1e-6)
    feas, obj = evaluate(inst, sol)
    return Result(sol, obj, bound, feas, rt, [(rt, obj)], solver=f"HiGHS {highspy.Highs().version()}")


def solve_pyvrp(inst: Instance, time_limit: float = 60.0, seed: int = 1) -> Result:
    """PyVRP hybrid genetic search, fleet fixed at the instance's k vehicles. Heuristic: no bound."""
    import pyvrp
    from pyvrp.stop import MaxRuntime
    d, t0 = inst.data, time.time()
    xy = d["node_coord"]
    m = pyvrp.Model()
    m.add_vehicle_type(num_available=d["vehicles"], capacity=int(d["capacity"]))
    locs = [m.add_location(x=float(px), y=float(py)) for px, py in xy]     # PyVRP >= 0.14 API
    m.add_depot(locs[0])
    for i in range(1, len(xy)):
        m.add_client(locs[i], delivery=int(d["demand"][i]))
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            if i != j:   # QOBLIB rounds each Euclidean edge, as its checker does
                m.add_edge(a, b, distance=round(math.dist(xy[i], xy[j])))
    res = m.solve(stop=MaxRuntime(time_limit), seed=seed, display=False)
    # Route activities: depots and clients; client idx is 0-based, QOBLIB customers are 1..n.
    sol = [[a.idx + 1 for a in r if a.is_client()] for r in res.best.routes()]
    feas, obj = evaluate(inst, sol)
    return Result(sol, obj, None, feas, time.time() - t0, [(time.time() - t0, obj)], solver="PyVRP")


DEFAULT_SOLVER = {"01-marketsplit": solve_highs, "07-independentset": solve_highs,
                  "09-routing": solve_pyvrp}


# ---------------------------------------------------------------- official checks

def official_checker(cls: str) -> Path | None:
    """Build the library's Rust checker once (needs cargo); return its path, or None."""
    chk = class_dir(cls) / "check"
    binary = chk / "target" / "release" / CHECKER[cls]
    if binary.exists():
        return binary
    if shutil.which("cargo") is None:
        return None
    subprocess.run(["cargo", "build", "--release", "-q"], cwd=chk, check=False)
    return binary if binary.exists() else None


def run_official_check(inst: Instance, solution_file: Path) -> tuple[int | None, str]:
    binary = official_checker(inst.cls)
    if binary is None:
        return None, "official checker not built (install Rust: https://rustup.rs), skipped"
    p = subprocess.run([str(binary), str(inst.path), str(solution_file)], capture_output=True, text=True)
    meaning = {0: "VALID", 20: "SUBOPTIMAL", 21: "INFEASIBLE", 10: "INVALID_FILE", 2: "USAGE"}
    return p.returncode, meaning.get(p.returncode, "UNEXPECTED")


# ---------------------------------------------------------------------- submission

TEMPLATE_HEADER = None   # read from the library's own template so the columns never drift


def submission_header() -> list[str]:
    with open(HOME / "misc" / "submission_template.csv", newline="", encoding="utf-8-sig") as fh:
        return next(csv.reader(fh))


def model_stats(inst: Instance) -> dict:
    d = inst.data
    if inst.cls == "01-marketsplit":
        nz = [a for row in d["A"] for a in row if a]
        return {"approach": "Binary linear program: Ax + s+ - s- = b, minimise total slack",
                "vars": d["n"] + 2 * len(d["b"]), "bin": d["n"], "int": 0, "cont": 2 * len(d["b"]),
                "nnz": len(nz) + 2 * len(d["b"]) + 2 * len(d["b"]), "ctype": "Integer",
                "crange": f"{min(nz)} - {max(nz)}"}
    if inst.cls == "07-independentset":
        return {"approach": "Binary linear program: max sum x, x_u + x_v <= 1 per edge",
                "vars": d["n"], "bin": d["n"], "int": 0, "cont": 0,
                "nnz": d["n"] + 2 * len(d["edges"]), "ctype": "Binary", "crange": "1 - 1"}
    n = len(d["node_coord"]) - 1
    return {"approach": "Giant-tour routing heuristic (no explicit MIP)", "vars": "N/A", "bin": "N/A",
            "int": "N/A", "cont": "N/A", "nnz": "N/A", "ctype": "Integer",
            "crange": f"customers={n}, vehicles={d['vehicles']}"}


def _cpus() -> str:
    """CPUs the container may actually use (cgroup quota), not the host's count."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return f"{int(quota) / int(period):g}"
    except OSError:
        pass
    return str(os.cpu_count())


def make_submission(inst: Instance, solution_file: Path, out_root: Path, *, name: str,
                    affiliation: str, reference: str, workflow: str, paradigm: str = "Classical",
                    algorithm_type: str = "Deterministic", runs: int = 1, bound=None,
                    runtime=None, cpu_runtime=None, gpu_runtime="N/A", qpu_runtime="N/A",
                    hardware: str | None = None, remarks: str = "") -> Path:
    """Write <out_root>/<instance>/{<instance>_summary.csv, <instance>_solution.<ext>}."""
    sol = read_solution(inst, solution_file)
    feasible, obj = evaluate(inst, sol)
    st = model_stats(inst)
    inst_dir = out_root / inst.name
    inst_dir.mkdir(parents=True, exist_ok=True)
    ext = {"01-marketsplit": "txt", "07-independentset": "txt", "09-routing": "sol"}[inst.cls]
    shutil.copy(solution_file, inst_dir / f"{inst.name}_solution.{ext}")
    hw = hardware or f"{platform.machine()}, {_cpus()} CPUs available to the run"
    row = {
        "Problem": inst.name, "Submitter": name, "Affiliation": affiliation,
        "Date": dt.date.today().isoformat(), "Reference": reference,
        "Best Objective Value": obj if inst.cls != "07-independentset" else int(obj),
        "Optimality Bound": "N/A" if bound is None else bound,
        "Modeling Approach": st["approach"], "# Decision Variables": st["vars"],
        "# Binary Variables": st["bin"], "# Integer Variables": st["int"],
        "# Continuous Variables": st["cont"], "# Non-Zero Coefficients": st["nnz"],
        "Coefficients Type": st["ctype"], "Coefficients Range": st["crange"],
        "Workflow": workflow, "Algorithm Type": algorithm_type, "Paradigm": paradigm,
        "# Runs": runs, "# Feasible Runs": runs if feasible else 0,
        "# Successful Runs": runs if feasible else 0, "Success Threshold": 0.0,
        "Hardware Specifications": hw, "Total Runtime": runtime if runtime is not None else "N/A",
        "Time to Solution": runtime if runtime is not None else "N/A",
        "CPU Runtime": cpu_runtime if cpu_runtime is not None else "N/A",
        "GPU Runtime": gpu_runtime, "QPU Runtime": qpu_runtime, "Other HW Runtime": "N/A",
        "Remarks": remarks,
    }
    header = submission_header()
    with open(inst_dir / f"{inst.name}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerow({k: row[k] for k in header})
    return inst_dir


def submission_dir(cls: str, method: str, submitter: str) -> Path:
    """Where a pull request puts it: <class>/submissions/<YYYYMMDD>_<Method>_<Surname>/.

    The validator only runs the solution checker for submissions inside this tree;
    anywhere else it checks structure alone, so always build submissions here.
    """
    surname = re.sub(r"\W", "", submitter.split()[-1]) or "Anon"
    tag = re.sub(r"\W", "", method) or "Method"
    return class_dir(cls) / "submissions" / f"{dt.date.today():%Y%m%d}_{tag}_{surname}"


def validate_submission(sub_root: Path) -> int:
    """Run the library's own submission validator (the same one its CI runs on every PR)."""
    cmd = [sys.executable, str(HOME / "misc" / "ci" / "check_submission.py"), str(sub_root)]
    if shutil.which("cargo") is None:
        cmd.append("--no-check")
    return subprocess.run(cmd).returncode


# ----------------------------------------------------------------------------- cli

def _solver(spec: str | None, cls: str):
    if not spec:
        return DEFAULT_SOLVER[cls]
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--classes", nargs="+", default=list(SUPPORTED))
    l = sub.add_parser("list"); l.add_argument("--class", dest="cls")
    s = sub.add_parser("solve"); s.add_argument("cls"); s.add_argument("instance")
    s.add_argument("--time", type=float, default=60.0); s.add_argument("--solver")
    s.add_argument("--out", type=Path, default=Path("results"))
    c = sub.add_parser("check"); c.add_argument("cls"); c.add_argument("instance"); c.add_argument("solution", type=Path)
    m = sub.add_parser("submit"); m.add_argument("cls"); m.add_argument("instance"); m.add_argument("solution", type=Path)
    for a in ("--name", "--affiliation", "--reference"):
        m.add_argument(a, required=True)
    m.add_argument("--workflow", default="Classical baseline from qoblib.py")
    m.add_argument("--paradigm", default="Classical", choices=["Classical", "Quantum Simulator", "Quantum Hardware"])
    m.add_argument("--algorithm-type", default="Deterministic", choices=["Deterministic", "Stochastic"])
    m.add_argument("--runs", type=int, default=1); m.add_argument("--bound"); m.add_argument("--runtime", type=float)
    m.add_argument("--qpu-runtime", default="N/A")
    m.add_argument("--method", default="Baseline", help="short method tag used in the submission folder name")
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        print(f"QOBLIB at {fetch(args.classes)}")
    elif args.cmd == "list":
        for cls in ([args.cls] if args.cls else SUPPORTED):
            bkv = best_known(cls)
            print(f"{cls}: {len(instances(cls))} instances")
            for name in instances(cls)[:400]:
                v, st = bkv.get(name, ("?", "?"))
                print(f"  {name:28s} best known {v:>10s} ({st})")
    elif args.cmd == "solve":
        inst = load(args.cls, args.instance)
        r = _solver(args.solver, args.cls)(inst, args.time)
        args.out.mkdir(parents=True, exist_ok=True)
        v, st = best_known(args.cls).get(inst.name, ("?", "?"))
        print(f"{inst.name}: {r.solver} objective {r.objective:g} (best known {v}, {st}); "
              f"bound {r.bound}; feasible {r.feasible}; {r.runtime:.1f} s")
        if r.solution is not None:
            out = write_solution(inst, r.solution, args.out / f"{inst.name}.sol")
            code, meaning = run_official_check(inst, out)
            print(f"solution -> {out}; official checker: {meaning}")
    elif args.cmd == "check":
        inst = load(args.cls, args.instance)
        feas, obj = evaluate(inst, read_solution(inst, args.solution))
        code, meaning = run_official_check(inst, args.solution)
        print(f"objective {obj:g}, feasible {feas}; official checker: {meaning}")
        sys.exit(0 if feas and code in (None, 0, 20) else 1)
    elif args.cmd == "submit":
        inst = load(args.cls, args.instance)
        feas, _ = evaluate(inst, read_solution(inst, args.solution))
        if not feas:
            print("WARNING: this solution is infeasible. It will be filed with '# Feasible Runs' = 0, "
                  "which QOBLIB accepts as a reported negative result, not as a solution.")
        root = submission_dir(args.cls, args.method, args.name)
        d = make_submission(inst, args.solution, root, name=args.name, affiliation=args.affiliation,
                            reference=args.reference, workflow=args.workflow, paradigm=args.paradigm,
                            algorithm_type=args.algorithm_type, runs=args.runs, bound=args.bound,
                            runtime=args.runtime, cpu_runtime=args.runtime, qpu_runtime=args.qpu_runtime)
        print(f"wrote {d}; validating with the library's own checker:")
        rc = validate_submission(root)
        if rc == 0:
            print(f"\nReady for a pull request to {REPO_URL}: commit {root.relative_to(HOME)} on a fork.")
        sys.exit(rc)


if __name__ == "__main__":
    main()
