"""Compare a Palace transmon run against Palace's published regression reference.

Reference: test/data/regression/ref/transmon/transmon_coarse in the Palace
source tree (v0.18.1). This is the same example, mesh and solver settings,
used by Palace's own CI.

Tolerances: eigenfrequency 1e-4 relative; Q, EPR and Q_ext 2e-2 relative
(these depend on small imaginary parts and differ across MPI partitionings and
BLAS builds). The run passes only if every row passes.

Usage:  python validate.py <run_postpro> <reference_dir> out.json
"""

import json
import sys

from derive_hamiltonian import read_csv

TOL = {"f": 1e-4, "Q": 2e-2, "p": 2e-2, "Qext": 2e-2}


def rows(run, ref):
    out = []

    def add(name, a, b, kind, digits):
        rel = abs(a - b) / abs(b)
        out.append({"name": name, "run": f"{a:.{digits}g}", "ref": f"{b:.{digits}g}",
                    "rel": f"{rel:.1e}", "rel_value": rel, "tol": TOL[kind], "pass": rel <= TOL[kind]})

    e1, e0 = read_csv(f"{run}/eig.csv"), read_csv(f"{ref}/eig.csv")
    p1, p0 = read_csv(f"{run}/port-EPR.csv"), read_csv(f"{ref}/port-EPR.csv")
    q1, q0 = read_csv(f"{run}/port-Q.csv"), read_csv(f"{ref}/port-Q.csv")
    labels = {0: "qubit mode", 1: "resonator mode"}
    for m in range(len(e0)):
        add(f"{labels[m]} f (GHz)", e1[m]["Re{f} (GHz)"], e0[m]["Re{f} (GHz)"], "f", 8)
        add(f"{labels[m]} Q", e1[m]["Q"], e0[m]["Q"], "Q", 5)
        add(f"{labels[m]} junction EPR p", p1[m]["p[3]"], p0[m]["p[3]"], "p", 5)
        add(f"{labels[m]} Q_ext port 1", q1[m]["Q_ext[1]"], q0[m]["Q_ext[1]"], "Qext", 4)
    return out


if __name__ == "__main__":
    run, ref, out = sys.argv[1:4]
    r = rows(run, ref)
    ok = all(x["pass"] for x in r)
    worst_f = max(x["rel_value"] for x in r if "f (GHz)" in x["name"])
    res = {"pass": ok, "reference": "Palace v0.18.1 test/data/regression/ref/transmon/transmon_coarse",
           "tolerances": TOL, "rows": r,
           "note": (f"{'All' if ok else 'NOT all'} {len(r)} quantities within tolerance. "
                    f"Largest eigenfrequency deviation {worst_f:.1e} relative. "
                    "Reference is Palace's own CI regression data for this example.")}
    json.dump(res, open(out, "w"), indent=2)
    for x in r:
        print(f"{'PASS' if x['pass'] else 'FAIL'}  {x['name']:<32} run {x['run']:>14}  ref {x['ref']:>14}  rel {x['rel']}")
    print("VALIDATION", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
