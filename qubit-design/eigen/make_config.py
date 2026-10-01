"""Turn DeviceLayout's eigenmode config into the run config for the final design.
usage: python make_config.py eig_base.json out.json <L_J nH> <C_J fF> <target GHz> [amr_its]"""
import json, sys
c = json.load(open(sys.argv[1])); LJ, CJ, tgt = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
amr = int(sys.argv[6]) if len(sys.argv) > 6 else 0
for p in c["Boundaries"]["LumpedPort"]:
    if "L" in p:
        p["L"], p["C"] = LJ * 1e-9, CJ * 1e-15
c["Solver"]["Eigenmode"].update({"N": 2, "Save": 2, "Target": tgt, "Tol": 1e-8})
c["Solver"]["Linear"]["Tol"] = 1e-10
c["Problem"]["OutputFormats"] = {"Paraview": True, "GridFunction": False}
c["Model"]["Refinement"] = {"MaxIts": amr, "Tol": 1e-3, "UpdateFraction": 0.7, "SaveAdaptIterations": False} if amr else {"MaxIts": 0}
if amr: c["Problem"]["Output"] += f"_amr{amr}"
json.dump(c, open(sys.argv[2], "w"), indent=1)
print(json.dumps([p for p in c["Boundaries"]["LumpedPort"] if "L" in p]))
