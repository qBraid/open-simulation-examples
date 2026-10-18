"""Build viewer.html for the SandboxAQ comparison. Pure Python, no chemistry stack needed.

Starts from the data inlined in ../drug-discovery/viewer.html (protein ribbons, pocket
surfaces, crystal and Boltz-2 ligands, the ligand chemical-space layout), then adds the
OpenFold3 poses and the AQAffinity and AQPotency affinities produced here.
usage: build_viewer.py   (run from this directory after the eval scripts)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = f"{HERE}/results"
src = open(f"{HERE}/../drug-discovery/viewer.html").read()
base, _ = json.JSONDecoder().raw_decode(src[src.index('{"stamp"'):])

of3 = json.load(open(f"{R}/poses_openfold3.json"))
bz = json.load(open(f"{HERE}/../drug-discovery/results/poses.json"))
show = json.load(open(f"{R}/of3_showcase.json"))
orow = {r["id"]: r for r in of3["per_complex"]}; brow = {r["id"]: r for r in bz["per_complex"]}
notes = {"8G43_ZU6": "Both models land on the crystal pose.",
         "7BLG_GAL": "Both models succeed on this sugar ligand.",
         "8BRO_R7E": "Both models miss: a confident wrong pose, the known co-folding failure mode."}
pick = lambda r: {k: r.get(k) for k in ("rmsd", "success", "pb_valid", "pb_failed", "confidence")}
complexes = []
for c in base["complexes"]:
    c = dict(c, of3=show.get(c["id"]), boltz=pick(brow[c["id"]]), of3m=pick(orow[c["id"]]), note=notes.get(c["id"], ""))
    complexes.append(c)
per_complex = [dict(id=i, b_rmsd=brow[i]["rmsd"], b_ok=brow[i]["success"], o_rmsd=orow[i]["rmsd"], o_ok=orow[i]["success"]) for i in sorted(orow)]
bench = dict(bz["summary"], openfold3=of3["summary"]["boltz2"])  # eval_poses.py labels its own model "boltz2"

aqa = json.load(open(f"{R}/aqaffinity.json")) if os.path.exists(f"{R}/aqaffinity.json") else None
aqp = json.load(open(f"{R}/aqpotency.json"))
run = (aqa or {}).get("primary_run")
ligs = []
for l in base["affinity"]["ligands"]:
    t, n = l["t"], l["name"]
    e = dict(l)
    e["aqp"] = next(x["aqpotency"] for x in aqp["targets"][t]["ligands"] if x["name"] == n)
    if aqa and run:
        x = next(x for x in aqa["targets"][t]["ligands"] if x["name"] == n)
        e["aqa"] = x.get(run)
    ligs.append(e)
metrics = {}
for t, m in base["affinity"]["metrics"].items():
    metrics[t] = dict(n=m["n"], exp_range_kcal=m["exp_range_kcal"], fep=m["fep_plus"], boltz=m["boltz2"], aqp=aqp["targets"][t]["aqpotency"])
    if aqa and run and run in aqa["targets"][t]["metrics"]: metrics[t]["aqa"] = aqa["targets"][t]["metrics"][run]["top"]
methods = [dict(k="fep", label="FEP+", note="Schrödinger's commercial physics method, hours of GPU per ligand pair; values from their public benchmark."),
           dict(k="boltz", label="Boltz-2", note="Open co-folding model with an affinity head, minutes per ligand on one L4 (the drug-discovery example)."),
           dict(k="aqa", label="AQAffinity", note="SandboxAQ's open (Apache-2.0) replication of the Boltz-2 affinity head on OpenFold3, run here on one A100."),
           dict(k="aqp", label="AQPotency", note="SandboxAQ's hosted proteochemometric model, called through their MCP server: UniProt ID plus SMILES, seconds per batch, about $0.11 per 16 ligands.")]
if not (aqa and run): methods = [m for m in methods if m["k"] != "aqa"]
data = dict(stamp=json.load(open(f"{R}/stamp.json")), bench=bench, published=base["published"], complexes=complexes, per_complex=per_complex,
            affinity=dict(ligands=ligs, metrics=metrics, methods=methods, seen={t: aqp["targets"][t]["seen_in_training"] for t in aqp["targets"]},
                          caveat="AQPotency reports 15 of 16 ligands per series as already in its training data (ligand_similarity 1.0), so its numbers here measure recall of known data. The series are also public in ChEMBL, which Boltz-2 and AQAffinity were trained on."))
html = open(f"{HERE}/viewer_template.html").read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
open(f"{HERE}/viewer.html", "w").write(html)
print("viewer.html", round(len(html) / 1e6, 2), "MB")
