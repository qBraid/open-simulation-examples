"""Put OpenFold3 predictions in the layout eval_poses.py reads.

For each query, the top sample by OpenFold3's own sample_ranking_score is
written as <id>_model_0.pdb (ligand stays chain L), with its aggregated
confidences as confidence_<id>_model_0.json. Selection uses only the model's
confidence, never the crystal, exactly as Boltz-2's model_0 was used.
"""
import glob, json, os, sys
import gemmi

SRC, DST = sys.argv[1:3]
os.makedirs(DST, exist_ok=True)
for qdir in sorted(glob.glob(f"{SRC}/*/")):
    i = os.path.basename(qdir.rstrip("/"))
    best = None
    for agg in glob.glob(f"{qdir}/**/*_confidences_aggregated.json", recursive=True):
        a = json.load(open(agg))
        s = a.get("sample_ranking_score", a.get("ranking_score"))
        if s is not None and (best is None or s > best[0]): best = (s, agg, a)
    if best is None: print("no samples", i); continue
    cif = best[1].replace("_confidences_aggregated.json", "_model.cif")
    st = gemmi.read_structure(cif); st.setup_entities()
    st.write_pdb(f"{DST}/{i}_model_0.pdb")
    a = best[2]
    json.dump(dict(confidence_score=a.get("sample_ranking_score"), iptm=a.get("iptm"),
                   ligand_iptm=a.get("chain_pair_iptm", {}).get("(A, L)") if isinstance(a.get("chain_pair_iptm"), dict) else None, raw=a),
              open(f"{DST}/confidence_{i}_model_0.json", "w"), indent=1)
    print(i, os.path.basename(cif), "ranking %.3f" % best[0])
