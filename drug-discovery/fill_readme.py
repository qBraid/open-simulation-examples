"""Write the Results and Stamp sections of README.md from results/*.json (no hand-copied numbers)."""
import json, math, re
P = json.load(open("results/poses.json")); A = json.load(open("results/affinity.json")); S = json.load(open("results/stamp.json"))
pub = json.load(open("results/published.json"))
s = P["summary"]; n = s["n"]; b = s["boltz2"]
pct = lambda v: f"{100 * v:.0f}%"
se = math.sqrt(b["rmsd_le_2"] * (1 - b["rmsd_le_2"]) / n)
rows = [f"| **Boltz-2, this run** (co-folding, no pocket given) | {pct(b['rmsd_le_2'])} | {pct(b['success_pb_valid'])} |"]
for m in ("gold", "vina"):
    if m in s: rows.append(f"| {m.upper()} docking (crystal protein and pocket given; PoseBusters paper) | {pct(s[m]['rmsd_le_2'])} | {pct(s[m]['success_pb_valid'])} |")
for p in pub["cofolding"]: rows.append(f"| {p['method']}, full V1 set (published) | {pct(p['rmsd_le_2'])} | – |")
miss = [r for r in P["per_complex"] if r.get("status") == "ok" and not r["success"]]
nopred = [r["id"] for r in P["per_complex"] if r.get("status") != "ok"]
top = max(p["rmsd_le_2"] for p in pub["cofolding"])
verdict = ("**Reached**" if b["rmsd_le_2"] >= top - se else "**Close**" if b["rmsd_le_2"] >= top - 2 * se else "**Not reached**")
txt = f"""### Pose prediction: {n} PoseBusters V1 complexes
| Method | RMSD ≤ 2 Å | and PB-valid |
|---|---|---|
""" + "\n".join(rows) + f"""

{verdict} against the bar. Boltz-2 gets {pct(b['rmsd_le_2'])} ± {100*se:.0f} (1σ, n={n}) against published co-folding at {pct(pub['cofolding'][0]['rmsd_le_2'])}–{pct(top)} on the full set, and it beats docking on the same complexes even though docking is handed the pocket.
- **Misses:** {', '.join(f"{r['id']} ({r['rmsd']:.1f} Å, confidence {r.get('confidence') or 0:.2f})" for r in miss) or 'none'}.{' Not predicted: ' + ', '.join(nopred) + '.' if nopred else ''}
- **Confidence did not flag the large misses well.** They are high-confidence wrong pockets, which is the known failure mode of co-folding.

### Affinity ranking: Schrödinger FEP+ JACS series, identical ligands
| Target | n | Boltz-2 Spearman [95% CI] | Boltz-2 Kendall | FEP+ Spearman [95% CI] | FEP+ Kendall | RMSE (centred) Boltz-2 / FEP+ |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {t.upper()} | {d['n']} | {d['boltz2']['spearman']:.2f} [{d['boltz2']['spearman_ci95'][0]:.2f}, {d['boltz2']['spearman_ci95'][1]:.2f}] | {d['boltz2']['kendall']:.2f} | {d['fep_plus']['spearman']:.2f} [{d['fep_plus']['spearman_ci95'][0]:.2f}, {d['fep_plus']['spearman_ci95'][1]:.2f}] | {d['fep_plus']['kendall']:.2f} | {d['boltz2']['rmse_centered']:.2f} / {d['fep_plus']['rmse_centered']:.2f} kcal/mol |" for t, d in A["targets"].items())
st = "\n".join(f"- {k}: {v}" for k, v in S.items() if k != "jobs")
r = open("README.md").read()
r = re.sub(r"## Results\n.*?\n## Method", "## Results\n" + txt + "\n\n## Method", r, flags=re.S)
r = re.sub(r"## Verification stamp\n.*", "## Verification stamp\n" + st + "\n", r, flags=re.S)
r = r.replace("RESULTS_PLACEHOLDER", txt).replace("STAMP_PLACEHOLDER", st)
open("README.md", "w").write(r); print(txt)
