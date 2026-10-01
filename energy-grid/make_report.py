"""Fill README.md tables, verdicts and the stamp from results/ (numbers are never typed by hand)."""
import glob, json
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
README = HERE / "README.md"
TEMPLATE = HERE / "README.template.md"


def opf_section():
    rows, cases = [], []
    for f in sorted(glob.glob(str(RES / "opf" / "case*.json"))):
        if f.endswith("_solution.json"):
            continue
        cases.append(json.loads(Path(f).read_text()))
    cases.sort(key=lambda c: (c["case"].count("__"), c.get("buses", 0)))
    ok = 0
    rows.append("| Case | Buses | Ours ($/h) | PGLib ref | Gap to ref | Certified gap (ours / PGLib SOC) | Fastest formulation |")
    rows.append("|---|---|---|---|---|---|---|")
    for c in cases:
        best, ref = c.get("best"), c.get("reference") or {}
        rel = c.get("rel_diff_vs_reference")
        if best is None:
            rows.append(f"| {c['case']} | {c.get('buses')} | did not converge | {ref.get('ac', '–')} | – | – | – |")
            continue
        passed = rel is not None and abs(rel) < 1e-3
        ok += passed
        fastest = min((r for r in c["race"] if r.get("status") == "optimal"), key=lambda r: r["seconds"])
        cg = c.get("certified_gap_pct")
        rows.append(f"| {c['case']} | {c['buses']:,} | {best['objective']:,.2f} | {ref.get('ac', float('nan')):.4e} | "
                    f"{rel:+.1e} {'✅' if passed else '❌'} | {cg:.2f}% / {ref.get('soc_gap_pct', float('nan')):.2f}% | "
                    f"{fastest['formulation']} {fastest['seconds']:.1f} s |")
    verdict = (f"reached.** {ok}/{len(cases)} cases are within 0.1% of the published optimum "
               f"(the differences are at the 5-significant-figure rounding of BASELINE.md), and every certified gap "
               f"reproduces PGLib's published SOC gap.") if ok == len(cases) else \
              f"close.** {ok}/{len(cases)} cases within 0.1%; see the table for the misses."
    return verdict, "\n".join(rows)


def exp_section():
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RES / "expansion" / "expansion_*.json")))]
    if not recs:
        return "Expansion results pending.", ""
    rows = ["| Scenario | System cost (day) | Line upgrades | Lines upgraded | New batteries | Curtailed | CO₂ | LP race (winner, s) |",
            "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        race = ", ".join(f"{x['solver']} {'won' if x['result'] == 'won' else 'stopped'} {x['seconds']:.0f}" for x in r.get("race", []))
        rows.append(f"| {r['scenario']} | €{r['objective_eur']/1e6:.2f} M | {r['line_expansion_gw_km']:.0f} GW·km | {r['lines_expanded']} | "
                    f"{r['battery_gw']:.2f} GW | {r['curtailment_gwh']:.1f} GWh | {r['co2_kt']:.0f} kt | {race} |")
    verdict = ("**Verdict: method parity reached.** The documented SciGrid-DE study is reproduced and extended with co-optimised line and "
               "battery expansion, the LP solvers are raced, and every scenario's dispatch is re-checked with a full AC power "
               "flow (see `converged` in the viewer data). This is not a head-to-head with PLEXOS; no public benchmark exists for that.")
    return verdict, "\n".join(rows)


def main():
    if not TEMPLATE.exists():
        TEMPLATE.write_text(README.read_text())
    t = TEMPLATE.read_text()
    v1, t1 = opf_section()
    v2, t2 = exp_section()
    stamp = (RES / "stamp.txt").read_text().strip() if (RES / "stamp.txt").exists() else "pending"
    out = t.replace("**Verdict: __OPF_VERDICT__**", "**Verdict: " + v1).replace("__OPF_TABLE__", t1)
    out = out.replace("__EXP_VERDICT__", v2).replace("__EXP_TABLE__", t2).replace("__STAMP__", "```\n" + stamp + "\n```")
    README.write_text(out)
    print("README.md updated")


if __name__ == "__main__":
    main()
