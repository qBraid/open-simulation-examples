"""Fill README_template.md with tables generated from the result files -> README.md."""
import json
from pathlib import Path

R = Path(__file__).parent / "results"
b = json.loads((R / "benchmarks.json").read_text())
v = json.loads((R / "validation.json").read_text())
t = (Path(__file__).parent / "README_template.md").read_text()


def row2d(k, x):
    if x.get("status", "complete") != "complete":
        return (f"| {k} | {x['compliance_python']:.4f} | stopped by cap at it. {x['compared_iterations']} | "
                f"{x['rel_diff_pct']:+.1e} % at it. {x['compared_iterations']} | – | {x['iterations_python']} / ≥{x['iterations_reference']} | "
                f"{x['identical_until_iter']} |")
    return (f"| {k} | {x['compliance_python']:.4f} | {x['compliance_reference']:.4f} | {x['rel_diff_pct']:+.1e} % | "
            f"{x['layout_max_abs_diff']:.1e} | {x['iterations_python']} / {x['iterations_reference']} | {x['identical_until_iter']} |")


hdr = "| case | compliance, Python | compliance, published code | Δ | max abs layout difference | iterations | identical until it. |\n|---|---|---|---|---|---|---|\n"
t = t.replace("RESULTS_TOP88", hdr + "\n".join(row2d(k, x) for k, x in b.items() if k.startswith("top88")))
t = t.replace("RESULTS_TOP3D", hdr + "\n".join(row2d(k, x) for k, x in b.items() if k.startswith("top3d")) +
              "\n\n`top3d_60x20x4` stops on the code's own rule (`change < 0.01`). The reference never meets that rule,"
              " because the OC update oscillates around 0.016, so it stops at its 200-iteration cap. `top3d_60x20x4_200it`"
              " forces the port to the same 200 iterations for a like-for-like comparison.")
c = v["continuum_3d"]
val = ("| quantity | value |\n|---|---|\n"
       f"| Euler–Bernoulli | {v['analytic']['euler_bernoulli']:.1f} |\n| Timoshenko (Cowper κ = {v['analytic']['kappa']:.4f}) | {v['analytic']['timoshenko']:.1f} |\n"
       f"| 3D continuum (P2 tetrahedra, extrapolated, order {c['observed_order']:.2f}) | {c['deflection']:.1f} ({c['vs_timoshenko_pct']:+.2f} % vs Timoshenko) |\n"
       f"| H8 Richardson (order {v['h8_richardson']['observed_order']:.2f}) | {v['h8_richardson']['extrapolated']:.1f} ({v['h8_richardson']['err_vs_continuum_pct']:+.2f} % vs continuum) |\n\n"
       "| H8 bricks through depth | unknowns | tip deflection | error vs 3D continuum |\n|---|---|---|---|\n" +
       "\n".join(f"| {r['n']} | {r['ndof']:,} | {r['deflection']:.1f} | {r['err_vs_continuum_pct']:+.2f} % |" for r in v["h8"]))
t = t.replace("RESULTS_VALIDATION", val)
tag = "" if (R / "bracket.json").exists() else "_2mm"
s = json.loads((R / f"bracket{tag}.json").read_text())
st = json.loads((R / "stl.json").read_text()) if (R / "stl.json").exists() else {}
br = (f"Design space {' × '.join(f'{d:g}' for d in s['design_space_mm'])} mm, {s['brick_mm']} mm bricks ({s['elements']:,} elements, {s['dofs']:,} unknowns), "
      f"Ti-6Al-4V, {s['volfrac'] * 100:.0f} % of the volume, three load cases on the pin minimised together, Heaviside projection to β = 16.\n\n"
      "| quantity | value |\n|---|---|\n"
      f"| mass | {s['mass_g']} g (solid block {s['full_block_mass_g']} g) |\n"
      f"| design iterations / wall time | {s['iterations']} / {s['seconds'] / 60:.1f} min |\n"
      f"| grey (0.1 < ρ < 0.9) elements | {s['grey_fraction'] * 100:.2f} % |\n"
      f"| von Mises p99 in the body | {s['vm_p99_MPa']} MPa (max {s['vm_max_body_MPa']} MPa) |\n"
      f"| linear-elastic safety factor vs 880 MPa yield (p99) | {s['safety_factor_p99']} |\n" +
      "".join(f"| compliance {k} | {c_:.1f} N·mm |\n" for k, c_ in s["compliance_Nmm"].items()) +
      "".join(f"| max displacement {k} | {d_:.4f} mm |\n" for k, d_ in s["tip_disp_mm"].items()) +
      (f"| printable STL | `{st.get('file')}`: {st['triangles']:,} triangles, watertight: {st['watertight']}, mesh mass {st['mass_g_ti64']} g |\n" if st else ""))
t = t.replace("RESULTS_BRACKET", br)
stamp = json.loads((R / "stamp.json").read_text()) if (R / "stamp.json").exists() else None
t = t.replace("STAMP", "**Verification stamp.** " + "; ".join(f"{k}: {x}" for k, x in stamp.items()) if stamp else "")
(Path(__file__).parent / "README.md").write_text(t)
print("README.md written")
