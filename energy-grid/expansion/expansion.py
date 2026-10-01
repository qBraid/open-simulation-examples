"""Transmission + storage capacity expansion on the real German grid (PyPSA SciGrid-DE, 24 h).

usage: python expansion.py <scigrid_de.nc> <out_dir> <scenario> [--race] [--pf]

A scenario is "w<wind_scale>_s<solar_scale>_c<co2_price>", e.g. w1.5_s1.0_c100.
--race  solve with HiGHS dual simplex and HiGHS interior point in parallel processes,
        keep the first to finish (both times are recorded).
--pf    after the linear optimisation, run the full non-linear AC power flow for every hour
        with the optimised dispatch (PyPSA's lopf-then-pf workflow), and export viewer data.
"""
import json, multiprocessing as mp, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

# t CO2 per MWh of electricity (typical fleet values), used with a CO2 price.
CO2 = {"Brown Coal": 1.10, "Hard Coal": 0.85, "Gas": 0.40, "Oil": 0.75}
LINE_COST = 400.0      # EUR per MW per km per year, annualised overhead line upgrade
BATTERY_COST = 120e3   # EUR per MW per year for a 4 h battery (power + energy, annualised)
HOURS = 24


def build(path, wind, solar, co2):
    n = pypsa.Network(path)
    # The standard SciGrid-DE preparation (PyPSA example): N-1 security margin, and the
    # three line upgrades the dataset needs to be feasible.
    n.lines["s_max_pu"] = 0.7
    n.lines.loc[["316", "527", "602"], "s_nom"] = 1715
    for car, k in (("Wind Onshore", wind), ("Wind Offshore", wind), ("Solar", solar)):
        n.generators.loc[n.generators.carrier == car, "p_nom"] *= k
    for car, f in CO2.items():
        n.generators.loc[n.generators.carrier == car, "marginal_cost"] += co2 * f
    scale = HOURS / 8760.0  # one representative day: annual costs scaled to the horizon
    n.lines["s_nom_min"] = n.lines["s_nom"]
    n.lines["s_nom_extendable"] = True
    n.lines["capital_cost"] = LINE_COST * n.lines["length"] * scale
    buses = n.buses.index[n.buses.v_nom >= 220]
    n.add("StorageUnit", "battery " + buses, bus=buses, carrier="battery", p_nom_extendable=True,
          max_hours=4, efficiency_store=0.95, efficiency_dispatch=0.95, cyclic_state_of_charge=True,
          capital_cost=BATTERY_COST * scale)
    return n


def parse(scn):
    w, s, c = re.fullmatch(r"w([\d.]+)_s([\d.]+)_c([\d.]+)", scn).groups()
    return float(w), float(s), float(c)


def _solve(args):
    path, scn, solver_name, q = args
    n = build(path, *parse(scn))
    t0 = time.time()
    status, cond = n.optimize(solver_name="highs", solver_options={"solver": solver_name, "threads": 1,
                                                                   "run_crossover": "on"})
    q.put((solver_name, time.time() - t0, status, cond, n.objective))


def race(path, scn):
    q = mp.Manager().Queue()
    procs = {s: mp.Process(target=_solve, args=((path, scn, s, q),)) for s in ("simplex", "ipm")}
    for p in procs.values():
        p.start()
    t0, results = time.time(), []
    while len(results) < 2 and time.time() - t0 < 900:
        try:
            results.append(q.get(timeout=5))
        except Exception:
            pass
    for p in procs.values():
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()
    rows = [{"solver": r[0], "seconds": round(r[1], 2), "status": r[2], "condition": r[3], "objective": r[4]}
            for r in results]
    ok = [r for r in rows if r["status"] == "ok"]
    return rows, (min(ok, key=lambda r: r["seconds"]) if ok else None)


def summarise(n, scn):
    gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().T
    avail = (n.generators_t.p_max_pu * n.generators.p_nom).reindex(columns=n.generators.index).fillna(
        n.generators.p_nom * n.generators.p_max_pu)
    renew = n.generators.carrier.isin(["Wind Onshore", "Wind Offshore", "Solar"])
    curtail = float((avail.loc[:, renew] - n.generators_t.p.loc[:, renew]).clip(lower=0).sum().sum())
    emis = sum(float(gen.get(c, pd.Series(0)).sum()) * f for c, f in CO2.items())
    ext = n.lines.s_nom_opt - n.lines.s_nom_min
    return {"scenario": scn, "objective_eur": float(n.objective),
            "line_expansion_gw_km": float((ext * n.lines.length).sum() / 1e3),
            "lines_expanded": int((ext > 1).sum()),
            "battery_gw": float(n.storage_units.p_nom_opt[n.storage_units.carrier == "battery"].sum() / 1e3),
            "curtailment_gwh": curtail / 1e3, "co2_kt": emis / 1e3,
            "generation_twh_by_carrier": {k: float(v.sum() / 1e6) for k, v in gen.items()}}


def export_viewer(n, out):
    """Hourly AC power flow on the optimised dispatch, packed for the three.js viewer."""
    for c in n.iterate_components(["Generator", "StorageUnit"]):
        pass
    n.generators_t.p_set = n.generators_t.p
    n.storage_units_t.p_set = n.storage_units_t.p
    n.lines["s_nom"] = n.lines["s_nom_opt"]
    info = n.pf(use_seed=True)
    conv = info["converged"]
    buses = n.buses
    lines = n.lines
    loading = (n.lines_t.p0.abs() / (lines.s_nom * lines.s_max_pu)).round(3)
    carriers = ["Nuclear", "Brown Coal", "Hard Coal", "Gas", "Oil", "Run of River", "Storage Hydro",
                "Wind Offshore", "Wind Onshore", "Solar", "Waste", "Geothermal", "Other", "Multiple"]
    g = n.generators_t.p.T.groupby(n.generators.carrier).sum().T.reindex(columns=carriers, fill_value=0)
    st = n.storage_units_t.p.T.groupby(n.storage_units.carrier).sum().T
    data = {
        "snapshots": [str(s) for s in n.snapshots],
        "converged": [bool(x) for x in np.asarray(conv).ravel()] if conv is not None else None,
        "buses": {"id": list(buses.index), "x": buses.x.round(4).tolist(), "y": buses.y.round(4).tolist(),
                  "vnom": buses.v_nom.tolist()},
        "v_mag": n.buses_t.v_mag_pu.round(4).T.values.tolist(),
        "lines": {"b0": [buses.index.get_loc(b) for b in lines.bus0], "b1": [buses.index.get_loc(b) for b in lines.bus1],
                  "s_nom_min": lines.s_nom_min.round(1).tolist(), "s_nom_opt": lines.s_nom_opt.round(1).tolist()},
        "line_p": n.lines_t.p0.round(1).T.values.tolist(),
        "line_loading": loading.T.values.tolist(),
        "mix": {c: g[c].round(1).tolist() for c in carriers if g[c].abs().sum() > 0},
        "storage": {c: st[c].round(1).tolist() for c in st.columns},
        "load": n.loads_t.p.sum(axis=1).round(1).tolist(),
        "battery": {"bus": [buses.index.get_loc(b) for b in n.storage_units.bus[n.storage_units.carrier == "battery"]],
                    "p_nom": n.storage_units.p_nom_opt[n.storage_units.carrier == "battery"].round(1).tolist()},
    }
    (out / "viewer_grid.json").write_text(json.dumps(data))


def main():
    path, out, scn = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    rec = {"scenario": scn}
    if "--race" in sys.argv:
        rec["race"], rec["winner"] = race(path, scn)
    n = build(path, *parse(scn))
    t0 = time.time()
    n.optimize(solver_name="highs", solver_options={"solver": (rec.get("winner") or {}).get("solver", "simplex"),
                                                    "threads": 1})
    rec["solve_seconds"] = round(time.time() - t0, 2)
    rec.update(summarise(n, scn))
    (out / f"expansion_{scn}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "generation_twh_by_carrier"}, indent=1))
    if "--pf" in sys.argv:
        export_viewer(n, out)


if __name__ == "__main__":
    main()
