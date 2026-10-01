"""Transmission + storage capacity expansion on the real German grid (PyPSA SciGrid-DE, 24 h).

usage: python expansion.py <scigrid_de.nc> <out_dir> <scenario> [--pf]

A scenario is "w<wind_scale>_s<solar_scale>_c<co2_price>", e.g. w1.5_s1.0_c100.
Always races HiGHS dual simplex against HiGHS interior point in two processes; the first to
finish wins (exclusive lock file) and writes the results; the other is stopped.
--pf    after the linear optimisation, run the full non-linear AC power flow for every hour
        with the optimised dispatch (PyPSA's lopf-then-pf workflow), and export viewer data.
"""
import json, multiprocessing as mp, os, re, sys, time
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


def _racer(path, scn, solver_name, out, do_pf):
    """Build, solve, and if first to finish claim the win (O_EXCL lock) and write the outputs."""
    n = build(path, *parse(scn))
    t0 = time.time()
    status, cond = n.optimize(solver_name="highs", solver_options={"solver": solver_name, "threads": 1})
    dt = time.time() - t0
    try:
        os.close(os.open(out / f".win_{scn}", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return
    rec = {"scenario": scn, "winner": solver_name, "solve_seconds": round(dt, 2), "status": status, "condition": cond}
    rec.update(summarise(n, scn))
    (out / f"expansion_{scn}.json").write_text(json.dumps(rec, indent=1))
    if do_pf and status == "ok":
        export_viewer(n, out, scn)


def race(path, scn, out, do_pf, solvers=("simplex", "ipm")):
    (out / f".win_{scn}").unlink(missing_ok=True)
    procs = {s: mp.Process(target=_racer, args=(path, scn, s, out, do_pf)) for s in solvers}
    t0 = time.time()
    for p in procs.values():
        p.start()
    winner_file = out / f"expansion_{scn}.json"
    winner_file.unlink(missing_ok=True)
    while not winner_file.exists() and any(p.is_alive() for p in procs.values()):
        time.sleep(2)
    t_win = time.time() - t0
    time.sleep(1)
    rec = json.loads(winner_file.read_text()) if winner_file.exists() else {"scenario": scn}
    rec["race"] = []
    for s, p in procs.items():
        if s == rec.get("winner"):
            rec["race"].append({"solver": s, "seconds": rec["solve_seconds"], "result": "won"})
        else:
            if p.is_alive():
                p.terminate()
            rec["race"].append({"solver": s, "seconds": round(t_win, 2), "result": "stopped (lost the race)"})
    for p in procs.values():
        p.join()
    winner_file.write_text(json.dumps(rec, indent=1))
    return rec


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


def export_viewer(n, out, scn):
    """Hourly AC power flow on the optimised dispatch, packed for the three.js viewer."""
    lp_p0 = n.lines_t.p0.copy()                 # linear (LOPF) flows, fallback for non-converged hours
    lp_gen = n.generators_t.p.copy()            # the optimised dispatch (PF only re-balances losses at the slack)
    lp_sto = n.storage_units_t.p.copy()
    n.generators_t.p_set = n.generators_t.p
    n.storage_units_t.p_set = n.storage_units_t.p
    n.lines["s_nom"] = n.lines["s_nom_opt"]
    # PyPSA's SciGrid lopf-then-pf recipe: all generators voltage-controlled (PV), a few PQ units at
    # bus 492 so the Jacobian stays well posed. Q set points are unknown in the dataset.
    n.generators["control"] = "PV"
    n.generators.loc[n.generators.bus == "492", "control"] = "PQ"
    info = n.pf(use_seed=True)
    conv = info["converged"]
    ok = np.asarray(conv).ravel().astype(bool) if conv is not None else np.ones(len(n.snapshots), bool)
    bad = n.snapshots[~ok]
    n.lines_t.p0.loc[bad] = lp_p0.loc[bad]
    n.buses_t.v_mag_pu.loc[bad] = 1.0
    buses = n.buses
    lines = n.lines
    loading = (n.lines_t.p0.abs() / (lines.s_nom * lines.s_max_pu)).round(3).fillna(0)
    carriers = ["Nuclear", "Brown Coal", "Hard Coal", "Gas", "Oil", "Run of River", "Storage Hydro",
                "Wind Offshore", "Wind Onshore", "Solar", "Waste", "Geothermal", "Other", "Multiple"]
    g = lp_gen.T.groupby(n.generators.carrier).sum().T.reindex(columns=carriers, fill_value=0)
    st = lp_sto.T.groupby(n.storage_units.carrier).sum().T
    data = {
        "snapshots": [str(s) for s in n.snapshots],
        "converged": [bool(x) for x in ok],
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
    (out / f"viewer_grid_{scn}.json").write_text(json.dumps(data))


def main():
    path, out, scn = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    rec = race(path, scn, out, "--pf" in sys.argv)
    print(json.dumps({k: v for k, v in rec.items() if k != "generation_twh_by_carrier"}, indent=1))


if __name__ == "__main__":
    main()
