"""Assemble viewer.html: inline the results into viewer_template.html.

usage: python build_viewer.py   (reads results/, writes viewer.html)
"""
import glob, json, math
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).parent
RES = HERE / "results"

SCENARIOS = {  # name -> (button label, tooltip)
    "w1.0_s1.0_c0": ("Base", "2011-01-01 as recorded, no carbon price"),
    "w1.0_s1.0_c100": ("CO₂ €100", "Same weather, carbon price 100 €/t"),
    "w1.5_s1.0_c0": ("Wind ×1.5", "Wind fleet scaled by 1.5, no carbon price"),
    "w2.0_s1.0_c100": ("Wind ×2 + CO₂", "Wind fleet doubled and carbon price 100 €/t: a 2030-like day"),
}
CASE_LABELS = {"case118_ieee": "IEEE 118", "case1354_pegase": "PEGASE 1354", "case2869_pegase": "PEGASE 2869"}
CITIES = {"Hamburg": (9.99, 53.55), "Berlin": (13.40, 52.52), "Munich": (11.58, 48.14),
          "Essen": (7.01, 51.46), "Frankfurt": (8.68, 50.11), "North Sea coast": (8.1, 53.75)}


def nearest_city(lon, lat):
    return min(CITIES, key=lambda c: (CITIES[c][0] - lon) ** 2 + (CITIES[c][1] - lat) ** 2)


def grid_payload():
    scen = {}
    for name, (label, title) in SCENARIOS.items():
        f = RES / "expansion" / f"viewer_grid_{name}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        summ = json.loads((RES / "expansion" / f"expansion_{name}.json").read_text())
        v = np.array(d["v_mag"], dtype=float)
        d["v_span"] = float(max(0.002, math.ceil(np.nanmax(np.abs(v - 1)) * 1000) / 1000))
        d["label"], d["title"], d["summary"] = label, title, summ
        d["date"] = d["snapshots"][0][:10]
        nc = sum(d["converged"])
        d["pf_note"] = (f"Voltages and flows: full AC power flow on the optimised dispatch, converged in {nc}/24 hours"
                        + ("" if nc == 24 else " (others show the linear-optimisation flows, voltage 1.0)") + ".")
        scen[name] = d
    if not scen:
        dev = RES / "expansion" / "viewer_grid_dev.json"
        d = json.loads(dev.read_text())
        v = np.array(d["v_mag"], dtype=float)
        d.update(v_span=float(max(0.002, math.ceil(np.nanmax(np.abs(v - 1)) * 1000) / 1000)), label="Dispatch (dev)",
                 title="development data", summary={"objective_eur": 0}, date=d["snapshots"][0][:10], pf_note="dev data")
        scen["dev"] = d
    default = "w1.0_s1.0_c0" if "w1.0_s1.0_c0" in scen else next(iter(scen))
    return {"scenarios": scen, "default": default, "borders": json.loads((HERE / "expansion" / "borders.json").read_text()),
            "subtitle": "Real German transmission grid (SciGrid-DE, 585 buses), 24 h, transmission + storage expansion with PyPSA and HiGHS",
            "tour": make_tour(scen, default)}


def make_tour(scen, default):
    S = scen[default]
    x, y = np.array(S["buses"]["x"]), np.array(S["buses"]["y"])
    mix = {k: np.array(v) for k, v in S["mix"].items()}
    wind = sum(mix.get(k, 0) for k in ("Wind Onshore", "Wind Offshore"))
    load = np.array(S["load"])
    h_wind = int(np.argmax(wind)) if np.ndim(wind) else 12
    h_load = int(np.argmax(load))
    ld = np.array(S["line_loading"])
    hot_hours = (ld >= 0.98).sum(axis=1)
    li = int(np.argmax(hot_hours))
    b0, b1 = S["lines"]["b0"][li], S["lines"]["b1"][li]
    mid = ((x[b0] + x[b1]) / 2, (y[b0] + y[b1]) / 2)
    tour = [
        {"title": "A real grid, a real winter day", "at": [10.4, 51.0], "dist": 70, "hour": h_wind,
         "text": f"585 buses and 852 lines of the German transmission system on {S['date']}. Peak wind is at "
                 f"{h_wind:02d}:00 with {wind[h_wind]/1000:.1f} GW. Particles show where the power goes; their density and speed follow the MW.",
         "cities": ["Hamburg", "Berlin", "Munich", "Essen", "Frankfurt"]},
        {"title": "North-Sea wind heads south", "at": [8.9, 53.3], "dist": 26, "hour": h_wind,
         "text": f"Offshore and coastal wind ({mix.get('Wind Offshore', np.zeros(24))[h_wind]/1000:.1f} GW offshore at this hour) has to cross the country to the load centres in the west and south.",
         "cities": ["Hamburg", "North Sea coast"]},
        {"title": "The bottleneck", "at": [float(mid[0]), float(mid[1])], "dist": 20,
         "hour": int(np.argmax(ld[li])),
         "text": f"The most congested line, near {nearest_city(*mid)}, sits at its N-1 limit for {int(hot_hours[li])} of 24 hours (pulsing). "
                 f"Congestion is what forces curtailment upstream and expensive generation downstream."},
    ]
    if "lines_expanded" in S["summary"]:
        add = np.array(S["lines"]["s_nom_opt"]) - np.array(S["lines"]["s_nom_min"])
        idx = np.where(add > 1)[0]
        if len(idx):
            cx = float(np.mean([(x[S['lines']['b0'][i]] + x[S['lines']['b1'][i]]) / 2 for i in idx]))
            cy = float(np.mean([(y[S['lines']['b0'][i]] + y[S['lines']['b1'][i]]) / 2 for i in idx]))
            tour.append({"title": "What the optimiser builds", "at": [cx, cy], "dist": 45, "hour": h_wind,
                         "text": f"Glowing teal: {S['summary']['lines_expanded']} line upgrades "
                                 f"({S['summary']['line_expansion_gw_km']:.0f} GW·km) chosen jointly with dispatch to minimise total cost. "
                                 f"Try the other scenarios to see how wind and a carbon price change the plan."})
    tour.append({"title": "Where the load is", "at": [7.2, 51.4], "dist": 24, "hour": h_load,
                 "text": f"Peak load {load[h_load]/1000:.1f} GW at {h_load:02d}:00. The Rhine-Ruhr cluster pulls power in from all sides; bus voltage pillars show the AC power-flow solution.",
                 "cities": ["Essen", "Frankfurt"]})
    return tour


def layout(case, sol):
    cache = RES / "opf" / f"layout_{case}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    g = nx.Graph()
    g.add_nodes_from(sol["bus"].keys())
    g.add_edges_from((b["f"], b["t"]) for b in sol["branch"])
    if len(g) <= 400:
        pos = nx.kamada_kawai_layout(g)
    else:  # ForceAtlas2 keeps large sparse grids spread out (spring layout curls them into a ring)
        pos = nx.forceatlas2_layout(g, max_iter=300, seed=7, scaling_ratio=2.0, gravity=2.0, linlog=False)
    xy = np.array([pos[n] for n in sol["bus"]])
    xy = xy - np.median(xy, 0)
    xy = np.clip(xy / np.percentile(np.abs(xy), 97), -1.15, 1.15)      # robust scale: outliers don't squash the core
    out = {n: [round(float(a), 4), round(float(b), 4)] for n, (a, b) in zip(sol["bus"], xy)}
    cache.write_text(json.dumps(out))
    return out


def opf_payload():
    cases = []
    for f in sorted(glob.glob(str(RES / "opf" / "case*.json"))):
        if f.endswith("_solution.json") or "/layout_" in f:
            continue
        cases.append(json.loads(Path(f).read_text()))
    order = lambda c: (c["case"].count("__"), c["buses"])
    cases.sort(key=order)
    nets = {}
    for case, label in CASE_LABELS.items():
        sf = RES / "opf" / f"{case}_solution.json"
        if not sf.exists():
            continue
        sol = json.loads(sf.read_text())
        pos = layout(case, sol)
        ids = list(sol["bus"].keys()); ix = {b: i for i, b in enumerate(ids)}
        gen = np.zeros(len(ids))
        for g in sol["gen"]:
            gen[ix[g["bus"]]] += max(0.0, g["pg"])
        nets[case] = {"label": label, "nodes": {
            "x": [pos[b][0] for b in ids], "y": [pos[b][1] for b in ids],
            "vm": [sol["bus"][b]["vm"] for b in ids], "vmin": [sol["bus"][b]["vmin"] for b in ids],
            "vmax": [sol["bus"][b]["vmax"] for b in ids], "gen": [round(float(v), 1) for v in gen],
            "load": [sol["load"].get(b, 0.0) for b in ids]},
            "edges": {"a": [ix[b["f"]] for b in sol["branch"]], "b": [ix[b["t"]] for b in sol["branch"]],
                      "p": [b["pf"] for b in sol["branch"]], "loading": [b["loading"] or 0.0 for b in sol["branch"]]}}
    return {"cases": cases, "nets": nets,
            "subtitle": "PGLib-OPF v23.07: AC optimal power flow, Ipopt, three formulations raced, SOC-certified gaps"}


def main():
    stamp = (RES / "stamp.txt").read_text().strip() if (RES / "stamp.txt").exists() else "stamp pending"
    data = {"grid": grid_payload(), "opf": opf_payload(), "stamp": stamp}
    html = (HERE / "viewer_template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (HERE / "viewer.html").write_text(html)
    print(f"viewer.html {len(html)/1e6:.2f} MB; scenarios {list(data['grid']['scenarios'])}; nets {list(data['opf']['nets'])}")


if __name__ == "__main__":
    main()
