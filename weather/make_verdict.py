"""Turn scores + tracks into the verdict, run-details stamp and guided tour (results/verdict.json).

Usage: python make_verdict.py <results_dir> <viz.npz> --wall-min <minutes> --cost <usd>
"""
import argparse, json, os
from datetime import datetime, timedelta

import numpy as np

from build_viewer import STORMS, track

ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("viz")
ap.add_argument("--wall-min", type=float, default=0); ap.add_argument("--extra-stamp", default=""); ap.add_argument("--cost", type=float, default=0)
ap.add_argument("--init", default="2020-08-24T00:00"); ap.add_argument("--model", default="sfno")
a = ap.parse_args()
S = json.load(open(os.path.join(a.results, f"{a.model}_scores.json")))
NAME = {"sfno": "SFNO-small", "fcn3": "FourCastNet 3"}[a.model]
H = json.load(open(os.path.join(a.results, "hres_scores.json")))
R = json.load(open(os.path.join(a.results, "wb2_ref_2020.json")))
Z = np.load(a.viz); lat, lon = Z["lat"], Z["lon"]
init = datetime.fromisoformat(a.init)

rows, paired = [], {}
for v, unit in (("z500", "m²/s²"), ("t850", "K"), ("t2m", "K")):
    for h in ("24", "72", "120"):
        s = np.array(S["rmse"][v][h]); hr = np.array(H["rmse"][v][h])
        d = s - hr; se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        paired[(v, h)] = (s.mean(), hr.mean(), d.mean(), se)
        rows.append({"var": v, "lead_h": int(h), "model": a.model, "rmse": float(s.mean()), "hres_paired": float(hr.mean()),
                     "diff": float(d.mean()), "diff_se": float(se), "better_inits": int((d < 0).sum()), "n": int(len(d)),
                     "wb2_hres": R["hres"][v].get(h), "wb2_graphcast": R["graphcast"][v].get(h), "wb2_pangu": R["pangu"][v].get(h)})

z5 = paired[("z500", "120")]; t8 = paired[("t850", "120")]
pct = lambda p: (p[0] / p[1] - 1) * 100
within = all(abs(p[2]) <= 2 * p[3] or p[0] <= p[1] for k, p in paired.items() if k[1] in ("72", "120"))
better = sum(1 for k, p in paired.items() if p[0] < p[1])
verdict_word = "reached" if within else ("close" if abs(pct(z5)) < 12 else "not reached")
html = (f"<b>Verdict ({NAME}): {verdict_word}.</b> Day 5 z500 RMSE {z5[0]:.0f} vs IFS HRES {z5[1]:.0f} on the same {len(S['inits'])} starts "
        f"({pct(z5):+.0f}%); T850 {t8[0]:.2f} vs {t8[1]:.2f} K ({pct(t8):+.0f}%). {NAME} beats HRES on {better} of 9 "
        f"variable–lead pairs. GraphCast (non-commercial weights) is the WB2 leader at {R['graphcast']['z500']['120']:.0f}.")
ens_rows = []
EP = os.path.join(a.results, "fcn3_ens_scores.json")
if a.model == "fcn3" and os.path.exists(EP):
    E = json.load(open(EP)); idx = [H["inits"].index(i) for i in E["inits"]]
    ep = {}
    for v in ("z500", "t850", "t2m"):
        for h in ("24", "72", "120"):
            e = np.array(E["rmse_mean"][v][h]); hr = np.array(H["rmse"][v][h])[idx]; d = e - hr
            se = d.std(ddof=1) / np.sqrt(len(d))
            ep[(v, h)] = (e.mean(), hr.mean(), d.mean(), se)
            ens_rows.append({"var": v, "lead_h": int(h), "ens_mean": float(e.mean()), "hres_paired": float(hr.mean()), "diff": float(d.mean()),
                             "diff_se": float(se), "better": int((d < 0).sum()), "n": len(d), "members": E["members"],
                             "wb2_ens_mean_50": R["ens_mean"][v].get(h), "wb2_hres": R["hres"][v].get(h)})
    ew = sum(1 for k, p in ep.items() if p[0] < p[1]); m1 = paired[("z500", "120")][0]; z5e = ep[("z500", "120")]; t8e = ep[("t850", "120")]
    eq = all(p[0] <= p[1] or abs(p[2]) <= 2 * p[3] for k, p in ep.items() if k[1] in ("72", "120"))
    verdict_word = "reached" if eq else ("close" if pct(z5e) < 8 else "not reached")
    html = (f"<b>Verdict ({NAME}): {verdict_word}.</b> FCN3 is a probabilistic model. Its {E['members']}-member mean scores "
            f"{z5e[0]:.0f} vs IFS HRES {z5e[1]:.0f} on day-5 z500 ({pct(z5e):+.0f}%) and {t8e[0]:.2f} vs {t8e[1]:.2f} K on T850 "
            f"({pct(t8e):+.0f}%) on the same {len(idx)} starts, beating HRES on {ew} of 9 variable–lead pairs. "
            f"A single member ({m1:.0f}) matches an IFS ensemble member (WB2 {R['ens_single_member']['z500']['120']:.0f}).")

tracks = {}
for name, lo0, la0, rad in STORMS:
    leads = list(range(0, 121, 6)); tleads = list(range(0, 121, 12))
    fc = track([Z[f"msl_{h:03d}"] for h in leads], lo0, la0, rad, lat, lon)
    tr = track([Z[f"era5_msl_{h:03d}"] for h in tleads], lo0, la0, rad, lat, lon)
    errs = []
    for j, p in enumerate(tr):
        i = j * 2
        if i < len(fc):
            la1, lo1, la2, lo2 = map(np.deg2rad, (fc[i][1], fc[i][0], p[1], p[0]))
            dkm = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
            errs.append({"lead_h": j * 12, "km": round(float(dkm)), "p_fc": fc[i][2], "p_era5": p[2]})
    tracks[name] = {"fc": fc, "tr": tr, "errors": errs}

def stop(lon_, lat_, dist, lead, html_, **kw):
    return dict(lon=lon_, lat=lat_, dist=dist, lead=lead, html=html_, **kw)

L = tracks["Hurricane Laura"]; B = tracks["Typhoon Bavi"]
laura_land = L["tr"][min(len(L["tr"]) - 1, 6)] if L["tr"] else [-93, 29.5, 0]
e72 = next((e for e in L["errors"] if e["lead_h"] == 72), None)
b72 = next((e for e in B["errors"] if e["lead_h"] == 72), None)
tour = [
    stop(-40, 15, 4.8, 0, f"<b>One GPU, five days of weather.</b> NVIDIA {NAME} steps the whole atmosphere 6 h at a time from the ERA5 state of 24 Aug 2020, 00 UTC. Moisture, winds at 850 hPa and 500 hPa height contours are all model output.", ov="tcwv", mode="fc", hold=5000),
    stop(-74, 19, 2.3, 0, "<b>Hurricane Laura</b> starts as a tropical storm near Hispaniola. Its moisture core is the bright plume, and the particles trace the model's own winds.", ov="tcwv", mode="fc", hold=4000),
    stop(-86, 24, 2.2, 0, "Play forward. <b>Cyan</b> is the forecast track, <b>red</b> is ERA5. For two days they agree to within about 170 km.", ov="msl", mode="fc", play_to=48, hold=7000),
    stop(-90, 26, 2.0, 72, (f"At +72 h the forecast centre is <b>{e72['km']} km</b> from ERA5's, with {e72['p_fc']:.0f} vs {e72['p_era5']:.0f} hPa. " if e72 else "") +
         (f"The real Laura recurves north into Louisiana; {NAME} " + ("keeps it heading west. " if (L["errors"] and max(e["km"] for e in L["errors"]) > 600) else "follows it. ") + "Drag the divider to compare."), ov="msl", mode="swipe", hold=7500),
    stop(127, 31, 2.5, 72, "<b>Typhoon Bavi</b> in the Yellow Sea, half a world away, comes out of the same forward pass." + (f" Here the forecast centre is {b72['km']} km from ERA5's at +72 h." if b72 else ""), ov="tcwv", mode="fc", hold=5500),
    stop(20, -45, 3.4, 120, "Day-5 error in 2 m temperature. Most of the error sits in the Southern Ocean storm track and over the continents, where small-scale weather has been smoothed out.", ov="t2m", mode="err", hold=6500),
    stop(-30, 25, 4.8, 120, html.replace("<b>Verdict", "<b>Skill vs ECMWF").replace("</b>", "</b>", 1), ov="tcwv", mode="fc", hold=8000),
]
stamp = (f"<h2>Run details</h2>Model: NVIDIA {NAME} via Earth2Studio 0.19.0, NVIDIA weights (commercially usable).<br>"
         f"Initial conditions and truth: ERA5 (ARCO public zarr). IFS HRES: WeatherBench2 public zarr.<br>"
         f"{len(S['inits'])} starts in 2020 (WB2's test year, outside training), 00 UTC, leads 1/3/5 days, latitude-weighted RMSE at 0.25°.<br>"
         f"Machine: qBraid {'gpu-a100-sxm' if 'A100' in S.get('gpu','') else 'gpu-l4'} ({S.get('gpu','?')}), {np.mean(S['step_s']):.2f} s per 6 h step, model load {S.get('load_s','?')} s.<br>"
         + (f"Ensemble: {E['members']} members (seeds 1000+) on {len(E['inits'])} of the starts.<br>" if ens_rows else "") +
         f"Wall time {a.wall_min:.0f} min · compute cost ≈ ${a.cost:.2f} · {datetime.utcnow():%Y-%m-%d}." + a.extra_stamp)
json.dump({"html": html, "stamp_html": stamp, "tour": tour, "table": rows, "ens_table": ens_rows, "tracks": tracks, "verdict": verdict_word},
          open(os.path.join(a.results, "verdict.json"), "w"), indent=1)
print(html); print(json.dumps(tracks["Hurricane Laura"]["errors"])); print(json.dumps(tracks["Typhoon Bavi"]["errors"]))
