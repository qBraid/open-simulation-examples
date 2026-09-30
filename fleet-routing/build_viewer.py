"""Build viewer.html: inline the results into a self-contained three.js page.

    python build_viewer.py            # reads results/*.json, writes viewer.html
"""

from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "results", name)) as f:
        return json.load(f)


def main():
    city = load("city.json")
    qaoa = load("qaoa.json")
    anchors = [load(f) for f in ("e-n22-k4.json", "a-n32-k5.json", "x-n101-k25.json")]

    lat0 = sum(c[0] for c in city["coords"]) / len(city["coords"])
    lon0 = sum(c[1] for c in city["coords"]) / len(city["coords"])
    kx = 111.320 * math.cos(math.radians(lat0))
    ky = 110.574

    def xy(p):  # (lat, lon) -> local km, x east, y north
        return [round((p[1] - lon0) * kx, 4), round((p[0] - lat0) * ky, 4)]

    solvers = {}
    for name, r in city["results"].items():
        routes = []
        for rt, geom in zip(r.get("routes") or [], city["geometry"].get(name, [])):
            routes.append({
                "stops": rt,
                "load": int(sum(city["demand"][i] for i in rt)),
                "m": int(sum(city["dist"][a][b] for a, b in zip([0, *rt], [*rt, 0]))),
                "path": [xy(p) for p in geom],
            })
        solvers[name] = {
            "cost": r.get("cost"),
            "lower_bound": r.get("lower_bound"),
            "trace": r.get("trace", []),
            "routes": routes,
        }

    data = {
        "city": city["meta"]["city"],
        "stamp": city["stamp"],
        "meta": city["meta"],
        "capacity": city["meta"]["capacity"],
        "stops": [xy(c) for c in city["coords"]],
        "demand": city["demand"],
        "roads": [[xy(p) for p in seg] for seg in city["roads"]],
        "solvers": solvers,
        "winner": city["winner"],
        "best": city["best_cost"],
        "lower_bound": city["lower_bound"],
        "gap": city.get("certified_gap_pct"),
        "anchors": [{
            "instance": a["instance"], "n": a["n_customers"], "bks": a["bks"],
            "best": a["best_cost"], "winner": a["winner"],
            "lower_bound": a["lower_bound"], "proven": a.get("proven_optimal", False),
            "gap": a.get("certified_gap_pct"), "time": a["stamp"]["time_limit_s"],
            "table": a["table"],
        } for a in anchors],
        "qaoa": qaoa,
    }
    with open(os.path.join(HERE, "viewer_template.html")) as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = os.path.join(HERE, "viewer.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
