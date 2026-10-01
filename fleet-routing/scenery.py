"""City scenery for the viewer: building footprints with heights, classed roads, water.

- Buildings: City of Chicago "Building Footprints" (data.cityofchicago.org, dataset
  syp8-uezg), official footprints with a `stories` field. Height = 3.7 m per storey
  + 2 m. Footprints with no storey count get a deterministic 2-3 storey estimate.
- Major/mid roads and water: OpenStreetMap via a small Overpass QL query.
  (overpass-api.de answers 406 from cloud IPs; the maps.mail.ru mirror works.)
  Minor streets come from the routing graph already in results/city.json.

Coordinates are local metres (x east, y north) around the viewer's origin
(mean of the stops), quantised to 0.5 m and packed as flat integer arrays.

    python scenery.py --out results/scenery.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import requests

CENTER = (41.8810, -87.6320)
RADIUS = 2600
OVERPASS = os.environ.get("OVERPASS_URL", "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
PORTAL = "https://data.cityofchicago.org/resource/syp8-uezg.geojson"


def bbox(pad=0):
    dlat = (RADIUS + pad) / 110574.0
    dlon = (RADIUS + pad) / (111320.0 * math.cos(math.radians(CENTER[0])))
    return CENTER[0] - dlat, CENTER[1] - dlon, CENTER[0] + dlat, CENTER[1] + dlon


def overpass(ql, tries=4):
    for k in range(tries):
        try:
            r = requests.post(OVERPASS, data={"data": ql}, timeout=300,
                              headers={"User-Agent": "qbraid-open-simulation-examples/1.0"})
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(10 * (k + 1))
    raise RuntimeError("Overpass query failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="results/city.json")
    ap.add_argument("--out", default="results/scenery.json")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from shapely.geometry import shape
    from shapely.geometry import MultiPolygon, Polygon

    city = json.load(open(a.city))
    lat0 = sum(c[0] for c in city["coords"]) / len(city["coords"])
    lon0 = sum(c[1] for c in city["coords"]) / len(city["coords"])
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110574.0

    def q(lon, lat):  # local metres, quantised to 0.5 m
        return int(round((lon - lon0) * kx * 2)), int(round((lat - lat0) * ky * 2))

    s, w, n, e = bbox()
    # Buildings (City of Chicago) ---------------------------------------------
    feats, off = [], 0
    while True:
        r = requests.get(PORTAL, timeout=300, params={
            "$where": f"within_box(the_geom, {n}, {w}, {s}, {e}) AND bldg_statu='ACTIVE'",
            "$select": "the_geom,stories,bldg_id", "$limit": 50000, "$offset": off})
        r.raise_for_status()
        page = r.json()["features"]
        feats += page
        if len(page) < 50000:
            break
        off += 50000
    rng = np.random.default_rng(3)
    verts, starts, heights, srcs = [], [], [], []
    n_tag = n_est = 0
    for f in feats:
        if not f.get("geometry"):
            continue
        g = shape(f["geometry"])
        try:
            st = float(f["properties"].get("stories") or 0)
        except ValueError:
            st = 0
        if 0 < st < 150:
            h, src = 3.7 * st + 2.0, 1
        else:
            h, src = 3.7 * float(rng.integers(2, 4)) + 2.0, 0
        for p in (g.geoms if isinstance(g, MultiPolygon) else [g]):
            if not isinstance(p, Polygon) or p.area * kx * ky < 25:
                continue
            p = p.simplify(4e-6, preserve_topology=True)
            ring = list(p.exterior.coords)[:-1]
            if len(ring) < 3:
                continue
            starts.append(len(verts) // 2)
            for lon, lat in ring:
                verts.extend(q(lon, lat))
            heights.append(round(h, 1))
            srcs.append(src)
            n_tag += src
            n_est += 1 - src

    # Roads (OSM, classed) ------------------------------------------------------
    roads = {"major": [], "mid": []}
    try:
        js = overpass(f'[out:json][timeout:180];way[highway~"^(motorway|trunk|primary|secondary|tertiary)(_link)?$"]'
                      f'({s},{w},{n},{e});out geom;')
        for el in js["elements"]:
            hw = el["tags"].get("highway", "")
            cls = "major" if hw.split("_")[0] in ("motorway", "trunk", "primary") else "mid"
            roads[cls].append([c for pt in el["geometry"] for c in q(pt["lon"], pt["lat"])])
    except RuntimeError as ex:
        print("roads: skipped,", ex)

    # Water (OSM): small lakes/river polygons + the Lake Michigan shoreline -------
    water, shore, piers = [], [], []
    try:
        s2, w2, n2, e2 = bbox(3000)
        js = overpass(f'[out:json][timeout:180];way[natural=water]({s2},{w2},{n2},{e2});out geom;')
        for el in js["elements"]:
            pts = [q(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            xs = np.array(pts, dtype=float)
            area = 0.5 * abs(np.dot(xs[:, 0], np.roll(xs[:, 1], 1)) - np.dot(xs[:, 1], np.roll(xs[:, 0], 1))) / 4
            if len(pts) > 3 and area > 2000:
                water.append([c for p in pts for c in p])
        # Lake Michigan is relation 1205149; take its outer ways clipped to the area
        js = overpass(f"[out:json][timeout:240];relation(1205149);out geom({s2},{w2},{n2},{e2});")
        segs = []
        for m in js["elements"][0]["members"]:
            cur = []
            for pt in m.get("geometry") or []:
                if pt is None:
                    if len(cur) > 1:
                        segs.append(cur)
                    cur = []
                else:
                    cur.append((round(pt["lat"], 7), round(pt["lon"], 7)))
            if len(cur) > 1:
                segs.append(cur)
        chains, changed = [s_[:] for s_ in segs], True
        while changed:  # stitch clipped way fragments end to end
            changed = False
            for i in range(len(chains)):
                for j in range(len(chains)):
                    if i == j or not chains[i] or not chains[j]:
                        continue
                    a_, b_ = chains[i], chains[j]
                    if a_[-1] == b_[0]:
                        chains[i], chains[j], changed = a_ + b_[1:], [], True
                    elif a_[-1] == b_[-1]:
                        chains[i], chains[j], changed = a_ + b_[::-1][1:], [], True
            chains = [c_ for c_ in chains if c_]
        chains.sort(key=len, reverse=True)
        if chains:
            c0 = chains[0] if chains[0][0][0] > chains[0][-1][0] else chains[0][::-1]  # north -> south
            shore = [c for lat, lon in c0 for c in q(lon, lat)]
            piers = [[c for lat, lon in ch for c in q(lon, lat)] for ch in chains[1:]
                     if ch[0] == ch[-1] and len(ch) > 8]
    except RuntimeError as ex:
        print("water: skipped,", ex)

    out = {
        "units": "0.5 m, x east / y north, origin = mean of stops",
        "origin": [lat0, lon0],
        "buildings": {"starts": starts, "verts": verts, "h": heights, "src": srcs},
        "roads": roads, "water": water, "shore": shore, "piers": piers,
        "stats": {"buildings": len(heights), "height_from_stories": n_tag, "height_estimated": n_est,
                  "roads": {k: len(v) for k, v in roads.items()}, "water": len(water), "shore_pts": len(shore) // 2, "piers": len(piers),
                  "sources": ["City of Chicago Building Footprints (syp8-uezg)",
                              "OpenStreetMap contributors (ODbL) via Overpass"]},
    }
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(json.dumps(out["stats"]), f"{os.path.getsize(a.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
