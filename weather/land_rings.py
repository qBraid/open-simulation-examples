"""Decode Natural Earth land TopoJSON (world-atlas, public domain) into lon/lat rings
for the globe's base map. Output: int16 arrays (degrees * 100) and ring lengths."""
import json
import numpy as np


def decode(path):
    t = json.load(open(path))
    sx, sy = t["transform"]["scale"]; tx, ty = t["transform"]["translate"]
    arcs = []
    for arc in t["arcs"]:
        x = y = 0; pts = []
        for dx, dy in arc:
            x += dx; y += dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)

    def ring(idx):
        out = []
        for i in idx:
            pts = arcs[i] if i >= 0 else arcs[~i][::-1]
            out.extend(pts if not out else pts[1:])
        return out

    rings = []
    for g in t["objects"]["land"]["geometries"]:
        polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
        for poly in polys:
            for r in poly:
                rings.append(ring(r))
    return rings


def pack(path):
    rings = decode(path)
    lens = np.array([len(r) for r in rings], dtype=np.int32)
    pts = np.round(np.array([p for r in rings for p in r]) * 100).astype(np.int16)
    return pts, lens


if __name__ == "__main__":
    import sys
    pts, lens = pack(sys.argv[1])
    print(len(lens), "rings", len(pts), "points", pts.nbytes, "bytes")
