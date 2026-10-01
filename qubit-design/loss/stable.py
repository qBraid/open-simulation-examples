"""Tabulate the 2D edge integrals S_i(g) for one film thickness / substrate / x0."""
import json, sys
import edge2d

def table(h, eps_s, x0, gaps=(3, 6, 10, 20, 30, 50, 100, 200, 500), t=0.003, eps_i=10.0):
    out = {}
    for g in gaps:
        r = edge2d.run(g=float(g), h=h, eps_s=eps_s, x0=x0, t=t, eps_i=eps_i)
        out[float(g)] = {k: float(r[k]) for k in ("S_MS", "S_MA", "S_SA", "K", "K_analytic_zero_thickness")}
        out[float(g)].update(MS=out[float(g)]["S_MS"], MA=out[float(g)]["S_MA"], SA=out[float(g)]["S_SA"])
    return out

if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    path = a.pop("out")
    json.dump({str(k): v for k, v in table(**a).items()}, open(path, "w"), indent=1)
