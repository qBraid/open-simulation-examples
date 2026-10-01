"""S&P 500 implied-volatility surfaces over 2011-2026, anchored to public CBOE indices.

Free historical option chains do not exist, so this is a *model* surface with
public anchors, and it is labelled that way in the viewer:
  * ATM level by tenor: VIX9D (9d), VIX (30d), VIX3M (93d), VIX6M (182d), VIX1Y (365d),
    interpolated in total variance (linear in sigma^2 T) to a 12-tenor grid.
  * Smile shape: CBOE SKEW gives the 30-day risk-neutral skewness,
    S = (100 - SKEW) / 10. Skewness is scaled across tenors as S(T) = S(30d) sqrt(30/T)
    (iid returns), excess kurtosis is set to K = 1.5 S^2, and the smile is the
    Gram-Charlier approximation of Backus, Foresi & Wu (2004):
        sigma(z) = sigma_atm (1 - (S/6) z' + (K/24) z'^2),  z' = -z,  z = ln(K/F) / (sigma_atm sqrt T)
    so downside strikes carry higher vol when S < 0.
Output: results/vol_surfaces.json (month-end frames plus stress days).
"""
import csv
import json
import math
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results"

TENORS = {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 182, "VIX1Y": 365}
GRID_T = [7, 14, 30, 45, 60, 91, 122, 152, 182, 243, 304, 365]          # days
GRID_Z = [round(-2.5 + 0.2 * i, 2) for i in range(21)]                  # -2.5 .. 1.5 std-moves
EVENTS = {"2011-08-08": "US downgrade", "2015-08-24": "China deval / flash crash",
          "2018-02-05": "Volmageddon", "2020-03-16": "COVID crash", "2022-06-13": "Fed 75bp shock",
          "2024-08-05": "Yen carry unwind", "2025-04-08": "Tariff shock"}


def read(name, col=-1):
    out = {}
    with open(DATA / f"{name}_History.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            try:
                d = datetime.strptime(row[0], "%m/%d/%Y").date()
                out[d] = float(row[col] if name != "SKEW" else row[1])
            except (ValueError, IndexError):
                continue
    return out


def atm_curve(levels):
    """Total-variance interpolation from the 5 CBOE tenors to GRID_T (flat extrapolation in vol)."""
    pts = sorted((TENORS[k], (v / 100) ** 2 * TENORS[k] / 365) for k, v in levels.items())
    vols = []
    for t in GRID_T:
        if t <= pts[0][0]:
            w = pts[0][1] / pts[0][0] * t
        elif t >= pts[-1][0]:
            w = pts[-1][1] / pts[-1][0] * t
        else:
            for (t0, w0), (t1, w1) in zip(pts, pts[1:]):
                if t0 <= t <= t1:
                    w = w0 + (w1 - w0) * (t - t0) / (t1 - t0)
                    break
        vols.append(math.sqrt(max(w, 1e-8) * 365 / t))
    return vols


def surface(levels, skew_index):
    s30 = (100 - skew_index) / 10.0
    atm = atm_curve(levels)
    rows = []
    for t, a in zip(GRID_T, atm):
        s = s30 * math.sqrt(30 / t)
        k = 1.5 * s * s
        rows.append([round(max(0.03, a * (1 - (s / 6) * (-z) + (k / 24) * z * z)), 5) for z in GRID_Z])
    return rows, atm, s30


def main():
    series = {k: read(k) for k in TENORS}
    skew = read("SKEW")
    days = sorted(set.intersection(*(set(v) for v in series.values())) & set(skew))
    keep = []
    for i, d in enumerate(days):
        nxt = days[i + 1] if i + 1 < len(days) else None
        if nxt is None or nxt.month != d.month or d.isoformat() in EVENTS:
            keep.append(d)
    frames = []
    for d in keep:
        lv = {k: series[k][d] for k in TENORS}
        surf, atm, s30 = surface(lv, skew[d])
        frames.append({"date": d.isoformat(), "event": EVENTS.get(d.isoformat()), "vix": lv["VIX"],
                       "skew_index": skew[d], "skewness30": round(s30, 3),
                       "atm": [round(a, 5) for a in atm], "iv": surf,
                       "inverted": lv["VIX9D"] > lv["VIX3M"]})
    missing = [e for e in EVENTS if e not in {f["date"] for f in frames}]
    out = {"tenors_days": GRID_T, "z_grid": GRID_Z, "frames": frames,
           "method": __doc__.strip().splitlines()[0], "events_missing": missing,
           "source": "CBOE index history (VIX9D, VIX, VIX3M, VIX6M, VIX1Y, SKEW), cdn.cboe.com"}
    (OUT / "vol_surfaces.json").write_text(json.dumps(out, separators=(",", ":")))
    inv = sum(f["inverted"] for f in frames)
    print(f"{len(frames)} frames {frames[0]['date']}..{frames[-1]['date']}; inverted term structure in {inv}; "
          f"max VIX {max(f['vix'] for f in frames)} on {max(frames, key=lambda f: f['vix'])['date']}; missing events {missing}")


if __name__ == "__main__":
    main()
