"""Inline results/*.json into viewer_template.html -> viewer.html (single self-contained file)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "results"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    pricing = load("pricing.json")
    bt = load("backtest.json")
    ser = load("backtest_series.json")
    vol = load("vol_surfaces.json")
    mc = load("mc_risk.json")
    stamp_txt = (R / "stamp.txt").read_text().strip() if (R / "stamp.txt").exists() else ""
    series = ser["series"]
    for k, v in series.items():
        if isinstance(v, dict):
            v.pop("weights_yearly", None)
    data = {"pricing": {k: pricing[k] for k in ("summary", "heston", "heston_sources", "american")},
            "backtest": bt, "series": series, "frontiers": ser["frontiers"], "vol": vol, "mc": mc,
            "stamp": stamp_txt}
    html = (HERE / "viewer_template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (HERE / "viewer.html").write_text(html)
    print(f"viewer.html {len(html)/1e6:.2f} MB (mc {'yes' if mc else 'pending'})")


if __name__ == "__main__":
    main()
