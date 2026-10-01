"""Build the self-contained viewer.html from training/eval outputs.

    python build_viewer.py --run <run_dir> --geometry geometry.json \
        --ghost early:<tag> [--ghost mid:<tag>] --bench results/benchmark.json --out viewer.html
Each --ghost names an eval tag (rollout_<tag>.npz / eval_<tag>.json) to replay as a
translucent ghost next to the final policy.
"""
import argparse
import base64
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--geometry", required=True)
p.add_argument("--final", default="final")
p.add_argument("--ghost", action="append", default=[], help="label:tag")
p.add_argument("--bench", required=True)
p.add_argument("--template", default=os.path.join(os.path.dirname(__file__), "viewer_template.html"))
p.add_argument("--live", default=None, help="live.json from export_live.py")
p.add_argument("--out", default="viewer.html")
a = p.parse_args()

b64 = lambda arr: base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()
geom = json.load(open(a.geometry))
# keep only what the viewer draws: visual meshes (group 2) + foot spheres
geom["geoms"] = [g for g in geom["geoms"] if g["body"] > 0 and (g["group"] == 2 or (g["type"] == 2 and g["name"] in ("FR", "FL", "RR", "RL")))]
used = sorted({g["mesh"] for g in geom["geoms"] if g.get("mesh") is not None})
remap = {m: i for i, m in enumerate(used)}
geom["meshes"] = [geom["meshes"][m] for m in used]
for g in geom["geoms"]:
    if g.get("mesh") is not None:
        g["mesh"] = remap[g["mesh"]]

curve = [json.loads(l) for l in open(os.path.join(a.run, "progress.jsonl"))]
curve = list({c["step"]: c for c in curve}.values())  # dedupe: a resumed job re-evaluates its start step
curve = [{"step": c["step"], "reward": c["reward"], "reward_std": c["reward_std"], "wall_s": c["wall_s"]} for c in curve]


def roll(tag, label):
    z = np.load(os.path.join(a.run, f"rollout_{tag}.npz"))
    ev = json.load(open(os.path.join(a.run, f"eval_{tag}.json")))
    return ev, {"frames": b64(z["frames"]), "n": int(z["frames"].shape[0]), "linvel": b64(z["linvel"]),
                "cmd": b64(z["cmds"]), "label": label, "fell_at": ev["rollout"]["fell_at_frame"]}


ev_final, r_final = roll(a.final, "final policy")
rollouts = {"final": r_final}
ghosts = []
for spec in a.ghost:
    label, tag = spec.split(":", 1)
    ev, r = roll(tag, label)
    rollouts[label] = r
    step = int(ev.get("step", 0)) if "step" in ev else None
    if step is not None:
        rew = min(curve, key=lambda c: abs(c["step"] - step))["reward"]
        ghosts.append({"name": label, "step": step, "reward": rew})

bench = json.load(open(a.bench))
data = {
    "geometry": geom, "nbody": len(geom["bodies"]), "dt": ev_final["rollout"]["dt"],
    "rollouts": rollouts, "script": ev_final["rollout"]["script"], "curve": curve, "ghosts": ghosts,
    "scatter": ev_final["tracking"]["cmd_vs_actual_speed"] if "tracking" in ev_final else [[], []],
    "bench": bench["rows"], "stamp": bench["stamp"], "screen_sub": bench.get("screen_sub", ""),
    "tour": bench["tour"], "vmax": 1.5,
    "live": json.load(open(a.live)) if a.live else None,
}
html = open(a.template).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open(a.out, "w").write(html)
print(a.out, f"{len(html)/1e6:.2f} MB", "rollouts:", list(rollouts))
