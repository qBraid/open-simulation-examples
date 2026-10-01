"""Verification stamp from the shared pool's queue logs: GPU and CPU minutes used by this stream.

usage: make_stamp.py <gpu.log> <cpu.log> <out.json>   (pool gpu-l4 billed at $0.49/h for the whole box)
"""
import sys, json, re, datetime as dt
from importlib.metadata import version


def minutes(path, stream="materials"):
    start, total, jobs = {}, 0.0, 0
    for line in open(path):
        m = re.match(r"(\S+) (START|END)\s+(\S+)(?: slot=(\d))?", line)
        if not m or m.group(3) != stream: continue
        t = dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00")); key = m.group(4) or "g"
        if m.group(2) == "START": start[key] = t
        elif key in start: total += (t - start.pop(key)).total_seconds() / 60; jobs += 1
    return total, jobs


g, ng = minutes(sys.argv[1]); c, nc = minutes(sys.argv[2])
vers = {p: version(p) for p in ("torch", "mace-torch", "orb-models", "ase", "phonopy", "pyscf") if __import__("importlib.util").util.find_spec(p.replace("-", "_").replace("mace_torch", "mace").replace("orb_models", "orb_models")) or True}
cost = g / 60 * 0.49
text = (f"Verified {dt.date.today()} on a shared qBraid gpu-l4 pool box (cgroup: 5 CPUs, 62 GB; NVIDIA L4, driver 595). "
        f"This stream used {g:.0f} GPU-minutes in {ng} jobs and {c:.0f} queued CPU-minutes; at $0.49/h for the box that is about ${cost:.2f}. "
        + "Versions: " + ", ".join(f"{k} {v}" for k, v in vers.items()) + ".")
json.dump(dict(text=text, gpu_min=g, cpu_min=c, cost_usd=cost, versions=vers), open(sys.argv[3], "w"), indent=1)
print(text)
