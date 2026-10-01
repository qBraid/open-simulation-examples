"""Verification stamp from the shared-pool GPU queue log: real GPU minutes and cost share."""
import json, sys, datetime as dt, subprocess
log, out = sys.argv[1], sys.argv[2]
start, mins = {}, 0.0; jobs = []
for line in open(log):
    p = line.split()
    if len(p) < 3 or p[2] != "drug-discovery": continue
    t = dt.datetime.fromisoformat(p[0].replace("Z", "+00:00"))
    if p[1] == "START": start["t"] = t; start["cmd"] = line.split("::", 1)[-1].strip()
    elif p[1] == "END" and "t" in start:
        m = (t - start.pop("t")).total_seconds() / 60; mins += m; jobs.append(dict(minutes=round(m, 1), rc=p[3], cmd=start.get("cmd", "")))
rate = 0.49 / 60  # gpu-l4 $/min (qbraid compute list); the whole box is billed, we report our GPU share
json.dump(dict(date=dt.date.today().isoformat(), machine="qBraid gpu-l4 shared pool (NVIDIA L4 24 GB, driver 595, ~5 vCPU cgroup)",
               env="Python 3.12, boltz 2.2.1, torch 2.14.1+cu130, posebusters 0.6.5, rdkit, gemmi",
               gpu_minutes=round(mins, 1), gpu_jobs=len(jobs), cost_usd_gpu_share=round(mins * rate, 2),
               note="CPU prep (MSAs via the public ColabFold server) ran at low priority on the same box; viewer built on the subscription pod at no extra cost",
               jobs=jobs), open(out, "w"), indent=1)
print(open(out).read()[:400])
