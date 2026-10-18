---
name: qbraid-long-jobs
description: Keep a long-running job on a qBraid on-demand CPU/GPU instance alive, and still capped. Use before any job expected to run over about 15 minutes on an on-demand instance (training, inference sweeps, CFD, benchmarks), when an instance stopped itself unexpectedly, or when the Lab Settings panel shows different timers from what `qbraid compute instances set` reported. Covers the two independent shutdown mechanisms, how to set both, and how to babysit jobs over ssh without killing your own shell.
metadata:
  version: "0.1.0"
  status: "provisional"
  verified: "2026-10-18"
---

# Long jobs on qBraid on-demand instances

An on-demand instance stops when **either** of two independent mechanisms says
so. Setting one does not change the other, and the stricter one wins.

| | qBraid server-side policy | In-pod Lab timers |
|---|---|---|
| Set with | `qbraid compute instances set <label> --max-session N --auto-stop N` | `~/.qbraid/qbraidrc` on the instance, section `[lab-settings]`, or the Lab's Settings panel ("Usage & Compute") |
| Enforced by | qBraid's backend | the `qbraid_lab` Jupyter server extension, a loop every 60 s, whether or not a browser tab is open |
| Clock | wall clock from when the instance started running | container uptime (`/proc/1` ctime): restarts with the container |
| Defaults on a fresh box | none set | `auto_cutoff_minutes = 120`, `idle_timeout_minutes = 15`, `default_session_length = 60` (the file doesn't exist, so these apply) |

What the in-pod timers mean (qbraid-lab 4.6.13, `qbraid_lab/handlers/compute.py`):

- **Auto shutdown** (`auto_cutoff_minutes`): stops the instance at N minutes of
  container uptime. A job you started over ssh dies at 120 minutes whatever
  `--max-session` says.
- **Idle timeout** (`idle_timeout_minutes`): idle means no browser input
  (mouse/keys in the Lab), **and** average CPU at or under 30%, **and** every
  GPU at or under 20%. Stops after N + 10 minutes (N + 5 with a tab open and
  the "Still there?" prompt unanswered). Jupyter kernel or terminal activity
  doesn't count. An I/O-bound or bursty GPU job can be killed as idle.
- **Session check-ins** (`default_session_length`): a prompt every N minutes,
  enforced only while a Lab tab is open.
- `0` disables a timer. The file is re-read every minute; no restart needed.

## Recipe

```bash
H=<ssh alias>; L=<instance label>          # both printed by `qbraid compute up <profile>`
# 1. server-side cap: the backstop
qbraid compute instances set $L --auto-stop 0 --max-session 360
# 2. in-pod timers: off (no CLI command writes these)
ssh $H 'python3 - <<EOF
import configparser, pathlib
p = pathlib.Path.home()/".qbraid"/"qbraidrc"; p.parent.mkdir(exist_ok=True)
c = configparser.ConfigParser(); c.read(p)
c.has_section("lab-settings") or c.add_section("lab-settings")
for k in ("auto_cutoff_minutes", "idle_timeout_minutes", "default_session_length"): c.set("lab-settings", k, "0")
c.write(open(p, "w"))
EOF
grep -A3 "^\[lab-settings\]" ~/.qbraid/qbraidrc'
# 3. when the job is done: copy results off, then
qbraid compute instances terminate $L --yes
```

`configparser` merges into an existing `qbraidrc`, so nothing else in the file
is lost. Never print the whole file: filter with
`grep -viE "token|key|secret|pass"` if you need to look at it.

With the in-pod timers off, `--max-session` is the only thing between a
forgotten box and a large bill. Always set it, sized to the job plus margin.
`qbraid compute instances show` hides an auto-stop of 0, so don't read its
absence as "not set".

## Running the job

- Start it detached so ssh can drop:
  `ssh $H 'cd <dir> && setsid nohup bash run.sh > run.out 2>&1 < /dev/null & echo started'`.
- Wait for it with a polling loop in the background (`until ssh $H '<check>'; do sleep 30; done`)
  and check exit codes, not just log words.
- **Never `pkill -f` or `pgrep -f` a pattern that your own command line also
  contains.** `ssh $H 'pkill -f "python main.py"'` matches the remote shell
  running that very command and kills it (exit 143/144), and a local
  `pkill -f` kills your own tool shell. Anchor the pattern to the start of the
  target's command line: `pgrep -f "^/home/jovyan/env/bin/python main.py"`.
- **Never print `ps aux` or `/proc/*/cmdline` on an instance:** Jupyter's
  token is in its arguments. Use `pgrep -fc` for counts, `ps -o pid,etimes,comm`
  for ages, `top -b -n1` for load.
- Credits: an A100 is 3.74 credits per minute (`qbraid compute list`). Report
  credits, or credits / 100 as USD; never a raw credit figure with a `$`.

## When an instance stopped by itself

1. `qbraid compute instances show <label>`: status and the server-side policy.
2. On a restarted box, container uptime
   (`echo $(( $(date +%s) - $(stat -c %Z /proc/1) ))`) tells you when it came back.
3. 120 minutes after start, or 25 minutes into a low-CPU/low-GPU phase,
   points at the in-pod timers. Set them as above and rerun.

Reported upstream as qBraid/qBraid-CLI#392: `instances set` reports success
while the in-pod timers still apply.
