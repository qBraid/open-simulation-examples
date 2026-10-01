"""stdin -> stdout log filter: drop known Warp spam, stop writing after a byte cap
(but keep draining stdin so the producer never gets SIGPIPE).
    python -u train.py ... 2>&1 | python logcap.py 5000000 > run.log"""
import sys

cap = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
DROP = ("linesearch iterations limit", "warn_overflow", "Cloning mujoco_menagerie")
n = 0
for line in sys.stdin:
    if any(d in line for d in DROP):
        continue
    if n < cap:
        sys.stdout.write(line)
        sys.stdout.flush()
        n += len(line)
        if n >= cap:
            sys.stdout.write(f"[logcap: {cap} bytes reached, further output dropped]\n")
