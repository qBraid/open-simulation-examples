#!/bin/bash
# GPU job: evaluate final + early policies, export live payload, browser-twin checks. Logs capped.
cd "${WORK:-/tmp/robotics}"
P=env/bin/python; C="env/bin/python logcap.py 2000000"
$P -u evaluate.py --run runs/go1_flat --params params_final.pkl --tag final --step 217907200 2>&1 | $C
$P -u evaluate.py --run runs/go1_flat --params params_000011468800.pkl --tag early --step 11468800 2>&1 | $C
JAX_PLATFORMS=cpu $P -u export_live.py --run runs/go1_flat --params params_final.pkl --out live.json 2>&1 | $C
JAX_PLATFORMS=cpu $P -u live_check.py --live live.json --run runs/go1_flat --out runs/go1_flat/live_check.json 2>&1 | $C
JAX_PLATFORMS=cpu $P -u live_check.py --live live.json --run runs/go1_flat --solver default --out runs/go1_flat/live_check_default.json 2>&1 | $C
echo EVALDONE
