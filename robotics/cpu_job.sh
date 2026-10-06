#!/bin/bash
cd "${WORK:-/tmp/robotics}"; P=env/bin/python; C="env/bin/python logcap.py 1000000"; export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
$P -u export_live.py --run runs/go1_flat --params params_final.pkl --out live.json 2>&1 | $C
$P -u live_check.py --live live.json --run runs/go1_flat --out runs/go1_flat/live_check.json 2>&1 | $C
$P -u live_check.py --live live.json --run runs/go1_flat --solver default --out runs/go1_flat/live_check_default.json 2>&1 | $C
echo CPUDONE
