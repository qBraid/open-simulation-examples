#!/bin/bash
# Run NVIDIA's physicsnemo-cfd surface benchmark, one model at a time, timed, one log per model.
# env: MODELS (default: all five), EXTRA (Hydra overrides), TAG (log/timing suffix)
cd /home/jovyan/aero/physicsnemo-cfd/workflows/benchmarking
export PHYSICSNEMO_CFD_MODEL_CACHE=/home/jovyan/aero/hfcache HF_HOME=/home/jovyan/aero/hfcache
PY=/home/jovyan/aero/pn/bin/python
mkdir -p /home/jovyan/aero/out
for m in ${MODELS:-domino_surface geotransolver_surface transolver_surface xmgn_surface fignet_surface}; do
  t0=$(date +%s)
  $PY main.py --config-name=q_$m ${EXTRA} > /home/jovyan/aero/out/$m$TAG.log 2>&1
  echo "$m$TAG exit=$? secs=$(( $(date +%s) - t0 ))" >> /home/jovyan/aero/out/timing.txt
done
