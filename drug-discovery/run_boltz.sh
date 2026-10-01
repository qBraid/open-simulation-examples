#!/bin/bash
# Boltz-2 co-folding / affinity in GPU-sized batches (each batch < 40 min on an L4).
# usage: run_boltz.sh <yaml_dir> <out_dir> <batch_size> <batch_index> [extra boltz flags]
# Settings: 5 diffusion samples, 3 recycles, 200 steps, physics steering potentials,
# seed 0, no cuEquivariance kernels (not installed). Top-ranked sample (model_0) is scored.
set -euo pipefail
Y=$1; O=$2; B=$3; K=$4; shift 4
D=$O/inputs_$K; mkdir -p $D
ls $Y/*.yaml | sort | sed -n "$((K*B+1)),$(((K+1)*B))p" | while read f; do ln -sf $(readlink -f $f) $D/; done
[ -z "$(ls -A $D)" ] && { echo "empty batch $K"; exit 0; }
boltz predict $D --out_dir $O/batch_$K --cache ${BOLTZ_CACHE:-$PWD/cache} --model boltz2 --output_format pdb \
  --diffusion_samples 5 --recycling_steps 3 --sampling_steps 200 --use_potentials --seed 0 --no_kernels --num_workers 1 "$@"
