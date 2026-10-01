#!/bin/bash
# End-to-end: data, prep (CPU), Boltz-2 (GPU), scoring, viewer. Run from this directory.
# GPU: one 24 GB card (tested on an NVIDIA L4). CPU steps are light.
set -euo pipefail
W=${WORK:-$PWD/work}; mkdir -p $W
[ -d $W/posebusters_benchmark_set ] || { curl -sL -o $W/pb.zip https://zenodo.org/api/records/8278563/files/posebusters_paper_data.zip/content; unzip -q -o $W/pb.zip "posebusters_benchmark_set/*" -d $W; }
[ -f $W/pbres.csv ] || curl -sL -o $W/pbres.csv https://zenodo.org/api/records/8278563/files/posebusters_paper_results.csv/content
mkdir -p $W/fep && cp fep_inputs/*.sdf $W/fep/ && cp results/fep_plus_tyk2.csv $W/fep/tyk2_out.csv && cp results/fep_plus_cdk2.csv $W/fep/cdk2_out.csv
B=https://raw.githubusercontent.com/schrodinger/public_binding_free_energy_benchmark/main/fep_benchmark_inputs/structure_inputs/jacs_set
for t in tyk2 cdk2; do [ -f $W/fep/${t}_protein.pdb ] || curl -sfo $W/fep/${t}_protein.pdb $B/${t}_protein.pdb; done
python prep_posebusters.py $W/posebusters_benchmark_set $W/pb      # 32 complexes + MSAs (ColabFold server)
python prep_affinity.py $W/fep $W/aff                               # 2 x 16 ligands + MSAs
for k in 0 1; do ./run_boltz.sh $W/pb/yaml $W/out_pb 16 $k; done   # ~2-3 min per complex on an L4
for k in 0 1; do ./run_boltz.sh $W/aff/yaml $W/out_aff 16 $k; done
python eval_poses.py $W/posebusters_benchmark_set $W/out_pb $W/pb/subset.json $W/pbres.csv $W/eval
python eval_affinity.py $W/out_aff $W/fep results/affinity.json
cp $W/eval/poses.json results/poses.json
python build_viewer.py $W/posebusters_benchmark_set $W/eval results
