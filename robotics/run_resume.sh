#!/bin/bash
# resume Go1 training from the latest saved params; metrics -> runs/go1_flat/progress.jsonl, stdout capped at 5 MB
cd /tmp/ose/robotics
last=$(ls runs/go1_flat/params_0*.pkl | tail -1); off=$(basename $last .pkl | sed "s/params_0*//"); off=${off:-0}
rem=$((200000000 - off)); evals=$(( rem / 10000000 + 1 ))
echo "resume from $last offset=$off remaining=$rem evals=$evals"
env/bin/python -u train.py --env Go1JoystickFlatTerrain --run runs/go1_flat --restore_pkl $last --step_offset $off --steps $rem --num_evals $evals 2>&1 | env/bin/python logcap.py 5000000
