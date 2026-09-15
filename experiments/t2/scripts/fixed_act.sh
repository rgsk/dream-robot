#!/usr/bin/env bash
# T2-fixed, ACT on the local machine: pull the pod's fixed-cube 50/50 dataset, train ACT (50 epochs,
# as ACT's 100-demo cells), eval with half shake, paths with z = 0 and z sampled.
#   bash experiments/t2/scripts/fixed_act.sh 2>&1 | command grep --line-buffered -v "Svt\|WARNING\|INFO\|Map:\|moov" > experiments/t2/fixed_act.log
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MUJOCO_GL=glfw PYTHONUNBUFFERED=1

DATA=data/robosuite/pick_place_two_bins_fixed_noise0025_n100
REPO=dream_robot/pick_place_two_bins
ENV=robosuite/pick_place_two_bins_fixed
ACT=experiments/t2_fixed_act_half100
S=experiments/t2/scripts/rollout_paths.py

echo "== pull dataset from pod $(date +%T)"
mkdir -p $DATA
rsync -az --no-owner --no-group -e "ssh -i $HOME/.ssh/rgsk_github_ssh -p 13763" \
    root@213.173.108.6:/workspace/dream-robot/$DATA/ $DATA/
du -sh $DATA

echo "== ACT train $(date +%T)"
uv run python -m dream_robot.policies.act.train --root $DATA --repo-id $REPO --out $ACT --epochs 50

echo "== ACT eval, half shake $(date +%T)"
uv run python -m dream_robot.core.run --env $ENV --policy act --checkpoint $ACT/checkpoint.pt \
    --noise-sigma 0.025 --experiments $ACT/eval

echo "== paths, z = 0 $(date +%T)"
uv run python $S run --env $ENV --policy act --checkpoint $ACT/checkpoint.pt --noise-sigma 0.025 \
    --out experiments/t2/paths/fixed_act.json
echo "== paths, z sampled $(date +%T)"
uv run python $S run --env $ENV --policy act --checkpoint $ACT/checkpoint.pt --noise-sigma 0.025 \
    --sample-z --out experiments/t2/paths/fixed_act_sampled_z.json
uv run python $S plot experiments/t2/paths/{fixed_act,fixed_act_sampled_z}.json \
    --out scratch/t2/paths_fixed_act.png

echo "== done $(date +%T)"
