#!/usr/bin/env bash
# 2026-11-08: sim sweeps, actions = decoder(dreamed z, joints), WAM-C seeds 0, 1, 2.
cd /home/rahul/Documents/codes/ml/dream-robot
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
L=experiments/lam
SUM=$L/sweep_decoder_results.txt
echo "sweep_decoder, started $(date)" > $SUM
for r in wamc_l3 wamc_l3_z_s1 wamc_l3_z_s2; do
  timeout 2400 .venv/bin/python $L/scripts/sweep_decoder.py $L/$r 2>&1 | grep --line-buffered -v Warn > $L/${r}_decoder.log \
    || echo "$r failed/timed out" | tee -a $SUM
  echo "$r: $(grep -E '^n_exec' $L/${r}_decoder.log | tr '\n' ' ') $(grep '^total' $L/${r}_decoder.log)" | tee -a $SUM
done
echo "finished $(date)" | tee -a $SUM
