#!/usr/bin/env bash
# 2026-11-08: decoder sweeps 2: bc (joints + now frame, no WAM) once; f (joints + now + dreamed frames) x 3 WAMs.
cd /home/rahul/Documents/codes/ml/dream-robot
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
L=experiments/lam
SUM=$L/sweep_decoder_results.txt
echo "sweep_decoder2 (bc, f), started $(date)" >> $SUM
run() {  # wam dir, kind
  local log=$L/$(basename $1)_decoder_$2.log
  timeout 2400 .venv/bin/python $L/scripts/sweep_decoder.py $1 --kind $2 2>&1 | grep --line-buffered -v Warn > $log \
    || echo "$1 $2 failed/timed out" | tee -a $SUM
  echo "$2 $(basename $1): $(grep -E '^n_exec' $log | tr '\n' ' ') $(grep '^total' $log)" | tee -a $SUM
}
run $L/wamc_l3 bc
for r in wamc_l3 wamc_l3_z_s1 wamc_l3_z_s2; do run $L/$r f; done
echo "finished2 $(date)" | tee -a $SUM
