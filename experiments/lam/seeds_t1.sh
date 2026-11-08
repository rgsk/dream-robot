#!/usr/bin/env bash
# 2026-11-08 overnight: WAM-C seeds 1, 2 with z and without (--no-z); same LAM z8_det_rev. Then power off.
cd /home/rahul/Documents/codes/ml/dream-robot
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
L=experiments/lam
SUM=$L/seeds_t1_results.txt
echo "seeds_t1, started $(date)" > $SUM
for s in 1 2; do
  for arm in z noz; do
    flag=$([ $arm = noz ] && echo --no-z)
    O=$L/wamc_l3_${arm}_s$s; mkdir -p $O
    timeout 5400 .venv/bin/python $L/scripts/wam_c.py $flag --seed $s --out $O 2>&1 | grep --line-buffered -v Warn > $O.log \
      || echo "$O failed/timed out" | tee -a $SUM
    echo "$arm seed $s: $(grep -E '^n_exec' $O.log | tr '\n' ' ') $(grep '^total' $O.log | cut -d'(' -f1)" | tee -a $SUM
  done
done
echo "finished $(date); powering off" | tee -a $SUM
sleep 60
systemctl poweroff
