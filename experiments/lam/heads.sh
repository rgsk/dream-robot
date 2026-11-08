#!/usr/bin/env bash
# 2026-11-08: 8 action heads (j, jn, jz, jnz, jf, jnf, jzf, jnzf) on each frozen WAM-C trunk, 3 seeds.
cd /home/rahul/Documents/codes/ml/dream-robot
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
L=experiments/lam
SUM=$L/heads_results.txt
echo "heads, started $(date)" >> $SUM
for r in ${RUNS:-wamc_l3 wamc_l3_z_s1 wamc_l3_z_s2}; do
  .venv/bin/python $L/scripts/wam_c_heads.py $L/$r 2>&1 | grep --line-buffered -v Warn > $L/${r}_heads.log \
    || echo "$r failed" | tee -a $SUM
  grep -E 'head ' $L/${r}_heads.log | tee -a $SUM
done
echo "finished $(date)" | tee -a $SUM
