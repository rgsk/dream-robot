#!/usr/bin/env bash
# 2026-11-08: jnzf with the now frame blanked in 50% of training samples, 3 trunks; then reliance probe.
cd /home/rahul/Documents/codes/ml/dream-robot
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
L=experiments/lam
SUM=$L/heads_results.txt
echo "heads_drop (jnzf, now-drop 0.5), started $(date)" >> $SUM
for r in wamc_l3 wamc_l3_z_s1 wamc_l3_z_s2; do
  .venv/bin/python $L/scripts/wam_c_heads.py $L/$r --heads jnzf --now-drop 0.5 2>&1 | grep --line-buffered -v Warn > $L/${r}_heads_drop.log \
    || echo "$r failed" | tee -a $SUM
  grep -E 'head ' $L/${r}_heads_drop.log | tee -a $SUM
done
.venv/bin/python $L/scripts/heads_reliance.py $L/wamc_l3 $L/wamc_l3_z_s1 $L/wamc_l3_z_s2 --heads jzf,jnzf,jnzf_d50 2>&1 | grep -v Warn > $L/heads_reliance_drop.log
echo "finished_drop $(date)" | tee -a $SUM
