#!/usr/bin/env bash
# 2026-11-08: T10 + 300 fruit video + LAM z target (seed 0) vs T10 + 300 fruit video (no z, exists).
# Tests: fruit, apple0, unseen (tomato/onion), 60 seeds each. Baseline gets apple0 + unseen too.
cd /home/rahul/Documents/codes/ml/dream-robot
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
U=experiments/pick_shapes/uwm_dino
L=experiments/pick_shapes/lam
S=experiments/pick_shapes/scripts
SUM=$L/lam_fruit_results.txt
echo "lam_fruit, started $(date)" > $SUM

ev() {  # checkpoint, pool, log
  timeout 1800 envs/robocasa/.venv/bin/python $S/uwm_cam_eval.py $1 --seeds 2000-2059 --videos 3 --pool $2 2>&1 \
    | grep --line-buffered -E "live|lifted|Error" > $3
  local by=$(grep -v live $3 | grep -o "'mesh': '[a-z]*_[0-9]*', 'ok': [A-Za-z]*" \
    | sed -E "s/'mesh': '([a-z]*)_[0-9]*', 'ok': (.*)/\1 \2/" | sort | uniq -c | grep True | awk '{print $2"="$1}' | tr '\n' ' ')
  echo "$1 on $2: $(tail -1 $3)   [lifted by type: $by]" | tee -a $SUM
}

H=$L/T_l10_vfruit300_lam_s0; mkdir -p $H
timeout 2400 .venv/bin/python $S/uwm_dino_target.py --labelled 10 --steps 8000 --save-every 1000 --seed 0 \
  --lam $L/z8 --more-video pick_fruit_video_n100 pick_fruit_video2_n200 --out $H > $H/train.log 2>&1 \
  || echo "$H train failed/timed out" | tee -a $SUM
for p in fruit apple0 unseen; do ev $H/step8000.pt $p $H/eval_step8000_${p}_60.log; done
B=$U/T_l10_vfruit300_s0
for p in apple0 unseen; do ev $B/step8000.pt $p $B/eval_step8000_${p}_60.log; done
echo "finished $(date)" | tee -a $SUM
