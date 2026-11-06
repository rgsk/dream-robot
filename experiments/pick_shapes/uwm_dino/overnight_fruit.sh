#!/usr/bin/env bash
# Overnight 2026-11-06: fruit-video test, seeds 1-2 + apple check + unseen fruit; then summary + power off.
# Every step has a timeout and a failure doesn't stop the chain.
cd /home/rahul/Documents/codes/ml/dream-robot
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
U=experiments/pick_shapes/uwm_dino
S=experiments/pick_shapes/scripts
SUM=$U/overnight_results.txt
echo "overnight fruit test, started $(date)" > $SUM

ev() {  # checkpoint, pool, log
  timeout 1800 envs/robocasa/.venv/bin/python $S/uwm_cam_eval.py $1 --seeds 2000-2059 --videos 3 --pool $2 2>&1 \
    | grep --line-buffered -E "live|lifted|Error" > $3
  local by=$(grep -v live $3 | grep -o "'mesh': '[a-z]*_[0-9]*', 'ok': [A-Za-z]*" \
    | sed -E "s/'mesh': '([a-z]*)_[0-9]*', 'ok': (.*)/\1 \2/" | sort | uniq -c | grep True | awk '{print $2"="$1}' | tr '\n' ' ')
  echo "$1 on $2: $(tail -1 $3)   [lifted by type: $by]" | tee -a $SUM
}

# 1. seeds 1, 2: train T10 + fruit video, then fruit-test it and the existing T10 seeds
for s in 1 2; do
  H=$U/T_l10_vfruit_s$s; mkdir -p $H
  timeout 1800 .venv/bin/python $S/uwm_dino_target.py --labelled 10 --steps 8000 --save-every 1000 --seed $s \
    --more-video pick_fruit_video_n100 --out $H > $H/train.log 2>&1 || echo "$H train failed/timed out" | tee -a $SUM
  ev $H/step8000.pt fruit $H/eval_step8000_fruit_60.log
  ev $U/T_l10_v0_s$s/step8000.pt fruit $U/T_l10_v0_s$s/eval_step8000_fruit_60.log
done

# 2. apple check: does fruit video hurt the original task? (seed 0)
ev $U/T_l10_vfruit/step8000.pt apple0 $U/T_l10_vfruit/eval_step8000_60.log

# 3. unseen fruit (tomato, onion: in neither labels nor video), seed 0
ev $U/T_l10_v0/step8000.pt unseen $U/T_l10_v0/eval_step8000_unseen_60.log
ev $U/T_l10_vfruit/step8000.pt unseen $U/T_l10_vfruit/eval_step8000_unseen_60.log

echo "finished $(date); powering off" | tee -a $SUM
sync
systemctl poweroff
