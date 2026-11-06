#!/usr/bin/env bash
# 2026-11-06: 200 more fruit video clips (300 total), T10 + 300 fruit clips x 3 seeds, fruit test.
# Every step has a timeout; a failure doesn't stop the chain. No power-off.
cd /home/rahul/Documents/codes/ml/dream-robot
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
U=experiments/pick_shapes/uwm_dino
S=experiments/pick_shapes/scripts
SUM=$U/fruit300_results.txt
echo "fruit300, started $(date)" > $SUM

ev() {  # checkpoint, pool, log
  timeout 1800 envs/robocasa/.venv/bin/python $S/uwm_cam_eval.py $1 --seeds 2000-2059 --videos 3 --pool $2 2>&1 \
    | grep --line-buffered -E "live|lifted|Error" > $3
  local by=$(grep -v live $3 | grep -o "'mesh': '[a-z]*_[0-9]*', 'ok': [A-Za-z]*" \
    | sed -E "s/'mesh': '([a-z]*)_[0-9]*', 'ok': (.*)/\1 \2/" | sort | uniq -c | grep True | awk '{print $2"="$1}' | tr '\n' ' ')
  echo "$1 on $2: $(tail -1 $3)   [lifted by type: $by]" | tee -a $SUM
}

timeout 7200 envs/robocasa/.venv/bin/python $S/rung0_record.py --episodes 200 --seed-start 1500 --pool fruit \
  --name pick_fruit_video2 2>&1 | grep --line-buffered -E "KEEP|DROP|kept|Error" > $U/record_fruit2_n200.log
echo "record: $(tail -1 $U/record_fruit2_n200.log)" | tee -a $SUM
timeout 600 .venv/bin/python $S/dino_cache.py --data pick_fruit_video2_n200 > $U/cache_fruit2.log 2>&1
echo "cache: $(tail -1 $U/cache_fruit2.log)" | tee -a $SUM

for s in 0 1 2; do
  H=$U/T_l10_vfruit300_s$s; mkdir -p $H
  timeout 2400 .venv/bin/python $S/uwm_dino_target.py --labelled 10 --steps 8000 --save-every 1000 --seed $s \
    --more-video pick_fruit_video_n100 pick_fruit_video2_n200 --out $H > $H/train.log 2>&1 \
    || echo "$H train failed/timed out" | tee -a $SUM
  ev $H/step8000.pt fruit $H/eval_step8000_fruit_60.log
done
echo "finished $(date)" | tee -a $SUM
