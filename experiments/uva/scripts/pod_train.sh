#!/usr/bin/env bash
# train_uva.py on the pod, detached, with glibc reusing freed memory.
# Why (experiments/uva/notes.md, c1 stall): on that host every fresh large allocation stalled in the
# kernel; their loader's per-window numpy copies went 5 ms -> 1.8 s and c1's steps 2 s -> 45 s.
# These must be set before the process starts (glibc reads them once).
#   NAME=t5_c2 bash experiments/uva/scripts/pod_train.sh --data data/uva/t5 --name t5 --image 224 --wrist ...
# Log: /workspace/dream-robot/uva_$NAME.log; run dir: /workspace/runs/$NAME
set -euo pipefail
NAME=${NAME:?set NAME}
cd /workspace/dream-robot
export PYTHONUNBUFFERED=1
export MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=17179869184 MALLOC_TOP_PAD_=1073741824
setsid nohup envs/uva/.venv/bin/python experiments/uva/scripts/train_uva.py --out "/workspace/runs/$NAME" "$@" \
    > "/workspace/dream-robot/uva_$NAME.log" 2>&1 < /dev/null &
echo "started $NAME (pid $!)"
