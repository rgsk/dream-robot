#!/usr/bin/env bash
# Dream YAM Sim (Dream Labs' MuJoCo station: two YAM arms, three cameras, fruit) in its own venv,
# with the dream library that calls Dream-1. Idempotent: every step skips itself when already done.
#
#   envs/dream_yam/setup.sh
#
# The sim ships inside the dreamlabs wheel (dream_examples/yam_mujoco); there is nothing to clone.
# The API key is not set here: run the key step of platform.dreamlabs.ai/docs yourself, it writes
# ~/.dreamlabs/credentials, which the client reads.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$HERE/.venv/bin/python"

step() { echo; echo "==> $*"; }

step "venv (Python 3.12)"
[[ -x "$PY" ]] || uv venv "$HERE/.venv" --seed --python 3.12

step "packages"
# Both named and pinned: dreamlabs alone can leave an older dreamlabs-client in place.
uv pip install --python "$PY" "dreamlabs==1.0.1" "dreamlabs-client==1.0.1"
# What our server needs on top: MCP, and video for recorded clips.
uv pip install --python "$PY" "mcp>=1.20" "imageio[ffmpeg]"

step "dream_robot on the path"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO" > "$SITE/dream_robot.pth"

step "offline checks (no API call)"
SIM="$SITE/dream_examples/yam_mujoco"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
(cd "$SIM" && "$PY" yam_env.py --headless >/dev/null && echo "sim OK")
(cd "$SIM" && "$PY" dream_control.py --dry-run --headless --quiet)

step "done"
"$PY" -c "import dream, mujoco, dream_robot; print('dream', dream.__version__, '| mujoco', mujoco.__version__)"
[[ -f ~/.dreamlabs/credentials ]] && echo "API key: ~/.dreamlabs/credentials found" \
    || echo "API key: none yet -- run the key step of platform.dreamlabs.ai/docs"
