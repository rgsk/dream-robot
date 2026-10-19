#!/usr/bin/env bash
# RoboCasa kitchens in their own venv. Idempotent: every step skips itself when already done.
#
#   envs/robocasa/setup.sh            venv + source + packages + macros + ~23 GB of assets
#   envs/robocasa/setup.sh --no-assets   everything but the asset download
#
# Pins come from RoboCasa's own setup.py (Python 3.11 per its docs, robosuite master).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$HERE/.venv/bin/python"
ASSETS=1
[[ "${1:-}" == "--no-assets" ]] && ASSETS=0

step() { echo; echo "==> $*"; }

step "venv (Python 3.11)"
[[ -x "$PY" ]] || uv venv "$HERE/.venv" --python 3.11

step "source clones"
mkdir -p "$HERE/src"
for repo in ARISE-Initiative/robosuite robocasa/robocasa; do
    dir="$HERE/src/${repo#*/}"
    if [[ -d "$dir/.git" ]]; then git -C "$dir" pull --ff-only; else git clone --depth 1 "https://github.com/$repo" "$dir"; fi
done

step "packages"
uv pip install --python "$PY" -e "$HERE/src/robosuite" -e "$HERE/src/robocasa"
# What our server needs on top: MCP, and perception's scipy/Pillow (already pulled in, kept explicit).
uv pip install --python "$PY" "mcp>=1.20" scipy pillow "imageio[ffmpeg]" pyyaml glfw

step "dream_robot on the path"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO" > "$SITE/dream_robot.pth"

step "robocasa / robosuite macros"
"$PY" -m robocasa.scripts.setup_macros

if [[ $ASSETS == 1 ]]; then
    step "kitchen assets (~23 GB)"
    # The script asks for confirmation; answer yes. `yes` dies of SIGPIPE when the download
    # exits, and pipefail would report that (exit 141) as a failure -- hence `|| true` inside.
    (yes || true) | "$PY" -m robocasa.scripts.download_kitchen_assets
fi

step "done"
"$PY" -c "import robocasa, robosuite, mujoco, dream_robot; print('robocasa', robocasa.__version__ if hasattr(robocasa, '__version__') else '?', '| robosuite', robosuite.__version__, '| mujoco', mujoco.__version__)"
