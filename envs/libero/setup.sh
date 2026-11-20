#!/usr/bin/env bash
# LIBERO (robosuite 1.4.0 benchmark: LIBERO-90 short tasks, LIBERO-10 long ones) in its own venv.
# Idempotent: every step skips itself when already done.
#
#   envs/libero/setup.sh
#
# Package: hf-libero (Hugging Face's pip packaging of LIBERO, what lerobot[libero] uses); it pins
# robosuite==1.4.0 and mujoco<3.9, which the main venv (robosuite 1.5.1, mujoco 3.9) can't share.
# Demos are not downloaded here: data/libero/<suite>/<task>_demo.hdf5 from the HF dataset
# yifengzhu-hf/LIBERO-datasets (0.4-1.4 GB per task, 50 demos each, images included).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$HERE/.venv/bin/python"

step() { echo; echo "==> $*"; }

step "venv (Python 3.11)"
[[ -x "$PY" ]] || uv venv "$HERE/.venv" --python 3.11

step "torch from RoboCasa's venv (hard links)"
# hf-libero pulls torch in via transformers: ~3.5 GB with its CUDA wheels. RoboCasa's venv (same Python 3.11) has
# torch 2.7.1+cu126; uv's current cache can't reuse it (an older uv cached it in a format uv 0.13 doesn't read),
# so link its files in directly: no download, no disk. uv then sees torch==2.7.1 installed and keeps it.
S="$HERE/../robocasa/.venv/lib/python3.11/site-packages"
D="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -d "$S/torch" && ! -d "$D/torch" ]]; then
    for x in "$S"/torch "$S"/torch-2.7.1.dist-info "$S"/torchgen "$S"/functorch "$S"/triton "$S"/triton-*.dist-info "$S"/nvidia "$S"/nvidia_*.dist-info; do
        [[ -e "$x" ]] && cp -al "$x" "$D/"
    done
fi

step "packages"
uv pip install --python "$PY" "hf-libero>=0.1.3,<0.2" "torch==2.7.1" h5py "imageio[ffmpeg]" pillow pyyaml "mcp>=1.20"

step "dream_robot on the path"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO" > "$SITE/dream_robot.pth"

step "LIBERO config (repo-local; ~/.libero/config.yaml may point at another project's install)"
# LIBERO reads its paths from $LIBERO_CONFIG_PATH/config.yaml (default ~/.libero, and asks interactively when
# missing). Ours: envs/libero/config/, generated here (absolute paths, gitignored); set LIBERO_CONFIG_PATH to it.
PKG="$D/libero/libero"
mkdir -p "$HERE/config"
cat > "$HERE/config/config.yaml" <<YAML
benchmark_root: $PKG
bddl_files: $PKG/bddl_files
init_states: $PKG/init_files
datasets: $REPO/data/libero
assets: $PKG/assets
YAML

step "done"
LIBERO_CONFIG_PATH="$HERE/config" MUJOCO_GL=egl "$PY" -c "import robosuite, mujoco, libero.libero as L; print('robosuite', robosuite.__version__, '| mujoco', mujoco.__version__, '| libero', L.__file__)"
