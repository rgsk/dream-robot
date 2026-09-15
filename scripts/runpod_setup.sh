#!/usr/bin/env bash
# Once per fresh pod, for dream-robot. See runpod.md.
#   local:  ssh -i ~/.ssh/rgsk_github_ssh -p <port> root@<ip> 'bash /workspace/dream-robot/scripts/runpod_setup.sh'
#
# Everything outside /workspace is wiped when a pod is terminated, so this runs on every new pod.
# The shared /workspace/setup.sh (from the llm repo) restores uv, rsync and the env vars; this adds
# what MuJoCo needs to render with no display, which the "RunPod PyTorch" image lacks.
set -euo pipefail

bash /workspace/setup.sh

# The image ships NVIDIA's EGL driver (libEGL_nvidia) but not the glvnd loader (libEGL.so.1) that
# MuJoCo opens with MUJOCO_GL=egl. ffmpeg for dataset and eval videos.
if ! ldconfig -p | grep -q "libEGL.so.1"; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libglvnd0 ffmpeg >/dev/null
fi

# Headless rendering for every login shell (ssh, bash -lc).
cat > /etc/profile.d/dream-robot.sh <<'EOF'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
EOF

echo "dream-robot setup done -- new ssh sessions load the env; this shell needs:"
echo "  source /etc/profile.d/workspace.sh && source /etc/profile.d/dream-robot.sh"
echo "  cd /workspace/dream-robot && uv sync   (~2.5 min)"
