# LIBERO

Next rung (2026-11-20): one Verne policy on many short LIBERO-90 skills; the LLM (MCP) chains them for
LIBERO-10's long tasks. Published LIBERO numbers are saturated (~97-99% for big VLAs, trained per suite on the
test tasks); ours is a different protocol: skills -> chained long tasks, small model from scratch.

## Step 1 · install + film demos
- envs/libero/setup.sh: Python 3.11 venv, hf-libero (robosuite 1.4.0, mujoco 3.8.1); torch 2.7.1 hard-linked from
  RoboCasa's venv (uv's cache couldn't reuse it, 3.5 GB at 0.1 MB/s otherwise); repo-local LIBERO config
  (envs/libero/config/, LIBERO_CONFIG_PATH) since ~/.libero points at robot-vla. Object assets: ~/.cache/libero
  (was a partial copy, completed from HF: 586 files, 409 MB).
- Data: data/libero/libero_90/, 3 tasks of 50 human demos each (HF yifengzhu-hf/LIBERO-datasets):
  open the top drawer (67-177 steps, mean 95), turn on the stove (80-105, mean 93), moka pot on the stove
  (126-218, mean 151). Obs: agentview + wrist 128x128, joints 7, gripper 2, ee pose; actions 7 (ee delta + grip);
  20 Hz.
- Replay of demo_0 in our sim from its saved first state: **success on all 3**. scripts/film_demos.py; films in
  scratch_nov/libero/films/.
