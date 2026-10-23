"""Does the sim hand UWM what it saw in training? Replay recorded demos and compare at every chunk.

Every UWM eval so far: ~0 successes, mostly no grasp, while offline action error is ~0.03 rad. If
the live inputs differ from the training data (image flip / shift / colour, frame order, joint
units), the model is fed something it never saw. This script:
- resets the sim to a demo's seed (data/cosmos_policy/t5_224/manifest.json) and plays the demo's
  *recorded* actions, so the sim should retrace the demo;
- at each step compares the live top / wrist frames and joints with the recorded ones;
- every 16 steps asks the UWM server for a chunk twice, through the eval client's own request
  code: once from the live sim, once from the recorded frames + joints; both are compared with
  the demo's next 16 actions (data units: radians for joints, 0..1 for the gripper).
Live ~ recorded everywhere -> the hookup is fine; the model fails off the demo path.

    # UWM server first (its own env), then:
    MUJOCO_GL=egl uv run --with h5py python experiments/uwm/scripts/replay_check.py \
        --checkpoint localhost:6001 --demos 3
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from dream_robot.policies.uwm.policy import CAMERAS, UWMPolicy
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import InstructedPickPlace, TaskConfig

DATA = Path("data/cosmos_policy/t5_224")
JPEG_KEYS = {"top": "agentview_rgb_jpeg", "wrist": "eye_in_hand_rgb_jpeg"}


def frames(group, camera: str) -> np.ndarray:
    return np.stack([np.array(Image.open(io.BytesIO(j.tobytes())).convert("RGB"))
                     for j in group[f"obs/{JPEG_KEYS[camera]}"]])


def ask(policy: UWMPolicy, request: dict) -> np.ndarray:
    policy._conn.send(request)
    return np.asarray(policy._conn.recv()["actions"], np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default="localhost:6001", help="the UWM server's host:port")
    p.add_argument("--demos", type=int, default=3, help="one per instruction file, in order")
    p.add_argument("--out", type=Path, default=Path("experiments/uwm/replay"))
    args = p.parse_args()

    config = TaskConfig.load()
    config = replace(config, camera_hw=(224, 224))
    env = InstructedPickPlace(config, pairs="seen")
    policy = UWMPolicy(env, address=args.checkpoint)
    manifest = json.loads((DATA / "manifest.json").read_text())
    picks = list({m["file"]: m for m in reversed(manifest)}.values())[: args.demos]  # demo 0 per file

    rows = []
    for m in picks:
        with h5py.File(DATA / m["file"]) as f:
            group = f[f"data/demo_{m['demo']}"]
            recorded = {c: frames(group, c) for c in CAMERAS}
            states, actions = group["robot_states"][:], group["actions"][:]
        observation = env.reset(seed=m["seed"])
        policy.reset()
        prompt_ok = env.prompt == m["file"][: -len("_demo.hdf5")].replace("_", " ")
        image_gap = {c: [] for c in CAMERAS}
        state_gap, chunks, success = [], [], False
        for t in range(len(actions)):
            for c in CAMERAS:
                image_gap[c].append(np.abs(observation.images[c].astype(np.int16)
                                           - recorded[c][t].astype(np.int16)).mean())
            state_gap.append(np.abs(np.asarray(observation.state) - states[t]).max())
            if t >= 1 and t % 16 == 1 and t + 16 <= len(actions):
                live = ask(policy, policy._request(observation))
                rec = ask(policy, {
                    "prompt": policy._prompt, "hw": [224, 224],
                    **{c: recorded[c][t - 1: t + 1].tobytes() for c in CAMERAS},
                    "proprio": states[t - 1: t + 1].tolist(),
                })
                truth = actions[t: t + 16]
                chunks.append({"t": t, "live_vs_demo": float(np.abs(live - truth).mean()),
                               "recorded_vs_demo": float(np.abs(rec - truth).mean()),
                               "live_vs_recorded": float(np.abs(live - rec).mean()),
                               "gripper_live_vs_demo": float(np.abs(live[:, 7] - truth[:, 7]).mean())})
            policy._previous = {c: np.ascontiguousarray(observation.images[c], np.uint8) for c in CAMERAS}
            policy._previous_state = np.asarray(observation.state, np.float32).tolist()
            result = env.step(actions[t])
            observation, success = result.observation, success or result.success
        row = {"file": m["file"], "seed": m["seed"], "prompt_matches": prompt_ok, "replay_success": success,
               "image_gap_mean": {c: round(float(np.mean(v)), 2) for c, v in image_gap.items()},
               "image_gap_max": {c: round(float(np.max(v)), 2) for c, v in image_gap.items()},
               "state_gap_max": round(float(np.max(state_gap)), 4),
               "chunks": len(chunks),
               **{k: round(float(np.mean([ch[k] for ch in chunks])), 4)
                  for k in ("live_vs_demo", "recorded_vs_demo", "live_vs_recorded", "gripper_live_vs_demo")}}
        rows.append({**row, "per_chunk": chunks})
        print(json.dumps(row), flush=True)
        # first frame, live | recorded, for eyes
        args.out.mkdir(parents=True, exist_ok=True)
        live0 = env.reset(seed=m["seed"]).images
        Image.fromarray(np.concatenate([np.concatenate([live0[c], recorded[c][0]], 1) for c in CAMERAS], 0)
                        ).save(args.out / f"seed{m['seed']}_live_vs_recorded.png")
    env.close()
    (args.out / "replay.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
