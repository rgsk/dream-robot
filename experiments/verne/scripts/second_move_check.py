# LLM-loop check (2026-11-20): can a T5 policy do the SECOND move of "red left, blue right" when it
# gets one clause at a time? Each eval seed's T5 reset, then the other (unnamed) cube is put into the
# bin the instruction does NOT name and settled, arm left at home (= a scripted home after move 1).
# The policy gets the usual one-clause prompt for the named cube. Same 50 eval seeds and scoring as a
# normal sweep, so the score compares with the run's sweep_8 / sweep_16.json.
# usage: python experiments/verne/scripts/second_move_check.py experiments/verne/runs/22_position_spread/t5/seed0
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dream_robot.policies.verne import rollouts
from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.run import load_task, to_gpu

SETTLE = 20  # control steps holding still while the dropped cube settles (~0.7 s)


class Prefilled:
    # the T5 env with the unnamed cube already in the other bin; everything else delegated
    def __init__(self, env):
        self.env = env
        self.placed = {}  # seed -> did the cube settle inside the opposite bin

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None):
        env = self.env
        obs = env.reset(seed=seed)
        other = 1 - env.instruction.target_slot
        far_bin = 1 - env.target_bin
        cx, cy, cz = env.bin_centers[far_bin]
        sim = env._env.sim
        joint = env._env.cubes[other].joints[0]
        q = np.array(sim.data.get_joint_qpos(joint))  # x y z + quaternion (keeps its spawn yaw)
        q[:3] = cx, cy, cz + 0.04  # 2 cm above the floor, falls flat
        sim.data.set_joint_qpos(joint, q)
        sim.data.set_joint_qvel(joint, np.zeros(6))
        sim.forward()
        hold = obs.state[:8].copy()  # joint targets = where the arm is; jaw as it is
        for _ in range(SETTLE):
            obs = env.step(hold).observation
        self.placed[seed] = env.which_bin(env.distractor_pos) == far_bin
        # the episode starts now: fresh clock and lift references (the settled cube's z is its spawn)
        env._t = 0
        env._env.timestep = 0  # robosuite's own clock too, or its horizon ends the episode 20 steps early
        env._z0 = env._cube_positions()[:, 2].copy()
        env._max_z = env._z0.copy()
        return obs


def main(run: Path) -> None:
    out = run / 'second_move'
    out.mkdir(exist_ok=True)
    state, action, episode, latent, instruction, slot, train_ids, _, _ = load_task('t5')
    d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
    net = rollouts.load(torch.load(run / 'weights.pt'))
    vae = vae_mod.load(vae_mod.WEIGHTS)
    env = Prefilled(rollouts.make_env('t5'))

    obs = env.reset(seed=rollouts.SEEDS[0])  # a look at the start scene
    Image.fromarray(np.concatenate(list(obs.images.values()), axis=1)).save(out / 'start_scene.png')

    for n_exec in (8, 16):
        res, side = {}, {}
        for seed in rollouts.SEEDS:
            res[seed] = rollouts.rollout(net, vae, d, env, seed, n_exec)
            side[seed] = env.instruction.side
            print(f'n_exec {n_exec} seed {seed} {side[seed]:5s} placed {env.placed[seed]}  {res[seed]}', flush=True)
        ok = sum(r[0] for r in res.values())
        fails = Counter(r[1] for r in res.values() if not r[0])
        by_side = {k: f'{sum(res[s][0] for s in res if side[s] == k)}/{sum(side[s] == k for s in res)}' for k in sorted(set(side.values()))}
        print(f'n_exec {n_exec:2d}: {ok}/{len(res)}  failures {dict(fails)}  by side {by_side}')
        print(f'  other cube settled in its bin: {sum(env.placed.values())}/{len(env.placed)}')
        normal = json.loads((run / f'sweep_{n_exec}.json').read_text())
        print(f'  normal T5 on the same seeds: {sum(v[0] for v in normal.values())}/{len(normal)}', flush=True)
        (out / f'sweep_{n_exec}.json').write_text(
            json.dumps({str(s): list(r) + [bool(env.placed[s])] for s, r in res.items()}, indent=1)
        )


if __name__ == '__main__':
    main(Path(sys.argv[1]))
