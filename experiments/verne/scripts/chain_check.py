# LLM-loop chain without the LLM (2026-11-20): "A to side S", scripted home, "B to the other side", in one
# episode. Per eval seed: move 1 = the reset's own instruction; move 2 = the other cube to the other bin.
# Each move runs the T5 policy until a robot-sensor stop rule (gripper closed, then reopened with the hand
# > LIFTED above the table, then TAIL more steps) or CAP steps. Home = joint interpolation back to the reset
# pose. Scored at the end from the sim (ground truth, scorer only): which cubes rest in their named bins.
# usage: python experiments/verne/scripts/chain_check.py experiments/verne/runs/22_position_spread/t5/seed0 [n_film (default 0)]
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dream_robot.core.video import observation_panel, write_video
from dream_robot.policies.verne import hand, rollouts
from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.data import one_hot
from dream_robot.policies.verne.run import load_task, to_gpu

REPO = Path(__file__).resolve().parents[3]
CAP, TAIL, HOME, HOLD, CARRY = 600, 20, 45, 10, 10  # steps
CLOSED, OPEN = 0.6, 0.85  # gripper opening (state[7]): below = holding / closed, above = released
LIFTED = 0.06  # m above the table at release: let go of a carried cube, not a missed grasp
N_EXECS = (16,)  # one is enough: 16 scored higher on normal T5 (49 vs 47) and the second move (45 vs 43)
HELD_OUT = {('yellow', 'left'), ('green', 'right')}


def move(net, vae, d, env, prompt, seed, n_exec, frames):
    # the policy on one clause until the stop rule or CAP -> 'stop' / 'cap', steps
    torch.manual_seed(seed * 100 + n_exec)
    ins, drv, plan = one_hot(prompt), hand.Driver(env), []
    obs = env._observe()
    # the reset jaw reads ~0.52 and opens over the first steps, so: armed once fully open; carried = closed
    # with the hand lifted for CARRY steps in a row; then a release
    prev, armed, carry, released_at = obs.state, False, 0, None
    table_z = env._env.table_offset[2]
    for step in range(CAP):
        if not plan:
            plan = drv.plan(obs.state[:7], rollouts.act(net, vae, d, obs, prev, ins)[:n_exec])
        if frames is not None:
            frames.append(observation_panel(env.render(), obs.images))
        r = env.step(drv.action(obs.state[:7], plan.pop(0)))
        prev, obs = obs.state, r.observation
        jaw, z = obs.state[7], env._raw['robot0_eef_pos'][2] - table_z
        armed |= jaw > OPEN
        if armed and carry < CARRY:
            carry = carry + 1 if (jaw < CLOSED and z > LIFTED) else 0
        if carry >= CARRY and released_at is None and jaw > OPEN and z > LIFTED:
            released_at = step
        if released_at is not None and step - released_at >= TAIL:
            return 'stop', step + 1
    return 'cap', CAP


def go_home(env, home, frames):
    # scripted: joints straight-line to the reset pose, gripper open, then hold
    start = env._observe().state[:7]
    for k in range(1, HOME + HOLD + 1):
        q = start + (home - start) * min(k / HOME, 1.0)
        obs = env.step(np.concatenate([q, [1.0]])).observation
        if frames is not None:
            frames.append(observation_panel(env.render(), obs.images))


def chain(net, vae, d, env, seed, n_exec, film):
    obs = env.reset(seed=seed)
    ins = env.instruction
    a, b = ins.target_slot, 1 - ins.target_slot
    side2 = [n for n in env.bin_names if n != ins.side][0]
    prompts = (ins.prompt, f'put the {ins.distractor_color} cube in the {side2} box')
    bins = (env.bin_names.index(ins.side), env.bin_names.index(side2))
    home, frames = obs.state[:7].copy(), [] if film else None
    ends = []
    for k, p in enumerate(prompts):
        ends.append(move(net, vae, d, env, p, seed, n_exec, frames))
        if k == 0:
            cubes = env._cube_positions()
            after1 = env.which_bin(cubes[a]) == bins[0]
            go_home(env, home, frames)
    cubes = env._cube_positions()
    done = (env.which_bin(cubes[a]) == bins[0], env.which_bin(cubes[b]) == bins[1])
    row = {
        'prompts': prompts,
        'move1_after_move1': bool(after1),
        'done': [bool(x) for x in done],
        'ends': ends,
        'held_out_2': (ins.distractor_color, side2) in HELD_OUT,
    }
    if film:
        tag = f'{ins.color}_{ins.side}_then_{ins.distractor_color}_{side2}_{sum(done)}of2'
        path = REPO / f'scratch_nov/verne_chain/seed{seed}_n{n_exec}_{tag}.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_video(frames + [frames[-1]] * 15, path, fps=env.control_hz)
    return row


def main(run: Path, n_film: int) -> None:
    out = run / 'chain'
    out.mkdir(exist_ok=True)
    state, action, episode, latent, instruction, slot, train_ids, _, _ = load_task('t5')
    d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
    net = rollouts.load(torch.load(run / 'weights.pt'))
    vae = vae_mod.load(vae_mod.WEIGHTS)
    env = rollouts.make_env('t5', horizon_seconds=90)  # room for move, home, move; we never stop on its flags
    for n_exec in N_EXECS:
        rows = {}
        for i, seed in enumerate(rollouts.SEEDS):
            rows[seed] = r = chain(net, vae, d, env, seed, n_exec, film=i < n_film)
            print(f'n_exec {n_exec} seed {seed} {r["prompts"][0].split()[2:7:4]} -> {r["prompts"][1].split()[2:7:4]}'
                  f'  done {r["done"]}  ends {r["ends"]}  held_out_2 {r["held_out_2"]}', flush=True)
        n = len(rows)
        score = Counter(sum(r['done']) for r in rows.values())
        m1 = sum(r['done'][0] for r in rows.values())
        m2 = sum(r['done'][1] for r in rows.values())
        seen = [r for r in rows.values() if not r['held_out_2']]
        print(f'n_exec {n_exec:2d}: both {score[2]}/{n}, one {score[1]}, none {score[0]};  move 1 {m1}/{n}, move 2 {m2}/{n}')
        print(f'  seen-pair move 2 only: both {sum(all(r["done"]) for r in seen)}/{len(seen)}'
              f', move 2 {sum(r["done"][1] for r in seen)}/{len(seen)}')
        print(f'  move 1 placed before home {sum(r["move1_after_move1"] for r in rows.values())}/{n};'
              f' ends move1 {dict(Counter(r["ends"][0][0] for r in rows.values()))},'
              f' move2 {dict(Counter(r["ends"][1][0] for r in rows.values()))}', flush=True)
        (out / f'chain_{n_exec}.json').write_text(json.dumps({str(s): r for s, r in rows.items()}, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
