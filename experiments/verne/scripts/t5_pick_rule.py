# Which cube does a T5 policy grasp, if not the named one? From a run's probe.json (gripper x at the first
# close; the cubes sit at x -7 / +7 cm) and the T5 env's draws on the same seeds: the cube it went for
# (slot, colour) against the named one, tallied by colour and by slot.
# usage: python experiments/verne/scripts/t5_pick_rule.py experiments/verne/runs/07_instruction/t5/seed0
import json
import sys
from collections import Counter
from pathlib import Path

from dream_robot.policies.verne.rollouts import make_env

p = json.loads((Path(sys.argv[1]) / 'probe.json').read_text())
env = make_env('t5')
by_colour, by_slot, rows = Counter(), Counter(), []
for seed, (cube, grip) in p["points"].items():
    if grip is None:  # never closed
        continue
    env.reset(seed=int(seed))
    ins = env.instruction
    xs = env._cube_positions()[:, 0]
    colours = [ins.distractor_color] * 2
    colours[ins.target_slot] = ins.color
    picked = int(abs(grip[0] - xs[1]) < abs(grip[0] - xs[0]))  # nearer cube slot
    rows.append((seed, ins.color, colours[picked], picked, ins.target_slot))
    by_colour[(colours[picked], 'named' if picked == ins.target_slot else 'other')] += 1
    by_slot[(picked, 'named' if picked == ins.target_slot else 'other')] += 1
env.close()
right = sum(r[3] == r[4] for r in rows)
print(f'grasped the named cube {right}/{len(rows)}')
print('slot picked (x of slot 0 = %.0f cm, slot 1 = %.0f cm):' % tuple(100 * xs), dict(sorted(by_slot.items())))
print('colour picked:', dict(sorted(by_colour.items())))
pairs = Counter((frozenset((r[1], r[2])) if r[1] != r[2] else None) for r in rows)
picked_colour_given_pair = Counter()
for seed, named, got, _, _ in rows:
    picked_colour_given_pair[(named, got)] += 1
print('(named, grasped colour):', dict(sorted(picked_colour_given_pair.items())))
