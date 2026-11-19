"""Step 19 diagnosis: does the arm use the frame and the colour word where the cube is chosen? Val action loss at
fixed flow times τ (0 = noise), on held-out moments from the decision window (the arm near home, before it shows
which cube; clips.decision_steps) and on uniform moments, with:

    true      the moment's own frame, joints and prompt
    frame     another moment's frame (rolled in the batch; joints and prompt kept)
    colour    the colour word changed to a different colour (side word kept)

gap = changed − true: how much the arm's prediction rests on that input. An arm that reads 'which cube' from the
frame should lose a lot when the colour word changes at decision moments (it should reach for another cube).

    python experiments/verne/scripts/arm_use_check.py name=path/to/weights.pt ...
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

from dream_robot.policies.verne import clips
from dream_robot.policies.verne.data import COLOURS, get_batch, to_gpu
from dream_robot.policies.verne.model import Verne
from dream_robot.policies.verne.run import load_task

state, action, episode, latent, instruction, slot, train_ids, val_ids, _ = load_task('t5')
d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
decision = clips.decision_steps(d, val_ids)
sets = {}
for name, early in (('decision', 1.0), ('uniform', 0.0)):
    rng = np.random.default_rng(0)
    sets[name] = [get_batch(d, val_ids, 256, rng, early, decision) for _ in range(4)]
TAUS = (0.0, 0.18, 0.4, 0.67)


def other_colour(ins: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    out = ins.clone()
    c = ins[:, COLOURS].argmax(1)
    shift = torch.randint(1, 4, c.shape, device=ins.device, generator=g)
    out[:, COLOURS] = 0
    out[torch.arange(len(ins)), COLOURS.start + (c + shift) % 4] = 1
    return out


for arg in sys.argv[1:]:
    name, _, path = arg.partition('=')
    net = Verne().cuda().eval()
    missing, unexpected = net.load_state_dict(torch.load(path), strict=False)
    assert not unexpected and all(
        k.startswith(('time_video', 'gen_out', 'future_heads', 'change_out')) for k in missing
    ), missing
    print(f'{name}')
    for which, batches in sets.items():
        g = torch.Generator('cuda').manual_seed(0)
        print(f'  {which:8s} tau    true    frame (gap)       colour (gap)')
        for tau in TAUS:
            acc = {'true': 0.0, 'frame': 0.0, 'colour': 0.0}
            for b in batches:
                noise = torch.randn(b.acts.shape, device='cuda', generator=g)
                t = torch.full((len(noise),), tau, device='cuda')
                x = (1 - tau) * noise + tau * b.acts
                swapped = other_colour(b.instruction, g)
                with torch.no_grad():
                    for key, frame, ins in (
                        ('true', b.frame, b.instruction),
                        ('frame', b.frame.roll(1, 0), b.instruction),
                        ('colour', b.frame, swapped),
                    ):
                        v = net.velocity(net.see(frame, b.state, ins), x, t, ins, b.state)
                        acc[key] += F.mse_loss(v, b.acts - noise).item() / len(batches)
            tr = acc['true']
            print(
                f'           {tau:.2f}  {tr:.3f}   {acc["frame"]:.3f} ({acc["frame"] - tr:+.3f})'
                f'    {acc["colour"]:.3f} ({acc["colour"] - tr:+.3f})'
            )
print('------ FINISHED ------')
