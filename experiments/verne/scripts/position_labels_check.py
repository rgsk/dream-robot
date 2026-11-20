# Step 23 check: are the position targets (step 22's: per scene and cube, its centre per camera from the latent
# erase renders, in patches of each camera's 8 x 8 grid) on the cube's centre? Renders val reset scenes as raw
# pixels (as the pixel run sees them) and draws each cube's target as a cross in that cube's colour, the 8 x 8
# patch grid faint, both cameras, 3x size.
#   python experiments/verne/scripts/position_labels_check.py [n]   -> scratch_nov/verne_pixels/position_labels.png
import sys

import numpy as np
from PIL import Image, ImageDraw

from dream_robot.policies.verne import position as pos
from dream_robot.policies.verne import vae
from dream_robot.policies.verne.data import SCENE_SEEDS, load_erased, load_scenes
from dream_robot.policies.verne.rollouts import make_env

n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
sc = load_scenes('val', vae.WEIGHTS, frames='latent')
cen = pos.centres(sc, load_erased('val', vae.WEIGHTS)).cpu().numpy()  # [N, 2 slots, 4]
env = make_env('t5')
S, tiles = 3, []
for i, seed in enumerate(list(SCENE_SEEDS['val'])[:n]):
    obs = env.reset(seed=seed)
    ins = env.instruction
    colours = [ins.distractor_color, ins.distractor_color]
    colours[ins.target_slot] = ins.color
    pic = np.concatenate([obs.images['top'], obs.images['wrist']], 1)
    im = Image.fromarray(pic).resize((256 * S, 128 * S), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for k in range(1, 16):  # patch grid: 16 px per patch
        d.line([(k * 16 * S, 0), (k * 16 * S, 128 * S)], fill=(200, 200, 200), width=1)
    for k in range(1, 8):
        d.line([(0, k * 16 * S), (256 * S, k * 16 * S)], fill=(200, 200, 200), width=1)
    for slot in (0, 1):
        rgb = tuple(int(255 * c) for c in env._cfg.palette[colours[slot]][:3])
        for cam in (0, 1):
            r, c = cen[i, slot, 2 * cam], cen[i, slot, 2 * cam + 1]
            x, y = (c + 8 * cam) * 16 * S, r * 16 * S
            for dx, dy, w in ((14, 0, 7), (0, 14, 7)):  # black outline, then the cube's colour
                d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(0, 0, 0), width=w)
            d.line([(x - 12, y), (x + 12, y)], fill=rgb, width=3)
            d.line([(x, y - 12), (x, y + 12)], fill=rgb, width=3)
    d.text((6, 4), f'seed {seed}: "{ins.color} {ins.side}"  (near {colours[0]}, far {colours[1]})', fill=(0, 0, 0))
    tiles.append(np.asarray(im))
env.close()
gap = np.full((8, tiles[0].shape[1], 3), 255, np.uint8)
out = np.concatenate([x for t in tiles for x in (t, gap)][:-1], 0)
Image.fromarray(out).save('scratch_nov/verne_pixels/position_labels.png')
print('scratch_nov/verne_pixels/position_labels.png', out.shape)
