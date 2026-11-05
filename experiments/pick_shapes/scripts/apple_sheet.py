"""The rung 1 apples side by side: apple_0 (trained on) + apple_1..10 (test), objaverse registry.

Same standalone render as contact_sheet.py (5 cm checker). RoboCasa's category scale (0.9 for
objaverse apples) is the same for all, so relative sizes are right; absolute size in the kitchen
is in the eval log's size_m.

    MUJOCO_GL=egl envs/robocasa/.venv/bin/python experiments/pick_shapes/scripts/apple_sheet.py
"""

import sys
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from contact_sheet import ASSETS, TILE, scene_xml  # noqa: E402

OUT = Path('experiments/pick_shapes/apples_rung1.png')
MESHES = [f'apple_{i}' for i in range(11)]


def render(model_path: Path):
    m = mujoco.MjModel.from_xml_string(scene_xml(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.0]
    cam.distance, cam.azimuth, cam.elevation = 0.3, 135, -35
    with mujoco.Renderer(m, TILE, TILE) as r:
        r.update_scene(d, cam)
        return r.render()


cols = 6
sheet = Image.new('RGB', (cols * TILE, 2 * TILE), 'white')
draw = ImageDraw.Draw(sheet)
for i, name in enumerate(MESHES):
    img = render(ASSETS / 'objaverse/apple' / name / 'model.xml')
    x, y = (i % cols) * TILE, (i // cols) * TILE
    sheet.paste(Image.fromarray(img), (x, y))
    draw.text((x + 6, y + 6), name + ('  (TRAINED)' if i == 0 else '  (test)'), fill='white')
sheet.save(OUT)
print('wrote', OUT)
