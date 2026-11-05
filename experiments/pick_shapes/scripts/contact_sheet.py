"""Contact sheet of RoboCasa graspable objects, grouped by grasp shape family.

Each object's model.xml is loaded on its own (no kitchen, no robot) on a 5 cm checker floor,
so shape and size read at a glance. One row per family, one mesh per category.

    MUJOCO_GL=egl envs/robocasa/.venv/bin/python experiments/pick_shapes/scripts/contact_sheet.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ASSETS = Path('envs/robocasa/src/robocasa/robocasa/models/assets/objects')
ROBOSUITE_TEX = Path('envs/robocasa/src/robosuite/robosuite/models/assets/textures').resolve()
OUT = Path('experiments/pick_shapes/contact_sheet.png')
FAMILIES = {
    'sphere': ['apple', 'orange', 'tomato', 'onion'],
    'cylinder': ['can', 'bottled_water', 'ketchup', 'spray'],
    'box': ['boxed_food', 'cereal', 'boxed_drink', 'butter_stick'],
    'long_thin': ['banana', 'carrot', 'baguette', 'wooden_spoon'],
    'handled_hollow': ['mug', 'bowl', 'pitcher', 'pan'],
    'irregular_soft': ['broccoli', 'croissant', 'fish', 'sponge'],
}
TILE = 256


def first_mesh(cat: str) -> Path:
    # out: model.xml of the first instance found, aigen first, then objaverse, lightwheel
    for reg in ('aigen_objs', 'objaverse', 'lightwheel'):
        d = ASSETS / reg / cat
        if d.is_dir():
            inst = sorted(p for p in d.iterdir() if (p / 'model.xml').exists())
            if inst:
                return inst[0] / 'model.xml'
    raise FileNotFoundError(cat)


def scene_xml(model: Path) -> str:
    # out: the object's MJCF with absolute asset paths, plus floor, light and camera
    root = ET.parse(model).getroot()
    for el in root.iter():
        if 'file' in el.attrib:
            p = (model.parent / el.get('file')).resolve()
            if not p.exists():  # some meshes point at their authors' machines; robosuite ships the textures
                p = ROBOSUITE_TEX / Path(el.get('file')).name
            el.set('file', str(p))
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', name='chk', type='2d', builtin='checker', width='64', height='64',
                  rgb1='.85 .85 .85', rgb2='.7 .7 .7')
    ET.SubElement(asset, 'material', name='chk', texture='chk', texrepeat='10 10')  # 0.5 m / 10 = 5 cm
    wb = root.find('worldbody')
    ET.SubElement(wb, 'geom', type='plane', size='0.25 0.25 .01', material='chk')
    ET.SubElement(wb, 'light', pos='0.3 -0.3 0.8', dir='-0.3 0.3 -0.8', diffuse='.9 .9 .9')
    ET.SubElement(wb, 'light', pos='-0.3 0.3 0.8', dir='0.3 -0.3 -0.8', diffuse='.4 .4 .4')
    return ET.tostring(root, encoding='unicode')


def render(cat: str) -> tuple[np.ndarray, str]:
    model_path = first_mesh(cat)
    m = mujoco.MjModel.from_xml_string(scene_xml(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # lift the object so its lowest collision point sits on the floor
    lo = min((d.geom_xpos[g][2] - m.geom_rbound[g]) for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'object')
    m.body_pos[m.body_parentid[body]][2] -= lo * 0.8  # rbound overestimates; 0.8 keeps it near the floor
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.04]
    cam.distance, cam.azimuth, cam.elevation = 0.45, 135, -35
    with mujoco.Renderer(m, TILE, TILE) as r:
        r.update_scene(d, cam)
        return r.render(), model_path.parent.name


def main() -> None:
    cols = max(len(v) for v in FAMILIES.values())
    sheet = Image.new('RGB', (cols * TILE + 140, len(FAMILIES) * TILE), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (fam, cats) in enumerate(FAMILIES.items()):
        draw.text((8, i * TILE + TILE // 2), fam, fill='black')
        for j, cat in enumerate(cats):
            img, name = render(cat)
            sheet.paste(Image.fromarray(img), (140 + j * TILE, i * TILE))
            draw.text((140 + j * TILE + 6, i * TILE + 6), name, fill='white')
    sheet.save(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
