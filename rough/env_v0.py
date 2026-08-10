import robosuite as suite
import numpy as np
from robosuite.controllers import load_composite_controller_config
import imageio
from robosuite.environments.manipulation.lift import Lift
from robosuite.utils.mjcf_utils import new_body, new_geom, array_to_string
import inspect
import xml.etree.ElementTree as ET
def t1():
    env = suite.make(
        "Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    )
    print(f'{env.action_dim=}')
    obs = env.reset()
    print({k: v.shape for k, v in obs.items()})


    assert np.array_equal(
        np.concatenate([obs["cube_pos"], obs["cube_quat"], obs["gripper_to_cube_pos"]]),
        obs["object-state"],
    )
    print("object-state == cube_pos ++ cube_quat ++ gripper_to_cube_pos ✓")
    proprio_state_built = []
    for key in obs:
        if key.startswith('robot0_') and key != 'robot0_proprio-state':
            proprio_state_built.append(obs[key])
    assert np.array_equal(
        np.concatenate(proprio_state_built),
        obs["robot0_proprio-state"],
    )
    print('robot0_proprio-state is concatenation of other keys starting with robot0_')

def t2():
    cfg = load_composite_controller_config(robot="Panda")
    print(cfg)
    cfg['body_parts']['right']['type'] = 'JOINT_POSITION'
    cfg['body_parts']['right']['input_type'] = 'absolute'

    env = suite.make(
        "Lift", robots="Panda", controller_configs=cfg,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    )
    print(f'{env.action_dim=}')
    obs = env.reset()
    # print({k: v.shape for k, v in obs.items()})


    print(f'{obs['robot0_joint_pos']=}')
    action = np.concatenate([obs["robot0_joint_pos"], [0.0]])
    obs, reward, done, info = env.step(action)
    print(f'{obs['robot0_joint_pos']=}')
    print(f'init  : {np.abs(action[:7] - obs["robot0_joint_pos"])}')
    action[3] += 0.1
    for i in range(20):
        obs, reward, done, info = env.step(action)
        print(f'iter {i}: {np.abs(action[:7] - obs["robot0_joint_pos"])}')
def t3t4():
    env = suite.make(
        "Lift",
        robots="Panda",
        camera_names=["agentview", "robot0_eye_in_hand"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    obs = env.reset()
    imageio.imwrite("rough/generated/t3-agent.png", obs["agentview_image"])
    imageio.imwrite("rough/generated/t3-agent-rev.png", obs["agentview_image"][::-1])
    imageio.imwrite("rough/generated/t3-robot.png", obs["robot0_eye_in_hand_image"])
    imageio.imwrite("rough/generated/t3-robot-rev.png", obs["robot0_eye_in_hand_image"][::-1])
    print({k: v.shape for k, v in obs.items()})

class BinLift(Lift):  
    def _load_model(self):
        super()._load_model()
        print(type(self.model))
        print(f'{self.table_full_size=}')
        print(f'{self.table_offset=}')
        print(inspect.signature(new_geom))
        g = new_geom(
            name="goal_bin_floor", type="box", size=[0.08, 0.08, 0.02], group=1
        )
        
        b = new_body(
            name="goal_bin",
            pos=[self.table_offset[0], self.table_offset[1] + 0.2, 0.8 + 0.02],
        )
        b.append(g)
        t = 0.005          # wall half-thickness
        hx = hy = 0.08     # floor half-extents
        hz = 0.025         # wall half-height
        z  = 0.02 + hz     # stand on the slab

        walls = [
            ("px", [ hx, 0.0, z], [t, hy, hz]),
            ("nx", [-hx, 0.0, z], [t, hy, hz]),
            ("py", [0.0,  hy, z], [hx, t, hz]),
            ("ny", [0.0, -hy, z], [hx, t, hz]),
        ]
        for suffix, pos, size in walls:
            b.append(new_geom(name=f"goal_bin_wall_{suffix}", type="box",
                            size=size, pos=pos, group=1))

        print(ET.tostring(b))
        self.model.worldbody.append(b)

def t5():
    env = BinLift(
        robots="Panda",
        camera_names=["agentview", "robot0_eye_in_hand"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    print(f'{"goal_bin_floor" in env.sim.model.geom_names=}')
    obs = env.reset()
    print(f'{obs["cube_pos"][2]=}, {env.table_offset[2]=}')
    # imageio.imwrite("rough/generated/t5-agent-group0.png", obs["agentview_image"][::-1])
    imageio.imwrite("rough/generated/t5-agent-group1.png", obs["agentview_image"][::-1])
    # imageio.imwrite("rough/generated/t5-robot.png", obs["robot0_eye_in_hand_image"][::-1])


t5()
