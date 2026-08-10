import robosuite as suite
import numpy as np
from robosuite.controllers import load_composite_controller_config
import imageio

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

def t5():
    pass

t3t4()