"""Step 3 visuals. Replays the recorded seeds (clean and sigma 0.05 -- same seeding as the recorder, so
these are the dataset's episodes) and measures gripper-to-cube xy error at grasp height while open.
Writes a clean-vs-noisy coverage plot and a video of the biggest correction. Result logged in ../notes.md."""
import json, os, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from dream_robot.core.record import JointNoise
from dream_robot.core.video import observation_panel, write_video
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy, EEF_SITE
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

LOW, OUT = 3.0, "scratch/noisy_expert"
BC_FAIL_ARRIVAL = [3.15, 3.60, 1.43, 1.14, 3.86, 2.73, 1.31, 3.37, 2.71, 2.13, 2.79, 3.32, 1.16, 1.19, 1.96]  # step 1
env = PickPlaceCube(); expert = ExpertActor(ExpertPolicy(env))

def kept_seeds(root):
    s = json.load(open(f"{root}/recording_summary.json"))
    return [e["seed"] for e in s["episodes"] if e["kept"]]

def replay(seed, sigma, film=False):
    noise = JointNoise(sigma, env.embodiment) if sigma > 0 else None
    obs = env.reset(seed=seed); expert.reset(); rng = np.random.default_rng(seed)
    m, d = env.mujoco; sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    rows, frames = [], []
    while True:
        g = d.site_xpos[sid] - env.cube_pos
        rows.append((np.linalg.norm(g[:2]) * 100, g[2] * 100, obs.state[7]))
        if film: frames.append(observation_panel(env.render(), obs.images))
        a = expert(obs)
        r = env.step(a if noise is None else noise(a, rng)); obs = r.observation
        if r.terminated or r.truncated: return np.array(rows), frames, r.success

near = {}
for label, root, sigma in [("clean demos", "data/robosuite/pick_place_cube", 0.0),
                           ("shaky demos (sigma 0.05)", "data/robosuite/pick_place_cube_noise005", 0.05)]:
    vals, worst, mismatches = [], (0.0, None), 0
    for seed in kept_seeds(root):
        rows, _, ok = replay(seed, sigma)
        mismatches += not ok
        sel = rows[(rows[:, 1] < LOW) & (rows[:, 2] > 0.9), 0]
        vals.extend(sel)
        if len(sel) and sel.max() > worst[0]: worst = (sel.max(), seed)
    vals = np.array(vals); near[label] = vals
    print(f"{label}: {len(vals)} grasp-height open steps | >1 cm {np.sum(vals > 1)}  >2 cm {np.sum(vals > 2)}  "
          f"max {vals.max():.2f} cm (seed {worst[1]}) | replay mismatches vs recording: {mismatches}", flush=True)
    if sigma > 0: worst_seed = worst[1]

fig, ax = plt.subplots(figsize=(9, 3.4))
bins = np.linspace(0, 4, 41)
for (label, v), c in zip(near.items(), ["tab:blue", "tab:orange"]):
    ax.hist(v, bins=bins, alpha=0.6, color=c, label=f"{label}: {np.sum(v > 1)} steps > 1 cm")
ax.set_yscale("log")
for x in BC_FAIL_ARRIVAL: ax.axvline(x, color="tab:red", alpha=0.35, lw=1.5)
ax.plot([], [], color="tab:red", alpha=0.5, label="where BC's failed episodes arrived")
ax.set_xlabel("gripper-to-cube xy error at grasp height, gripper open (cm)")
ax.set_ylabel("steps in demos (log)"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/demo_coverage_clean_vs_shaky.png", dpi=130)

rows, frames, ok = replay(worst_seed, 0.05, film=True)
write_video(frames, f"{OUT}/shaky_expert_recovery_seed{worst_seed}.mp4", fps=env.control_hz)
print(f"video: seed {worst_seed}, {len(frames)} frames, success={ok}")
env.close()
