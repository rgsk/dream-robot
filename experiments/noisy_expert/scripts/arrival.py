import os, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy, EEF_SITE
from dream_robot.policies.bc.policy import BCPolicy
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

OUT = "scratch/noisy_expert"; LOW = 3.0
env = PickPlaceCube()
pols = {"expert": ExpertActor(ExpertPolicy(env)), "bc": BCPolicy.load("experiments/bc_pick_place_cube/checkpoint.pt")}

def run(p, seed):
    obs = env.reset(seed=seed); p.reset(); rows, Q, wrist = [], [], []
    m, d = env.mujoco; sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    while True:
        g = d.site_xpos[sid] - env.cube_pos
        rows.append((np.linalg.norm(g[:2]) * 100, g[2] * 100, obs.state[7]))
        Q.append(obs.state[:7].copy()); wrist.append(obs.images["wrist"].copy())
        r = env.step(p(obs)); obs = r.observation
        if r.terminated or r.truncated: return np.array(rows), np.array(Q), wrist, r

def classify(rows, Q, r, a):
    if r.success: return "success"
    after = rows[a:, 2]
    if after.min() < 0.1: return "closed_on_air"
    if after.min() < 0.9: return "closed_then_lost"
    last = int(np.nonzero(np.abs(np.diff(Q, axis=0)).max(1) > 1e-3)[0][-1])
    return (f"frozen_open@{last}" if last < len(Q) - 100 else f"hovering_open@{last}")

data = {}
for name, p in pols.items():
    for seed in range(1000, 1020):
        rows, Q, wrist, r = run(p, seed)
        a = int(np.nonzero(rows[:, 1] < LOW)[0][0])
        kind = classify(rows, Q, r, a)
        data[(name, seed)] = (rows[a, 0], kind, wrist[min(a + 15, len(wrist) - 1)], len(rows))
        print(f"{name:6} {seed}  arrival xy {rows[a,0]:5.2f} cm at t={a:3}  {kind}", flush=True)

for name in pols:
    xs = np.array([data[(name, s)][0] for s in range(1000, 1020)])
    kinds = [data[(name, s)][1].split("@")[0] for s in range(1000, 1020)]
    print(f"\n{name}: arrival xy median {np.median(xs):.2f}, range {xs.min():.2f}..{xs.max():.2f} cm; "
          f"{ {k: kinds.count(k) for k in sorted(set(kinds))} }")

# strip plot
fig, ax = plt.subplots(figsize=(8, 3))
colors = {"success": "tab:green", "frozen_open": "tab:red", "closed_on_air": "tab:orange",
          "closed_then_lost": "tab:purple", "hovering_open": "tab:brown"}
rowy = {"expert": 1, "bc": 0}
for (name, seed), (xy, kind, _, _) in data.items():
    ax.scatter(xy, rowy[name] + np.random.default_rng(seed).uniform(-0.15, 0.15), c=colors[kind.split("@")[0]], s=40)
ax.axvline(0.25, ls="--", c="k"); ax.text(0.3, 1.35, "widest error in any demo (0.25 cm)", fontsize=8)
for k, c in colors.items(): ax.scatter([], [], c=c, label=k)
ax.set_yticks([0, 1], ["BC", "expert"]); ax.set_ylim(-0.5, 1.6)
ax.set_xlabel("gripper-to-cube xy error on arriving at grasp height (cm), eval seeds 1000-1019")
ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(f"{OUT}/arrival_xy.png", dpi=130)

# wrist frames 15 steps after arrival: expert vs BC on the two failure seeds + one BC success
pick = [("expert", 1000), ("bc", 1000), ("expert", 1005), ("bc", 1005), ("bc", 1010)]
strip = np.concatenate([np.kron(data[k][2], np.ones((3, 3, 1), dtype=np.uint8)) for k in pick], axis=1)
Image.fromarray(strip).save(f"{OUT}/wrist_at_arrival.png")
print("\nwrist strip order:", pick)
env.close()
