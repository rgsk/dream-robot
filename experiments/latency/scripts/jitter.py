"""Arm jitter with vs without execution noise: |dq| and |d2q| of measured joints, FM T2-fixed."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "experiments/latency/scripts")
from lag_sweep import CHECKPOINTS, ENV, LaggedPlanner
from dream_robot.core.record import JointNoise
from dream_robot.core.registry import make_env, make_policy

env = make_env(ENV)
policy = make_policy("flow_matching", env, checkpoint=CHECKPOINTS["flow_matching"])
arm = env.embodiment.arm_slice
EXECUTE = {0.0: 8, 0.16: 8, 0.3: 7}  # delay + execute <= 16
for mode, latency in [("hold", 0.0), ("async", 0.16), ("rtc", 0.16), ("async", 0.3), ("rtc", 0.3)]:
    for sigma in (0.0,):
        d1, d2 = [], []
        for seed in range(1000, 1005):
            planner = LaggedPlanner(policy, execute=EXECUTE[latency], delay=round(latency * 30), mode=mode)
            obs = env.reset(seed=seed); planner.reset(); policy.seed(seed)
            rng, noise, qs = np.random.default_rng(seed), JointNoise(sigma, env.embodiment), []
            while True:
                qs.append(np.asarray(obs.state)[arm])
                r = env.step(noise(planner(obs), rng)); obs = r.observation
                if r.terminated or r.truncated: break
            q = np.array(qs)
            d1.append(np.abs(np.diff(q, axis=0)).mean()); d2.append(np.abs(np.diff(q, 2, axis=0)).mean())
        print(f"{mode:5s} lag {latency:4.2f}  noise {sigma:5.3f}:  |dq| {1000*np.mean(d1):5.2f} mrad/step"
              f"   wobble |d2q| {1000*np.mean(d2):5.2f} mrad/step^2", flush=True)
env.close()
