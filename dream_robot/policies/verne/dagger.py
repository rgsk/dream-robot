"""DAgger for Verne: a trained policy drives, the state-based expert labels the states it reaches.

Demos only show the expert's own path, so a policy that drifts off it lands in states no demo labels
(a crooked hold, a square wrist that never closes, a missed grasp) and stalls there. DAgger records
the policy's own episodes with the expert's action at every step as the label: the next training run
learns what to do exactly where the last policy went wrong.

The expert is StateExpert (sims/robosuite/tasks/pick_place_cube/expert.py): it reads its phase from
the scene each step, so it can label any state, and it retries (open, rise clear, re-aim) after a
missed grasp or a slip, which no recorded demo contains.

The recorder is the usual one (core/record.py), used the other way round: the "policy" it records is
the expert (labels), and its perturbation hook executes the Verne policy's action instead. Every
episode is kept, failures included: their labels are still right, and they are the states that matter.
"""

from types import SimpleNamespace

import numpy as np
import torch

from dream_robot.core.record import record_episodes

from . import rollouts, vae
from .data import REPO, one_hot
from .run import RUNS, load_task


class DaggerActor:
    # records the expert's label; keeps the Verne policy's action for RunPolicy to execute
    def __init__(self, expert, net, enc, d, n_exec: int = 8, seed_start: int = 0):
        self.expert, self.net, self.enc, self.d, self.n_exec = expert, net, enc, d, n_exec
        self.episode, self.plan, self.executed = seed_start - 1, [], None

    def reset(self) -> None:
        self.expert.reset()
        self.episode += 1
        torch.manual_seed(self.episode * 100 + self.n_exec)  # as a sweep episode's noise
        self.plan, self.executed = [], None

    def __call__(self, observation) -> np.ndarray:
        label = self.expert(observation)
        if not self.plan:
            ins = one_hot(self.expert._env.prompt)  # T1: all zeros
            self.plan = list(
                rollouts.act(self.net, self.enc, self.d, observation, ins)[: self.n_exec]
            )
        self.executed = self.plan.pop(0)
        return label


class RunPolicy:
    # the recorder's perturbation hook: execute the policy's action, not the recorded label
    def __init__(self, actor: DaggerActor, source: str):
        self.actor, self.source = actor, source

    def __call__(self, action, rng):
        return self.actor.executed

    def to_dict(self) -> dict:
        return {'type': 'dagger', 'policy': self.source}


def collect(policy_run: str, base_task: str, out_name: str, episodes: int, seed_start: int):
    # policy_run: a run folder under RUNS, e.g. '05_square/t1-wide-square-noise/seed0'; base_task: the
    # task it was trained on (any load_task name). Writes data/robosuite/<out_name>.
    from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import StateExpert
    from dream_robot.sims.robosuite.tasks.pick_place_cube.record import REPO_ID

    # the stats the policy was trained with: its whole task's training rows (base + any extra data)
    state, action, episode, _, _, _, train_ids, _, scene = load_task(base_task)
    lab = np.isin(episode, train_ids)
    stats = dict(
        s_mean=state[lab].mean(0),
        s_std=state[lab].std(0, ddof=1),
        a_mean=action[lab].mean(0),
        a_std=action[lab].std(0, ddof=1),
    )
    d = SimpleNamespace(**{k: torch.tensor(v, device='cuda') for k, v in stats.items()})
    net = rollouts.load(torch.load(RUNS / policy_run / 'weights.pt'))
    env = rollouts.make_env(scene)
    actor = DaggerActor(
        StateExpert(env, square_up=True), net, vae.load(vae.WEIGHTS), d, seed_start=seed_start
    )
    try:
        return record_episodes(
            env,
            actor,
            repo_id=REPO_ID,
            root=REPO / 'data/robosuite' / out_name,
            task_prompt=env._cfg.prompt,
            episodes=episodes,
            robot_type='panda',
            seed_start=seed_start,
            keep_failures=True,  # every policy episode: its states are the point
            # the recorder checks action[t] against state[t+1]; here the action is the expert's label
            # and the policy's action is what ran, so they differ by design
            tracking_tolerance=float('inf'),
            perturb=RunPolicy(actor, policy_run),
        )
    finally:
        env.close()
