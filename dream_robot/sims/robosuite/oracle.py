"""A perfect video model, faked with the simulator: the expert's future, from *here*.

ROADMAP "Next: our own executor", step 1. The executor is split in two -- a video
model imagines the frames ahead, an inverse dynamics model (IDM) reads the arm
motion that gets there. Before any video model exists, this file stands in for
one: at a decision point it saves the whole simulation, lets the scripted expert
carry on for ``k`` steps, keeps the frame it ends on, and rewinds.

**From the current state, not from a recorded demo.** Replaying demo frames would
test a video model that ignores the present: once the IDM is a little off, the
arm is no longer where the demo was, and the demo's next frame is a future that
cannot be reached from here. Forking the live sim gives the future *of this
state*, which is what a real video model will be asked for.

**The expert is shadowed, not restarted.** The expert is a phase machine
(approach, descend, grasp, ...). A shadow copy is fed every real observation, so
its phase follows what the IDM actually did; each imagination forks that shadow
and is thrown away with the rewind.

Privileged access throughout -- simulator state, the expert's ground truth -- on
purpose: this is the oracle a learned model will replace, and it lives on the sim
side of the seam. Nothing in ``policies/`` imports it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from dream_robot.core.schema import Observation

# Episode bookkeeping on our task wrappers that a rewind must also undo: the last
# raw observation, the step counter, and the lift references the failure
# diagnosis reads. Whichever of these a task has get saved.
_WRAPPER_FIELDS = ("_raw", "_t", "_max_z", "_z0", "_max_cube_z", "_cube_z0")

# robosuite's own clock: done flips when timestep reaches the horizon, and a
# step after that raises.
_ROBOSUITE_FIELDS = ("timestep", "cur_time", "done")



def _stateful_parts(env) -> list:
    """robosuite objects that carry state between steps outside MuJoCo's.

    - gripper models rate-limit their command: each step moves ``current_action``
      a little toward open or closed, so a rewind that misses it closes the
      fingers faster the second time;
    - part controllers cache what they last read from the sim (joint positions,
      mass matrix, goal) and act on it;
    - observables sample on their own timers, so a rewound clock with unrewound
      timers hands back a reading taken at a different substep.
    """
    parts = list(env._env._observables.values())
    for robot in env._env.robots:
        parts += list(robot.gripper.values())
        parts += list(robot.composite_controller.part_controllers.values())
    return parts


_PLAIN = (np.ndarray, bool, int, float, type(None))


def _plain_fields(obj) -> dict:
    """An object's array and scalar attributes, copied. Handles and configs are skipped."""
    return {k: copy.deepcopy(v) for k, v in vars(obj).items() if isinstance(v, _PLAIN)}


@dataclass
class SimSnapshot:
    """Everything that decides what the next ``env.step`` does, for one robosuite task."""

    physics: mujoco.MjData
    robosuite: dict
    wrapper: dict
    parts: list[dict]

    @classmethod
    def take(cls, env) -> SimSnapshot:
        model, data = env.mujoco
        # All of MjData, not mj_getState's integration state: that left 1e-5 rad
        # of drift per rewind (derived quantities the next step starts from).
        physics = mujoco.MjData(model)
        mujoco.mj_copyData(physics, model, data)
        inner = env._env
        return cls(
            physics=physics,
            robosuite={f: copy.deepcopy(getattr(inner, f)) for f in _ROBOSUITE_FIELDS},
            wrapper={
                f: copy.deepcopy(getattr(env, f)) for f in _WRAPPER_FIELDS if hasattr(env, f)
            },
            parts=[_plain_fields(p) for p in _stateful_parts(env)],
        )

    def restore(self, env) -> None:
        model, data = env.mujoco
        mujoco.mj_copyData(data, model, self.physics)
        for f, v in self.robosuite.items():
            setattr(env._env, f, copy.deepcopy(v))
        for f, v in self.wrapper.items():
            setattr(env, f, copy.deepcopy(v))
        for part, fields in zip(_stateful_parts(env), self.parts, strict=True):
            for k, v in fields.items():
                setattr(part, k, copy.deepcopy(v))


class ExpertFuture:
    """``observe(obs, imagine=True)`` -> the observation the expert reaches ``k`` steps on.

    Call once per real control step, with that step's observation, before the
    action is sent. ``imagine=False`` only advances the shadow expert (the IDM
    is still executing its last chunk and needs no new future).

    ``k`` is how far ahead the stand-in video model sees. Near the end of an
    episode the imagination stops at the horizon and returns the last frame it
    reached, so the future can be fewer than ``k`` steps ahead.
    """

    def __init__(self, env, expert, *, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._env = env
        self._expert = expert                  # the task's ExpertPolicy: (obs) -> (action, command)
        self.k = k
        self.imagined_steps: list[int] = []    # per imagination, how far it actually got

    def reset(self) -> None:
        """Call after env.reset(). Latches the expert onto this episode."""
        self._expert.reset()
        self.imagined_steps = []

    def observe(self, observation: Observation, *, imagine: bool) -> Observation | None:
        future = self._imagine(observation) if imagine else None
        self._expert(observation)              # the shadow follows the real step
        return future

    def _imagine(self, observation: Observation) -> Observation:
        snapshot = SimSnapshot.take(self._env)
        machine = copy.deepcopy(self._expert._machine)
        horizon = self._env.horizon_steps
        obs, steps = observation, 0
        try:
            while steps < self.k and self._env._t < horizon:
                action, _ = self._expert(obs)
                obs = self._env.step(action).observation
                steps += 1
        finally:
            snapshot.restore(self._env)
            self._expert._machine = machine
        self.imagined_steps.append(steps)
        return obs
