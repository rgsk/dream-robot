"""UWM, rolled out as a policy through its action server.

The model lives in its own env (Python 3.10, their torch 2.2) as
``experiments/uwm/scripts/serve.py``; the socket and chunk playing are Cosmos's
(``policies/cosmos_policy``). What UWM needs on top: the **last two frames** per camera,
consecutive steps, as it was trained (their rollouts double the first frame at episode
start; so does this), and the joints + gripper at the same two steps (read by models trained
with ``--joints``). Their rollouts play all 16 actions of a chunk, so ``execute``
defaults to the whole chunk here.
"""

from __future__ import annotations

import numpy as np

from dream_robot.core.schema import Observation
from dream_robot.policies.cosmos_policy.policy import CosmosPolicy

CAMERAS = ("top", "wrist")


class UWMPolicy(CosmosPolicy):
    def __init__(self, env, *, address: str = "localhost:6001", execute: int | None = None):
        super().__init__(env, address=address, execute=execute)
        if execute is None:
            self.execute = self.chunk
        self._previous: dict[str, np.ndarray] | None = None
        self._previous_state: list[float] | None = None

    def reset(self) -> None:
        super().reset()
        self._previous = None
        self._previous_state = None

    def __call__(self, observation: Observation) -> np.ndarray:
        action = super().__call__(observation)
        self._previous = {k: np.ascontiguousarray(observation.images[k], np.uint8) for k in CAMERAS}
        self._previous_state = np.asarray(observation.state, np.float32).tolist()
        return action

    def _request(self, observation: Observation) -> dict:
        now = {k: np.ascontiguousarray(observation.images[k], np.uint8) for k in CAMERAS}
        before = self._previous or now
        state = np.asarray(observation.state, np.float32).tolist()
        return {
            "prompt": self._prompt, "hw": list(now["top"].shape[:2]),
            "proprio": [self._previous_state or state, state],
            **{k: np.stack([before[k], now[k]]).tobytes() for k in CAMERAS},
        }
