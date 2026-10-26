"""UVA, rolled out as a policy through its action server.

The model lives in its own env (Python 3.9, their torch 2.1) as
``experiments/uva/scripts/serve.py``; the socket and chunk playing are Cosmos's
(``policies/cosmos_policy``). What UVA needs on top: the **last 16 steps** of the top camera
(+ the wrist camera, if the server's model was trained with it) and of the joints + gripper,
consecutive, as it was trained (it looks at steps 3, 7, 11, 15 of them). At episode start
the first frame is repeated to fill the window. Their LIBERO rollouts play 8 of each
16-action chunk, so ``execute`` defaults to 8.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dream_robot.core.schema import Observation
from dream_robot.policies.cosmos_policy.policy import CosmosPolicy

HISTORY = 16


class UVAPolicy(CosmosPolicy):
    def __init__(self, env, *, address: str = "localhost:6003", execute: int | None = None):
        super().__init__(env, address=address, execute=execute or 8)
        self._cameras = ("top", "wrist") if self.server.get("wrist") else ("top",)
        self._frames = {k: deque(maxlen=HISTORY) for k in self._cameras}
        self._states: deque[list[float]] = deque(maxlen=HISTORY)

    def reset(self) -> None:
        super().reset()
        for frames in self._frames.values():
            frames.clear()
        self._states.clear()

    def __call__(self, observation: Observation) -> np.ndarray:
        # every step goes into the window, not only the ones that ask for a chunk;
        # episode start (empty window): fill it with the first step
        state = np.asarray(observation.state, np.float32).tolist()
        steps = 1 if self._states else HISTORY
        for k in self._cameras:
            self._frames[k].extend([np.ascontiguousarray(observation.images[k], np.uint8)] * steps)
        self._states.extend([state] * steps)
        return super().__call__(observation)

    def _request(self, observation: Observation) -> dict:
        return {
            "prompt": self._prompt, "hw": list(self._frames["top"][-1].shape[:2]),
            "proprio": list(self._states),
            **{k: np.stack(self._frames[k]).tobytes() for k in self._cameras},
        }
