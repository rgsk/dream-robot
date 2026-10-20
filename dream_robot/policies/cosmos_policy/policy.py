"""Cosmos Policy, rolled out as a policy through its action server.

Satisfies ``core.record.Policy``. The model lives in its own env (Python 3.10, their
torch) as ``experiments/cosmos_policy/scripts/serve.py``; this file only packs our
``Observation`` into the server's plain-dict message and plays the chunk it sends back.

**Chunks.** Each call to the server returns ``chunk`` actions (32); the first ``execute``
(default 16, half) are played, then it asks again from the frame it is at. The eval
pauses physics during the call, so thinking time does not show here (it did in
experiments/latency).

**What it imagines** comes back with every chunk: the top and wrist frames the model drew
``chunk`` steps ahead. They go in ``aim_images``, which the eval video shows as a third column
beside the real cameras -- a wrong-cube episode either imagines the wrong cube, or imagines the
right one and acts wrongly.

**What it sends** is what training saw: the top and wrist frames as uint8 at the env's
size (record and eval both at 224 px), ``observation.state`` as proprio, and the prompt
string -- the server looks its T5 embedding up, so every eval wording must be in its pickle.
"""

from __future__ import annotations

from multiprocessing.connection import Client

import numpy as np

from dream_robot.core.schema import Observation

AUTHKEY = b"dream-robot"


class CosmosPolicy:
    def __init__(self, env, *, address: str = "localhost:6000", execute: int | None = None):
        host, port = address.rsplit(":", 1)
        self._conn = Client((host, int(port)), authkey=AUTHKEY)
        self._conn.send("hello")
        self.server = self._conn.recv()
        self.chunk = self.server["chunk"]
        self.execute = execute or self.chunk // 2
        if not 1 <= self.execute <= self.chunk:
            raise ValueError(f"execute must be in 1..{self.chunk}, got {self.execute}")
        self._env = env
        self._queue: list[np.ndarray] = []
        self._prompt: str | None = None
        self.call_seconds: list[float] = []
        self.aim_images: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        self._queue = []
        self._prompt = self._env.prompt
        self.aim_images = None

    def __call__(self, observation: Observation) -> np.ndarray:
        if self._prompt is None:
            raise RuntimeError("call reset() after env.reset() before stepping")
        if not self._queue:
            top, wrist = (np.ascontiguousarray(observation.images[k], np.uint8) for k in ("top", "wrist"))
            self._conn.send({
                "prompt": self._prompt,
                "top": top.tobytes(), "wrist": wrist.tobytes(), "hw": list(top.shape[:2]),
                "proprio": np.asarray(observation.state, np.float32).tolist(),
            })
            reply = self._conn.recv()
            self.call_seconds.append(reply["seconds"])
            self._queue = list(np.asarray(reply["actions"], np.float32)[: self.execute])
            self.aim_images = {
                name: np.frombuffer(reply[f"future_{name}"], np.uint8).reshape(top.shape)
                for name in ("top", "wrist")
            }
        action = self._queue.pop(0)
        grip = self._env.embodiment.gripper_slice
        action[grip] = np.clip(action[grip], 0.0, 1.0)    # 0 closed .. 1 open
        return action
