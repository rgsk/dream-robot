"""T5 two-cube scene driven by an LLM through 4 tools, the arm moved by the Verne policy (2026-11-20).

The simplest LLM loop: the driver reads a two-clause instruction ("put the yellow cube in the right box,
then the green cube in the left box") and has only

    reset(seed)         a new episode; returns the instruction
    look(camera)        an image: scene (wide), top, wrist
    home()              scripted: the arm's joints back to the start pose, gripper open
    put(colour, side)   the Verne policy runs on "put the <colour> cube in the <side> box" for SECONDS,
                        then returns the scene image; call again to let it continue

No gripper reading, no success flag: the driver judges from the images (see camera-only driving).
The referee (which cube rests in which box) and the episode's film (scene view, every 2nd step) are written
to --out-dir after every call, never returned by a tool.

    MUJOCO_GL=egl uv run python -m dream_robot.agent.verne_server \
        --run experiments/verne/runs/22_position_spread/t5/seed0
"""

import argparse
import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server.mcpserver import Image, MCPServer

from dream_robot.agent.server import _on_sim_thread, _png, _quiet

COLOURS = ('red', 'green', 'blue', 'yellow')
SIDES = ('left', 'right')
HOME_STEPS, HOLD_STEPS = 45, 10

INSTRUCTIONS = """You drive a robot arm at a table with two cubes and two boxes (left and right, as seen in the
"scene" camera). A learned policy does the actual moving: `put(colour, side)` runs it on "put the <colour>
cube in the <side> box" for a few seconds and returns the scene image. It usually needs several calls in a
row to finish one cube; keep calling the same put while it is making progress. `home()` moves the arm back to
its start pose (scripted). `look(camera)` shows "scene" (wide), "top" or "wrist" without moving.
You get no sensor readings and no success signal: judge everything from the images. Work through the
instruction one cube at a time. If the arm stalls (e.g. hovering with an empty gripper), call home() and
then put again. When done, say for each cube whether it is in its box, judged from the images."""


class Session:
    def __init__(self, run: Path, seconds: float, out_dir: str):
        import torch

        from dream_robot.policies.verne import rollouts
        from dream_robot.policies.verne import vae as vae_mod
        from dream_robot.policies.verne.run import load_task, to_gpu

        state, action, episode, latent, instruction, slot, train_ids, _, _ = load_task('t5')
        self.d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
        self.net = rollouts.load(torch.load(run / 'weights.pt'))
        self.vae = vae_mod.load(vae_mod.WEIGHTS)
        self.env = rollouts.make_env('t5', horizon_seconds=3600)  # the driver decides when it's over
        self.obs = self.env.reset(seed=0)
        self.steps_per_call = int(round(seconds * self.env.control_hz))
        self.out_dir = out_dir
        self.seed, self.calls, self.frames = None, [], []
        self.t0 = time.time()

    # --- the referee: ground truth, never returned by a tool ---------------------------------
    def log_score(self) -> None:
        # rewritten after every tool call (claude -p kills the server at the end, so nothing runs at exit):
        # <score_dir>/seed<seed>_<start time>.json and .mp4, one pair per episode
        if self.seed is None:
            return
        env, (want, side2) = self.env, self.want
        cubes = env._cube_positions()
        named = env.instruction.target_slot
        got = [env.which_bin(cubes[named]), env.which_bin(cubes[1 - named])]
        done = [got[0] == env.bin_names.index(want[1]), got[1] == env.bin_names.index(side2)]
        row = {'seed': self.seed, 'instruction': self.instruction, 'done': done, 'calls': self.calls,
               'sim_steps': env._t, 'wall_s': round(time.time() - self.t0, 1)}
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = out / f'seed{self.seed}_{time.strftime("%H%M%S", time.localtime(self.t0))}'
        stem.with_suffix('.json').write_text(json.dumps(row, indent=1))
        if self.frames:
            from dream_robot.core.video import write_video

            write_video(self.frames, stem.with_suffix('.mp4'), fps=self.env.control_hz / 2)

    # --- tools --------------------------------------------------------------------------------
    def reset(self, seed: int) -> dict[str, Any]:
        with _quiet():
            self.obs = self.env.reset(seed=seed)
        ins = self.env.instruction
        side2 = [n for n in self.env.bin_names if n != ins.side][0]
        self.want = ((ins.color, ins.side), side2)
        self.instruction = (f'put the {ins.color} cube in the {ins.side} box, '
                            f'then the {ins.distractor_color} cube in the {side2} box')
        self.seed, self.calls, self.frames, self.t0 = seed, [], [], time.time()
        self.home_q = self.obs.state[:7].copy()
        self.prev = self.obs.state
        return {'instruction': self.instruction}

    def _step(self, action) -> None:
        r = self.env.step(action)
        self.prev, self.obs = self.obs.state, r.observation
        if self.env._t % 2 == 0:
            self.frames.append(self.env.render())

    def put(self, colour: str, side: str) -> np.ndarray:
        import torch

        from dream_robot.policies.verne import hand, rollouts
        from dream_robot.policies.verne.data import one_hot

        self._check()
        self.calls.append(f'put {colour} {side}')
        torch.manual_seed(self.seed * 1000 + len(self.calls))
        ins, drv, plan = one_hot(f'put the {colour} cube in the {side} box'), hand.Driver(self.env), []
        for _ in range(self.steps_per_call):
            if not plan:
                chunk = rollouts.act(self.net, self.vae, self.d, self.obs, self.prev, ins)[:16]
                plan = drv.plan(self.obs.state[:7], chunk)
            self._step(drv.action(self.obs.state[:7], plan.pop(0)))
        self.log_score()
        return self.env.render()

    def home(self) -> np.ndarray:
        self._check()
        self.calls.append('home')
        start = self.obs.state[:7].copy()
        for k in range(1, HOME_STEPS + HOLD_STEPS + 1):
            q = start + (self.home_q - start) * min(k / HOME_STEPS, 1.0)
            self._step(np.concatenate([q, [1.0]]))
        self.log_score()
        return self.env.render()

    def look(self, camera: str) -> np.ndarray:
        self._check()
        self.calls.append(f'look {camera}')
        return self.env.render() if camera == 'scene' else self.obs.images[camera]

    def _check(self) -> None:
        if self.seed is None:
            raise ValueError('call reset(seed) first')


def build(run: Path, seconds: float, out_dir: str) -> MCPServer:
    # the session (demo stats, VAE, policy, sim: ~10-20 s) loads on the sim thread in the background, so the
    # server answers the client's handshake at once (the Inspector gives up otherwise and its proxy crashes);
    # tools run on the same one thread, so the first call simply waits behind the load
    sim = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sim')
    box = {}

    def load():
        box['s'] = Session(run, seconds, out_dir)  # its prints and C output land in the log (see main)

    sim.submit(load)

    class _Lazy:
        def __getattr__(self, name):
            return getattr(box['s'], name)

    s = _Lazy()
    server = MCPServer('verne-t5', instructions=INSTRUCTIONS)
    on_sim = _on_sim_thread(sim)

    @server.tool()
    @on_sim
    def reset(seed: int) -> dict[str, Any]:
        """Start a new episode (same seed, same scene). Returns the instruction."""
        return s.reset(seed)

    @server.tool()
    @on_sim
    def look(camera: Literal['scene', 'top', 'wrist'] = 'scene') -> Image:
        """An image without moving. scene = wide view; top and wrist = the robot's own cameras."""
        return _png(s.look(camera))

    @server.tool()
    @on_sim
    def home() -> Image:
        """Scripted: the arm back to its start pose, gripper open. Returns the scene image."""
        return _png(s.home())

    @server.tool()
    @on_sim
    def put(colour: Literal[COLOURS], side: Literal[SIDES]) -> Image:  # type: ignore[valid-type]
        """The policy runs on "put the <colour> cube in the <side> box" for a few seconds, then returns the
        scene image. Call again to let it continue."""
        return _png(s.put(colour, side))

    return server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', type=Path, default=Path('experiments/verne/runs/22_position_spread/t5/seed0'))
    p.add_argument('--seconds', type=float, default=3.0, help='policy time per put call')
    p.add_argument('--out-dir', default='scratch_nov/verne_llm/episodes', help='referee json + film per episode')
    p.add_argument('--log', default='scratch_nov/verne_llm/server.log', help="the server's stderr and stray stdout")
    a = p.parse_args()
    _wire_only_stdout(a.log)
    build(a.run, a.seconds, a.out_dir).run('stdio')


class _TextToLog:
    # sys.stdout for the server's life: the SDK writes the protocol to .buffer (the wire); text written by
    # anything else (print, logging to stdout) goes to stderr = the log
    def __init__(self, wire):
        self.buffer = wire

    def write(self, text):
        return sys.stderr.write(text)

    def flush(self):
        sys.stderr.flush()


def _wire_only_stdout(log: str) -> None:
    # The MCP Inspector's proxy forwards a stdio server's stderr to the browser and crashes ("Not connected")
    # if a line arrives before the browser's stream is up; loading the policy writes plenty. So fd 2 -> the
    # log file, the wire -> a private copy of fd 1, fd 1 itself -> the log (C prints, e.g. EGL / MuJoCo).
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    f = open(log, 'a')
    os.dup2(f.fileno(), 2)
    wire = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    sys.stdout = _TextToLog(wire)


if __name__ == '__main__':
    main()
