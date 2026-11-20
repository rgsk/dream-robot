"""LIBERO driven by an LLM through skill tools, the arm moved by a trained Verne (2026-11-20; runs in envs/libero).

The driver gets a long LIBERO task in words ("turn on the stove and put the frying pan on it") and only

    reset(episode)            a new episode from the task's official start state `episode` (0-49); returns the task
    look(camera)              an image: scene (front, wide), agentview (the policy's), wrist
    home()                    scripted: the hand back to where it started, gripper open
    turn_on(object)           the policy runs "turn_on(stove)" for SECONDS, then the agentview image
    turn_off(object)          same, "turn_off(stove)"
    put(object, target)       same, "put(moka_pot | frying_pan, stove)"

Call a skill again to let it continue. No success flag, no gripper reading: the driver judges from images. The
referee (the task's own check, LIBERO's predicates) and a film go to --out-dir after every call, never to a tool.
Same plumbing as verne_server: stderr and stray stdout to --log, the protocol on a private copy of stdout, the
policy loading in the background so the client's handshake is answered at once.

    envs/libero/.venv/bin/python -m dream_robot.agent.libero_server
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server.mcpserver import Image, MCPServer

from dream_robot.agent.server import _on_sim_thread, _png
from dream_robot.agent.verne_server import _wire_only_stdout

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault('LIBERO_CONFIG_PATH', str(REPO / 'envs/libero/config'))
SETTLE = 10  # no-op steps after a reset, as LIBERO's eval
HOME_STEPS = 40

INSTRUCTIONS = """You drive a robot arm in a kitchen scene (a stove with a knob, a moka pot, a frying pan). A learned
policy does the moving; you choose WHICH skill, from three tools: turn_on(stove), turn_off(stove),
put(object, target) with object moka_pot or frying_pan and target stove. Each call runs the policy for a few seconds
and returns the agentview image; one skill usually needs several calls in a row, so keep calling the same skill while
it is making progress. home() moves the arm back to its start (scripted). look(camera) shows "scene" (wide front
view), "agentview" (what the policy sees) or "wrist" without moving. You get no sensor readings and no success
signal: judge from the images (the stove's burner glows red when it is on). Work through the task one skill at a time.
If a skill stalls (e.g. the arm hovers with an empty gripper), call home() and try again. When done, say whether the
task is complete, judged from the images."""


class Session:
    def __init__(self, run: Path, weights: str, task: str, suite: str, seconds: float, out_dir: str):
        import torch
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        from dream_robot.policies.verne.libero import Runner

        bddl = Path(get_libero_path('bddl_files')) / suite / f'{task}.bddl'
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
        self.env.seed(0)
        self.init = torch.load(Path(get_libero_path('init_states')) / suite / f'{task}.pruned_init', weights_only=False)
        self.language = self.env.language_instruction if hasattr(self.env, 'language_instruction') else task
        self.runner = Runner(run, weights)
        self.steps_per_call = int(round(seconds * 20))  # LIBERO runs at 20 Hz
        self.task, self.out_dir = task, Path(out_dir)
        self.episode, self.obs, self.calls, self.frames = None, None, [], []

    # --- the referee: ground truth, never returned by a tool ---------------------------------------------------
    def log_score(self) -> None:
        if self.episode is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.out_dir / f'{self.task}_ep{self.episode}_{time.strftime("%H%M%S", time.localtime(self.t0))}'
        row = {'task': self.task, 'episode': self.episode, 'success': bool(self.env.check_success()),
               'calls': self.calls, 'sim_steps': self.t, 'wall_s': round(time.time() - self.t0, 1)}
        stem.with_suffix('.json').write_text(json.dumps(row, indent=1))
        if self.frames:
            import imageio.v3 as iio

            iio.imwrite(stem.with_suffix('.mp4'), np.stack(self.frames), fps=20)

    # --- tools ---------------------------------------------------------------------------------------------------
    def reset(self, episode: int) -> dict[str, Any]:
        if not 0 <= episode < len(self.init):
            raise ValueError(f'episode must be 0..{len(self.init) - 1}')
        self.env.reset()
        self.obs = self.env.set_init_state(self.init[episode])
        for _ in range(SETTLE):
            self.obs, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        self.runner.reset()
        self.home_pose = self.runner.hand(self.env)
        self.episode, self.calls, self.frames, self.t, self.t0 = episode, [], [], 0, time.time()
        return {'instruction': self.language}

    def _step(self, action) -> None:
        self.obs, _, _, _ = self.env.step(action)
        self.t += 1
        self.frames.append(np.concatenate([self._image('scene', 256), self._image('agentview', 256)], 1))

    def _image(self, camera: str, size: int = 256) -> np.ndarray:
        if camera == 'scene':
            return self.env.env.sim.render(width=size, height=size, camera_name='frontview')[::-1]
        key = 'agentview_image' if camera == 'agentview' else 'robot0_eye_in_hand_image'
        img = self.obs[key][::-1]
        return img.repeat(size // img.shape[0], 0).repeat(size // img.shape[1], 1)

    def skill(self, name: str, *args: str) -> np.ndarray:
        self._check()
        key = (name, *args)
        from dream_robot.policies.verne.libero import SKILLS

        if key not in SKILLS:
            raise ValueError(f'no trained skill {name}({", ".join(args)}); have {list(SKILLS)}')
        self.calls.append(f'{name}({", ".join(args)})')
        for _ in range(self.steps_per_call):
            self._step(self.runner.act(self.env, self.obs, SKILLS[key]))
        self.log_score()
        return self._image('agentview')

    def home(self) -> np.ndarray:
        self._check()
        self.calls.append('home')
        self.runner.reset()
        p, r = self.home_pose
        for _ in range(HOME_STEPS):
            a = self.runner.command(*self.runner.hand(self.env), p, r, -1.0)
            self._step(a)
        self.log_score()
        return self._image('agentview')

    def look(self, camera: str) -> np.ndarray:
        self._check()
        self.calls.append(f'look {camera}')
        return self._image(camera)

    def _check(self) -> None:
        if self.episode is None:
            raise ValueError('call reset(episode) first')


def build(run: Path, weights: str, task: str, suite: str, seconds: float, out_dir: str) -> MCPServer:
    sim = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sim')
    box = {}
    sim.submit(lambda: box.update(s=Session(run, weights, task, suite, seconds, out_dir)))

    class _Lazy:
        def __getattr__(self, name):
            return getattr(box['s'], name)

    s = _Lazy()
    server = MCPServer('libero', instructions=INSTRUCTIONS)
    on_sim = _on_sim_thread(sim)

    @server.tool()
    @on_sim
    def reset(episode: int = 0) -> dict[str, Any]:
        """Start the task from its official start state `episode` (0-49). Returns the instruction."""
        return s.reset(episode)

    @server.tool()
    @on_sim
    def look(camera: Literal['scene', 'agentview', 'wrist'] = 'scene') -> Image:
        """An image without moving: scene = wide front view, agentview = the policy's camera, wrist = the hand's."""
        return _png(s.look(camera))

    @server.tool()
    @on_sim
    def home() -> Image:
        """Scripted: the hand back to where it started, gripper open. Returns the agentview image."""
        return _png(s.home())

    @server.tool()
    @on_sim
    def turn_on(object: Literal['stove'] = 'stove') -> Image:
        """The policy turns the stove's knob on for a few seconds; call again to continue. Agentview image."""
        return _png(s.skill('turn_on', object))

    @server.tool()
    @on_sim
    def turn_off(object: Literal['stove'] = 'stove') -> Image:
        """The policy turns the stove's knob off for a few seconds; call again to continue. Agentview image."""
        return _png(s.skill('turn_off', object))

    @server.tool()
    @on_sim
    def put(object: Literal['moka_pot', 'frying_pan'], target: Literal['stove'] = 'stove') -> Image:
        """The policy puts the object on the target for a few seconds; call again to continue. Agentview image."""
        return _png(s.skill('put', object, target))

    return server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', type=Path, default=REPO / 'experiments/libero/runs/skills_free/seed0')
    p.add_argument('--weights', default='weights.pt')
    p.add_argument('--task', default='KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it')
    p.add_argument('--suite', default='libero_90', help='libero_90 or libero_10')
    p.add_argument('--seconds', type=float, default=3.0, help='policy time per skill call')
    p.add_argument('--out-dir', default=str(REPO / 'scratch_nov/libero_llm/episodes'))
    p.add_argument('--log', default=str(REPO / 'scratch_nov/libero_llm/server.log'))
    a = p.parse_args()
    _wire_only_stdout(a.log)
    build(a.run, a.weights, a.task, a.suite, a.seconds, a.out_dir).run('stdio')


if __name__ == '__main__':
    main()
