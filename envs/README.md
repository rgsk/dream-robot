# envs/ — one folder per external simulator

The main venv (repo root, Python 3.14) runs robosuite and every policy. A simulator that cannot
share it — pinned Python, pinned MuJoCo, its own robosuite fork — gets a folder here. ROADMAP's rule
holds: the seam is a dataset on disk or an MCP server, never an import across venvs.

```- 
envs/<name>/
  setup.sh      idempotent: venv, source clones, packages, assets. Re-run any time.
  README.md     what it is, what it pins, quirks
  .venv/        gitignored
  src/          gitignored: editable source clones
  setup.log     gitignored
```

**Our code is on the venv's path, not installed in it.** `setup.sh` writes a `.pth` file pointing at
the repo root, so `dream_robot` imports under any Python without the root pyproject's
`requires-python` getting in the way. Only what a simulator's server needs has to import there —
`core`, `agent`, and that simulator's `sims/<name>/` folder.

**Each simulator is an MCP server like any other.** Its `.mcp.json` entry runs
`envs/<name>/.venv/bin/python -m dream_robot.agent.server --env <name>/<task>`, so the same tools,
the same viewer and the same drivers (you, me, a planner) work unchanged.

**Every new simulator gets its own venv; the main venv takes no more.** The main venv is the stable
bench — every policy, the results matrix, and robosuite: the one simulator that lives there, as the
home backend that T1-T6, the recorder, the eval harness and the agent server were built on (its pins
in the root `pyproject.toml` exist to keep it and lerobot together). It stays unless it ever
conflicts with something the bench needs. No other simulator's pins get to move it. Each `envs/<name>/` is disposable: delete the folder and it is
gone; re-run `setup.sh` and it is back.

- **Disk is not the cost.** uv hard-links packages from one cache on the same filesystem, so a
  second torch is a link, not another few GB.
- **Policies still run in-process.** Rule 1 (policies depend on torch + core only) makes them
  portable: a simulator's venv adds torch, and the `.pth` puts our code on its path.
- **The real cost: shared code runs on the oldest Python in use** (3.11 for RoboCasa). `core`,
  `agent` and `policies` must not use syntax newer than that, whatever the root pyproject says; run
  the fast test suite in each simulator's venv to keep it so.

**Adding one:** copy a `setup.sh`, change the clones and pins, add its tasks under
`dream_robot/sims/<name>/` and in `core/registry.py`.

| folder | simulator | Python | why separate |
|---|---|---|---|
| `robocasa/` | RoboCasa kitchens (robosuite master) | 3.11 | pins MuJoCo 3.3.1, numpy 2.2.5, lerobot 0.3.3 |
| `dream_yam/` | Dream YAM Sim (Dream Labs, ships in the `dreamlabs` wheel) + the Dream-1 client | 3.12 | pins numpy 2.2.6, jax 0.7.1, MuJoCo >= 3.3.5 |
