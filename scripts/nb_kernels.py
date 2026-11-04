"""List the running Jupyter kernels (VS Code's and others), or shut down all but the listed notebooks'.

    uv run python scripts/nb_kernels.py                                  # notebook, pid, GPU MiB of each
    uv run python scripts/nb_kernels.py --keep rough/uwm2.ipynb [...]    # shut the others down

A kernel's notebook is read from its `__vsc_ipynb_file__` (as nb_exec.py does). A kernel busy
running a cell can't answer, so its notebook is unknown: it is never shut down, only reported.
Shutdown is a clean shutdown_request; VS Code then shows that notebook's kernel as stopped.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from jupyter_client import BlockingKernelClient
from jupyter_core.paths import jupyter_runtime_dir


def gpu_mib() -> dict[int, int]:
    # out: {pid: MiB} for every process on the GPU
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {int(p): int(m) for p, m in (line.split(", ") for line in out.splitlines() if line)}


def pid_of(conn: Path) -> int | None:
    # the kernel process is the one started with this connection file on its command line
    for d in Path("/proc").iterdir():
        try:
            if d.name.isdigit() and conn.name in (d / "cmdline").read_bytes().decode(errors="ignore"):
                return int(d.name)
        except OSError:
            pass
    return None


def kernels() -> list[tuple[Path, BlockingKernelClient, str | None]]:
    # out: (connection file, client, notebook path or None if busy/unknown) per live connection file
    dirs = {Path(jupyter_runtime_dir()), Path(f"/run/user/{os.getuid()}/jupyter/runtime")}
    found = []
    for f in sorted(f for d in dirs for f in d.glob("kernel-*.json")):
        if pid_of(f) is None:
            continue  # stale connection file, its kernel is gone
        kc = BlockingKernelClient(connection_file=str(f))
        kc.load_connection_file()
        kc.start_channels()
        nb = None
        try:
            reply = kc.execute_interactive(
                "", silent=True, store_history=False,
                user_expressions={"p": "globals().get('__vsc_ipynb_file__')"}, timeout=3,
            )
            p = reply["content"]["user_expressions"]["p"]["data"]["text/plain"].strip("'\"")
            nb = None if p == "None" else str(Path(p).resolve())
        except Exception:
            pass  # busy, or not a VS Code kernel
        found.append((f, kc, nb))
    return found


def main(argv: list[str]) -> None:
    keep = None
    if argv[:1] == ["--keep"]:
        if len(argv) < 2:
            sys.exit("--keep needs at least one notebook")
        keep = {str(Path(a).resolve()) for a in argv[1:]}
    elif argv:
        sys.exit(__doc__)
    mib = gpu_mib()
    cwd = Path.cwd()
    for conn, kc, nb in kernels():
        pid = pid_of(conn)
        name = os.path.relpath(nb, cwd) if nb else "? (busy or not VS Code's)"
        line = f"pid {pid}  GPU {mib.get(pid, 0):5d} MiB  {name}"
        if keep is None or nb is None or nb in keep:
            print(line + ("   kept" if keep is not None else ""))
            continue
        kc.shutdown()
        for _ in range(20):  # up to 5 s for it to exit
            if pid is None or not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.25)
        else:
            os.kill(pid, 15)  # didn't exit on request: SIGTERM
        print(line + "   SHUT DOWN")
        kc.stop_channels()


if __name__ == "__main__":
    main(sys.argv[1:])
