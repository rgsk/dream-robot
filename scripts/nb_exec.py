"""Run notebook cells on the kernel VS Code started for that notebook, and write their outputs into the .ipynb.

The kernel is VS Code's own, so every variable the user's cells made is there, and a cell run
here is visible to them (and theirs to it). VS Code sets `__vsc_ipynb_file__` in each kernel to
its notebook's path; this script asks every running kernel for it and picks the match.
The notebook must be open in VS Code, its kernel started, and saved (VS Code then reloads
the outputs written here).

    uv run python scripts/nb_exec.py rough/act.ipynb 42 43     # these cell indices
    uv run python scripts/nb_exec.py rough/act.ipynb all       # every code cell, in order
    uv run python scripts/nb_exec.py rough/act.ipynb from:40   # cell 40 to the end
    uv run python scripts/nb_exec.py rough/act.ipynb 18d       # only cell 18's imports, defs and classes

The `d` form skips a cell's long work (training, sweeps) when only its definitions are needed;
its outputs in the .ipynb are left as they were.

Stops at the first cell that raises. Indices count every cell, markdown too, as in
the .ipynb; markdown cells in a range are skipped.
"""

import os
import sys
import time
from pathlib import Path

import ast

import nbformat
from jupyter_client import BlockingKernelClient
from jupyter_core.paths import jupyter_runtime_dir


def find_kernel(nb_path: Path) -> BlockingKernelClient:
    # out: a client for the kernel whose __vsc_ipynb_file__ is nb_path
    # VS Code writes its connection files to /run/user/<uid>/jupyter/runtime, which may not be jupyter_runtime_dir()
    dirs = {Path(jupyter_runtime_dir()), Path(f"/run/user/{os.getuid()}/jupyter/runtime")}
    for f in sorted(f for d in dirs for f in d.glob("kernel-*.json")):
        kc = BlockingKernelClient(connection_file=str(f))
        kc.load_connection_file()
        kc.start_channels()
        try:
            # silent: no execution count, no history, nothing shown in the notebook
            reply = kc.execute_interactive(
                "",
                silent=True,
                store_history=False,
                user_expressions={"p": "globals().get('__vsc_ipynb_file__')"},
                timeout=3,
            )
            p = reply["content"]["user_expressions"]["p"]["data"]["text/plain"].strip("'\"")
            if p != "None" and Path(p).resolve() == nb_path.resolve():
                return kc
        except Exception:
            pass  # dead, busy or not a VS Code kernel
        kc.stop_channels()
    sys.exit(f"no running VS Code kernel for {nb_path}: open it in VS Code and run a cell")


def run_cell(kc: BlockingKernelClient, source: str, on_output=None) -> tuple[list, int | None, bool]:
    # on_output(outputs, count): called after every new output, so long cells can show progress
    # out: (outputs in nbformat shape, execution count, whether it raised)
    msg_id = kc.execute(source)
    outputs: list = []
    count = None
    failed = False
    while True:
        msg = kc.get_iopub_msg()
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        kind, content = msg["msg_type"], msg["content"]
        if kind == "execute_input":
            count = content["execution_count"]
        elif kind == "stream":
            # consecutive prints to the same stream join into one output, as Jupyter shows them
            if outputs and outputs[-1].output_type == "stream" and outputs[-1].name == content["name"]:
                outputs[-1].text += content["text"]
            else:
                outputs.append(nbformat.v4.new_output("stream", name=content["name"], text=content["text"]))
        elif kind in ("execute_result", "display_data"):
            extra = {"execution_count": content["execution_count"]} if kind == "execute_result" else {}
            outputs.append(
                nbformat.v4.new_output(kind, data=content["data"], metadata=content["metadata"], **extra)
            )
        elif kind == "error":
            failed = True
            outputs.append(
                nbformat.v4.new_output(
                    "error", ename=content["ename"], evalue=content["evalue"], traceback=content["traceback"]
                )
            )
        elif kind == "clear_output":
            outputs = []
        elif kind == "status" and content["execution_state"] == "idle":
            return outputs, count, failed
        else:
            continue
        if on_output and kind != "execute_input":
            on_output(outputs, count)


def summary(outputs: list) -> str:
    # what this script prints back: text as-is, images as a placeholder
    parts = []
    for o in outputs:
        if o.output_type == "stream":
            parts.append(o.text.rstrip("\n"))
        elif o.output_type == "error":
            parts.append(f"ERROR {o.ename}: {o.evalue}")
        else:
            data = o.get("data", {})
            parts.append("[image]" if "image/png" in data else data.get("text/plain", ""))
    return "\n".join(parts)


def main() -> None:
    path, *which = sys.argv[1:]
    nb = nbformat.read(path, as_version=4)
    if which == ["all"]:
        indices = range(len(nb.cells))
    elif len(which) == 1 and which[0].startswith("from:"):
        indices = range(int(which[0][5:]), len(nb.cells))
    else:
        indices = which
    jobs = [(int(str(i).rstrip("d")), str(i).endswith("d")) for i in indices]  # (cell, defs only)
    jobs = [(i, d) for i, d in jobs if nb.cells[i].cell_type == "code" and nb.cells[i].source.strip()]

    kc = find_kernel(Path(path))
    for i, defs_only in jobs:
        if defs_only:
            keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
            tree = ast.parse(nb.cells[i].source)
            src = "\n".join(ast.unparse(n) for n in tree.body if isinstance(n, keep))
            outputs, count, failed = run_cell(kc, src)
            print(f"--- cell {i} defs only")
            print(summary(outputs))
            if failed:
                sys.exit(1)
            continue
        last = [0.0]

        def write(outputs: list, count: int | None, i: int = i) -> None:
            # re-read before writing, so an edit made while this cell ran isn't lost
            nb_now = nbformat.read(path, as_version=4)
            nb_now.cells[i].outputs = outputs
            nb_now.cells[i].execution_count = count
            nbformat.write(nb_now, path)
            last[0] = time.time()

        def progress(outputs: list, count: int | None) -> None:
            # partial outputs at most every 2 s, so VS Code shows a long cell's prints as they come
            if time.time() - last[0] > 2:
                write(outputs, count)

        outputs, count, failed = run_cell(kc, nb.cells[i].source, progress)
        write(outputs, count)
        print(f"--- cell {i} [{count}]")
        print(summary(outputs))
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
