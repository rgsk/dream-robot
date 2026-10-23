# Disk cache for notebook functions, which also replays what they print.
# Copy this file into any project with joblib, then
#     cache = NotebookCache("../artifacts/cache/<notebook>")
#     @cache
#     def train(steps): ...
# Results are keyed by the arguments and the source of the function's whole cell. A cell text
# never run before recomputes; one that ran before, even restored after other edits, replays
# that old result. So the one sure way to recompute is bumping the version (below).
# Defining one prints whether this cell text has results stored: "cached" or "not cached".
# Keep each cached function alone in its cell. Edits to other cells it uses don't recompute;
# to recompute after those, bump its version: `@cache` is version 1, then `@cache(2)`, ...
# So a version above 1 reads: this cell was rerun for changes in other cells, not its own.
# Going back below the highest version a cell has run with is refused: it would replay a result
# computed against the other cells as they were back then. Only for that same cell, though: edit
# anything else in it and it's a new cell, which starts again at any version, plain `@cache` too.
import contextlib
import functools
import hashlib
import inspect
import io
import json
import linecache
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO, cast, overload

from joblib import Memory
from joblib.memory import MemorizedFunc

VERSION = re.compile(r"(@\w+)\(\s*\d+\s*\)")  # `@cache(3)`, read as `@cache` when comparing cells


class Tee(io.StringIO):
    # keeps what is printed and still shows it live
    def __init__(self, live: TextIO) -> None:
        super().__init__()
        self.live = live

    def write(self, s: str, /) -> int:
        self.live.write(s)
        return super().write(s)


class NotebookCache:
    def __init__(self, path: str | Path) -> None:
        self.memory = Memory(location=path, verbose=0)
        self.path = Path(path)
        self.versions = Path(path) / "versions.json"  # each cell's highest version so far

    def check_version(self, func: Callable, version: int) -> None:
        # refuses a version below the highest one func's cell has run with, and records this one.
        # A cell is its text with the version taken out, so a bump keeps it the same cell
        cell = VERSION.sub(r"\1", "".join(linecache.getlines(inspect.getfile(func))))
        name = f"{func.__qualname__} {hashlib.sha1(cell.encode()).hexdigest()[:12]}"
        seen: dict[str, int] = json.loads(self.versions.read_text()) if self.versions.exists() else {}
        latest = seen.get(name, version)
        if version < latest:
            raise ValueError(
                f"{func.__qualname__}: version {version} is below {latest}, the highest this cell"
                " has run with, so it would replay a stale result;"
                f" use @cache({latest + 1}) to recompute"
            )
        seen[name] = version
        self.versions.parent.mkdir(parents=True, exist_ok=True)
        self.versions.write_text(json.dumps(seen, indent=1) + "\n")

    @overload
    def __call__[**P, R](self, arg: Callable[P, R], /) -> Callable[P, R]: ...
    @overload
    def __call__[**P, R](self, arg: int, /) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
    def __call__[**P, R](
        self, arg: Callable[P, R] | int, /
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        # @cache is version 1, @cache(n) version n
        if isinstance(arg, int):
            return lambda func: self.wrap(func, arg)
        return self.wrap(arg, 1)

    def wrap[**P, R](self, func: Callable[P, R], version: int) -> Callable[P, R]:
        self.check_version(func, version)
        ran = False

        def captured(*args: P.args, **kwargs: P.kwargs) -> tuple[R, str]:
            nonlocal ran
            ran = True
            printed = Tee(sys.stdout)
            with contextlib.redirect_stdout(printed):
                result = func(*args, **kwargs)
            return result, printed.getvalue()

        # joblib names the cache folder by the code's file, so file it under func's cell
        captured.__code__ = captured.__code__.replace(co_filename=inspect.getfile(func))
        # func's name, and its source for joblib's change check
        functools.update_wrapper(captured, func)
        memorized = cast(MemorizedFunc, self.memory.cache(captured))
        cached = cast(Callable[P, tuple[R, str]], memorized)
        # this cell text's stored calls, one folder per set of arguments
        folder = self.path / "joblib" / memorized.func_id
        stored = sum((d / "output.pkl").exists() for d in folder.iterdir()) if folder.exists() else 0
        name = func.__qualname__
        print(f"{name}: cached, {stored} call(s) stored" if stored else f"{name}: not cached, will compute")

        @functools.wraps(func)
        def replay(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal ran
            ran = False
            result, printed = cached(*args, **kwargs)
            if not ran:
                print(printed, end="")
            return result

        return replay

    def clear(self) -> None:
        # drops every result cached under path, and the versions with them
        self.memory.clear(warn=False)
        self.versions.unlink(missing_ok=True)
