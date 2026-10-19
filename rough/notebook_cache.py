# Disk cache for notebook functions, which also replays what they print.
# Copy this file into any project with joblib, then
#     cache = NotebookCache("../artifacts/cache/<notebook>")
#     @cache
#     def train(steps): ...
# Results are keyed by the arguments and the source of the function's whole cell,
# so editing anything in that cell recomputes, and restoring a cell restores its results.
import contextlib
import functools
import inspect
import io
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO, cast

from joblib import Memory


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

    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]:
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
        cached = cast(Callable[P, tuple[R, str]], self.memory.cache(captured))

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
        # drops every result cached under path
        self.memory.clear(warn=False)
