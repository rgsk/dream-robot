# Disk cache for notebook functions, which also replays what they print.
# Copy this file into any project with joblib, then
#     cache = NotebookCache("../artifacts/cache/<notebook>")
#     @cache
#     def train(steps): ...
# Results are keyed by the arguments and the code of the function's whole cell, as Python parses
# it: comments, layout and docstrings don't count, so editing only those replays the old result.
# A cell code never run before recomputes; one that ran before, even restored after other edits,
# replays that old result. So the one sure way to recompute is bumping the version (below).
# Defining one prints whether this cell code has results stored: "cached" or "not cached".
# Keep each cached function alone in its cell. Edits to other cells it uses don't recompute;
# to recompute after those, bump its version: `@cache` is version 1, then `@cache(2)`, ...
# So a version above 1 reads: this cell was rerun for changes in other cells, not its own.
# `@cache(uses=[module, ...])`: the listed modules' code (parsed, as cells are) joins the cell's,
# so editing a module the function relies on recomputes, and editing it back replays.
# Going back below the highest version a cell has run with is refused: it would replay a result
# computed against the other cells as they were back then. Only for that same cell, though: edit
# anything else in it and it's a new cell, which starts again at any version, plain `@cache` too.
# `@cache(keep=True)` (or `@cache(2, keep=True)`): when this cell code has no result for a call,
# replay one stored by an OLDER version of the cell (same function name, same arguments; a
# parameter added since counts as matching at its default), newest first. It is
# copied in, so later runs replay it as usual,
# also after removing keep (one-shot: run once with keep, then drop it). For edits you know
# don't change the result; left on, it would replay stale results after a real change.
# Every replay of a copied result says so: `copied by keep=True from older code (cell ...)`.
# keep only copies results stored at the same version: `@cache(3, keep=True)` right after a
# bump finds none and recomputes, as the bump asks.
import ast
import contextlib
import functools
import hashlib
import inspect
import io
import json
import linecache
import re
import shutil
import sys
import textwrap
import types
from types import ModuleType
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO, cast, overload

from joblib import Memory
from joblib.func_inspect import filter_args
from joblib.memory import MemorizedFunc

# `@cache(3)`, read as `@cache` when comparing cells
VERSION = re.compile(r'(@\w+)\(\s*\d+\s*\)')
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
# `keep=...` in the decorator is not part of the cell: `@cache(keep=True)` reads as `@cache`,
# `@cache(2, keep=True)` as `@cache(2)`, so removing it after one run keeps the copied results
KEEP_ONLY = re.compile(r'(@\w+)\(\s*keep\s*=\s*\w+\s*\)')
KEEP_ARG = re.compile(r'(@\w+\(\s*\d+)\s*,\s*keep\s*=\s*\w+\s*\)')


def without_keep(code: str) -> str:
    return KEEP_ARG.sub(r'\1)', KEEP_ONLY.sub(r'\1', code))


def normalized(code: str) -> str:
    # code as Python parses it, without comments, layout or docstrings: edits to only
    # those give the same text. Code that doesn't parse (an untransformed magic) stays
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if isinstance(node, DOCSTRING_OWNERS) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def module_code(uses) -> str:
    # the listed modules' code as Python parses it: formatting, comments and docstrings in
    # them don't count, the same rule as for cells
    return ''.join(normalized(inspect.getsource(m)) for m in uses)


def digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


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
        # each cell's highest version so far
        self.versions = Path(path) / 'versions.json'

    def check_version(self, func: Callable, version: int, uses=()) -> None:
        # refuses a version below the highest one func's cell has run with, and records this one.
        # A cell is its code with the version taken out, so a bump keeps it the same cell
        cell = normalized(
            VERSION.sub(r'\1', without_keep(''.join(linecache.getlines(inspect.getfile(func)))))
        ) + module_code(uses)
        name = f'{func.__qualname__} {digest(cell)}'
        seen: dict[str, int] = (
            json.loads(self.versions.read_text()) if self.versions.exists() else {}
        )
        latest = seen.get(name, version)
        if version < latest:
            raise ValueError(
                f'{func.__qualname__}: version {version} is below {latest}, the highest this cell'
                ' has run with, so it would replay a stale result;'
                f' use @cache({latest + 1}) to recompute'
            )
        seen[name] = version
        self.versions.parent.mkdir(parents=True, exist_ok=True)
        self.versions.write_text(json.dumps(seen, indent=1) + '\n')

    @overload
    def __call__[**P, R](self, arg: Callable[P, R], /) -> Callable[P, R]: ...
    @overload
    def __call__[**P, R](
        self, arg: int = 1, /, *, keep: bool = False, uses: Sequence[ModuleType] = ()
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
    def __call__[**P, R](
        self,
        arg: Callable[P, R] | int = 1,
        /,
        *,
        keep: bool = False,
        uses: Sequence[ModuleType] = (),
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        # @cache is version 1, @cache(n) version n; keep=True: replay older code's results;
        # uses=[module, ...]: those modules' code is part of the cell, so editing it recomputes
        if isinstance(arg, int):
            return lambda func: self.wrap(func, arg, keep, uses)
        return self.wrap(arg, 1, False)

    def wrap[**P, R](
        self, func: Callable[P, R], version: int, keep: bool = False, uses=()
    ) -> Callable[P, R]:
        self.check_version(func, version, uses)
        ran = False

        def captured(*args: P.args, **kwargs: P.kwargs) -> tuple[R, str]:
            nonlocal ran
            ran = True
            printed = Tee(sys.stdout)
            with contextlib.redirect_stdout(printed):
                result = func(*args, **kwargs)
            return result, printed.getvalue()

        # joblib names the cache folder by the code's file, and wipes it when the
        # function's source changes. So both go through a stand-in file: it holds func's
        # normalized source and is named by the normalized cell (version kept, so a bump
        # is a new folder). It sits next to the real cell, as joblib reads the folder
        # ipykernel_<pid>/ the same in every session
        file = inspect.getfile(func)
        cell = normalized(without_keep(''.join(linecache.getlines(file)))) + module_code(uses)
        stand_in = str(Path(file).with_name(f'cell-{digest(cell)}.py'))
        source = normalized(without_keep(textwrap.dedent(inspect.getsource(func)))) + '\n'
        linecache.cache[stand_in] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            stand_in,
        )
        # func, read from the stand-in: joblib takes its arguments and source through
        # __wrapped__
        shim = types.FunctionType(
            func.__code__.replace(co_filename=stand_in, co_firstlineno=1),
            func.__globals__,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )
        shim.__kwdefaults__ = func.__kwdefaults__
        captured.__code__ = captured.__code__.replace(co_filename=stand_in)
        functools.update_wrapper(captured, func)  # func's name
        captured.__wrapped__ = shim
        memorized = cast(MemorizedFunc, self.memory.cache(captured))
        cached = cast(Callable[P, tuple[R, str]], memorized)
        # this cell code's stored calls, one folder per set of arguments
        folder = Path(memorized.store_backend.location) / memorized.func_id
        stored = (
            sum((d / 'output.pkl').exists() for d in folder.iterdir()) if folder.exists() else 0
        )
        name = func.__qualname__
        # each cell folder records its version, so keep never copies across a bump
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'cache_version').write_text(str(version))
        print(
            f'{name}: cached, {stored} call(s) stored'
            if stored
            else f'{name}: not cached, will compute'
            + (" (or replay older code's result: keep)" if keep else '')
        )
        defaults = {
            k: repr(p.default)
            for k, p in inspect.signature(func).parameters.items()
            if p.default is not inspect.Parameter.empty
        }

        def adopt(args: tuple, kwargs: dict) -> None:
            # no result for this call in this cell code: copy the newest matching one stored
            # by an older version of the cell, with this call's argument key
            args_id = memorized._get_args_id(*args, **kwargs)
            if (folder / args_id / 'output.pkl').exists():
                return
            want = {
                k: repr(v)
                for k, v in filter_args(memorized.func, memorized.ignore, args, kwargs).items()
            }
            found = []
            for meta in folder.parent.parent.glob(f'*/{folder.name}/*/metadata.json'):
                if meta.parent.parent == folder or not (meta.parent / 'output.pkl').exists():
                    continue
                stamp = meta.parent.parent / 'cache_version'
                if (int(stamp.read_text()) if stamp.exists() else 1) != version:
                    continue  # another version: a bump means recompute
                old = json.loads(meta.read_text())['input_args']
                if set(old) <= set(want) and all(
                    old[k] == v if k in old else defaults.get(k) == v for k, v in want.items()
                ):
                    found.append(meta.parent)
            if not found:
                return
            src = max(found, key=lambda d: (d / 'output.pkl').stat().st_mtime)
            memorized._check_previous_func_code()  # writes this cell's func_code.py
            dst = folder / args_id
            shutil.copytree(src, dst, dirs_exist_ok=True)
            meta = json.loads((dst / 'metadata.json').read_text())
            old_cell = src.parent.parent.name.rsplit('cell-', 1)[-1]
            meta |= {
                'input_args': want,
                'copied_by_keep_from': meta.get('copied_by_keep_from', old_cell),
            }
            (dst / 'metadata.json').write_text(json.dumps(meta))

        @functools.wraps(func)
        def replay(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal ran
            ran = False
            if keep:
                adopt(args, kwargs)
            result, printed = cached(*args, **kwargs)
            if not ran:
                # a result keep=True copied in says so on every replay
                meta = folder / memorized._get_args_id(*args, **kwargs) / 'metadata.json'
                copied = json.loads(meta.read_text()).get('copied_by_keep_from')
                if copied:
                    print(f'{name}: result copied by keep=True from older code (cell {copied})')
                print(printed, end='')
            return result

        return replay

    def clear(self) -> None:
        # drops every result cached under path, and the versions with them
        self.memory.clear(warn=False)
        self.versions.unlink(missing_ok=True)
