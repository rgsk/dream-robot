"""frozendict (the PyPI package, for Python < 3.15) with per-key types.

Each line has what basedpyright says TODAY (stock 1.40.2) and what the local basedpyright fork
(../../projects/basedpyright-frozendict) will say after adding support for this package:
  step 1: recognize frozendict.frozendict like the 3.15 built-in (per-key types, typo errors,
          completion, printing, annotations, merging, .get(), .copy())
Only what the built-in also has gets per-key types, so moving to 3.15 later is just the import:
.set() and .delete() exist only in this package and keep its stub's plain types.

Needs `uv add frozendict` to run, and the fork's wrapper in .venv/bin to see the step 1 types.
"""

from frozendict import frozendict


def f() -> frozendict[{'ok': bool}]:
    return frozendict(ok=True)


print(f())

from typing import reveal_type

# %% Per-key types (step 1)
user = frozendict(name='rahul', age=27)
reveal_type(user)  # today: frozendict[str, str | int]
#   step 1: frozendict[{'name': str, 'age': int}]
reveal_type(user['age'])  # today: str | int   step 1: int
try:
    user['nme']  # today: no error   step 1: error, "nme" is not a defined key
except KeyError as e:
    print('KeyError:', e)

cfg = frozendict({'lr': 1e-3, 'layers': (64, 64)})
reveal_type(cfg['layers'])  # today: Unknown   step 1: tuple[int, int]

# %% The shape in annotations (step 1). Works at runtime on 3.14 too
# today: 3 errors on this line ("Dictionary expression not allowed in type expression", ...)
type User = frozendict[{'name': str, 'age': int}]


def load() -> User:
    return frozendict(name='rahul', age=27)


reveal_type(load()['age'])  # today: Unknown   step 1: int

# %% Merging, .get(), .copy() (step 1)
reveal_type(user | {'age': 28})  # today: frozendict[str, str | int]
#   step 1: frozendict[{'name': str, 'age': int}]
reveal_type(user.get('age'))  # today: str | int | None   step 1: int (the key is always there)
reveal_type(user.get('nme'))  # today: str | int | None
#   step 1: None (no error: it doesn't fail at runtime)
reveal_type(user.copy())  # today: frozendict[str, str | int]
#   step 1: frozendict[{'name': str, 'age': int}]
# frozendict(user, city='delhi') runs, but this package's stub has no (mapping, **kwargs)
# signature, so it's an error today and after step 1. Use user | {'city': 'delhi'}.

# %% .set() and .delete(): this package only, so no per-key types (the built-in has neither)
reveal_type(user.set('city', 'delhi'))  # frozendict[str, str | int], today and with the fork
reveal_type(user.delete('age'))  # frozendict[str, str | int], today and with the fork
# Portable instead of .set(): `|` keeps the keys with the fork, and works on the 3.15 built-in
reveal_type(user | {'city': 'delhi'})  # today: frozendict[str, str | int]
#   step 1: frozendict[{'name': str, 'age': int, 'city': str}]
