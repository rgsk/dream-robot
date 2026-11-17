# Helpers for probe cells. Part of nbtools (copyable alone into any project with IPython, which
# brings `executing`):
#     from nbtools import probe, show
#
#     show(ins, ins.shape, ins.nonzero())
#     ins = tensor([[0., 1., 0., 0., 1., 0.]])
#     ins.shape = torch.Size([1, 6])
#     ins.nonzero() =
#     tensor([[0, 1],
#             [0, 4]])
#     show('hello', [1, 2], f'{n} rows')   # literals print as print() would
#     hello
#     [1, 2]
#     3 rows
#
# A probe's own names stay out of the notebook's globals in either of these:
#     class _:        # one line; but a comprehension or lambda inside can't see names made in it
#         show(...)
#     @probe          # a function: no such catch
#     def _():
#         show(...)

import ast
import sys

import executing


def show(*values):
    # one line per argument, as print(f'{x = }') (repr); a value that prints over several lines
    # starts on the line below its name. executing finds the call in the caller's source, so the
    # arguments are plain code the IDE completes. No source (python -c): names print as '?'
    # A literal argument ('hello', 1, [1, 2], an f-string) has no name worth showing: it prints
    # as print() would, str and all
    ex = executing.Source.executing(sys._getframe(1))
    if ex.node is None:
        args = [None] * len(values)
    else:
        args = ex.node.args
    for arg, value in zip(args, values, strict=True):
        if arg is not None and is_literal(arg):
            print(value)
            continue
        # each argument's text as written (ast.unparse would respace it: 7/4 -> 7 / 4)
        name = '?' if arg is None else ast.get_source_segment(ex.source.text, arg)
        text = repr(value)
        print(f'{name} =' + ('\n' if '\n' in text else ' ') + text)


def is_literal(node: ast.expr) -> bool:
    # a constant, an f-string, or a list/tuple/set/dict built only of those (-1 included)
    if isinstance(node, (ast.Constant, ast.JoinedStr)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return is_literal(node.operand)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(is_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and is_literal(k) for k in node.keys) and all(
            is_literal(v) for v in node.values
        )
    return False


def probe(f):
    # @probe on a def: call it right away
    return f()
