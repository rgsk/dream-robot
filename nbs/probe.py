# Helpers for probe cells. Copy this file into any project with IPython (it brings `executing`), then
#     sys.path.insert(0, '..')  # nbs/probe.py
#     from probe import probe, show
#
#     show(ins, ins.shape, ins.nonzero())
#     ins = tensor([[0., 1., 0., 0., 1., 0.]])
#     ins.shape = torch.Size([1, 6])
#     ins.nonzero() =
#     tensor([[0, 1],
#             [0, 4]])
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
    ex = executing.Source.executing(sys._getframe(1))
    if ex.node is None:
        names = ['?'] * len(values)
    else:  # each argument's text as written (ast.unparse would respace it: 7/4 -> 7 / 4)
        names = [ast.get_source_segment(ex.source.text, a) for a in ex.node.args]
    for name, value in zip(names, values):
        text = repr(value)
        print(f'{name} =' + ('\n' if '\n' in text else ' ') + text)


def probe(f):
    # @probe on a def: call it right away
    return f()
