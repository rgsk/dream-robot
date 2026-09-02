"""Behaviour cloning: predict the expert's action from what the expert could see.

The baseline every other method is measured against, and deliberately the
dumbest thing that could work -- one observation in, one action out, no
chunking, no temporal model, no generative head. Its failures are the argument
for everything above it on the ladder.
"""
