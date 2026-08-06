"""Sim-agnostic core: schema, dataset, env protocol, recorder, eval.

Nothing in this package may import a simulator. Policies depend on torch + core
only (ROADMAP rule 1), and that is what lets Isaac -- which pins its own Python
runtime -- coexist with robosuite in one repo.
"""
