"""Policies. Each depends on torch and ``dream_robot.core`` -- never on a simulator.

ROADMAP rule 1. It is what survives the Isaac venv split: recording happens
inside a backend's own Python runtime, training happens here, and the two meet
at a directory on disk. If anything under this package ever imports robosuite,
the seam has been bypassed and cross-sim comparability is gone.
"""
