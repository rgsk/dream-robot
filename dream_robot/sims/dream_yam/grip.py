"""Grasp and release from the gripper's own sensor: what a real robot knows without cameras.

A position-servoed gripper told to close to opening ``c`` (0 closed, 1 open) gets there unless
something is between the fingers: then it stops wider, at the object. So **holding** is the
measured opening staying above the commanded one, with the fingers still. **Released** is that gap
gone: the hand opened, or the object slipped out (the camera tells which).

Both need the condition to last ``hold_steps`` (0.3 s at 20 Hz): fingers in motion lag their
command for a moment, and that is not a grasp.
"""

GAP_HOLD = 0.08      # measured - commanded opening above this: something is in the way
GAP_FREE = 0.01      # below this: nothing is (a loosened grip on a held banana sat at 0.02-0.03)
STILL = 0.005        # fingers moving less than this per step count as stopped


class GripWatch:
    """Per-arm holding state from (commanded, measured) openings, one step at a time."""

    def __init__(self, hold_steps=6):
        self.hold_steps = hold_steps
        self.holding = {"left": False, "right": False}
        self.width = {"left": None, "right": None}      # opening when the grasp was detected
        self._prev = {"left": None, "right": None}
        self._run = {"left": 0, "right": 0}             # steps the opposite state has lasted

    def update(self, commanded, measured):
        """One step. ``commanded`` / ``measured``: openings by arm. Returns events this step:
        ``[(arm, "grasp" | "release", opening)]``."""
        events = []
        for arm in ("left", "right"):
            c, o = commanded[arm], measured[arm]
            prev, self._prev[arm] = self._prev[arm], o
            still = prev is not None and abs(o - prev) < STILL
            gap = o - c
            if not self.holding[arm]:
                flip = gap > GAP_HOLD and still
            else:
                flip = gap < GAP_FREE
            self._run[arm] = self._run[arm] + 1 if flip else 0
            if self._run[arm] >= self.hold_steps:
                self._run[arm] = 0
                self.holding[arm] = not self.holding[arm]
                self.width[arm] = round(o, 2) if self.holding[arm] else None
                events.append((arm, "grasp" if self.holding[arm] else "release", round(o, 2)))
        return events
