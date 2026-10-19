"""core.camera.backproject and agent.perception.colour_mask, on a synthetic overhead camera."""

import numpy as np
import pytest
from fakes import overhead_rgbd

from dream_robot.agent.perception import colour_mask
from dream_robot.core.camera import backproject

BLUE, RED = (25, 60, 230), (220, 25, 25)


def test_backproject_lands_on_the_floor_below_the_camera():
    frame = overhead_rgbd(hw=64, height=1.0)
    mask = np.zeros((64, 64), dtype=bool)
    mask[31:33, 31:33] = True                     # the four centre pixels
    points = backproject(frame, mask)
    np.testing.assert_allclose(points.mean(axis=0), [0, 0, 0], atol=1e-9)


def test_image_up_is_world_plus_y_and_right_is_plus_x():
    frame = overhead_rgbd(hw=64, height=1.0)
    mask = np.zeros((64, 64), dtype=bool)
    mask[0, 63] = True                            # top-right pixel
    (x, y, z), = backproject(frame, mask)
    assert x > 0.9 and y > 0.9 and z == pytest.approx(0)


def test_white_and_grey_are_no_colour():
    grey = np.array([[[255, 255, 255], [140, 140, 140], [20, 20, 20]]], dtype=np.uint8)
    for colour in ("red", "green", "blue", "yellow"):
        assert not colour_mask(grey, colour).any()
