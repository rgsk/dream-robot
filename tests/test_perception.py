"""core.camera.backproject and agent.perception.find, on a synthetic overhead camera."""

import numpy as np
import pytest
from fakes import overhead_rgbd

from dream_robot.agent.perception import colour_mask, find
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


def test_find_returns_the_top_centre_of_the_named_square():
    frame = overhead_rgbd(hw=128, squares=[
        (BLUE, (0.2, -0.1), 0.05, 0.04),
        (RED, (-0.2, 0.1), 0.05, 0.04),
    ])
    (found,) = find(frame, colour_mask(frame.rgb, "blue"))
    np.testing.assert_allclose(found.point, [0.2, -0.1, 0.04], atol=0.01)


def test_absent_colour_is_an_empty_list_not_an_error():
    frame = overhead_rgbd(hw=64, squares=[(BLUE, (0, 0), 0.1, 0.04)])
    assert find(frame, colour_mask(frame.rgb, "red")) == []


def test_two_same_coloured_objects_are_two_candidates_never_their_average():
    """The step-3 bug: one mask, one median, a point on the bigger cube or between."""
    frame = overhead_rgbd(hw=256, squares=[
        (BLUE, (0.2, 0.0), 0.06, 0.04),     # bigger in the image: listed first
        (BLUE, (-0.2, 0.0), 0.04, 0.04),
    ])
    found = find(frame, colour_mask(frame.rgb, "blue"))
    assert len(found) == 2
    np.testing.assert_allclose(found[0].point, [0.2, 0.0, 0.04], atol=0.01)
    np.testing.assert_allclose(found[1].point, [-0.2, 0.0, 0.04], atol=0.01)
    assert not (found[0].mask & found[1].mask).any()


def test_specks_below_the_noise_floor_are_not_objects():
    frame = overhead_rgbd(hw=128, squares=[
        (BLUE, (0.0, 0.0), 0.06, 0.04),
        (BLUE, (0.3, 0.3), 0.01, 0.04),     # a few pixels
    ])
    assert len(find(frame, colour_mask(frame.rgb, "blue"))) == 1


def test_white_and_grey_are_no_colour():
    grey = np.array([[[255, 255, 255], [140, 140, 140], [20, 20, 20]]], dtype=np.uint8)
    for colour in ("red", "green", "blue", "yellow"):
        assert not colour_mask(grey, colour).any()
