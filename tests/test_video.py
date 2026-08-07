import numpy as np
import pytest

from dream_robot.core.video import observation_panel, upscale_nearest, write_video


def frame(h, w, value=0):
    return np.full((h, w, 3), value, dtype=np.uint8)


# --- upscaling --------------------------------------------------------------

def test_upscale_repeats_exactly():
    img = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    out = upscale_nearest(img, 2)
    assert out.shape == (4, 4, 3)
    # Nearest-neighbour, so every source pixel appears verbatim in a 2x2 block.
    assert np.array_equal(out[0:2, 0:2], np.tile(img[0, 0], (2, 2, 1)))


def test_upscale_introduces_no_new_values():
    """Any interpolation would invent colours the policy never saw."""
    img = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    assert set(np.unique(upscale_nearest(img, 3))) == set(np.unique(img))


def test_upscale_rejects_zero():
    with pytest.raises(ValueError, match="factor must be >= 1"):
        upscale_nearest(frame(4, 4), 0)


# --- panel composition ------------------------------------------------------

def test_panel_places_cameras_beside_the_wide_view():
    panel = observation_panel(frame(256, 256, 10), {"top": frame(128, 128, 20),
                                                    "wrist": frame(128, 128, 30)})
    # Two 128px cameras stack into a 128-wide column: 256 + 128 = 384.
    assert panel.shape == (256, 384, 3)
    assert panel[0, 0, 0] == 10          # wide view on the left
    assert panel[0, 300, 0] == 20        # top camera, upper right
    assert panel[200, 300, 0] == 30      # wrist camera, lower right


def test_panel_handles_a_single_camera():
    panel = observation_panel(frame(128, 128), {"top": frame(128, 128)}, order=("top",))
    assert panel.shape == (128, 256, 3)


def test_panel_ignores_cameras_not_in_order():
    panel = observation_panel(frame(256, 256), {"top": frame(128, 128)})
    # Only `top` is known, so it scales to the full 256 height: 256 + 256 = 512.
    assert panel.shape == (256, 512, 3)


def test_panel_needs_at_least_one_known_camera():
    with pytest.raises(ValueError, match="none of"):
        observation_panel(frame(256, 256), {"agentview": frame(128, 128)})


def test_panel_rejects_non_integer_scaling():
    """Integer scaling only, so the panel cannot misrepresent the resolution."""
    with pytest.raises(ValueError, match="does not divide"):
        observation_panel(frame(256, 256), {"top": frame(100, 100), "wrist": frame(100, 100)})


def test_panel_rejects_float_frames():
    with pytest.raises(ValueError, match="wide frame must be uint8"):
        observation_panel(np.zeros((8, 8, 3), np.float32), {"top": frame(4, 4)})


# --- writing ----------------------------------------------------------------

def test_write_video_roundtrips(tmp_path):
    import imageio.v3 as iio

    frames = [frame(32, 32, v) for v in (0, 40, 80, 120)]
    path = write_video(frames, tmp_path / "nested" / "out.mp4", fps=30)
    assert path.exists()
    assert iio.imread(path).shape[0] == len(frames)


def test_write_video_rejects_empty():
    with pytest.raises(ValueError, match="no frames"):
        write_video([], "unused.mp4", fps=30)


def test_write_video_rejects_float_frames(tmp_path):
    with pytest.raises(ValueError, match="frames must be uint8"):
        write_video([np.zeros((8, 8, 3), np.float32)], tmp_path / "x.mp4", fps=30)
