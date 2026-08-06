import ast
import inspect

import numpy as np
import pytest

from dream_robot.core import schema
from dream_robot.core.schema import (
    PANDA,
    Embodiment,
    build_observation,
    build_state,
    joint_tracking_error,
    validate_action,
    validate_images,
)


def frame(h=8, w=8):
    return np.zeros((h, w, 3), dtype=np.uint8)


def ok_state(embodiment=PANDA):
    return dict(
        joint_positions=np.zeros(embodiment.arm_joints),
        gripper_openings=np.full(embodiment.grippers, 0.5),
        embodiment=embodiment,
    )


# --- embodiment ------------------------------------------------------------

def test_panda_matches_spec_dim_8():
    assert PANDA.dim == 8


def test_bimanual_is_just_a_bigger_dim():
    """ROADMAP rule 2: bimanual must not require a policy change."""
    bi = Embodiment(arm_joints=14, grippers=2)
    assert bi.dim == 16
    assert bi.arm_slice == slice(0, 14)
    assert bi.gripper_slice == slice(14, 16)


@pytest.mark.parametrize("kwargs", [{"arm_joints": 0}, {"arm_joints": 7, "grippers": 0}])
def test_degenerate_embodiment_rejected(kwargs):
    with pytest.raises(ValueError):
        Embodiment(**kwargs)


# --- state -----------------------------------------------------------------

def test_build_state_layout_and_dtype():
    s = build_state(
        joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        gripper_openings=[1.0],
        embodiment=PANDA,
    )
    assert s.dtype == np.float32 and s.shape == (8,)
    assert np.allclose(s[PANDA.arm_slice], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert s[PANDA.gripper_slice] == pytest.approx([1.0])


def test_wrong_joint_count_rejected():
    with pytest.raises(ValueError, match="expected 7 joint positions"):
        build_state(joint_positions=np.zeros(6), gripper_openings=[0.0], embodiment=PANDA)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 42.0])
def test_unnormalised_gripper_rejected(bad):
    """Two mirrored finger joints in raw metres must be normalised in the adapter."""
    with pytest.raises(ValueError, match="gripper_openings must lie"):
        build_state(joint_positions=np.zeros(7), gripper_openings=[bad], embodiment=PANDA)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_state_rejected(bad):
    joints = np.zeros(7)
    joints[3] = bad
    with pytest.raises(ValueError, match="non-finite"):
        build_state(joint_positions=joints, gripper_openings=[0.0], embodiment=PANDA)


# --- images ----------------------------------------------------------------

def test_canonical_camera_names_accepted():
    imgs = validate_images({"top": frame(), "wrist": frame()})
    assert set(imgs) == {"top", "wrist"}


def test_backend_camera_name_rejected():
    """agentview / robot0_eye_in_hand must be renamed in the adapter."""
    with pytest.raises(ValueError, match="non-canonical camera name"):
        validate_images({"agentview": frame()})


def test_no_cameras_rejected():
    with pytest.raises(ValueError, match="at least one camera"):
        validate_images({})


def test_float_images_rejected():
    with pytest.raises(ValueError, match="expected uint8"):
        validate_images({"top": np.zeros((8, 8, 3), dtype=np.float32)})


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 1), (3, 8, 8)])
def test_wrong_image_shape_rejected(shape):
    """(3, H, W) is the common one -- torch layout leaking into the recorder."""
    with pytest.raises(ValueError, match=r"expected \(H, W, 3\)"):
        validate_images({"top": np.zeros(shape, dtype=np.uint8)})


# --- the structural guarantee ----------------------------------------------

@pytest.mark.parametrize("fn", [build_state, build_observation])
def test_no_channel_for_privileged_state(fn):
    """spec.md bans object pose from observation.* and says it is enforced
    structurally. The enforcement is this: the constructors take named numeric
    arguments and have no dict, no **kwargs, no *args through which a backend's
    raw observation could arrive. If someone adds one, this fails.
    """
    params = inspect.signature(fn).parameters
    kinds = {p.kind for p in params.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds
    assert inspect.Parameter.VAR_POSITIONAL not in kinds
    allowed = {"joint_positions", "gripper_openings", "images", "embodiment"}
    assert set(params) <= allowed, f"unexpected parameter(s): {set(params) - allowed}"


def test_core_imports_no_simulator():
    """ROADMAP rule 1. core must stay importable inside a policy-only venv.

    Checks the import statements, not the source text -- the docstrings talk
    about robosuite on purpose.
    """
    tree = ast.parse(inspect.getsource(schema))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "numpy", "dataclasses", "typing", "collections", "__future__",
    }, imported


# --- action ----------------------------------------------------------------

def test_validate_action_roundtrip():
    a = validate_action(np.zeros(8), PANDA)
    assert a.dtype == np.float32 and a.shape == (8,)


def test_action_dim_mismatch_rejected():
    with pytest.raises(ValueError, match="expected action of dim 8"):
        validate_action(np.zeros(7), PANDA)


def test_plus_minus_one_gripper_convention_rejected():
    """robosuite uses -1 open / +1 close; the contract uses [0, 1]."""
    a = np.zeros(8)
    a[PANDA.gripper_slice] = -1.0
    with pytest.raises(ValueError, match="gripper entries must lie"):
        validate_action(a, PANDA)


# --- the recorder integrity check ------------------------------------------

def test_perfect_tracking_is_zero_error():
    states = np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 8))
    actions = np.roll(states, -1, axis=0)  # action[t] == state[t+1], exactly
    mean, mx = joint_tracking_error(actions, states, PANDA)
    assert mean == 0.0 and mx == 0.0


def test_off_by_one_recorder_is_caught():
    """The failure this check exists for: writing action[t] == state[t]."""
    states = np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 8))
    mean, _ = joint_tracking_error(states, states, PANDA)
    assert mean == pytest.approx(1.0)
