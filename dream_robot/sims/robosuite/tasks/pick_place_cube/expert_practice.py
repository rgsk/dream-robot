from dataclasses import dataclass
from pathlib import Path
import yaml

TASK_YAML = Path(__file__).with_name("task.yaml")


@dataclass(frozen=True)
class IKConfig:
    damping: float
    max_joint_step: float


@dataclass(frozen=True)
class ExpertConfig:
    approach_height: float
    grasp_dz: float
    lift_height: float
    drop_height: float
    reach_tolerance: float
    fine_tolerance: float
    carry_tolerance: float
    lift_clearance: float
    release_opening: float
    grip_settle_window: int
    grip_settle_epsilon: float
    phase_timeout_seconds: float
    ik: IKConfig

    @classmethod
    def load(cls, path: Path = TASK_YAML) -> ExpertConfig:
        raw = yaml.safe_load(path.read_text())["expert"]
        return cls(
            approach_height=float(raw["approach_height"]),
            grasp_dz=float(raw["grasp_dz"]),
            lift_height=float(raw["lift_height"]),
            drop_height=float(raw["drop_height"]),
            reach_tolerance=float(raw["reach_tolerance"]),
            fine_tolerance=float(raw["fine_tolerance"]),
            carry_tolerance=float(raw["carry_tolerance"]),
            lift_clearance=float(raw["lift_clearance"]),
            release_opening=float(raw["release_opening"]),
            grip_settle_window=int(raw["grip_settle_window"]),
            grip_settle_epsilon=float(raw["grip_settle_epsilon"]),
            phase_timeout_seconds=float(raw["phase_timeout_seconds"]),
            ik=IKConfig(
                damping=float(raw["ik"]["damping"]),
                max_joint_step=float(raw["ik"]["max_joint_step"]),
            ),
        )

class ExpertPolicy:
    """Wires the pure phase machine to the robosuite task.

    Privileged state and MuJoCo handles in, contract-shaped actions out. This is
    the only place the two halves meet, and it is on the sim side of the seam:
    ``core`` never learns that an IK solver exists.
    """

    def __init__(self, env, config: ExpertConfig | None = None):
        self._env = env
        self._cfg = config or ExpertConfig.load()