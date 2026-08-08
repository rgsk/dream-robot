from dataclasses import dataclass
from pathlib import Path
import yaml

TASK_YAML = Path(__file__).with_name("task_practice.yaml")

@dataclass(frozen=True)
class BinSpec:
    offset_xy: tuple[float, float]
    inner_half: float
    wall_half_thickness: float
    wall_half_height: float

@dataclass(frozen=True)
class TaskConfig:
    control_hz: float
    horizon_seconds: float
    camera_hw: tuple[int, int]
    camera_mapping: dict[str, str]
    render_camera: str
    render_hw: tuple[int, int]
    cube_xy_range: tuple[float, float]
    bin: BinSpec
    resting_z_margin: float
    lift_threshold: float
    dropped_radius: float                    
    
    @classmethod
    def load(cls, path: Path = TASK_YAML) -> TaskConfig:
        raw = yaml.safe_load(path.read_text())
        b = raw["scene"]["bin"]
        return cls(
            control_hz=float(raw["control"]["hz"]),
            horizon_seconds=float(raw["control"]["horizon_seconds"]),
            camera_hw=(int(raw["cameras"]["height"]), int(raw["cameras"]["width"])),
            camera_mapping=dict(raw["cameras"]["mapping"]),
            render_camera=str(raw["render"]["camera"]),
            render_hw=(int(raw["render"]["height"]), int(raw["render"]["width"])),
            cube_xy_range=tuple(raw["scene"]["cube_xy_range"]),
            bin=BinSpec(
                offset_xy=tuple(b["offset_xy"]),
                inner_half=float(b["inner_half"]),
                wall_half_thickness=float(b["wall_half_thickness"]),
                wall_half_height=float(b["wall_half_height"]),
            ),
            resting_z_margin=float(raw["success"]["resting_z_margin"]),
            lift_threshold=float(raw["failure"]["lift_threshold"]),
            dropped_radius=float(raw["failure"]["dropped_radius"]),
        )


class PickPlaceCube:
    """Implements ``core.env_api.Env``. Structurally, without inheriting it."""

    def __init__(self, config: TaskConfig | None = None):
        pass