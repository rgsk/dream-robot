from pathlib import Path

import numpy as np
import torch

from dream_robot.core.dataset import open_dataset
from dream_robot.policies.bc.data import read_spec

ROOT = Path("data/robosuite/pick_place_cube")
REPO_ID = "dream_robot/pick_place_cube"

torch.manual_seed(0)
np.random.seed(0)
print(f"torch {torch.__version__}  ·  dataset present: {ROOT.exists()}")

spec = read_spec(REPO_ID, ROOT)
print(spec)

ds = open_dataset(REPO_ID, ROOT)
item = ds[0]
for k, v in item.items():
    a = np.asarray(v)
    print(f"{k:28s} {str(tuple(a.shape)):16s} {a.dtype}")