"""Does BC predict 'close' at the true close moment on demo frames? Per model, on its own dataset,
held-out (val) and train episodes, frames first_close-10 .. first_close+10."""
import json, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from dream_robot.policies.bc.policy import BCPolicy

cases = [("clean BC", "experiments/bc_pick_place_cube", "data/robosuite/pick_place_cube"),
         ("shaky BC", "experiments/bc_noise005", "data/robosuite/pick_place_cube_noise005")]
for name, exp, root in cases:
    summ = json.load(open(f"{exp}/training_summary.json"))
    pol = BCPolicy.load(f"{exp}/checkpoint.pt"); net, dev = pol._net, pol._device
    ds = LeRobotDataset("dream_robot/pick_place_cube", root=root)
    ep = np.asarray(ds.hf_dataset["episode_index"]); act = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    for split in ("val_episodes", "train_episodes"):
        before, after, hit = [], [], []
        for e in summ[split]:
            idx = np.nonzero(ep == e)[0]; c = idx[np.nonzero(act[idx, 7] < 0.5)[0][0]]
            for i in range(c - 10, c + 11):
                item = ds[int(i)]
                imgs = {k: item[f"observation.images.{k}"][None].to(dev) for k in net.cameras}
                with torch.no_grad():
                    g = float(net.predict(imgs, item["observation.state"][None].to(dev))[0, 7])
                (before if i < c else after).append(g)
            hit.append(np.mean(np.array(after[-11:]) < 0.5))
        print(f"{name} {split[:-9]:5}: {len(summ[split])} eps | predicted gripper, 10 frames before close "
              f"(label 1) mean {np.mean(before):.2f} | close frame +0..10 (label 0) mean {np.mean(after):.2f}, "
              f"predicted <0.5 on {np.mean(np.array(after) < 0.5):.0%} of frames; eps with any close predicted "
              f"{sum(h > 0 for h in hit)}/{len(hit)}", flush=True)
