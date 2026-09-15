"""Copycat test: does BC shrink the corrective part of the action (target - current pose) toward zero?
On each model's held-out demo frames, gripper-open only, binned by the label's correction size."""
import json, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from dream_robot.policies.bc.policy import BCPolicy

cases = [("clean BC", "experiments/bc_pick_place_cube", "data/robosuite/pick_place_cube"),
         ("shaky BC", "experiments/bc_noise005", "data/robosuite/pick_place_cube_noise005"),
         ("shaky BC 100", "experiments/bc_noise005_n100", "data/robosuite/pick_place_cube_noise005_n100")]
bins = [(0, 10), (10, 25), (25, 45), (45, 80)]   # mrad, L2 over the 7 arm joints
for name, exp, root in cases:
    val = json.load(open(f"{exp}/training_summary.json"))["val_episodes"]
    pol = BCPolicy.load(f"{exp}/checkpoint.pt"); net, dev = pol._net, pol._device
    ds = LeRobotDataset("dream_robot/pick_place_cube", root=root)
    ep = np.asarray(ds.hf_dataset["episode_index"]); act = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    idx = [int(i) for i in np.nonzero(np.isin(ep, val) & (act[:, 7] > 0.5))[0]]
    S, L, P = [], [], []
    for b in range(0, len(idx), 64):
        items = [ds[i] for i in idx[b:b + 64]]
        imgs = {k: torch.stack([it[f"observation.images.{k}"] for it in items]).to(dev) for k in net.cameras}
        st = torch.stack([it["observation.state"] for it in items]).to(dev)
        with torch.no_grad(): P.append(net.predict(imgs, st)[:, :7].cpu().numpy())
        S.append(st[:, :7].cpu().numpy()); L.append(np.stack([it["action"][:7].numpy() for it in items]))
    S, L, P = map(np.concatenate, (S, L, P))
    dl, dp = (L - S) * 1000, (P - S) * 1000                     # mrad
    ml = np.linalg.norm(dl, axis=1)
    print(f"\n{name}: {len(S)} held-out open-gripper frames")
    print(f"  {'correction size':>16} {'frames':>6} {'|pred corr|/|label corr|':>25} {'cosine':>7} {'model err':>10} {'copy-pose err':>14}")
    for lo, hi in bins:
        m = (ml >= lo) & (ml < hi)
        if m.sum() < 5: continue
        ratio = np.linalg.norm(dp[m], axis=1).mean() / ml[m].mean()
        cos = np.mean(np.sum(dp[m] * dl[m], 1) / (np.linalg.norm(dp[m], axis=1) * ml[m] + 1e-9))
        err = np.linalg.norm(dp[m] - dl[m], axis=1).mean(); copy = ml[m].mean()
        print(f"  {f'{lo}-{hi} mrad':>16} {m.sum():>6} {ratio:>25.2f} {cos:>7.2f} {err:>9.1f}m {copy:>13.1f}m")
