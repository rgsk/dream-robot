"""What a trained WAM-C imagines: chunk 1's 2 frames (t+8, t+16) dreamed from the now frame + joints,
decoded by the VAE, next to the real frames. Val demos only.

    .venv/bin/python experiments/lam/scripts/wam_c_dream.py experiments/lam/wamc_l3 scratch/wamc_dream_s0.png
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from lam import NAME, REPO, load, split  # noqa: E402
from wam_c import H, K, M, WAMC, mask, tokens, vae_dec, vae_enc  # noqa: E402

dev = 'cuda'


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f'dream_robot/{NAME}', root=REPO / 'data/robosuite' / NAME)
    action = np.asarray(ds.hf_dataset['action'], dtype=np.float32)
    state, episode, top = load(NAME)
    eps = split()
    lab10 = np.isin(episode, eps['labelled'])
    s_mean, s_std = torch.tensor(state[lab10].mean(0), device=dev), torch.tensor(state[lab10].std(0), device=dev)

    vl = torch.load(REPO / 'cache/lam/vae_lat.pt', weights_only=False)
    vae = torch.nn.ModuleDict(dict(enc=vae_enc(), dec=vae_dec())).to(dev).eval()
    vae.load_state_dict(vl['weights'])
    ck = torch.load(run / 'model.pt', weights_only=False)
    no_z = ck['args'].get('no_z', False)
    net = WAMC(8).to(dev).eval()
    net.load_state_dict(ck['state_dict'])
    from wam_c import is_z
    blocked = ~mask.to(dev) | (is_z.to(dev)[None, :] if no_z else False)

    # 4 val demos, 4 moments each
    picks = []
    for e in eps['val'][:4]:
        rows = np.flatnonzero(episode == e)
        picks += [rows[0] + k for k in (30, 70, 110, 150)]
    picks = np.array(picks)
    torch.manual_seed(0)
    with torch.no_grad():
        now = torch.tensor(top[picks], device=dev).float() / 255 * 2 - 1
        B = len(picks)
        seen = torch.zeros(B, 1 + M * K, 64, 16, device=dev)
        seen[:, 0] = tokens(vae['enc'](now).chunk(2, dim=1)[0])
        st = ((torch.tensor(state[picks], device=dev) - s_mean) / s_std)[:, None].expand(B, M, -1)
        x_f = torch.randn(B, M, K * 64, 16, device=dev)
        x_z = torch.randn(B, M, K, 8, device=dev)
        x_a = torch.randn(B, M, H, 8, device=dev)
        tau = torch.zeros(B, M, device=dev)
        for i in range(10):
            tau[:, 0] = i / 10
            v_f, v_z, v_a = net(seen, st, x_f, x_z, x_a, tau, blocked)
            x_f[:, 0] += v_f[:, 0] / 10
            x_z[:, 0] += v_z[:, 0] / 10
            x_a[:, 0] += v_a[:, 0] / 10
        from einops import rearrange
        lat = rearrange(x_f[:, 0].view(B, K, 64, 16), 'b k (h w) (p q c) -> (b k) c (h p) (w q)', h=8, p=2, q=2)
        dreams = ((vae['dec'](lat).clamp(-1, 1) + 1) / 2).view(B, K, 3, 128, 128).cpu().numpy()

    fig, axs = plt.subplots(B, 5, figsize=(9, 1.9 * B))
    for b, r in enumerate(picks):
        ims = [top[r], top[min(r + 8, r)] if False else top[r + 8], top[r + 16], dreams[b, 0], dreams[b, 1]]
        names = ['now', 'real +8', 'real +16', 'dream +8', 'dream +16']
        for ax, im, name in zip(axs[b], ims, names):
            im = im.transpose(1, 2, 0)
            ax.imshow(im if im.dtype != np.uint8 else im)
            ax.set_title(f'{name}' + (f'  (demo {episode[r]}, step {r - np.flatnonzero(episode == episode[r])[0]})' if name == 'now' else ''), fontsize=7)
            ax.axis('off')
    fig.suptitle(f'{run.name}: dreamed chunk-1 frames on val demos ({"z hidden" if no_z else "with z"})', fontsize=9)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    real = np.stack([top[picks + 8], top[picks + 16]], 1).astype(np.float32) / 255
    print(f'{run.name}: dream vs real pixel MSE {((dreams - real) ** 2).mean():.4f}   '
          f'copy now {((top[picks][:, None].astype(np.float32) / 255 - real) ** 2).mean():.4f}  -> {out}')


if __name__ == '__main__':
    main()
