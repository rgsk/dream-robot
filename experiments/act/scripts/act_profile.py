import time, numpy as np, torch
from dream_robot.policies.act.model import ACTConfig, ACTPolicyNet, kl_divergence, masked_l1
from dream_robot.policies.bc.model import NormalizationStats
stats = NormalizationStats(np.zeros(8, np.float32), np.ones(8, np.float32), np.zeros(8, np.float32), np.ones(8, np.float32), np.full(3, .5, np.float32), np.full(3, .25, np.float32))
base = dict(state_dim=8, action_dim=8, arm_joints=7, grippers=1, cameras=("top", "wrist"), image_hw=(128, 128), chunk=50)
variants = {
    "current (16x16 grid, 4 enc / 4 vae)": {},
    "8x8 grid": dict(conv_channels=(32, 64, 128, 128)),
    "2 enc / 2 vae layers": dict(encoder_layers=2, vae_layers=2),
    "8x8 grid + 2/2 layers": dict(conv_channels=(32, 64, 128, 128), encoder_layers=2, vae_layers=2),
}
dev = torch.device("cuda"); B = 64
imgs = {c: torch.rand(B, 3, 128, 128, device=dev) for c in ("top", "wrist")}
st = torch.randn(B, 8, device=dev); act = torch.randn(B, 50, 8, device=dev); pad = torch.zeros(B, 50, dtype=torch.bool, device=dev)
for name, kw in variants.items():
    net = ACTPolicyNet(ACTConfig(**base, **kw), stats).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-4)
    for i in range(13):
        if i == 3: torch.cuda.synchronize(); t = time.time()
        pred, mu, lv = net(imgs, st, act, pad); loss = masked_l1(pred, act, pad) + 10 * kl_divergence(mu, lv)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); ms = (time.time() - t) / 10 * 1000
    print(f"{name:38} {net.parameter_count/1e6:5.2f}M params  {ms:6.1f} ms/step  -> {ms*66/1000:5.1f} s/epoch GPU (25 demos), {ms*333/1000:6.1f} s (100 demos)", flush=True)
    del net, opt; torch.cuda.empty_cache()
