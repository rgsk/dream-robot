# step 10 local check: (1) the present's keys and values are the same with and without future tokens (the mask
# works, inference unchanged); (2) 2000 steps on t5: flow and future losses both fall
import numpy as np
import torch

from dream_robot.policies.verne.data import get_batch, to_gpu
from dream_robot.policies.verne.model import Verne
from dream_robot.policies.verne.run import load_task
from dream_robot.policies.verne.train import train

state, action, episode, latent, instruction, train_ids, val_ids, _ = load_task('t5')
d = to_gpu(latent, state, instruction, action, episode, train_ids)
b = get_batch(d, train_ids, 8, np.random.default_rng(0))
net = Verne().cuda()
with torch.no_grad():
    kv0, _ = net.see(b.frame, b.state, b.instruction)
    kv1, pred = net.see(b.frame, b.state, b.instruction, torch.randn_like(b.future), torch.rand(8, device='cuda'))
print('future', tuple(b.future.shape))
print('max kv diff with future tokens', max((a - c).abs().max().item() for (a, _), (c, _) in zip(kv0, kv1)),
      'pred', tuple(pred.shape))
train(d, train_ids, val_ids, 2000, seed=0)
