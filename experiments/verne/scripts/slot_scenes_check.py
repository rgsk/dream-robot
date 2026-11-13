# step 09b local check: render the demo-less T5 scenes (train 1000, val 200; cached under data/robosuite/t5_scenes)
# and train 3000 steps on t5-scenes: flow loss as before, slot head on demo rows + scenes, val scene accuracy
from dream_robot.policies.verne import vae
from dream_robot.policies.verne.data import load_scenes, to_gpu
from dream_robot.policies.verne.run import load_task
from dream_robot.policies.verne.train import train

scenes = load_scenes('train', vae.WEIGHTS), load_scenes('val', vae.WEIGHTS)
print('scenes', [len(s.slot) for s in scenes], 'slot-1 share', [round(s.slot.float().mean().item(), 2) for s in scenes])
state, action, episode, latent, instruction, slot, train_ids, val_ids, _ = load_task('t5-scenes')
d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
train(d, train_ids, val_ids, 3000, seed=0, scenes=scenes)
