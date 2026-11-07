from frozendict import frozendict

CFG = frozendict(epochs=20, lr=1e-3, name='dino')


def train(cfg=CFG):  # no annotation: cfg's type comes from the default
    return cfg['epochs'] * 2, cfg['name'].upper()  # hover: int, str; cfg[ completes the keys


print(train({'epochs': 20, 'lr': 0.001, 'name': 'dino'}))  # a merge keeps the shape
# cfg['epoch'] inside train would be a typo error (in a .py; notebooks don't show diagnostics)
print(dict(epochs=20, lr=1e-3, name='dino'))
