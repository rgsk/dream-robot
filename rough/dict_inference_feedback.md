# Using the fork: notes from act_dict.ipynb and act_namespace.ipynb (Claude, 2026-11-05, updated 2026-11-14)

## The rule now (2026-11-14): namespaces mostly

[act_namespace.ipynb](act_namespace.ipynb) redid act_dict (steps 0-20) with `SimpleNamespace` in place of
dict/frozendict. Same results (ChunkNet ensembled 18/20, ACTNet 11/20 = act_dict's 11/20), shorter reads:
`stats.action.std`, not `stats['action']['std']`; several lines the formatter split over three now fit on one.

```python
from types import SimpleNamespace as namespace
```

| Use | When | act_namespace |
|---|---|---|
| namespace | a bundle read by name later; the names are known when the code is written | `data`, `stats`, each batch, `train` results |
| `-> namespace` | on every function returning one; the fork keeps the attributes the body builds | `to_gpu`, `mean_std`, `get_batch`, `train` |
| Protocol | a namespace crossing into a function parameter, whose attributes the function reads | (example below) |
| tuple | a few ordered values unpacked on the spot | `load_table`, `make_chunks`, `load_images` |
| NamedTuple | a small ordered result that is also a parameter type | `Outcome` |
| dict | the keys are data (seeds, step counts, a `state_dict`) | sweep `results` |

frozendict is dropped: its lock only stops swapping a key, the Tensors inside stay mutable.

**Protocol: names a namespace's shape at a parameter.** It never builds one (`HasName('rahul')` raises
`TypeError: Protocols cannot be instantiated`); the namespace is the value, the Protocol the parameter type:

```python
class Batch(Protocol):
    top: Tensor  # [B, 3, 128, 128]
    chunks: Tensor  # [B, K, 8]
    pad: Tensor  # [B, K] bool


def batch_loss(b: Batch) -> Tensor:
    return chunk_l1(net(b.top), b.chunks, b.pad)  # b.pad: Tensor, inside the function


b = get_batch('train', i)  # -> namespace: SimpleNamespace[{top: Tensor, chunks: Tensor, pad: Tensor, ...}]
batch_loss(b)  # fits: extra attributes are fine
batch_loss(namespace(top=x, chunks=y))  # error: "pad" is not present
```

Checked with the fork's CLI. Before writing one, check whether a return type on a small helper is enough. In
act_namespace, the split chosen by a variable needed no Protocol:

```python
def get_batch(split: Literal['train', 'val'], i: Tensor) -> namespace:
    def take(x) -> Tensor:  # x: a train/val pair; -> Tensor types every field of the batch
        return (x.train if split == 'train' else x.val)[i]

    return namespace(top=take(data.top).float() / 255, pad=take(data.pad), ...)
```

**Where dicts still win:**
- A name picked by a variable: `data['top'][split]` with `split: Literal['train', 'val']` is typed by the fork
  (below); `getattr(data.top, split)` is untyped, hence `take`. A fork idea: type `getattr(ns, lit)` for a Literal
  union as the union of those attributes.
- Keys that are data, and library edges (`obs.images`, `state_dict()`, JSON, `**kwargs`): convert with `vars(ns)` /
  `namespace(**d)`.
- Merging: `a | {'k': v}` has no namespace form; `namespace(**vars(a), k=v)`. Never needed in act_namespace.

## Earlier: act_dict.ipynb (2026-11-05)

How the fork's features held up in [act_dict.ipynb](act_dict.ipynb) (ACT, steps 0-21). I write notebooks through
scripts, not in the editor, so I saw the fork through the CLI (`reveal_type` on the notebook's code cells, pulled
out into a `.py` file), not through hover or completion. "Stock" below is basedpyright 1.40.2.

### What worked

**Inferred returns carry the shape, nested** (step 4). No annotations:

```python
def to_gpu(x: NDArray):
    return frozendict(train=torch.tensor(x[~is_val]).to(dev), val=torch.tensor(x[is_val]).to(dev))

def mean_std(x: Tensor, dims: tuple[int, ...]):
    return frozendict(mean=x.mean(dims, keepdim=True), std=x.std(dims, keepdim=True))

data = frozendict(
    top=to_gpu(top),
    wrist=to_gpu(wrist),
    state=to_gpu(state),
    chunks=to_gpu(chunks),
    pad=to_gpu(pad),
)
stats = frozendict(
    state=mean_std(data['state']['train'], (0,)),
    action=mean_std(data['chunks']['train'][:, 0], (0,)),
    top=mean_std(data['top']['train'][::20].float() / 255, (0, 2, 3)),
    wrist=mean_std(data['wrist']['train'][::20].float() / 255, (0, 2, 3)),
)

data['chunks']['train']   # Tensor
stats['action']['std']    # Tensor
```

**`@cache` keeps it** (step 8). `NotebookCache.__call__` is typed `Callable[P, R] -> Callable[P, R]`:

```python
@cache(2)
def train(seed: int = 0, steps: int = 3000, shift: int = 4, lr: float = 1e-3):
    ...
    return frozendict(weights={k: v.cpu() for k, v in trained.items()}, err=err)

run = train(seed=0)
run['weights']   # fork: dict[str, Tensor]    stock: dict[str, Tensor] | Tensor
run['err']       # fork: Tensor               stock: dict[str, Tensor] | Tensor
```

**Globals read inside functions and methods** (steps 6, 7, 20). In a notebook nearly everything is a global read
from a function, and every read kept its key's type:

```python
class ChunkNet(nn.Module):
    def forward(self, top: Tensor, wrist: Tensor, state: Tensor) -> Tensor:
        f_top: Tensor = self.top((top - stats['top']['mean']) / stats['top']['std'])
        ...
        return out * stats['action']['std'] + stats['action']['mean']
```

**Literal-union keys** (step 7). The split name is itself a key, so there's no helper and no annotation:

```python
def get_batch(split: Literal['train', 'val'], i: Tensor):
    return frozendict(
        top=data['top'][split][i].float() / 255,
        state=data['state'][split][i],
        ...
    )

b = get_batch('train', i)
b['pad']   # Tensor
```

On a frozendict with mixed values, the fork gives `data['top'][split]` the type `Tensor`. Stock gives
`Tensor | Unknown` plus a **false error**: it merges all the values into one union, finds an `int` in it, and
reports `__getitem__ method not defined on type "int"`.

**Frozendicts inside tuples inside a list** (steps 17, 20):

```python
cams = [  # (backbone, image, its stats, its tag)
    (self.top, top, stats['top'], self.cam_tag[0]),
    (self.wrist, wrist, stats['wrist'], self.cam_tag[1]),
]
for backbone, image, cam_stats, tag in cams:
    grid: Tensor = backbone((image - cam_stats['mean']) / cam_stats['std'])
# cam_stats   fork: frozendict[{'mean': Tensor, 'std': Tensor}]   stock: frozendict[str, Tensor]
```

**Runtime:** the PyPI `frozendict` pickles through `NotebookCache` with no trouble.

### Where it got in the way

**The typo error prints the whole shape:**

```python
data['chunk']
# error: Could not access item in frozendict
#   "chunk" is not a defined key in "frozendict[{'top': frozendict[{'train': Tensor, 'val': Tensor}], 'wrist': frozendict[{...
```

With five nested keys it runs to several hundred characters on one line. A shorter form would help, listing just
the top-level keys:

```
"chunk" is not a key; keys: top, wrist, state, chunks, pad
```

**Brackets in loop-heavy cells** (steps 8, 21). The training loop reads five fields every step:

```python
b = get_batch('train', i)
pred: Tensor = net(random_shift(b['top'], shift), random_shift(b['wrist'], shift), b['state'])
loss = chunk_l1(pred, b['chunks'], b['pad'])
```

That's Python, not the fork. It's why small ordered bundles stayed tuples, unpacked in the loop header (step 12):

```python
preds = np.stack([chunk[step - made] for made, chunk in plans])
```

**Many bundles don't need the fork at all.** `data`, `stats` and the batch hold only Tensors, so stock basedpyright
already types every read correctly. The fork's types differ only for mixed values, such as `train`'s result. On
the all-Tensor bundles, its value is key completion, the typo error and hover, not the value types.

### What I didn't use

- `|` merging, `.get()`, `match`, narrowing by key, `fromkeys`: these bundles are built once and only read.
- Hover and completion in VS Code. I only checked them indirectly, through `reveal_type`.
- Plain dict inference: every `dict` in the notebook is keyed by data (`{seed: confirm_run(seed) ...}`). A side
  test with every `frozendict(` changed to `dict(` gave the same types at all eight reads checked, losing only the
  typo error.

### Design notes it confirmed

**No shape annotations.** The one place a shape had to cross a parameter wanted a class anyway (step 10):

```python
class Outcome(NamedTuple):
    success: bool
    mode: str
    steps: int

def score(results: list[Outcome]) -> str: ...

return Outcome(r.success, str(r.failure_mode), step + 1)   # positional reads like a sentence
ok, mode, steps = rollout(1000, n_exec=10)                  # and unpacks in the same order
```

**The working rule then** (superseded by the table at the top):

| Use | When | act_dict |
|---|---|---|
| frozendict | read by key later, or fields of the same type (mean/std) | `data`, `stats`, `get_batch`, `train` |
| tuple | a few ordered values of different types | `load_table`, `make_chunks`, `Plan`, `cams` |
| class (NamedTuple, dataclass, TypedDict) | the shape needs an annotation | `Outcome` |
| dict | the keys are data | `runs`, `results` |
