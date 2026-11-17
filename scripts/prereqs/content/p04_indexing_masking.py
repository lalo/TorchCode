"""Prerequisite 4: Indexing & masking.

Content for templates/00_p04_indexing_masking.ipynb and its solution notebook.
Edit here, then run `make prereqs`; never edit the generated notebooks by hand.
"""

NOTEBOOK = {
    "prereq_id": "P4",
    "file": "00_p04_indexing_masking",
    "task_id": "p04_indexing_masking",
    "title": "Indexing & Masking",
    # Shape symbols used in this notebook, shown in the "📐 Notation" box
    "notation": [
        ('B', 'batch size'),
        ('C', 'number of classes (`pick_per_row`)'),
        ('V', 'vocabulary size (`keep_topk`, `unsort`)'),
        ('S_q', 'number of queries'),
        ('S_k', 'number of keys'),
        ('n', 'sequence length for `attention_mask`'),
        ('w', 'window size'),
        ('G', 'number of groups (`group_mean`)'),
        ('S', 'sequence length (tokens)'),
        ('D', 'feature dimension'),
        ('N', 'number of `True` entries in a boolean mask (§3)'),
    ],
    "intro": r'''
# 🔵 Prerequisite 4: Indexing & Masking

Picking the logit of the correct class, looking up embeddings, hiding future tokens from attention, keeping only the top-k logits when sampling, routing tokens to experts: all of these **select** or **overwrite** parts of a tensor by index or by mask.

PyTorch has several ways to do this (slicing, integer "fancy" indexing, boolean masks, `masked_fill`, `topk`, `gather`, `scatter`). Picking the right one keeps the code loop-free and differentiable.

### What you'll learn
1. Slicing: `:` vs `...` vs `None`, steps, and negative indices
2. Integer (fancy) indexing: `weight[indices]` and `x[rows, cols]`
3. Boolean masks: reading `x[mask]` and writing `x[mask] = ...`
4. `masked_fill` with `-inf`, and building masks with `triu` / comparisons
5. `topk` and `sort`: values *and* indices
6. `gather` and `scatter`: index-based reads and writes along a dim
7. Grouping by id: `unique`, `bincount`, `index_add`
8. More selection tools: `one_hot`, `nonzero`, `searchsorted`, `take_along_dim`, `index_select`, `scatter_add`
9. Many ways to do it: equivalent indexing patterns and their trade-offs

### Exercises
| # | Function | Shows up in |
|---|----------|-------------|
| 1 | `pick_per_row(logits, targets)` | #16 Cross-Entropy |
| 2 | `attention_mask(n, kind, w)` | #09 Causal, #11 Sliding Window, #13 GPT-2 Block, #14 KV Cache |
| 3 | `apply_mask(scores, mask)` | every masked attention problem |
| 4 | `keep_topk(logits, k)` | #32 Top-k Sampling, #28 MoE |
| 5 | `unsort(sorted_values, sort_idx)` | #32 Top-p Sampling |
| 6 | `group_mean(values, group_ids)` | #38 GRPO |

### Rules
- **No Python loops** over tensor elements, or over groups (Exercise 6).
- Do **not** use `F.cross_entropy` / `F.nll_loss` (Exercise 1) or `torch.argsort` on the indices to undo the sort (Exercise 5). Use `scatter`.
''',
    "lesson": [
        ("md", r'''
## 1. Slicing: `:`, `...`, `None`, and steps

Inside `x[...]`, each entry addresses dimensions from the left:
- **`:`** keeps **exactly one** dimension whole. `0` (an integer) picks one position and **removes** that dimension.
- **`...`** (Ellipsis) stands for **as many `:` as needed** (zero or more), so the entries after it address the **last** dimensions. At most one `...` per index.
- **`None`** inserts a new size-1 dimension at that spot (P2).

Think **coordinates**, not steps: every entry addresses one dimension of the **original** tensor, chosen by its **position between the commas**. `0` always means "the first one". Where you put it decides *which* dimension loses its axis:

```python
#    B  S  D
x[   0, :, :]   # first sequence -> (S, D)
x[   :, 0, :]   # first token    -> (B, D)   B kept, S picked, D kept (the trailing : is implied)
x[   :, :, 0]   # first feature  -> (B, S)
```

With `x` of shape `(B, S, D)` (batch, tokens, features) these look alike but are not:

| Index | Shape | Meaning |
|-------|-------|---------|
| `x[:, 0]` | `(B, D)` | the **first token** of every sequence |
| `x[..., 0]` | `(B, S)` | the **first feature** of every token |
| `x[0]` or `x[0, ...]` (same thing) | `(S, D)` | the first sequence: still `S` tokens of `D` numbers each |
| `x[..., None]` | `(B, S, D, 1)` | a new size-1 dim at the end |
| `x[:, None]` | `(B, 1, S, D)` | a new size-1 dim after dim 0 |

Because `...` absorbs any number of leading dims, `x[..., 0::2]` (every other feature, as RoPE uses) works unchanged for `(S, D)` and `(B, H, S, D)`.

**Steps and negatives:** `start:stop:step` selects every `step`-th element; negative **indices** count from the end (`x[:, -1]` is the last token). But PyTorch does **not** allow negative **steps**: NumPy's `x[::-1]` raises an error; use `x.flip(dim)` (which copies). Plain slices return **views**, so they share memory with `x`.
'''),
        ("code", r'''
x = torch.arange(2 * 3 * 4).view(2, 3, 4)     # (B, S, D) = (2, 3, 4): every size different
print(x[:, 0].shape)                  # torch.Size([2, 4]): token 0 of each sequence
print(x[..., 0].shape)                # torch.Size([2, 3]): feature 0 of every token
print(x[0].shape, x[0, ...].shape)    # both torch.Size([3, 4])
print(x[..., None].shape, x[:, None].shape)   # (2, 3, 4, 1) and (2, 1, 3, 4)
print(x[..., 0::2].shape)             # torch.Size([2, 3, 2]): even features
print(x[:, -1])                       # the last token of each sequence

try:
    x[..., ::-1]
except ValueError as e:
    print("ValueError:", e)
print(x[0, 0], x[0, 0].flip(0))       # [0, 1, 2, 3] -> [3, 2, 1, 0]
'''),
        ("md", r'''
## 2. Integer (fancy) indexing

Indexing with an integer **tensor** selects rows: `weight[indices]` is an embedding lookup, and works for any shape of `indices`.
Indexing with **two** index tensors pairs them up element-wise: `x[rows, cols]` returns `x[rows[0], cols[0]], x[rows[1], cols[1]], …`. This is **not** the same as `x[rows][:, cols]`.

- **One** index tensor: each number in it is **replaced by a whole row**. So the result has the index's shape, plus the row's shape at the end.
- **Two** index tensors: each `(row, col)` **pair** picks a **single number**, like coordinates.

Every index must exist in its dimension (`0 … size-1`, or negative from the end). One bad index anywhere raises an `IndexError` for the whole operation.
'''),
        ("code", r'''
weight = torch.arange(10.).view(5, 2)          # 5 embeddings of size 2
idx = torch.tensor([[0, 4], [2, 2]])
print(weight[idx].shape)                       # (2, 2, 2): idx shape + embedding dim
print(weight[idx])                             # [[row 0, row 4], [row 2, row 2]]

x = torch.tensor([[10., 11., 12.],
                  [20., 21., 22.]])
print(x[torch.tensor([0, 1]), torch.tensor([2, 0])])   # tensor([12., 20.]): x[0, 2] and x[1, 0]
try:
    x[torch.tensor([0, 1]), torch.tensor([2, 3])]       # column 3 doesn't exist
except IndexError as e:
    print("IndexError:", e)
'''),
        ("md", r'''
## 3. Boolean masks

- **Reading** `x[mask]` returns a **1-D** tensor of the selected elements. The shape is lost.
- **Writing** `x[mask] = value` (or `x[mask] += ...`) updates only the selected positions.
- `torch.where(mask, a, b)` picks element-wise and **keeps the shape**. It returns a new tensor; there is no `where_`. To replace in place, use `x.masked_fill_(mask, v)` or `x[mask] = v`.

Two separate steps happen in `x[x < 0]`. First, `x < 0` compares **every** number, so the mask has the **same shape as `x`**. No indexing has happened yet. Then `x[mask]` **filters and collects** the numbers marked `True`. The result is 1-D because rows can have different numbers of matches, and those can't form a rectangle without inventing padding.

A boolean mask is **not** a single index slot like `0` or `:`. **It covers as many dimensions as it has**, starting where you put it. For `x` of shape `(B, S, D)` and `N` = the number of `True`s:

| Mask shape | Each `True` keeps | `x[mask]` shape |
|------------|-------------------|-----------------|
| `(B,)` | a whole sequence | `(N, S, D)` |
| `(B, S)` | one token's feature vector | `(N, D)` |
| `(B, S, D)`, e.g. `x < 0` | one number | `(N,)` |

The rule: **`(N,)` + whatever dimensions the mask doesn't cover.** A mask must line up with the dimensions it covers; it does **not** broadcast. A `(S, S)` mask on a `(B, S, S)` tensor needs `y[..., mask]`, and plain `y[mask]` raises an error. `masked_fill` and `torch.where` **do** broadcast (§4).
'''),
        ("code", r'''
x = torch.tensor([[1., -2.], [-3., 4.]])
mask = x < 0
print(x[mask])                        # tensor([-2., -3.]): flattened
print(torch.where(mask, 0.0, x))      # same shape, negatives replaced
y = x.clone(); y[mask] = 0.0; print(y)

x = torch.tensor([[[10, 11], [12, 13], [14, 15]],       # (B=2, S=3, D=2)
                  [[20, 21], [22, 23], [24, 25]]])
print(x[torch.tensor([True, False])].shape)            # mask (B,): torch.Size([1, 3, 2]), all of sequence 0
mask_BS = torch.tensor([[True, False, True],
                        [False, True, False]])
print(x[mask_BS])                                      # mask (B, S): 3 whole tokens -> (3, 2)
print(x[x > 20])                                       # mask (B, S, D): single numbers -> 1-D

scores = torch.zeros(2, 3, 3)                          # (B, S, S)
future = torch.ones(3, 3, dtype=torch.bool).triu(1)    # (S, S)
try:
    scores[future] = -1.0                              # boolean indexing does not broadcast
except IndexError as e:
    print("IndexError:", str(e)[:70])
scores[..., future] = -1.0                             # ... skips B, the mask covers the last two dims
print(scores[0])
'''),
        ("md", r'''
## 4. `masked_fill` and building masks

`scores.masked_fill(mask, float("-inf"))` returns a copy with `-inf` wherever `mask` is `True`. The mask **broadcasts**, so one `(S, S)` mask can cover a whole `(B, H, S, S)` score tensor. After softmax, `-inf` becomes probability `0`.
Masks are built from comparisons, or with `torch.triu` / `torch.tril`. `diagonal=k` shifts which diagonal is kept.

> ⚠️ If an entire row is `-inf`, softmax returns `NaN` for that row. Make sure every query can see at least one key (P9 covers this).

> ⚠️ **Mask conventions differ between APIs.** With `masked_fill`, `True` means **blocked**. PyTorch's `F.scaled_dot_product_attention` takes a boolean `attn_mask` where `True` means **allowed** (the position takes part in attention). Handing one API's mask to the other attends to exactly the wrong positions, without any error. Convert with `~mask`.
'''),
        ("code", r'''
S = 4
future = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)   # True above the diagonal
print(future)
i = torch.arange(S)
print(torch.equal(future, i[None, :] > i[:, None]))   # same mask, from a comparison (P2)

scores = torch.zeros(2, S, S)
print(torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)[0])

import torch.nn.functional as F
q = k = v = torch.randn(1, S, 8)
ours = torch.softmax((q @ k.transpose(-2, -1) / 8 ** 0.5).masked_fill(future, float("-inf")), dim=-1) @ v
sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=~future)   # SDPA: True = ALLOWED, so invert
print(torch.allclose(ours, sdpa, atol=1e-6))                         # True
'''),
        ("md", r'''
## 5. `topk` and `sort`

Both return `(values, indices)` along a dim (default: last). `topk(k)` gives the `k` largest, in descending order. `sort` orders everything.
The **indices** tell you where each value came from, and you need them to put values back (§6).
'''),
        ("code", r'''
logits = torch.tensor([0.1, 2.0, -1.0, 3.0, 0.5])
vals, idx = logits.topk(2)
print(vals, idx)                    # tensor([3., 2.]) tensor([3, 1])
s_vals, s_idx = logits.sort(descending=True)
print(s_vals, s_idx)
'''),
        ("md", r'''
## 6. `gather` and `scatter`

- `x.gather(dim, idx)` **reads**: `out[i][j] = x[i][idx[i][j]]` (for `dim=1`). `out` has the shape of `idx`.
- `x.scatter(dim, idx, src)` **writes**: `out[i][idx[i][j]] = src[i][j]`. It returns a copy of `x` with those positions overwritten (`scatter_` is the in-place version).

They are inverses: `gather` pulls values *from* positions, `scatter` pushes values *to* positions.

The first argument is **`dim`**: which dimension the **values** in `idx` point into. With `dim=1` on a 2-D tensor, the **row** comes from *where* a number sits in `idx`, and the **column** comes from the number *itself*.

> ⚠️ Use the same `dim` the indices came from. Indices from `x.sort(dim=-1)` or `x.topk(k, dim=-1)` are **column** positions, so `scatter(-1, ...)`. With `scatter(0, ...)` they would be read as row numbers, and you get errors like *index 7 is out of bounds for dimension 0 with size 3*.
'''),
        ("code", r'''
x = torch.tensor([[10., 11., 12.],
                  [20., 21., 22.]])
idx = torch.tensor([[2, 0],
                    [1, 1]])
print(x.gather(1, idx))                          # [[12., 10.], [21., 21.]]

base = torch.zeros(2, 3)
print(base.scatter(1, torch.tensor([[2], [0]]), torch.tensor([[5.], [7.]])))
# [[0., 0., 5.], [7., 0., 0.]]
'''),
        ("md", r'''
## 7. Grouping by id

GRPO (#38) normalizes each reward against the other rewards **from the same prompt**. Group ids can be arbitrary integers, like `[7, 3, 7, 10, 3]`.

- `torch.unique(ids, return_inverse=True)` returns `(unique_ids, inverse)`, where `inverse[i]` is the position of `ids[i]` in `unique_ids`. That relabels the groups as `0 … G-1`.
- `torch.bincount(inverse)` counts the members of each group.
- `torch.zeros(G).index_add(0, inverse, values)` adds every value into its group's slot (`bincount(inverse, weights=values)` also works).
- `group_stat[inverse]` broadcasts a per-group result back to every element.

No loop over groups needed, no matter how many there are.

The cell below is **one pipeline**, not a list of alternatives: relabel → count → sum → divide → expand back. Two details:

- `bincount` counts **any** non-negative integers, and the output has one slot for every value from `0` up to the largest: `bincount([3, 3, 5])` is `[0, 0, 0, 2, 0, 1]`. That's why you count `inv` (compact `0 … G-1`), not the raw ids. With ids like `1000` you'd get mostly empty slots. (`ids.unique(return_inverse=True, return_counts=True)` gives the counts in the same call.)
- The last step, `(sums / counts)[inv]`, is the integer indexing from §2. It doesn't reorder; it **expands** `G` group means into one mean per element.
'''),
        ("code", r'''
ids = torch.tensor([7, 3, 7, 10, 3])
vals = torch.tensor([1., 2., 3., 4., 5.])
uniq, inv = ids.unique(return_inverse=True)
print(uniq, inv)                                  # tensor([ 3,  7, 10]) tensor([1, 0, 1, 2, 0])
counts = torch.bincount(inv)
sums = torch.zeros(len(uniq)).index_add(0, inv, vals)
print(counts, sums)                               # [2, 2, 1]  [7., 4., 4.]
print((sums / counts)[inv])                       # each element's group mean: [2., 3.5, 2., 4., 3.5]
print(torch.bincount(torch.tensor([3, 3, 5])))    # tensor([0, 0, 0, 2, 0, 1]): slot = value, entry = how often
'''),
        ("md", r'''
## 8. More selection tools

- **`F.one_hot(idx, num_classes)`** turns class ids into one-hot rows (as `int64`; convert with `.float()`).
- **`torch.nonzero(mask)`** / **`torch.where(mask)`** (one argument) return the *positions* of `True` entries. The output size depends on the data, so on a GPU the CPU must wait for the result (a **sync**), and tools like `torch.compile` can't plan around it. Prefer a mask or `masked_fill` when you can.
- **`torch.searchsorted(sorted_seq, values)`** finds where each value would be inserted. With a cumulative distribution it samples by **inverse CDF**: draw `u ~ U(0, 1)` and find the first bucket whose cumulative probability exceeds `u`.
- **`torch.take_along_dim(x, idx, dim)`** is `gather` that also broadcasts `idx`; **`x.index_select(dim, idx)`** picks whole slices along one dim.
- **`scatter_add` / `scatter_reduce`** write **many** values into the **same** slot, combining them (sum, max, mean…). Plain `scatter` would keep only one of them.
'''),
        ("code", r'''
import torch.nn.functional as F
print(F.one_hot(torch.tensor([0, 2, 1]), num_classes=3))
print(torch.nonzero(torch.tensor([[0, 1], [1, 1]])))         # (row, col) of each nonzero

probs = torch.tensor([0.1, 0.6, 0.3])
cdf = probs.cumsum(0)                                         # [0.1, 0.7, 1.0]
u = torch.tensor([0.05, 0.5, 0.95])
print(torch.searchsorted(cdf, u))                             # [0, 1, 2]: inverse-CDF sampling

x = torch.tensor([[10., 11., 12.], [20., 21., 22.]])
print(torch.take_along_dim(x, torch.tensor([[2], [0]]), dim=1))   # [[12.], [20.]]
print(x.index_select(1, torch.tensor([0, 2])))                     # columns 0 and 2

slots = torch.zeros(3).scatter_add(0, torch.tensor([0, 0, 2]), torch.tensor([1., 2., 5.]))
print(slots)                                                  # [3., 0., 5.]: both writes to slot 0 are summed
'''),
        ("md", r'''
## 9. Many ways to do it

| Goal | Equivalent forms | Watch out for |
|------|------------------|---------------|
| One value per row | `x[torch.arange(B), idx]`, `x.gather(1, idx[:, None]).squeeze(1)`, `torch.take_along_dim(x, idx[:, None], 1).squeeze(1)`, `(F.one_hot(idx, C) * x).sum(1)` | the one-hot form builds a full `(B, C)` tensor: wasteful for large vocabularies |
| Sums per group | `zeros(G).index_add(0, inv, v)`, `zeros(G).scatter_add(0, inv, v)`, `torch.bincount(inv, weights=v)` | `bincount` has **no gradient**; on GPU, `index_add`/`scatter_add` use atomic adds, so float results can differ slightly run to run |
| Replace where a condition holds | `x.masked_fill(mask, v)`, `torch.where(mask, v, x)`, `y = x.clone(); y[mask] = v` | the last one is in place on a copy: fine, but easy to forget the `clone()`. It also doesn't broadcast: a `(S, S)` mask on `(B, S, S)` needs `y[..., mask] = v` |
| Sample from probabilities | `torch.multinomial(p, n)`, `torch.searchsorted(p.cumsum(0), torch.rand(n))` | `multinomial` handles rounding at the top end of the CDF; with `searchsorted`, clamp to `len(p) - 1` |
'''),
        ("code", r'''
torch.manual_seed(0)
B, C = 4, 6
x = torch.randn(B, C)
idx = torch.tensor([0, 5, 2, 2])
a = x[torch.arange(B), idx]
b = x.gather(1, idx[:, None]).squeeze(1)
c = (F.one_hot(idx, C) * x).sum(1)
print(torch.equal(a, b), torch.allclose(a, c))

inv = torch.tensor([0, 1, 0, 2])
v = torch.tensor([1., 2., 3., 4.])
print(torch.zeros(3).index_add(0, inv, v), torch.zeros(3).scatter_add(0, inv, v), torch.bincount(inv, weights=v))
'''),
    ],
    "exercises_intro": r'''
---
# ✏️ Exercises

Implement the six functions below. Each one has a debug cell you can use before you submit.
''',
    "exercises": [
        {
            "spec": r'''
### Exercise 1: `pick_per_row(logits, targets)`

For each row, pick the value at that row's target column.

- `logits`: `(B, C)`
- `targets`: `(B,)` integer class ids
- returns: `(B,)` with `out[b] = logits[b, targets[b]]`

*Cross-entropy (#16) is `-pick_per_row(log_probs, targets).mean()`.*
''',
            "stub": r'''
def pick_per_row(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def pick_per_row(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, targets]
''',
            "debug": r'''
logits = torch.tensor([[0.1, 0.7, 0.2],
                       [0.5, 0.3, 0.2]])
print(pick_per_row(logits, torch.tensor([1, 0])))   # expect tensor([0.7000, 0.5000])
''',
        },
        {
            "spec": r'''
### Exercise 2: `attention_mask(n, kind, w=0)`

Build an `(n, n)` **boolean** mask for attention, where `True` means **blocked** (query `i` may *not* attend to key `j`).

- `kind="causal"`: block the future, `j > i`
- `kind="sliding"`: two-sided window (Longformer-style, #11), block `abs(i - j) > w`
- `kind="causal_sliding"`: causal window (Mistral-style), block `j > i` **or** `i - j > w`
- any other `kind`: raise `ValueError`

*Hint: P2's `relative_position(n)` (`R[i, j] = i - j`) makes each kind a single comparison. Rows are queries `i` (`arange(n)[:, None]`, shape `(n, 1)`), columns are keys `j` (`arange(n)[None, :]`, shape `(1, n)`), and comparing the two broadcasts to `(n, n)`.*
''',
            "stub": r'''
def attention_mask(n: int, kind: str, w: int = 0) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def attention_mask(n: int, kind: str, w: int = 0) -> torch.Tensor:
    idx = torch.arange(n)
    R = idx[:, None] - idx[None, :]          # R[i, j] = i - j
    if kind == "causal":
        return R < 0
    if kind == "sliding":
        return R.abs() > w
    if kind == "causal_sliding":
        return (R < 0) | (R > w)
    raise ValueError(f"unknown mask kind: {kind!r}")
''',
            "debug": r'''
for kind in ("causal", "sliding", "causal_sliding"):
    print(kind)
    print(attention_mask(5, kind, w=1).int())
# causal_sliding with w=1 should leave exactly 2 visible keys per row (itself and the previous one),
# except row 0, which only sees itself.
''',
        },
        {
            "spec": r'''
### Exercise 3: `apply_mask(scores, mask)`

Set blocked attention scores to `-inf` so softmax gives them zero probability.

- `scores`: `(..., S_q, S_k)` float, any number of leading dims (batch, heads)
- `mask`: `(S_q, S_k)` bool, `True` = blocked
- returns: same shape and dtype as `scores`
- do **not** modify `scores` in place
''',
            "stub": r'''
def apply_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def apply_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(mask, float("-inf"))
''',
            "debug": r'''
scores = torch.zeros(2, 3, 3)
masked = apply_mask(scores, attention_mask(3, "causal"))
print(masked[0])
print(torch.softmax(masked, dim=-1)[0])   # row i spreads evenly over keys 0..i
''',
        },
        {
            "spec": r'''
### Exercise 4: `keep_topk(logits, k)`

Keep the `k` largest logits in each row (last dim) and set every other entry to `-inf`.

- `logits`: `(..., V)` float
- `k`: int, `1 <= k`; if `k >= V`, keep everything
- returns: same shape; kept entries **unchanged**, others `-inf`
- do **not** modify `logits` in place

*Hint: `topk` gives you the values **and** where they came from (§5). Instead of removing the losers, think about starting from all `-inf` and writing the winners back to their positions (§6).*

*#32 samples only from the top-k tokens; #28 MoE routes each token to its top-k experts.*
''',
            "stub": r'''
def keep_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def keep_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.shape[-1])
    vals, idx = logits.topk(k, dim=-1)
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, vals)
''',
            "debug": r'''
logits = torch.tensor([[0.1, 2.0, -1.0, 3.0, 0.5],
                       [9.0, 8.0, 7.0, 6.0, 5.0]])
print(keep_topk(logits, 2))
# expect [[-inf, 2., -inf, 3., -inf],
#         [ 9.,  8., -inf, -inf, -inf]]
''',
        },
        {
            "spec": r'''
### Exercise 5: `unsort(sorted_values, sort_idx)`

Undo a sort: given `sorted_values, sort_idx = x.sort(dim=-1)` (possibly after editing `sorted_values`), put each value back at its original position.

- `sorted_values`: `(..., V)`
- `sort_idx`: `(..., V)` long, from `x.sort(dim=-1)`; `sorted_values[..., j]` came from position `sort_idx[..., j]`
- returns: `(..., V)` with `out[..., sort_idx[..., j]] = sorted_values[..., j]`

*Top-p sampling (#32) sorts the logits, masks the tail in sorted order, then has to put them back.*
''',
            "stub": r'''
def unsort(sorted_values: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def unsort(sorted_values: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(sorted_values).scatter(-1, sort_idx, sorted_values)
''',
            "debug": r'''
x = torch.tensor([0.3, -1.0, 2.0, 0.0])
s, i = x.sort(descending=True)
print(s, i)
print(unsort(s, i))                      # expect the original x
s[2:] = float("-inf")                    # drop the two smallest, in sorted order
print(unsort(s, i))                      # expect tensor([0.3, -inf, 2.0, -inf])
''',
        },
        {
            "spec": r'''
### Exercise 6: `group_mean(values, group_ids)`

Replace every value with the mean of its group.

- `values`: `(B,)` float
- `group_ids`: `(B,)` integer ids; arbitrary values, not necessarily `0 … G-1` or sorted
- returns: `(B,)` with `out[i] = mean of values[j] over all j with group_ids[j] == group_ids[i]`
- same dtype as `values`, gradients must flow back to `values`, and **no Python loop over groups**

*#38 GRPO computes this (and a per-group std) to turn rewards into advantages.*
''',
            "stub": r'''
def group_mean(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    pass  # Replace this
''',
            "solution": r'''
def group_mean(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    sums = torch.zeros(len(uniq), dtype=values.dtype, device=values.device).index_add(0, inv, values)
    counts = torch.bincount(inv, minlength=len(uniq)).to(values.dtype)
    return (sums / counts)[inv]
''',
            "debug": r'''
print(group_mean(torch.tensor([1., 2., 3., 4., 5.]), torch.tensor([7, 3, 7, 10, 3])))
# expect tensor([2.0000, 3.5000, 2.0000, 4.0000, 3.5000])
''',
        },
    ],
    # Problem number -> what it reuses from this notebook ("What this unlocks" section)
    "unlocks": [
        (9, "`attention_mask(n, 'causal')` plus `apply_mask`"),
        (11, "`attention_mask(n, 'sliding', w)`; `'causal_sliding'` is the Mistral variant"),
        (13, "The causal mask and `masked_fill`, inside a full transformer block"),
        (14, "A causal mask shifted by the cache length: `torch.triu(..., diagonal=S_past + 1)` (lesson §4)"),
        (16, "`pick_per_row`: the log-probability of the correct class"),
        (18, "`weight[indices]`: the embedding lookup (lesson §2)"),
        (24, "Strided slicing `x[..., 0::2]` and `x[..., 1::2]` for RoPE pairs (lesson §1)"),
        (28, "`topk` routing, then boolean-mask reads and writes like `output[mask] += ...` (lesson §3)"),
        (32, "`keep_topk` for top-k, then `sort`, mask, and `unsort` for top-p"),
        (38, "`group_mean`: per-group statistics with `unique` / `bincount` / `index_add`, no loop over groups"),
    ],
}
