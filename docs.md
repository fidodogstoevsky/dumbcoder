# gridumbcoder

A fork of [DumbCoder](https://github.com/maxreciprocate/dumbcoder) (a simplified implementation of [DreamCoder](https://github.com/ellisk42/ec)), for matrix targets.

Given a target numpy array `X` and a set of primitive operations `D`, dumbcoder searches for a program — a composition of primitives — that produces `X`. It gets better at this over iterations by growing the primitive library and training a neural model to guide search.

---

## Core concepts

### Delta

A `Delta` is a single node in a program tree. It has:

- `head` — the function or value (e.g. `fill`, `rep_t`, `2`)
- `type` — the return type (e.g. `mat`, `int`)
- `tailtypes` — expected argument types, or `None` for terminals
- `tails` — filled-in arguments (other `Delta` nodes)
- `hiddentail` — for invented primitives, the underlying expression it abbreviates

Calling `delta()` evaluates the program it represents.

A `Delta` is **terminal** if it has no arguments, or if all its arguments are themselves terminal. The function `isterminal(d)` checks this.

### Deltas

`Deltas` is the DSL — a collection of `Delta` primitives. It has two parts:

- `core` — the hand-written primitives you provide
- `invented` — new primitives discovered during compression

It indexes primitives by type for efficient lookup during search.

### Program trees

Programs are expression trees of `Delta` nodes. For example, the program that builds a 2x3x3 matrix filled with zeros could be represented as:

```
(rt (fill 0 3 3) 2)
```

Which means: fill a 1-frame 3x3 grid with 0, then repeat it 2 times along the time axis.

---

## The ECD loop

`ECD(X, D)` runs the three-stage loop until a program is found for `X`.

```
while not solved:
    Explore  →  find programs that produce X
    Compress →  invent new primitives to compress found programs
    Dream    →  train a model to predict which primitives are useful for X
               and use it to bias search in the next iteration
```

### Stage 1: Explore

`solve_enumeration(X, D, Q)` searches for programs that evaluate to `X`.

`Q` is a log-probability distribution over all primitives in `D`. Programs are enumerated in order of decreasing probability using a budget-based scheme: the search iterates over successive probability windows `[gap*i, gap*(i+1)]`, covering all programs from most to least probable without repetition.

For each candidate program tree:
1. Evaluate it
2. If it matches `X`, save it (keeping the shortest match found so far)
3. Stop when `X` is found or timeout is reached

The core enumeration is `cenumerate` → `cenumerate_fold`, which expand program trees recursively while tracking accumulated log-probability against the budget window. An alternative generator-based enumerator `penumerate` / `groom` is used when a fixed budget ceiling is passed.

### Stage 2: Compress

`saturate(D, sols)` takes the found programs and grows the DSL by inventing new primitives.

It repeatedly:
1. Enumerates all subtrees ("ghosts") across all found programs using `spenumerate` / `count_jive`
2. For each ghost, computes a compression ratio: how much shorter would all programs be if this subtree were replaced by a single named primitive?
3. Picks the ghost with the best compression ratio
4. Creates a new `Delta` for it, with a `hiddentail` recording its underlying definition
5. Rewrites all trees to use the new primitive
6. Repeats until no ghost improves compression

The invented primitive can be a constant (no arguments) or a function (with holes that become arguments). Holes are tracked using `ishole` / `isarg` flags on `Delta` nodes, and `typize` extracts their types to form the new primitive's signature.

`freeze` is called on trees after rewriting to lock in the invented primitive's structure — preventing further substitution into its internals.

### Stage 3: Dream

`dream(D, soltrees)` trains a neural recognition model to look at a matrix and predict which DSL primitives were used to produce it.

It works by:
1. Sampling random program trees from the DSL using `newtree` / `needle`
2. Evaluating each tree to get an output matrix
3. Training the model on `(output matrix → primitive index)` pairs for every node in the tree

The model (`MatRecognitionModel`) processes the matrix frame by frame:
- A 2D CNN encodes each frame
- A GRU updates a hidden state across frames
- The final hidden state is projected to DSL logits (one per primitive)
- An auxiliary loss predicts the next frame, providing a richer training signal

After training, the model is run on `X` to produce a new `Q` — a biased distribution over primitives that reflects what the model thinks is relevant for `X`. This guides the next Explore phase toward more promising programs.

---

## Matrix type

The `mat` type is a 3D numpy array of shape `(T, H, W)` — `T` frames of `H×W` integer grids. The DSL primitives for matrices are defined in `dsl.py`:

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `fill` | `int, int, int → mat` | 1-frame grid filled with a value |
| `mset` | `mat, int, int, int → mat` | set a cell value in every frame |
| `cell` | `int → mat` | 1×1×1 grid |
| `hconcat` | `mat, mat → mat` | concatenate along width |
| `vconcat` | `mat, mat → mat` | concatenate along height |
| `tconcat` | `mat, mat → mat` | concatenate along time |
| `rot90` | `mat → mat` | rotate each frame 90° CCW |
| `fliph` | `mat → mat` | flip each frame horizontally |
| `flipv` | `mat → mat` | flip each frame vertically |
| `rep_t` | `mat, int → mat` | repeat along time |
| `rep_h` | `mat, int → mat` | repeat along height |
| `rep_w` | `mat, int → mat` | repeat along width |

---

## Example

```python
from ecd import ECD, mat_key
from dsl import *

D = Deltas([
    Delta(fill, mat, [int, int, int], repr='fill'),
    Delta(mset, mat, [mat, int, int, int], repr='mset'),
    Delta(rep_t, mat, [mat, int], repr='rt'),
    Delta(0, int),
    Delta(1, int),
    Delta(2, int),
    Delta(3, int),
])

# target: a 3x3 grid with 1 at center, repeated over 2 frames
X = np.tile(np.array([[[0,0,0],[0,1,0],[0,0,0]]]), (2,1,1))

Z = ECD(X, D, timeout=120)

xkey = mat_key(X)
print(Z[xkey])       # the program tree
print(Z[xkey]())     # evaluates back to X
```

---

## File overview

| File | Contents |
|------|----------|
| `dsl.py` | `Delta`, `Deltas`, matrix primitives, tree utilities |
| `ecd.py` | `ECD`, `solve_enumeration`, `saturate`, `dream`, enumeration helpers |
| `gpt.py` | `YOGPT` transformer, used as a building block for recognition models |
