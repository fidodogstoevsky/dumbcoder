# dumbcoder — Implementation Documentation

*Synthesizing mental abstractions (belief / agency) from non-mental primitives
via program induction + library compression.*

This document describes the implementation as it currently stands. The **ECD
architecture** (Section 2) is stable and unlikely to change. The **DSL/interpreter**
(Section 3), the **task corpus and harness** (Section 4), the primitive granularity,
and the **recognition ("Q") model** (Section 2.4) are still under active revision —
where a design is provisional it is flagged as such.

---

## 1. Research question and the core constraint

The project asks whether the *structure of an intentional agent* — an agent that
acts on a **private model** of the world that can diverge from the world itself
(i.e. holds a **false belief**) — can be **discovered by compression** rather than
**stipulated by the interpreter**.

The governing constraint, which every design decision answers to:

> **Belief must be a discoverable *compound* of general, non-mental parts.**
> It may never be a primitive, and it may never be a special case baked into the
> interpreter's evaluation rules.

The failure mode this rules out is *intensionality by construction*. A hypothetical
`believe(...)` primitive that conjures a private model, computes over it, and commits
the agent's move is disqualifying **even if it produces the right trajectories** —
it re-buries mentality in the DSL and leaves the compression step nothing to
discover. The problem is **opacity, not ephemerality**: an ephemeral private model
is fine, *provided* its construct / compute / commit steps are visible, separately
typed program steps that the searcher assembles and the library learner re-extracts.

Consequently:

- **Minimality is sought in the interpreter, not in the decomposition.** The
  interpreter is shrunk toward a single-grid `unfold` (Section 3.1); the agent's
  second grid is introduced *in program space* by general combinators.
- **Two coexisting grids** (world + private model) is treated as the irreducible
  structural footprint of intensionality. You cannot reach a single-grid,
  single-type *interpreter* while keeping belief synthesizable — so the second grid
  is pushed out of the interpreter and into the *program*.

---

## 2. The ECD architecture (`ecd.py`)

ECD is a wake–sleep library-learning loop in the DreamCoder lineage: **E**numerate
(solve tasks against the current library), **C**ompress (abstract shared structure
into new library primitives), **D**ream (train a neural recognition model to guide
the next enumeration). Each round makes programs that were previously out of budget
reachable, because their sub-parts have been compressed into single tokens.

```
        ┌─────────────────────────────────────────────────────────┐
        │  library D = core primitives + invented abstractions     │
        └─────────────────────────────────────────────────────────┘
                    │                    ▲                    │
       (wake) enumerate           (sleep) compress      (sleep) dream
       solve_enumeration          saturate_stitch       dream / MatRecognitionModel
                    │                    ▲                    │
                    ▼                    │                    ▼
         solutions per task ──────▶ new abstractions    recognition model Q
                                                          guides next wake
```

The `ECD(...)` driver (`ecd.py:764`) runs this loop; `experiment.run_phase`
(Section 4) wraps it with the concrete corpus, the per-round dreaming, and the
reporting. The pieces:

### 2.1 Program representation — `Delta` (`dsl.py:708`)

Every program, primitive, and invented abstraction is a `Delta`: a node in an
expression tree.

- `head` — the underlying Python callable or literal value.
- `tails` — the argument sub-trees (`None` for an unfilled node, `[]`/list once
  applied).
- `type` / `tailtypes` — the return type and the expected argument types. The type
  system is **monomorphic** and string-based (`'fn'`, `'fn_p_g'`, `'coord'`, …);
  there is no polymorphism, so distinct arrows over grids/pairs need distinct
  typed composer primitives.
- `repr` — the s-expression token used for serialization and parsing.
- `hiddentail` — for an *invented abstraction*, the body tree with `$i` argument
  holes. Calling the Delta (`Delta.__call__`, `dsl.py:739`) deep-copies the body
  and substitutes each `$i` with the corresponding tail (`replace_hidden`), then
  evaluates — so an abstraction is genuinely *inlined*, not opaque.

Supporting machinery on trees: `length` (node count, the MDL tiebreak),
`normalize` (fully inline all abstractions back to primitives),
`simplify` (a semantics-preserving rewrite that collapses spurious `fork` nesting),
`typize` (collect hole types), `freeze` (lock holes so a body can't be mutated),
and the s-expression reader/printer (`getast` / `todelta` / `tr` / `__repr__`).

### 2.2 The library — `Deltas` (`ecd.py:26`)

`Deltas` holds `core` primitives + `invented` abstractions and maintains the
indices enumeration needs: `bytype` / `bytype_terminal` (candidate symbols per
type), O(1) lookup by repr and by `(head, type)`, and terminal flags. `add`
registers a new abstraction (and re-`infer`s the indices); `reset` drops all
inventions back to the bare core (called at the start of every `ECD`).

### 2.3 Wake: enumeration (`solve_enumeration`, `ecd.py:618`)

Enumeration produces candidate programs **cheapest-first** under a cost model
`Q` (a log-probability per symbol) and matches each against the task trajectories.

- **Cost model.** A program's cost is the summed `-logp` of its nodes. `Q` is
  either the type-conditioned uniform prior, the content-aware prior, or the
  dreamed recognition model (Section 2.4).
- **Budget windows.** `cenumerate` (`ecd.py:151`) walks the search in expanding
  `-logp` bands `(LOGPGAP*idx, LOGPGAP*(idx+1))`, firing a callback for every
  program whose cost lands in the current window. `solve_enumeration` opens
  successive windows until the task is solved or the timeout expires. This is
  best-first enumeration by description length.
- **Root type / interpreter dispatch.** A program's *root type* selects how it is
  run against a task (`cb`, `ecd.py:650`):
  - `fn` → `unfold(g, T, f)` — the single-grid interpreter (the main path).
  - `fn_p_g` → `unfold_with_template(g, template, T, c)` — a pair-consumer run
    against a *given* external template (registration-style tasks).
  - `sfn` → `unfold_state` and `machine` → `unfold_m` — legacy pair/bundle
    interpreters from earlier iterations, retained for reference/baselines.
  Enumeration is necessarily **per root type** (one budget walk can't produce two
  different arrows), but the *library and the compression step are shared* across
  root types.
- **Matching.** A program is a solution for a task iff running it from the task's
  frame-0 grid for `T` frames reproduces the exact trajectory (`mat_key` compares
  shape + bytes). On a hit the solution is `simplify`-ed and stored; ties are
  broken by `length` (shorter wins).
- **Parallelism.** `ECD` dispatches the unsolved tasks across a
  `ProcessPoolExecutor` (`_solve_one_task`, `ecd.py:749`), one task per worker,
  with per-worker thread pinning to avoid OpenMP oversubscription.

### 2.4 Sleep I: compression (`saturate_stitch`, `ecd.py:285`)

Compression is delegated to **`stitch_core`** (the Rust stitch library) for
top-down library learning. The flow:

1. Serialize every solution to an s-expression string, after `normalize`
   (inline all prior abstractions) and `simplify`. *Normalized* programs are
   passed to stitch deliberately — feeding compressed programs (with `fn_0` as an
   opaque token) causes a naming collision with stitch's own `fn_0` discoveries.
2. Run `stitch_core.compress(programs, iterations, max_arity)`; it returns a set
   of `abstractions` (reusable fragments with `#i` holes) and the corpus
   `rewritten` in terms of them.
3. For each abstraction: remap its name to a globally-unique `fn_k` (offset past
   previously invented names), inline-expand any references to *skipped*
   abstractions, convert `#i` holes to `$i`, parse the body, infer hole types
   (`typize` + `_annotate_holes`), inject explicit arg nodes for
   partial-application slots (`_unsatisfied_tailtypes`), and register the result
   in `D`.
4. Parse stitch's rewritten programs as the new tree corpus for training.

A substantial amount of the code here is defensive plumbing around stitch's
encoding (skipped abstractions, partial applications, bare-reference substitution,
longest-name-first remapping). The **research payload** is what the abstraction
learner *invents*: across the phases, the pooled stitch reliably discovers a single
**agent constructor** `fn_0 = (fork (compose (wall_at $3 $2) (optimize (neg_dist
$1) $0)) (sync_to_world $0))` with `$0` (the agent value) **shared between the
policy and the commit** — that shared binding is the structural signature of agency,
and it is *discovered by MDL*, not stipulated.

### 2.5 Sleep II: dreaming (`dream` + `MatRecognitionModel`, `ecd.py:1043` / `860`)

The recognition model learns to predict, from a task's rendered trajectory, which
library symbols its solution uses — turning the uniform prior into a **learned,
task-conditioned `Q`** that steers enumeration toward promising programs.

- **`MatRecognitionModel`** is a *flat, matrix-conditioned* Q. It encodes a
  trajectory into a fixed vector by pooling learned row/column embeddings over
  role-identified cells — where roles are read from **motion**, not hardcoded
  values (agent = the cell vacated between frame 0 and the last frame; goal = the
  entity present at both ends), plus a wall-position pool, a whole-trajectory mean
  (the detour signal that distinguishes a false-belief path from a straight one),
  and a path-length embedding. A single linear head maps this to a distribution
  over the whole DSL. Crucially the Q is computed **once per task** and reused for
  every enumeration decision — there is no per-node tree-context GRU. *(This flat
  design replaced an earlier tree-conditioned model that was ~20× slower to
  enumerate under and generalized poorly across depths; the encoder specifics are
  still being tuned.)*
- **`dream`** trains it wake-sleep style on a mix of **replays** (the round's real
  solutions) and **fantasies** (programs sampled from the current library and run
  on fresh random grids), so the model sees both solved structure and the library's
  generative range.
- **`dreamed_q`** (`ecd.py:1010`) places the model's output on the *same
  type-conditional cost scale* as the uniform/content prior and **floors it at
  uniform** (`q = max(model, uniform)`): the model can make a symbol *cheaper*
  (tried earlier) but never push one below its uniform reachability, so dreaming
  can only ever help — a mis-calibrated model can't render a still-unsolved family
  (belief) unreachable. Visible integer literals are forced to cost 0 (the
  content trick).

### 2.6 Two priors worth naming

- **`uniform_type_q`** — `logp[i] = -log(#symbols sharing i's type)`; the neutral
  MDL prior.
- **content-aware Q** — additionally forces integer literals whose value is
  *visible in the grid* to cost 0. This is what lets tasks with different agent
  values solve in proportionate time in the first iteration, and it is on the
  critical path for belief's tractability. (The `coord` vs `cellvalue` type split
  in the DSL exists precisely so that the *invisible* wall coordinate stays a
  bounded latent while *visible* cell values are content-priced — Section 3.)

---

## 3. The DSL / interpreter (`dsl.py`)

The DSL is the substrate the searcher composes over. Its central claim is embodied
here: there is **no mental primitive**; the private model is introduced by general
combinators. The primitive *granularity* is provisional and is exactly the axis the
phases vary (Section 4).

### 3.1 The single-grid interpreter and `fork` / `sync_to_world`

The base interpreter is `unfold(g, T, f)` (`dsl.py:280`): thread a single grid
through `f : grid -> grid` for `T` frames, rendering each. A program is a plain
grid endomorphism — no world/model pair in the interpreter's state.

The agent's private model is introduced *in the program* by two combinators:

- **`fork(derive, commit)`** (`dsl.py:216`): `w ↦ commit((w, derive(w)))`. `derive`
  builds a private grid from a copy of the world (e.g. stamp a phantom wall, run the
  policy on it); `commit` reconciles the `(world, derived)` pair back to a single
  grid. The second grid lives only for the duration of the call. This is the S/fork
  combinator — completely general, nothing mental about it.
- **`sync_to_world(v)`** (`dsl.py:228`): a `fn_p_g` (pair→grid) commit that moves
  value `v` from its world position to its position in the derived grid — a
  grid-diff.

**Belief** is then the *composition*
`(fork (compose (wall_at r c) (optimize (neg_dist gv) av)) (sync_to_world av))`,
with `av` (the agent value) appearing **twice** — in the policy and in the commit.
That coincidence is the agency signature the compression step is meant to surface.

### 3.2 A general pair interface (defending against "gerrymandering")

If `fork` only ever fed `sync_to_world` and `sync` only ever ate `fork`, the split
would be a disguised `believe` primitive. To make the pair interface a genuine,
independently-populated calculus, the DSL adds non-mental inhabitants:

- **`overlay`** (`dsl.py:254`) — a second `fn_p_g` commit (graphics union, not
  mind): consumes `fork`'s pair without any sync.
- **`unfold_with_template`** (`dsl.py:268`) — a second *producer* of pairs, where
  the second channel is a **given external template**, not a derived model. A
  program using `sync_to_world` here is doing image *registration*, not holding a
  belief.

So the `pair_gg` / `fn_p_g` interface is populated from both sides by both mental
and non-mental families. Belief is one path through a general calculus.

### 3.3 The "cube" — symmetric complements

Every choice `sync_to_world`/`overlay`/`dup` bakes in has an opposite corner. The
cube (`make_symmetric_prims`) adds all of them, along independent axes:

| axis | belief's corner | complement(s) |
|---|---|---|
| direction | `sync_to_world` (read model → write world) | `sync_to_model` (perception) |
| scope | one value | `sync_all`, `sync_except` |
| z-order | `overlay` (model wins) | `underlay` (inpainting) |
| projection | `fst_gg` (keep world) | `snd_gg` (readout) |
| bifunctor | `mapsnd`/`on_model` | `mapfst`, `bimap` |
| pairing | `dup` (diagonal Δ) | `pair_blank` (fresh scratch) |
| utility | `neg_distance` (attract) | `distance` (flee) |
| grid-edit | `wall_at` (add) | `clear_at`, `erase` (remove) |

None of these help a theory-of-mind task; each is the natural tool for some
non-mental one. The experimental claim: joint MDL still selects exactly the
(read-model, write-world, single-`av`) corner for belief, while every complement
attaches to *its own* non-mental family — so the agency signature is **discovered**,
not gerrymandered into the primitive set. (This is why the corpus includes one
minds-free task *per corner* — an unused distractor would be trivially "avoided".)

### 3.4 Decomposition of `fork` and `sync` (product-category combinators)

`fork` is not atomic in general — it factors into textbook combinators of the
product (×) category:

```
fork(derive, commit)  ≡  commit ∘ mapsnd(derive) ∘ dup
    dup      :: grid -> pair       w ↦ (w, w)          (the diagonal Δ)
    mapsnd f :: pair -> pair       (a,b) ↦ (a, f(b))   (bifunctor 'second')
    commit   :: pair -> grid                            (the eliminator)
```

The DSL provides these (`dup`, `mapsnd`, and typed composers `compose_gp` /
`pipe_gpg`, since the monomorphic type system has no generic `compose` across
grid/pair arrows). Likewise `sync_to_world(v)` factors on the key `v` into
`register(locate v, place v)` — read a coordinate off one channel, impose it on the
other. **Phase 2** hands the searcher these decomposed parts and *removes* atomic
`fork`/`sync`, forcing belief to be **rediscovered** as a deeper compound (now with
`av` shared *three* ways: `optimize` + `locate` + `place`). The `fork_decomposed` /
`sync_decomposed` helpers exist to prove the decompositions are numerically
identical to the atomic versions before search begins.

### 3.5 The grid-stack — arity as a free parameter

The cube fixes *role* symmetry but every combinator is hardwired to **arity 2**
(`pair_gg`). "Why one world + one model, not three?" would then be answered by an
interpreter commitment. The `gstack` calculus (`dsl.py:580`) replaces the fixed pair
with a single recursive type — `gstack ::= () | (grid, *gstack)` — and depth-
polymorphic, n-ary lifts of every cube op (`dup_top`, `blank_top`, `swap_top`,
`map_top`, `zip_top`, `commit_top`, `peek`, and composers `compose_gs` / `pipe_gsg`).
Depth-1 reproduces `fork`+`sync` exactly (`fork_stack_decomposed`). This makes the
number of private channels a discoverable structural feature — **Phase 3** shows
that joint MDL never *selects* arity > 1 even though the stack makes it expressible.

### 3.6 Legacy interpreters

`unfold_state` (`sfn`, world/model pair threaded in interpreter state) and
`unfold_m` (`machine`, explicit `(kind, init, step, render)` bundles) are earlier
formulations kept as baselines. They put the second grid in the *interpreter*; the
single-grid `unfold` line (3.1) is the current design and the one the constraint in
Section 1 pushes toward.

---

## 4. The experiment harness (`experiment.py`, `prims.py`, tasks)

### 4.1 Primitive sets (`prims.py`)

One canonical home for every DSL handed to the searcher:

- `make_core_prims()` — the bare atomic DSL (fork/sync interface + grid core, no
  symmetric field).
- `make_symmetric_prims(decomposed=False)` — the cube. `decomposed=False` is
  **Phase 1** (atomic fork + sync); `decomposed=True` is **Phase 2** (fork and
  sync spelled out into product-category + register/locate/place plumbing; the
  scope complements stay atomic because they fold over an unbounded value set and
  have no locate/place spelling).
- `make_stack_prims()` — **Phase 3**'s depth-polymorphic grid-stack.

The `coord` / `cellvalue` terminal split lives here: `coord` reprs are distinct
(`c0`, `c1`, …) so the s-expression parser can disambiguate a grid *position* from a
cell *value* that share an integer head.

### 4.2 Task families

Tasks are `(trajectory, metadata)` pairs; each family is generated *through the
real interpreter* with a **necessity check** that the intended program is the unique
cheapest explanation (so a solution can't be a coincidence of an under-determined
scene).

- **Minds tasks** (`tasks_minds.py`): `physics`, `desire` (utility-driven motion),
  `belief` (false-belief detour around an *invisible* wall), plus the deeper
  `witness_belief`, `goal_displacement`, and `dual_belief` variants. Rejection
  filters (`_physically_explainable`, `_displaced_goal_explainable`,
  `_witness_rival_explainable`, …) discard scenes reproducible by a cheaper
  non-belief rival, so the wall is the *unique* explanation.
- **Minds-free tasks** (`tasks_world.py`): `overlay`, `registration`, and one task
  per cube corner — `flee`, `deletion`, `denoise`, `obstacle` (fn-rooted) and
  `perception`, `multi_registration`, `registration_except`, `inpainting`,
  `readout` (pair-rooted). These give each complement genuine non-mental work.

### 4.3 `run_phase` — one phase of the curriculum (`experiment.py:593`)

`run_phase(decomposed=...)` is the single entry point; `phase1.py` / `phase2.py` are
thin wrappers (`decomposed=False` / `True`), and `phase3_arity.py` runs the stack
variant. A phase:

1. **Generates the mixed corpus** (all minds + minds-free families) and dedupes it.
2. **Builds the library** (`Deltas(make_symmetric_prims(decomposed=))`) and
   verifies every task's ground-truth program runs; in Phase 2 it first proves the
   decomposition identities.
3. Runs **several full ECD rounds**. Enumeration is per-root-type (`fn` for the
   trajectory families, `fn_p_g` for registration), but **the library and the
   stitch are shared** — one joint `saturate_stitch` over *all* solutions pooled
   across both root types. This joint compression is the crux: it is what makes
   belief an MDL win over one objective, rather than an artefact of searching each
   family in its own silo.
4. **Dreams** after each round (unless `--no-dream`): trains the recognition model
   on the round's replays + fantasies and enumerates the next `DREAM_USE_ROUNDS`
   rounds under `dreamed_q`, then reverts to the uniform/content prior as a
   completeness mop-up.
5. **Reports** the verdict: a *usage census* (which family reaches for which
   primitive), whether the joint stitch invented the agent constructor, and whether
   that constructor is belief-specific (used by belief, absent from the non-mental
   rewrites).

Knobs: `--smoke` (fast, tiny corpus), `--samples` (dump/export example
trajectories), `--ecd-iters N`, `--t-fn SECONDS` (belief is the long pole),
`--no-dream`, `--plain-belief` / `--curriculum` (diagnostics that trade the
false-belief uniqueness guarantee for a shallower first solve).

---

## 5. What is stable vs. in flux

| Component | Status |
|---|---|
| ECD wake-sleep loop, budget-window enumeration, stitch integration | **Stable** — the architecture this chapter describes. |
| `Delta` / `Deltas` representation, monomorphic type system | **Stable.** |
| Single-grid `unfold` + `fork`/`sync` as the belief substrate | **Stable in spirit**; exact primitive set varies by phase. |
| DSL granularity, cube contents, decomposition depth | **Provisional** — the primitive-granularity question is the experiment. |
| Recognition model (`MatRecognitionModel`) encoder + dreaming | **In flux** — flat matrix-conditioned Q is current; encoder features being tuned. |
| Task corpus, necessity filters, per-family counts, timeouts | **In flux** — being calibrated. |

---

## 6. File map

| File | Role |
|---|---|
| `ecd.py` | The ECD architecture: `Deltas` library, enumeration, `saturate_stitch` compression, `dream` + `MatRecognitionModel` recognition model, the `ECD` driver. |
| `dsl.py` | Types, the `Delta` tree, the interpreters (`unfold`, `unfold_with_template`, legacy `unfold_state`/`unfold_m`), and every primitive (fork/sync, pair interface, cube, stack, decompositions). |
| `prims.py` | The three primitive sets handed to the searcher (`make_core_prims`, `make_symmetric_prims`, `make_stack_prims`). |
| `tasks_minds.py` | Minds task generators + necessity filters + `COMBOS`/`SIZE`/`DIRS`. |
| `tasks_world.py` | Minds-free task generators (pair-interface + one per cube corner). |
| `experiment.py` | Shared harness: `run_phase`, ground-truth verification, decomposition-identity checks, the usage-census / abstraction-generality reporting, CLI parsing. |
| `phase1.py` / `phase2.py` / `phase3_arity.py` | Thin phase drivers (atomic / decomposed / arity-stack). |
| `file11.py` / `file12.py` | Earlier `sfn` / `machine` baselines. |
| `run.job` / `template.job` / `sync.sh` | HPC batch-run scaffolding. |
</content>
</invoke>
