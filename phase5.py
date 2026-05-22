"""Phase 5: goal algebra — sequential desire, utility-decomposed DSL.

The DSL uses phase 3's utility layer: desires are assembled from composable
utility primitives before being compiled into actions.

  neg_dist($gv)       → util    "proximity to gv" — a desire as utility
  optimize(u, av)     → fn      greedy step maximising utility u for agent av
  fn_want($gv)        → fn      pre-compiled desire→action = optimize(neg_dist($gv), 1)
                                 fn_want is a CORE PRIMITIVE so ECD can use it
                                 directly without assembling it each time.
                                 Having it as a primitive makes sequential
                                 desire programs 14 nodes (vs 21 if optimize
                                 is assembled from scratch each occurrence),
                                 which is what makes enumeration tractable.

The full optimize/neg_dist/distance/add_util layer remains in the DSL so
stitch can discover the relationship fn_want = optimize(neg_dist, 1) and so
future blended/avoidance tasks can compose utilities before optimizing.

─────────────────────────────────────────────────────────────────────────────
Simple desire (tier 1):
  ig terminal (mode='agent') zeros the goal; ECD enumerates (gr, gc, gv).
  Solution (7 nodes):
    unfold_auto(gset($ig, $gr, $gc, $gv), fn_want($gv))
  gv × 2 (world + fn_want) — single-desire signature.
  Stitch discovers:
    fn_desire($ig, $gr, $gc, $gv) = unfold_auto(gset($ig,...), fn_want($gv))

─────────────────────────────────────────────────────────────────────────────
Sequential desire (tier 2):
  ig terminal zeros BOTH goals; ECD enumerates (r1, c1, gv1, r2, c2, gv2).
  Agent pursues gv1 while it exists, then switches to gv2.

  Solution (14 nodes):
    unfold_auto(
      gset(gset($ig, $r1, $c1, $gv1), $r2, $c2, $gv2),
      if_fn(exists($gv1), fn_want($gv1), fn_want($gv2))
    )

  gv1 × 3 (gset + exists + fn_want): world, trigger, and action share the value.
  gv2 × 2 (gset + fn_want): world and fallback action share the value.

  Stitch discovers the goal algebra combinator (level 3):
    fn_cond_desire($gv1, $gv2) =
      if_fn(exists($gv1), fn_want($gv1), fn_want($gv2))

  Then the full task abstraction (level 4):
    fn_seq_desire($ig, $r1, $c1, $gv1, $r2, $c2, $gv2) =
      unfold_auto(gset(gset($ig,...), ...), fn_cond_desire($gv1, $gv2))

─────────────────────────────────────────────────────────────────────────────
Goal algebra hierarchy:
  neg_dist($gv)             — desire as utility   (level 1, DSL primitive)
  fn_want($gv)              — desire as action     (level 2, core primitive)
  fn_cond_desire($gv1,$gv2) — sequential desire    (level 3, discovered)
  fn_seq_desire(...)        — full task abstraction (level 4, discovered)

Run:
  python phase5.py
"""

from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    make_desire_tasks, make_sequential_desire_tasks,
    task_terminals,
    normalize, mat_key,
)
from dsl import (
    mat, grid, fn, fn_pred, util,
    unfold_auto, gset, approach_from,
    optimize, neg_distance, distance, neg_util, add_util,
    if_fn, exists,
)

# ── Goal values ────────────────────────────────────────────────────────────────
GOAL_VALS  = (2, 4, 5)                 # simple desire
SEQ_COMBOS = ((2, 4), (4, 5), (2, 5)) # (gv1, gv2): first desire then fallback

# ── Tasks ─────────────────────────────────────────────────────────────────────
raw_simple  = make_desire_tasks(n_per_goal=8, goal_vals=GOAL_VALS, size=4, seed=0)
Xs_simple   = [x for x, _ in raw_simple]
meta_simple = [m for _, m in raw_simple]

raw_seq  = make_sequential_desire_tasks(n_per_combo=8, goal_combos=SEQ_COMBOS, size=4, seed=1)
Xs_seq   = [x for x, _ in raw_seq]
meta_seq = [m for _, m in raw_seq]

print(f"\n{len(Xs_simple)} simple desire tasks (goal_vals {list(GOAL_VALS)})")
print(f"{len(Xs_seq)} sequential desire tasks (combos {list(SEQ_COMBOS)})")

# ── DSL ───────────────────────────────────────────────────────────────────────
# fn_want = approach_from(1): goal_val → fn.
# Semantically: fn_want(gv) = optimize(neg_dist(gv), 1).
# It is a CORE PRIMITIVE so ECD doesn't need to assemble it from scratch each
# time, keeping sequential desire programs at 14 nodes (tractable) rather than
# 21 (too long to enumerate within the per-task timeout).
# The full optimize/neg_dist layer is also present so stitch can discover
# fn_want = optimize(neg_dist($gv), 1) and future tasks can compose utilities.
fn_want = approach_from(1)

core_prims = [
    Delta(unfold_auto,  mat,     [grid, fn],           repr='unfold'),
    Delta(gset,         grid,    [grid, int, int, int], repr='gset'),
    #Delta(fn_want,      fn,      [int],                 repr='fn_want'),
    Delta(optimize,     fn,      [util, int],           repr='optimize'),
    Delta(neg_distance, util,    [int],                 repr='neg_dist'),
    Delta(distance,     util,    [int],                 repr='distance'),
    Delta(neg_util,     util,    [util],                repr='neg_util'),
    Delta(add_util,     util,    [util, util],          repr='add_util'),
    Delta(if_fn,        fn,      [fn_pred, fn, fn],     repr='if_fn'),
    Delta(exists,       fn_pred, [int],                 repr='exists'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
]

# mode='agent' zeros out all non-agent, non-wall cells.
# Simple desire: one goal zeroed → ECD enumerates (gr, gc, gv).
# Sequential desire: two goals zeroed → ECD enumerates (r1, c1, gv1, r2, c2, gv2).
Xs     = Xs_simple + Xs_seq
ig_sim = task_terminals(Xs_simple, mode='agent')
ig_seq = task_terminals(Xs_seq,    mode='agent')
for i, d in enumerate(ig_seq):
    d.repr = f'ig_{len(Xs_simple) + i}'

ig = ig_sim + ig_seq
D  = Deltas(core_prims + ig)
print(f"\nDSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total")
print(f"  simple desire:  7 nodes/solution, 3 ints to enumerate")
print(f"  sequential:    14 nodes/solution, 6 ints to enumerate\n")

# ── ECD ───────────────────────────────────────────────────────────────────────
print("Running ECD…\n")
Z, rewritten = ECD(Xs, D, per_task_timeout=60, max_iterations=8, max_arity=8)

# ── Report ────────────────────────────────────────────────────────────────────
n_simple = sum(1 for x in Xs_simple if mat_key(x) in Z and Z[mat_key(x)] is not None)
n_seq    = sum(1 for x in Xs_seq    if mat_key(x) in Z and Z[mat_key(x)] is not None)
print(f"\n=== Results ===")
print(f"  simple desire:     {n_simple}/{len(Xs_simple)}")
print(f"  sequential desire: {n_seq}/{len(Xs_seq)}")

print("\n=== Invented primitives ===")
if not D.invented:
    print("  (none)")
else:
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}

        print(f"  {d.repr}  [{argtypes}]  -> {d.type}")
        print(f"    body: {body_str}")

        if 'if_fn' in body_str and 'exists' in body_str:
            shared_str = ', '.join(f'{v} (×{c})' for v, c in shared.items()) if shared else '—'
            print(f"    *** GOAL COMBINATOR — sequential desire; shared: {shared_str} ***")
        elif 'fn_want' in body_str and shared:
            label = ', '.join(f'{v} (×{c})' for v, c in shared.items())
            print(f"    *** DESIRE — fn_want shared variable {label} ***")
        elif shared:
            label = ', '.join(f'{v} (×{c})' for v, c in shared.items())
            print(f"    *** DESIRE — shared variable {label} ***")

print("\n=== Sample sequential desire solutions (2 per combo) ===")
for gv1, gv2 in SEQ_COMBOS:
    shown = 0
    for x, m in zip(Xs_seq, meta_seq):
        if m['gv1'] != gv1 or m['gv2'] != gv2 or shown >= 2:
            continue
        k   = mat_key(x)
        tag = (f"({gv1}→{gv2})  agent={m['agent']}  "
               f"goal1={m['goal1']} goal2={m['goal2']}  T={m['T']}")
        if k in Z and Z[k] is not None:
            sol = normalize(deepcopy(Z[k]))
            rw  = rewritten.get(k, '')
            print(f"  {tag}")
            print(f"    found:     {sol}")
            if rw:
                print(f"    rewritten: {rw}")
        else:
            print(f"  {tag}  → unsolved")
        shown += 1
