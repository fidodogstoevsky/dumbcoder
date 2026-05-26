"""File 4: simpler programs discovered first — physics before intention.

Each task shows agent (1) drifting in a straight line to goal (2) at the
opposite edge.  Both explanations produce the exact same trajectory:

  Physics:     (unfold ig (step 1 dir))              — 5 symbols, depth 2
  Intentional: (unfold ig (optimize (neg_dist 2) 1)) — 6 symbols, depth 3

approach(av, gv) is *not* in the DSL.  Instead the intentional explanation must
be assembled from two parts:
  neg_distance(gv) — utility: how close is the agent to cells of value gv?
  optimize(u, av)  — action rule: move one step to maximise that utility

step(1, dir) is shallower by one nesting level.  With any depth-penalising prior
it is the cheaper hypothesis, and ECD finds it first.

Stitch abstracts:
  fn_physics($ig, $dir) = (unfold $ig (step 1 $dir))

Direction is an explicit free parameter — the genuine physical degree of freedom.
Assembling the intentional equivalent would require stitch to also discover:
  fn_nav($ig) = (unfold $ig (optimize (neg_dist 2) 1))
which is a strictly deeper tree.

Run:
  python file4.py
"""

from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    make_physics_tasks,
    task_terminals,
    normalize, mat_key,
)
from dsl import (
    mat, grid, fn, direction, util,
    unfold_auto, step, optimize, neg_distance,
    RIGHT, LEFT, UP, DOWN,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
raw  = make_physics_tasks(n_per_dir=6, size=4, seed=0)
Xs   = [x for x, _ in raw]
meta = [m for _, m in raw]
print(f"\n{len(Xs)} physics tasks (4 directions × 6, agent drifts linearly to goal)")

# ── DSL ────────────────────────────────────────────────────────────────────
# The intentional explanation requires assembling two primitives:
#   (optimize (neg_dist 2) 1)  — 4 symbols, depth 2
# The physics explanation needs only one:
#   (step 1 DIR)               — 3 symbols, depth 1
# step is shallower, so ECD finds it first with any depth-sensitive prior.
core_prims = [
    Delta(unfold_auto,  mat,  [grid, fn],       repr='unfold'),
    Delta(step,         fn,   [int, direction], repr='step'),
    Delta(optimize,     fn,   [util, int],      repr='optimize'),
    Delta(neg_distance, util, [int],            repr='neg_dist'),
    Delta(RIGHT, direction, repr='RIGHT'),
    Delta(LEFT,  direction, repr='LEFT'),
    Delta(UP,    direction, repr='UP'),
    Delta(DOWN,  direction, repr='DOWN'),
    Delta(1, int, repr='1'),
    Delta(2, int, repr='2'),
]

ig = task_terminals(Xs, mode='full')
D  = Deltas(core_prims + ig)
print(f"DSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total\n")

# ── ECD ────────────────────────────────────────────────────────────────────
print("Running ECD…\n")
Z, rewritten = ECD(Xs, D, per_task_timeout=30, max_iterations=3)

# ── Report ─────────────────────────────────────────────────────────────────
n_solved      = sum(v is not None for v in Z.values())
step_count    = sum(1 for v in Z.values() if v is not None and 'step'     in str(v))
intent_count  = sum(1 for v in Z.values() if v is not None and 'optimize' in str(v))
print(f"\n=== Results: {n_solved}/{len(Xs)} solved ===")
print(f"  via step:     {step_count}   (physics — shallow)")
print(f"  via optimize: {intent_count}   (intentional — assembled from 2 parts)")

print("\n=== Invented primitives ===")
if not D.invented:
    print("  (none)")
else:
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        print(f"  {d.repr}  [{argtypes}]")
        print(f"    body: {body_str}")
        if 'step' in body_str:
            print(f"    *** PHYSICS — direction is explicit free parameter ***")
        elif 'optimize' in body_str:
            print(f"    (intentional — assembled from optimize + neg_dist)")

print("\n=== Sample solutions (first per direction) ===")
dir_names = {(-1, 0): 'UP', (1, 0): 'DOWN', (0, -1): 'LEFT', (0, 1): 'RIGHT'}
seen_dirs = set()
for x, m in zip(Xs, meta):
    d = tuple(m['direction'])
    if d in seen_dirs:
        continue
    seen_dirs.add(d)
    k = mat_key(x)
    dname = dir_names.get(d, str(d))
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  {dname:<5}  agent={m['agent']}  goal={m['goal']}")
        print(f"    found:     {sol}")
        if rw:
            print(f"    rewritten: {rw}")
    else:
        print(f"  {dname:<5}  agent={m['agent']}  goal={m['goal']}  → unsolved")
