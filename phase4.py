"""Phase 4: simpler programs discovered first — physics before intention.

Each task shows agent (1) drifting in a straight line to goal (2) at the
opposite edge.  Both explanations produce the exact same trajectory:

  Physics:     (unfold ig (step 1 dir))     — move in fixed direction
  Intentional: (unfold ig (approach 1 2))   — navigate toward goal

ECD finds the physics explanation first because step appears before approach
in the DSL, so step programs are enumerated first in every budget window.
Once step(1, dir) solves a task, approach is never tried for it.

Stitch abstracts:
  fn_physics($ig, $dir) = (unfold $ig (step 1 $dir))

Direction is an explicit free parameter — the physical degree of freedom.
The intentional alternative fn_nav($ig) = (unfold $ig (approach 1 2)) would
hide it, collapsing all directions into "approaching 2."

Run:
  python phase4.py
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
    mat, grid, fn, direction,
    unfold_auto, step, approach,
    RIGHT, LEFT, UP, DOWN,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
raw  = make_physics_tasks(n_per_dir=6, size=4, seed=0)
Xs   = [x for x, _ in raw]
meta = [m for _, m in raw]
print(f"\n{len(Xs)} physics tasks (4 directions × 6, agent drifts linearly to goal)")

# ── DSL ────────────────────────────────────────────────────────────────────
# step is listed BEFORE approach — step programs are always enumerated first.
# Both step(1,dir) and approach(1,2) produce the same trajectory for linear
# tasks, but ECD never reaches approach because step is found first.
core_prims = [
    Delta(unfold_auto, mat,       [grid, fn],       repr='unfold'),
    Delta(step,        fn,        [int, direction], repr='step'),
    Delta(approach,    fn,        [int, int],       repr='approach'),
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
n_solved       = sum(v is not None for v in Z.values())
step_count     = sum(1 for v in Z.values() if v is not None and 'step'     in str(v))
approach_count = sum(1 for v in Z.values() if v is not None and 'approach' in str(v))
print(f"\n=== Results: {n_solved}/{len(Xs)} solved ===")
print(f"  via step:     {step_count}   (physics — direction explicit)")
print(f"  via approach: {approach_count}   (intention — direction hidden)")

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
        elif 'approach' in body_str:
            print(f"    (intentional — direction collapsed into goal value)")

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
