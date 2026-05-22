"""Phase 3: desire tasks, bootstrapped with nav tasks.

Nav tasks (mode='full') give ECD easy wins in iteration 1.  Stitch creates
fn_0($grid) = (unfold $grid (approach_1 2)), which acts as a building block.

Desire tasks (mode='agent'): ig_i has the goal cell zeroed out, so ECD must
search over (gr, gc, gv).  After fn_0 exists, desire tasks with gv=2 shrink
to (fn_0 (gset ig_i gr gc 2)) — only 2 integers.  Once those are solved,
stitch generalises across goal_vals and discovers the desire abstraction:

  fn_desire($ig, $gr, $gc, $gv) = (unfold (gset $ig $gr $gc $gv) (approach_1 $gv))

$gv appears TWICE — in gset (world) and approach_1 (behaviour).  That
shared variable is desire.

Run:
  python phase3.py
"""

from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    make_nav_tasks, make_desire_tasks,
    task_terminals,
    normalize, mat_key,
)
from dsl import (
    mat, grid, fn, util,
    unfold_auto, gset,
    neg_distance, distance, neg_util, add_util, optimize,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
GOAL_VALS = (2, 4, 5)

Xs_nav  = make_nav_tasks(n=8, size=4, n_walls=0, seed=1)
raw_des = make_desire_tasks(n_per_goal=8, goal_vals=GOAL_VALS, size=4, seed=0)
Xs_des  = [x for x, _ in raw_des]
meta    = [m for _, m in raw_des]

print(f"\n{len(Xs_nav)} nav tasks + {len(Xs_des)} desire tasks "
      f"(goal_vals: {list(GOAL_VALS)})")

# ── DSL ────────────────────────────────────────────────────────────────────
core_prims = [
    Delta(unfold_auto,  mat,  [grid, fn],            repr='unfold'),
    Delta(gset,         grid, [grid, int, int, int],  repr='gset'),
    Delta(optimize,     fn,   [util, int],            repr='optimize'),
    Delta(neg_distance, util, [int],                  repr='neg_dist'),
    Delta(distance,     util, [int],                  repr='distance'),
    Delta(neg_util,     util, [util],                 repr='neg_util'),
    Delta(add_util,     util, [util, util],           repr='add_util'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
]

# Nav terminals: mode='full' (goal visible) → solution needs no gset.
# Desire terminals: mode='agent' (goal zeroed) → ECD must enumerate gr,gc,gv.
# Rename desire terminals so their indices don't collide with nav terminals.
ig_nav = task_terminals(Xs_nav, mode='full')
ig_des = task_terminals(Xs_des, mode='agent')
for i, d in enumerate(ig_des):
    d.repr = f'ig_{len(Xs_nav) + i}'

Xs = Xs_nav + Xs_des

for idx, task in enumerate(Xs):
    print('-----------------')
    print(f'task no.: {idx}')
    print(task)

ig = ig_nav + ig_des
D  = Deltas(core_prims + ig)
print(f"DSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total\n")

# ── ECD ────────────────────────────────────────────────────────────────────
print("Running ECD…\n")
Z, rewritten = ECD(Xs, D, per_task_timeout=60, max_iterations=8)

# ── Report ─────────────────────────────────────────────────────────────────
n_nav_solved = sum(1 for x in Xs_nav if mat_key(x) in Z and Z[mat_key(x)] is not None)
n_des_solved = sum(1 for x in Xs_des if mat_key(x) in Z and Z[mat_key(x)] is not None)
print(f"\n=== Results ===")
print(f"  nav:    {n_nav_solved}/{len(Xs_nav)}")
print(f"  desire: {n_des_solved}/{len(Xs_des)}")

print("\n=== Invented primitives ===")
if not D.invented:
    print("  (none)")
else:
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        print(f"  {d.repr}  [{argtypes}]")
        print(f"    body: {body_str}")
        if shared:
            label = ', '.join(f'{v} (×{c})' for v, c in shared.items())
            print(f"    *** DESIRE — shared variable {label}: same value in world and behaviour ***")

print("\n=== Sample desire solutions (2 per goal_val) ===")
for gv in GOAL_VALS:
    shown = 0
    for x, m in zip(Xs_des, meta):
        if m['goal_val'] != gv or shown >= 2:
            continue
        k   = mat_key(x)
        tag = f"goal_val={gv}  agent={m['agent']}  goal={m['goal']}"
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
