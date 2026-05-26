"""File 2b: false-belief tasks with a belief-attributing DSL.

Same tasks as file2.py but the DSL replaces mask+place_wall with two primitives
that make the agent's mental model explicit:

  add_phantom_wall(g, r, c) -> belief
    Constructs the agent's false model: the actual grid plus a wall it mistakenly
    believes is there.

  unfold_belief(actual_g, believed_g, f) -> mat
    Runs f on the *believed* world to determine the agent's moves, then replays
    those exact moves on the *actual* world.  The output trajectory shows reality
    — no mask needed, because the phantom wall never existed there.

False-belief solution:
  (unfold_belief ig_i (add_phantom_wall ig_i pwr pwc) (approach 1 2))

ig_i appears TWICE: once as the actual world (what the observer sees) and once
as the anchor for the belief (the agent's model is reality + one false feature).
Stitch discovers:

  fn_belief($ig, $r, $c) = (unfold_belief $ig (add_phantom_wall $ig $r $c) (approach 1 2))

$ig appearing twice is the structural signature of belief: the agent's model is
not arbitrary — it is grounded in the actual world with a single false addition.

Compare to file2.py:
  fn_belief($grid, $r, $c) = (mask (unfold (place_wall $grid $r $c) (approach 1 2)) 3)
  → world-level description: "navigate a modified world, hide the modification"

Run:
  python file2b.py
"""

from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    make_nav_tasks, make_false_belief_tasks,
    task_terminals,
    normalize, mat_key,
)
from dsl import (
    mat, grid, fn, belief,
    unfold_auto, unfold_belief,
    add_phantom_wall,
    approach,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
Xs_nav  = make_nav_tasks(n=8, size=4, n_walls=0, seed=0)
fb_meta = make_false_belief_tasks(n=20, size=4, n_phantoms=1, seed=42, return_meta=True)
Xs_fb   = [x for x, _ in fb_meta]
Xs      = Xs_nav + Xs_fb
print(f"\n{len(Xs)} tasks: {len(Xs_nav)} nav + {len(Xs_fb)} false-belief")

# ── DSL ────────────────────────────────────────────────────────────────────
core_prims = [
    Delta(unfold_auto,      mat,    [grid, fn],           repr='unfold'),
    Delta(unfold_belief,    mat,    [grid, belief, fn],   repr='unfold_belief'),
    Delta(add_phantom_wall, belief, [grid, int, int],     repr='add_phantom_wall'),
    Delta(approach,         fn,     [int, int],           repr='approach'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
]

ig = task_terminals(Xs, mode='full')
D  = Deltas(core_prims + ig)
print(f"DSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total")

# ── ECD ────────────────────────────────────────────────────────────────────
print("\nRunning ECD…\n")
Z, rewritten = ECD(Xs, D, per_task_timeout=60, max_iterations=6)

# ── Report ─────────────────────────────────────────────────────────────────
n_nav_solved = sum(1 for x in Xs_nav if mat_key(x) in Z and Z[mat_key(x)] is not None)
n_fb_solved  = sum(1 for x in Xs_fb  if mat_key(x) in Z and Z[mat_key(x)] is not None)
print(f"\n=== Results ===")
print(f"  nav:          {n_nav_solved}/{len(Xs_nav)}")
print(f"  false-belief: {n_fb_solved}/{len(Xs_fb)}")

print("\n=== Invented primitives ===")
if not D.invented:
    print("  (none)")
else:
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        has_ub   = 'unfold_belief'    in body_str
        has_apw  = 'add_phantom_wall' in body_str
        has_app  = 'approach'         in body_str
        print(f"  {d.repr}  [{argtypes}]")
        print(f"    body: {body_str}")
        if has_ub and has_apw and has_app:
            if shared:
                label = ', '.join(f'{v} (×{c})' for v, c in shared.items())
                print(f"    *** BELIEF — {label} grounds belief in actual world ***")
            else:
                print(f"    *** BELIEF — unfold_belief(actual, add_phantom_wall, approach) ***")
        elif has_app and not has_ub:
            print(f"        navigate — (approach 1 2)")

print("\n=== False-belief solutions (first 4) ===")
for x in Xs_fb[:4]:
    k = mat_key(x)
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  found:     {sol}")
        if rw:
            print(f"  rewritten: {rw}")
    else:
        print(f"  unsolved")
