"""Phase 2: false-belief tasks, discovered from scratch.

No bootstrapping.  ECD searches from the ground up.

init_grid design
----------------
Each task gets a task-specific terminal ig_i = x[0] (first frame).
This removes coordinate enumeration from the search entirely:

  Nav solution:          (unfold ig_i navigate)
  False-belief solution: (mask (unfold (place_wall ig_i pwr pwc) navigate) 3)

For nav, ECD searches only over T ∈ {1..9} and step functions — trivially fast.
For false-belief, it additionally searches over pwr,pwc ∈ {0..3} (phantom wall
position, invisible in x[0]) — 16 × 9 = 144 candidate programs per task.

Because ig_X tokens differ across tasks, stitch creates grid-typed holes and
discovers the pattern that matters:

  fn_nav($grid)
    = (unfold $grid navigate)

  fn_belief($grid, $pwr, $pwc)
    = (mask (unfold (place_wall $grid $pwr $pwc) navigate) 3)

The mask(…, 3) wrapper is the semantic signature of belief:
  "navigate on the believed grid (with phantom wall), hide the wall from output"

Run:
  python phase2.py
"""

from copy import deepcopy
from ecd import (
    Deltas, Delta, ECD,
    make_nav_tasks, make_false_belief_tasks,
    task_terminals,
    normalize, mat_key,
)
from dsl import (
    mat, grid, fn,
    mask, unfold_auto, place_wall,
    approach,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
Xs_nav = make_nav_tasks(n=8,  size=4, n_walls=1, seed=0)
fb_meta = make_false_belief_tasks(n=20, size=4, n_phantoms=1, seed=42, return_meta=True)
Xs_fb   = [x for x, _ in fb_meta]
Xs      = Xs_nav + Xs_fb
print(f"\n{len(Xs)} tasks: {len(Xs_nav)} nav + {len(Xs_fb)} false-belief")

# ── DSL ────────────────────────────────────────────────────────────────────
# Task terminals come first so their indices are stable across D.reset() calls.
# (D.reset() only clears D.invented, not D.core.)
core_prims = [
    Delta(mask,             mat,  [mat, int],            repr='mask'),
    Delta(unfold_auto,      mat,  [grid, fn],            repr='unfold'),
    Delta(place_wall,       grid, [grid, int, int],      repr='place_wall'),
    Delta(approach(1, 2),   fn,                          repr='navigate'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
    Delta(6, int, repr='6'), Delta(7, int, repr='7'),
    Delta(8, int, repr='8'), Delta(9, int, repr='9'),
]

# ig_i = x[0] (agent + goal + real walls; phantom walls absent by construction)
ig = task_terminals(Xs, mode='full')
D = Deltas(core_prims + ig)
print(f"DSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total")

# ── ECD ────────────────────────────────────────────────────────────────────
# Nav: searches T only → near-instant.
# False-belief: searches (pwr, pwc, T) → ~144 candidates, fast.
print("\nRunning ECD…\n")
Z, rewritten = ECD(Xs, D, per_task_timeout=30, max_iterations=6)

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
        has_mask = 'mask'       in body_str
        has_wall = 'place_wall' in body_str
        has_nav  = 'navigate'   in body_str
        tag = None
        if has_mask and has_wall and has_nav:
            tag = '*** BELIEF — mask(navigate-under-hidden-wall) ***'
        elif has_wall and has_nav:
            tag = '    nav-with-wall'
        print(f"  {d.repr}  [{argtypes}]")
        print(f"    body: {body_str}")
        if tag:
            print(f"    {tag}")

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
