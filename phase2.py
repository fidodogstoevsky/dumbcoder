"""Phase 2: false-belief tasks, discovered from scratch.

No bootstrapping. No hand-coded mental primitives. ECD discovers everything.

init_grid design
----------------
Each task gets a task-specific terminal ig_i = x[0] (first frame).

  Nav solution:          (unfold ig_i (approach 1 2))
  False-belief solution: (map_mat (replace_val 3 0) (unfold (place_wall ig_i pwr pwc) (approach 1 2)))

Neither mask nor navigate appear in the DSL — they must be discovered.

Expected discovery (layered across ECD iterations)
---------------------------------------------------
Iteration 1 — shared subexpressions across all 28 programs:
  fn_navigate  = (approach 1 2)           — the step function used everywhere
  fn_hide      = (replace_val 3 0)        — zero out walls (used in every false-belief solution)

Iteration 2 — structure visible now that subexpressions are atomic:
  fn_nav($grid)
    = (unfold $grid fn_navigate)

  fn_belief($grid, $pwr, $pwc)
    = (map_mat fn_hide (unfold (place_wall $grid $pwr $pwc) fn_navigate))

The belief primitive is built entirely from non-mental sub-primitives:
  map_mat, replace_val, unfold, place_wall, approach carry no mental semantics.
  Mental content emerges from their composition.

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
    unfold_auto, place_wall,
    approach, replace_val, map_mat,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
Xs_nav = make_nav_tasks(n=8,  size=4, n_walls=1, seed=0)
fb_meta = make_false_belief_tasks(n=20, size=4, n_phantoms=1, seed=42, return_meta=True)
Xs_fb   = [x for x, _ in fb_meta]
Xs      = Xs_nav + Xs_fb
print(f"\n{len(Xs)} tasks: {len(Xs_nav)} nav + {len(Xs_fb)} false-belief")

# ── DSL ────────────────────────────────────────────────────────────────────
core_prims = [
    Delta(map_mat,      mat,  [fn, mat],           repr='map_mat'),
    Delta(unfold_auto,  mat,  [grid, fn],           repr='unfold'),
    Delta(place_wall,   grid, [grid, int, int],     repr='place_wall'),
    Delta(replace_val,  fn,   [int, int],           repr='replace_val'),
    Delta(approach,     fn,   [int, int],           repr='approach'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
    Delta(6, int, repr='6'), Delta(7, int, repr='7'),
    Delta(8, int, repr='8'), Delta(9, int, repr='9'),
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
        has_map_mat     = 'map_mat'     in body_str
        has_replace_val = 'replace_val' in body_str
        has_wall        = 'place_wall'  in body_str
        has_approach    = 'approach'    in body_str
        tag = None
        if has_map_mat and has_replace_val and has_wall and has_approach:
            tag = '*** BELIEF — map_mat(hide-walls, navigate-under-phantom-wall) ***'
        elif has_approach and not has_wall and not has_map_mat:
            tag = '    navigate — (approach 1 2)'
        elif has_replace_val and not has_wall:
            tag = '    hide_walls — (replace_val 3 0)'
        elif has_wall and has_approach:
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
