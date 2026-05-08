"""Phase 3: desire abstraction, discovered from scratch.

No bootstrapping.  ECD searches from the ground up.

init_grid design (agent mode)
------------------------------
Each task terminal ig_i = x[0] with goal cells zeroed out — only the agent
(and walls, if any) are encoded.  The goal must be added explicitly:

  Solution: (unfold (gset ig_i gr gc gv) (approach 1 gv))

This preserves the shared-variable structure of desire:
  gv appears in BOTH gset (what is placed in the world)
            AND approach (what the agent pursues)

Stitch discovers:
  fn_desire($grid, $gr, $gc, $gv)
    = (unfold (gset $grid $gr $gc $gv) (approach 1 $gv))

The same $gv hole fills both slots — that shared variable IS desire:
  "the thing placed in the world is the same thing the agent moves toward"

goal_vals ∈ {2, 4, 5}  (3 reserved for walls in BFS)
8 tasks per goal_val = 24 tasks total.

Search space per task: T ∈ {1..9}, gr/gc ∈ {0..3}, gv ∈ {2..9}
  = 9 × 4 × 4 × 8 = ~1152 candidate programs — still far smaller than
  the full coordinate search without init_grid.

Run:
  python phase3.py
"""

from copy import deepcopy
from ecd import (
    Deltas, Delta, ECD,
    make_desire_tasks,
    task_terminals,
    normalize, mat_key,
)
from dsl import (
    mat, grid, fn, fn_pred,
    mask, unfold, unfold_auto, blank44, gset, place_agent_goal, place_wall,
    approach, if_fn, exists,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
desire_meta = make_desire_tasks(n_per_goal=8, goal_vals=(2, 4, 5), size=4, seed=7)
Xs_desire   = [x for x, _ in desire_meta]
print(f"\n{len(Xs_desire)} desire tasks (goal_vals 2, 4, 5)")

# ── DSL ────────────────────────────────────────────────────────────────────
core_prims = [
    Delta(mask,             mat,  [mat, int],                 repr='mask'),
    Delta(unfold_auto,      mat,  [grid, fn],                  repr='unfold'),
    Delta(blank44,          grid,                             repr='blank'),
    Delta(gset,             grid, [grid, int, int, int],      repr='gset'),
    Delta(place_agent_goal, grid, [grid, int, int, int, int], repr='place_ag'),
    Delta(place_wall,       grid, [grid, int, int],           repr='place_wall'),
    Delta(approach,         fn,   [int, int],                 repr='approach'),
    Delta(approach(1, 2),   fn,                               repr='navigate'),
    Delta(if_fn,            fn,   [fn_pred, fn, fn],          repr='if'),
    Delta(exists,           fn_pred, [int],                   repr='exists'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
    Delta(6, int, repr='6'), Delta(7, int, repr='7'),
    Delta(8, int, repr='8'), Delta(9, int, repr='9'),
]

# ig_i = x[0] with goal zeroed — agent position only.
# goal_val (gv) must be placed explicitly, appearing in both gset and approach.
ig = task_terminals(Xs_desire, mode='agent')
D  = Deltas(core_prims + ig)
print(f"DSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total")

# ── ECD ────────────────────────────────────────────────────────────────────
print("\nRunning ECD…\n")
Z, rewritten = ECD(Xs_desire, D, per_task_timeout=60, max_iterations=6)

# ── Report ─────────────────────────────────────────────────────────────────
n_solved = sum(v is not None for v in Z.values())
print(f"\nsolved {n_solved}/{len(Xs_desire)} tasks")

print("\n=== Invented primitives ===")
if not D.invented:
    print("  (none)")
else:
    for d in D.invented:
        argtypes  = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str  = str(normalize(deepcopy(d)))
        has_approach = 'approach' in body_str
        has_gset     = 'gset'     in body_str
        has_wall     = 'place_wall' in body_str
        has_mask     = 'mask'     in body_str
        tag = None
        if has_approach and has_gset and not has_wall and not has_mask:
            tag = '*** DESIRE — approach(1, gv) + gset(…, gv) shared variable ***'
        print(f"  {d.repr}  [{argtypes}]")
        print(f"    body: {body_str}")
        if tag:
            print(f"    {tag}")

print("\n=== Sample solutions (first 4) ===")
for x, meta in desire_meta[:4]:
    k = mat_key(x)
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  goal_val={meta['goal_val']}: {sol}")
        if rw:
            print(f"  rewritten:         {rw}")
    else:
        print(f"  goal_val={meta['goal_val']}: unsolved")
