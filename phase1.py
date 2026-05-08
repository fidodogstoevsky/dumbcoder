"""Phase 1 MVP: simple structural abstraction, no bootstrapping.

Tasks: nav on a 4×4 grid with a fixed 2-cell vertical barrier at (1,2)+(2,2).
Each task gets a task-specific terminal ig_i = x[0] (agent + goal + walls).

Search space per task: (unfold ig_i T navigate) — only T ∈ {1..9} to enumerate.
ECD solves all tasks near-instantly, then stitch compresses:

  10 programs of the form (unfold ig_0 3 navigate), (unfold ig_1 5 navigate), …
  → fn_0($grid, $T) = (unfold $grid $T navigate)

This is the simplest possible abstraction: a nav primitive parameterised over
the initial grid and path length.  The walls are already encoded in ig_i, so
stitch doesn't need to see place_wall at all to compress these programs.

Run:
  python phase1.py
"""

from copy import deepcopy
from ecd import (
    Deltas, Delta, ECD,
    make_fixed_wall_tasks, task_terminals,
    normalize, mat_key,
)
from dsl import (
    mat, grid, fn, fn_pred,
    mask, unfold, unfold_auto, blank44, gset, place_agent_goal, place_wall,
    approach, if_fn, exists,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
WALLS = [(1, 2), (2, 2)]
Xs = make_fixed_wall_tasks(n=10, walls=WALLS, size=4, seed=0)

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

ig = task_terminals(Xs, mode='full')
D  = Deltas(core_prims + ig)
print(f"DSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total")

# ── ECD ────────────────────────────────────────────────────────────────────
print("\nRunning ECD…\n")
Z, rewritten = ECD(Xs, D, per_task_timeout=30, max_iterations=3)

# ── Report ─────────────────────────────────────────────────────────────────
n_solved = sum(v is not None for v in Z.values())
print(f"\nsolved {n_solved}/{len(Xs)} tasks")

print("\n=== Invented primitives ===")
if not D.invented:
    print("  (none — stitch found no useful abstractions)")
else:
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body = normalize(deepcopy(d))
        print(f"  {d.repr}  [{argtypes}]")
        print(f"    body: {body}")

print("\n=== Rewritten programs ===")
for s in list(rewritten.values()):
    print(f"  {s}")
