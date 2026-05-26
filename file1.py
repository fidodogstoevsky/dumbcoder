"""File 1 MVP: simple structural abstraction, no bootstrapping.

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
  python file1.py
"""

from copy import deepcopy
from ecd import (
    Deltas, Delta, ECD,
    make_fixed_wall_tasks, task_terminals,
    normalize, mat_key, make_nav_tasks,
)
from dsl import (
    mat, grid, fn,
    unfold_auto, place_wall,
    approach,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
Xs = make_nav_tasks(n=8,  size=4, n_walls=2, seed=0)
# for idx, task in enumerate(Xs):
#     print('-----------------')
#     print(f'task no.: {idx}')
#     print(task)

# ── DSL ────────────────────────────────────────────────────────────────────
core_prims = [
    Delta(unfold_auto,      mat,  [grid, fn],            repr='unfold'),
    Delta(place_wall,       grid, [grid, int, int],      repr='place_wall'),
    Delta(approach,       fn,   [int, int],            repr='approach'),
    #Delta(approach(1, 2),   fn,                          repr='navigate'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
]

ig = task_terminals(Xs, mode='full')
D  = Deltas(core_prims + ig)
print(f"DSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total")

# ── ECD ────────────────────────────────────────────────────────────────────
print("\nRunning ECD…\n")
Z, rewritten = ECD(Xs, D, per_task_timeout=30, max_iterations=3, run_dream=False)

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
