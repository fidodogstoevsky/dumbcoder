"""Phase 4: physics step functions.

Tasks: objects drifting in a fixed direction each frame (gravity/magnetism).
No goals, no walls, no BFS. The correct step function is step(val, dir),
which is strictly simpler than approach(val, goal) for these trajectories.

init_grid design
----------------
ig_i = x[0] (object positions, mode='full').
Solution: (unfold ig_i (step 1 $dir))

direction is the only free search variable — 4 candidates (UP/DOWN/LEFT/RIGHT).
ECD solves each task in at most 4 enumeration steps.

Expected inventions
-------------------
After seeing tasks across all four directions, stitch should discover:

  fn_0($dir) = (unfold $grid (step 1 $dir))

showing that direction is the single degree of freedom — a directional primitive.
This is distinct from navigate/approach: there is no goal, no BFS, no desired state.
The agent (object) just drifts. Direction replaces desire.

If tasks from multiple directions are compressed together, stitch may also find:

  fn_down($grid) = (unfold $grid (step 1 DOWN))   — specialised per-direction

but fn_0 is the more general and interesting invention.

Run:
  python phase4.py
"""

from copy import deepcopy
from dsl import (
    mat, grid, fn, direction,
    unfold_auto, step, compose,
    UP, DOWN, LEFT, RIGHT,
)
from ecd import (
    Deltas, Delta, ECD,
    make_physics_tasks, task_terminals,
    normalize, mat_key,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
# 8 tasks per direction × 4 directions = 32 tasks.
# Diverse starting positions ensure stitch generalises over ig_i.
# All four directions ensure stitch abstracts over dir rather than baking it in.
physics_meta = make_physics_tasks(n_per_dir=8, size=4, n_objects=3, seed=0)
Xs = [x for x, _ in physics_meta]
print(f"\n{len(Xs)} physics tasks (UP/DOWN/LEFT/RIGHT, 8 each)")

# ── DSL ────────────────────────────────────────────────────────────────────
core_prims = [
    Delta(unfold_auto,  mat,  [grid, fn],        repr='unfold'),
    Delta(step,         fn,   [int, direction],  repr='step'),
    Delta(compose,      fn,   [fn, fn],          repr='compose'),
    Delta(1, int, repr='1'),
    Delta(UP,    direction, repr='UP'),
    Delta(DOWN,  direction, repr='DOWN'),
    Delta(LEFT,  direction, repr='LEFT'),
    Delta(RIGHT, direction, repr='RIGHT'),
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
    print("  (none)")
else:
    dir_names = {str(UP):'UP', str(DOWN):'DOWN', str(LEFT):'LEFT', str(RIGHT):'RIGHT'}
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        has_step    = 'step'    in body_str
        has_compose = 'compose' in body_str
        tag = None
        if has_step and '$' in body_str and not has_compose:
            tag = '*** DIRECTIONAL — step(val, $dir) free direction ***'
        elif has_step and has_compose:
            tag = '    composed physics'
        elif has_step:
            tag = '    specialised step (direction baked in)'
        print(f"  {d.repr}  [{argtypes}]")
        print(f"    body: {body_str}")
        if tag:
            print(f"    {tag}")

print("\n=== Sample solutions (one per direction) ===")
dir_names = {(-1,0):'UP', (1,0):'DOWN', (0,-1):'LEFT', (0,1):'RIGHT'}
seen_dirs = set()
for x, meta in physics_meta:
    d = tuple(meta['direction'])
    if d in seen_dirs:
        continue
    seen_dirs.add(d)
    k = mat_key(x)
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  {dir_names[d]:5}: {sol}")
        if rw:
            print(f"         rewritten: {rw}")
    else:
        print(f"  {dir_names[d]:5}: unsolved")
