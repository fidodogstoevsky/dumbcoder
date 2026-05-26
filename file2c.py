"""File 2c: false-belief via belief search (root_type=grid).

unfold is never in the DSL. The synthesizer searches over believed initial
worlds (programs of type grid). The evaluator implicitly wraps each candidate:

    unfold_belief_steps(actual_g, believed_g, T, approach(1, 2))

For nav tasks the solution is just ig_i (believed world = actual world).
For false-belief tasks the solution is place_wall(ig_i, r, c).

mask is also gone: unfold_belief_steps returns the actual-world trajectory,
so phantom walls are never visible in the output.

Stitch discovers:
  fn_nav($ig)            = $ig
  fn_belief($ig, $r, $c) = (place_wall $ig $r $c)

Run:
  python file2c.py
"""

from copy import deepcopy

from ecd import (
    Deltas, Delta, ECD,
    make_nav_tasks, make_false_belief_tasks,
    task_terminals,
    normalize, mat_key,
)
from dsl import (
    grid,
    place_wall,
    approach,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
Xs_nav  = make_nav_tasks(n=8, size=4, n_walls=1, seed=0)
fb_meta = make_false_belief_tasks(n=20, size=4, n_phantoms=1, seed=42, return_meta=True)
Xs_fb   = [x for x, _ in fb_meta]
Xs      = Xs_nav + Xs_fb
print(f"\n{len(Xs)} tasks: {len(Xs_nav)} nav + {len(Xs_fb)} false-belief")

# ── DSL ────────────────────────────────────────────────────────────────────
# Only grid-producing primitives. unfold, mask, approach are all implicit.
# Nav solution:          ig_i          (believed world = actual world)
# False-belief solution: place_wall(ig_i, r, c)
core_prims = [
    Delta(place_wall, grid, [grid, int, int], repr='place_wall'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
]

ig = task_terminals(Xs, mode='full')
D  = Deltas(core_prims + ig)
print(f"DSL: {len(core_prims)} core prims + {len(ig)} task terminals = {len(D)} total")
print("  nav solution (1 node): ig_i")
print("  false-belief solution (4 nodes): (place_wall ig_i r c)\n")

# ── ECD ────────────────────────────────────────────────────────────────────
# root_type=grid: enumerate programs that produce a grid (the believed initial world).
# step_fn=approach(1,2): the transition function, held constant and implicit.
print("Running ECD…\n")
Z, rewritten = ECD(
    Xs, D,
    per_task_timeout=60,
    max_iterations=6,
    max_arity=4,
    root_type=grid,
    step_fn=approach(1, 2),
)

# ── Report ─────────────────────────────────────────────────────────────────
n_nav = sum(1 for x in Xs_nav if mat_key(x) in Z and Z[mat_key(x)] is not None)
n_fb  = sum(1 for x in Xs_fb  if mat_key(x) in Z and Z[mat_key(x)] is not None)
print(f"\n=== Results ===")
print(f"  nav:          {n_nav}/{len(Xs_nav)}")
print(f"  false-belief: {n_fb}/{len(Xs_fb)}")

print("\n=== Invented primitives ===")
if not D.invented:
    print("  (none)")
else:
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        print(f"  {d.repr}  [{argtypes}]  -> {d.type}")
        print(f"    body: {body_str}")
        if 'place_wall' in body_str:
            print(f"    *** BELIEF — phantom wall grounds agent's false model ***")
        else:
            print(f"    *** NAV — believed world = actual world ***")

print("\n=== False-belief solutions (first 4) ===")
for x, meta in list(zip(Xs_fb, [m for _, m in fb_meta]))[:4]:
    k = mat_key(x)
    tag = f"agent={meta['agent']} goal={meta['goal']} pw={meta['phantom_walls']}"
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  {tag}")
        print(f"    found:     {sol}")
        if rw:
            print(f"    rewritten: {rw}")
    else:
        print(f"  {tag}  → unsolved")
