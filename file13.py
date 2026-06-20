"""File 13: single-grid interpreter; belief as a discoverable `fork` compound.

file11 threaded a (world, model) pair through the interpreter; file12 kept the
pair but moved init/render into program space as machine constructors.  Both
hardwired a *second grid* into the interpreter's state so that intensionality
could be expressed as a compound and then discovered by compression.

Here the interpreter shrinks to the simplest possible loop — it threads a single
grid and renders every frame:

    unfold(g, T, f)    f :: fn = grid -> grid

The agent's private model is no longer interpreter state.  It is introduced
*locally in program space* by a general, non-mental combinator and collapsed
back to one grid in the same step:

    fork(derive, commit) :: fn, fn_p_g -> fn      w |-> commit((w, derive(w)))
    sync_to_world(v)     :: int      -> fn_p_g     move v in world to its pos in derived

Neither primitive denotes a mental state on its own.  `fork` is the S/fork
combinator (apply a derived transform to a copy, then reconcile against the
original); `sync_to_world` is a grid-diff that transfers one value's position.
Belief is the *composition* — and that composition is what stitch is meant to
extract as the agent type constructor.  A `believe(...)` *primitive* would have
been intensional by construction (perceive-represent-act packaged opaquely,
re-burying the file6-9 mistake in the DSL); the whole point is that belief stays
a compound of general parts.

Expected solutions (one shared DSL across the curriculum):

  physics  (3 nodes):  (step v d)
  desire   (4 nodes):  (optimize (neg_dist gv) av)
  belief  (11 nodes):  (fork (compose (wall_at r c) (optimize (neg_dist gv) av))
                             (sync_to_world av))

av appears twice — in optimize (who acts on the model) and in sync_to_world
(whose move is committed to the world) — the structural signature of agency.
The hoped-for stitch discovery is the agent type constructor:

  fn_agent($r, $c, $gv, $av) =
    (fork (compose (wall_at $r $c) (optimize (neg_dist $gv) $av))
          (sync_to_world $av))

Run:
    python file13.py            # full run
    python file13.py --smoke    # tiny corpus, short timeouts
"""

import sys
import re as _re
from collections import Counter
from copy import deepcopy

import numpy as np

from ecd import Deltas, Delta, ECD, normalize, mat_key
from dsl import (
    fn, util, direction, fn_p_g,
    RIGHT, LEFT, UP, DOWN,
    fork, sync_to_world,
    compose, step, optimize, neg_distance, wall_at,
    unfold, tr, simplify,
)

# ── Configuration (same combos as file11/12) ────────────────────────────────────
COMBOS = [(1, 2), (1, 5), (4, 2), (4, 5)]
SIZE   = 5
DIRS   = {'right': RIGHT, 'left': LEFT, 'up': UP, 'down': DOWN}


# ── Task generation ────────────────────────────────────────────────────────────
# Every task is generated through the same `unfold` the searcher uses, so any
# solvability failure is search, never encoding.

def _agent_pos(frame, av):
    pos = np.argwhere(frame == av)
    return (int(pos[0][0]), int(pos[0][1])) if len(pos) else None


def _physically_explainable(x, g):
    "True if x is reproduced by a single physical fn (step or optimize)."
    T = x.shape[0]
    vals = [int(v) for v in np.unique(g) if v != 0]
    for v in vals:
        for d in DIRS.values():
            try:
                if np.array_equal(unfold(g, T, step(v, d)), x):
                    return True
            except Exception:
                pass
        for u in vals:
            if u == v:
                continue
            try:
                if np.array_equal(unfold(g, T, optimize(neg_distance(u), v)), x):
                    return True
            except Exception:
                pass
    return False


def _displaced_goal_explainable(x, g):
    """True if x is reproduced by a 'displaced-goal' belief instead of a wall.

    The wall scenes are under-determined: an agent that detours around a phantom
    wall can produce the exact same trajectory as an agent who simply believes
    the goal sits one cell from where it really is, then seeks that displaced
    goal.  That rival explanation is structurally isomorphic to the wall one
    (1 fn + 2 ints) yet usually *cheaper* under content-aware Q (it reuses the
    visible goal value), so the searcher reaches it first whenever the geometry
    permits.  We reject such scenes here so that only trajectories that *uniquely*
    require an obstacle survive — keeping the phantom wall the sole explanation.

    The rival family is, for an agent av seeking goal gv:

        (fork (compose MODIFY (optimize (neg_dist gv) av)) (sync_to_world av))

    where MODIFY nudges gv one step in the model: either a fixed direction
    (step gv d) or greedily toward the nearest u (optimize (neg_dist u) gv),
    u including 0 (empty space) — the form the searcher actually found.
    """
    T = x.shape[0]
    vals = [int(v) for v in np.unique(g) if v != 0]
    for av in vals:
        for gv in vals:
            if av == gv:
                continue
            seek = optimize(neg_distance(gv), av)
            modifications = [step(gv, d) for d in DIRS.values()]
            modifications += [optimize(neg_distance(u), gv)
                              for u in [0] + vals if u != gv]
            for modify in modifications:
                prog = fork(compose(modify, seek), sync_to_world(av))
                try:
                    if np.array_equal(unfold(g, T, prog), x):
                        return True
                except Exception:
                    pass
    return False


def make_physics_tasks(n, size=SIZE, vals=(1, 4), seed=0):
    "Ground truth: (step v d)."
    rng = np.random.default_rng(seed)
    tasks = []
    while len(tasks) < n:
        v     = int(rng.choice(vals))
        dname = str(rng.choice(list(DIRS)))
        dr, dc = DIRS[dname]
        T = int(rng.integers(4, 6))
        r_lo, r_hi = max(0, -dr * (T - 1)), min(size - 1, size - 1 - dr * (T - 1))
        c_lo, c_hi = max(0, -dc * (T - 1)), min(size - 1, size - 1 - dc * (T - 1))
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r, c = int(rng.integers(r_lo, r_hi + 1)), int(rng.integers(c_lo, c_hi + 1))
        g = np.zeros((size, size), dtype=int)
        g[r, c] = v
        x = unfold(g, T, step(v, DIRS[dname]))
        tasks.append((x, {'kind': 'physics', 'val': v, 'dir': dname}))
    return tasks


def make_desire_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0):
    "Ground truth: (optimize (neg_dist gv) av)."
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in combos:
        made = 0
        while made < n_per_combo:
            ar, ac = int(rng.integers(size)), int(rng.integers(size))
            gr, gc = int(rng.integers(size)), int(rng.integers(size))
            if ar == gr or ac == gc:
                continue
            L = abs(ar - gr) + abs(ac - gc)
            if not (3 <= L <= 5):
                continue
            g = np.zeros((size, size), dtype=int)
            g[ar, ac] = av
            g[gr, gc] = gv
            T = L + 1
            x = unfold(g, T, optimize(neg_distance(gv), av))
            if x[-1][gr, gc] != av:
                continue
            tasks.append((x, {'kind': 'desire', 'av': av, 'gv': gv}))
            made += 1
    return tasks


def make_belief_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8):
    """False-belief: the agent detours around a wall that exists only in its model.

      (fork (compose (wall_at pr pc) (optimize (neg_dist gv) av)) (sync_to_world av))

    The phantom wall is placed on the true-belief BFS path; scenes are rejected
    unless the wall actually causes a detour AND the trajectory is not
    explainable by any single physical program.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in combos:
        made, attempts = 0, 0
        while made < n_per_combo and attempts < 5000:
            attempts += 1
            ar, ac = int(rng.integers(size)), int(rng.integers(size))
            gr, gc = int(rng.integers(size)), int(rng.integers(size))
            if (ar, ac) == (gr, gc) or abs(ar - gr) + abs(ac - gc) < 3:
                continue
            g = np.zeros((size, size), dtype=int)
            g[ar, ac] = av
            g[gr, gc] = gv

            # true-belief trajectory: source of on-path phantom-wall candidates
            direct = unfold(g, max_T, optimize(neg_distance(gv), av))
            path  = [_agent_pos(direct[t], av) for t in range(max_T)]
            inter = [p for p in path if p and p != (ar, ac) and p != (gr, gc)]
            if not inter:
                continue
            pr, pc = inter[int(rng.integers(len(inter)))]

            gt = fork(compose(wall_at(pr, pc), optimize(neg_distance(gv), av)),
                      sync_to_world(av))
            x_full = unfold(g, max_T, gt)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (gr, gc)), None)
            if t_arrive is None or t_arrive < 3:
                continue
            T = t_arrive + 1
            x = x_full[:T].copy()
            if np.array_equal(x, direct[:T]):     # phantom wall must cause a detour
                continue
            if _physically_explainable(x, g):
                continue
            if _displaced_goal_explainable(x, g):  # reject scenes a displaced goal also explains
                continue
            tasks.append((x, {'kind': 'belief', 'av': av, 'gv': gv, 'pw': (pr, pc)}))
            made += 1
    return tasks


# ── DSL ────────────────────────────────────────────────────────────────────────

def make_core_prims():
    return [
        # Fork combinator + commit (program-space pair plumbing; no interpreter state)
        Delta(fork,          fn,     [fn, fn_p_g],       repr='fork'),
        Delta(sync_to_world, fn_p_g, [int],              repr='sync_to_world'),

        # Grid-state primitives (fn = grid -> grid)
        Delta(compose,      fn,   [fn, fn],          repr='compose'),
        Delta(step,         fn,   [int, direction],  repr='step'),
        Delta(optimize,     fn,   [util, int],       repr='optimize'),
        Delta(wall_at,      fn,   [int, int],        repr='wall_at'),

        # Utility
        Delta(neg_distance, util, [int],             repr='neg_dist'),

        # Direction terminals
        Delta(RIGHT, direction, repr='right'),
        Delta(LEFT,  direction, repr='left'),
        Delta(UP,    direction, repr='up'),
        Delta(DOWN,  direction, repr='down'),

        # Int terminals
        *[Delta(i, int, repr=str(i)) for i in range(6)],
    ]


def verify_ground_truth(D, tasks_meta):
    """Re-express each kind's ground truth as a Delta tree (the searcher's own
    encoding) and check it reproduces the task — catches eager/typing mistakes
    in the Delta encoding rather than in the raw python closures."""
    for x, m in tasks_meta:
        if m['kind'] == 'physics':
            prog = f"(step {m['val']} {m['dir']})"
        elif m['kind'] == 'desire':
            prog = f"(optimize (neg_dist {m['gv']}) {m['av']})"
        else:
            pr, pc = m['pw']
            prog = (f"(fork (compose (wall_at {pr} {pc}) "
                    f"(optimize (neg_dist {m['gv']}) {m['av']})) "
                    f"(sync_to_world {m['av']}))")
        tree = tr(D, prog)
        out  = unfold(x[0], x.shape[0], tree())
        assert np.array_equal(out, x), f"ground truth failed for {m}: {prog}"
    print(f"ground-truth check: {len(tasks_meta)} tasks verified via Delta trees")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(smoke=False):
    n_phys = 2 if smoke else 6
    n_des  = 1 if smoke else 2   # per combo
    n_bel  = 1 if smoke else 6   # per combo
    per_task_timeout = 10 if smoke else 300
    max_iterations   = 2  if smoke else 10

    print("Generating tasks…")
    phys = make_physics_tasks(n_phys, seed=0)
    des  = make_desire_tasks(n_des, COMBOS, seed=1)
    bel  = make_belief_tasks(n_bel, COMBOS, seed=2)

    # dedupe across the whole corpus (identical mats would skew counts)
    seen, all_tasks = set(), []
    for x, m in phys + des + bel:
        k = mat_key(x)
        if k in seen:
            continue
        seen.add(k)
        all_tasks.append((x, m))

    Xs   = [x for x, _ in all_tasks]
    meta = [m for _, m in all_tasks]
    by_kind = Counter(m['kind'] for m in meta)
    print(f"  {by_kind['physics']} physics, {by_kind['desire']} desire, "
          f"{by_kind['belief']} belief — {len(Xs)} total\n")

    D = Deltas(make_core_prims())
    print(f"DSL: {len(D)} primitives")
    print("  expected physics solution (3 nodes):  (step v d)")
    print("  expected desire solution  (4 nodes):  (optimize (neg_dist gv) av)")
    print("  expected belief solution (11 nodes):")
    print("    (fork (compose (wall_at r c) (optimize (neg_dist gv) av))")
    print("          (sync_to_world av))")
    print("  expected stitch discovery — the agent type constructor:")
    print("    fn_agent($r,$c,$gv,$av) with $av shared ×2 (optimize + sync_to_world)\n")

    verify_ground_truth(D, all_tasks)

    print("\nRunning ECD…\n")
    Z, rewritten = ECD(
        Xs, D,
        per_task_timeout=per_task_timeout,
        max_iterations=max_iterations,
        max_arity=5,
        stitch_iterations=4,
        root_type=fn,
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n=== Results ===")
    for kind in ('physics', 'desire', 'belief'):
        ks = [x for x, m in all_tasks if m['kind'] == kind]
        n  = sum(1 for x in ks if mat_key(x) in Z and Z[mat_key(x)] is not None)
        print(f"  {kind:8s}: {n}/{len(ks)}")

    print("\n=== Invented primitives ===")
    if not D.invented:
        print("  (none)")
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(simplify(normalize(deepcopy(d))))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        print(f"  {d.repr}  [{argtypes}]  -> {d.type}")
        print(f"    body: {body_str}")
        if 'fork' in body_str and 'sync_to_world' in body_str:
            print(f"    *** AGENT TYPE CONSTRUCTOR (belief) ***")
            print(f"        policy runs on a private model, move committed via sync_to_world")
            if shared:
                shared_str = ', '.join(f'{v} (×{c})' for v, c in shared.items())
                print(f"        shared: {shared_str}  — actor AND committer")
        elif 'optimize' in body_str or 'neg_dist' in body_str:
            print(f"    *** desire fragment (goal-directed motion) ***")
        elif 'step' in body_str:
            print(f"    *** physics fragment ***")

    # Structural census of the belief solutions: programs with ints stripped.
    print("\n=== Belief solution shapes ===")
    skeletons = Counter()
    for x, m in all_tasks:
        if m['kind'] != 'belief':
            continue
        k = mat_key(x)
        if k in Z and Z[k] is not None:
            sol = simplify(normalize(deepcopy(Z[k])))
            skeletons[_re.sub(r'\b\d+\b', '_', str(sol))] += 1
    for shape, cnt in skeletons.most_common():
        print(f"  ×{cnt}  {shape}")

    print("\n=== Sample solutions ===")
    kind_seen = Counter()
    for x, m in all_tasks:
        # print all belief solutions (to see shape variants); 2 samples otherwise
        if m['kind'] != 'belief' and kind_seen[m['kind']] >= 2:
            continue
        kind_seen[m['kind']] += 1
        k = mat_key(x)
        tag = {k2: v for k2, v in m.items() if k2 != 'kind'}
        if k in Z and Z[k] is not None:
            sol = simplify(normalize(deepcopy(Z[k])))
            rw  = rewritten.get(k, '')
            print(f"  [{m['kind']}] {tag}")
            print(f"    found:     {sol}")
            if rw:
                print(f"    rewritten: {rw}")
        else:
            print(f"  [{m['kind']}] {tag}  → unsolved")


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)
