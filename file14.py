"""File 14: belief's parts earn their place on non-mental tasks.

file13 makes belief the compound `fork(policy, sync_to_world av)` — never a
primitive.  But a compound is only honest if its parts have *independent
extension*: otherwise `fork`/`sync` could just be a `believe` primitive split
into two gears that only ever re-mesh into `believe` ("reverse-engineered, then
rediscovered").  The defence is to show the gears turn other machines.

This file gives `fork` and `sync` non-mental work, then shows belief reusing them:

    fork without sync  — a value leaves a trail (motion blur):
        (fork (step v d) overlay)
      output depends on BOTH the grid and its shift, so fork is *required*, and
      the commit is `overlay`, not `sync` — no mind in sight.

    sync without fork  — snap an object onto an external template (registration):
        (sync_to_world v)            applied to (working, template)
      the pair is two *given* grids, NOT a privately derived model.  sync just
      transfers v's position from the template — a coordinate join, not a belief.

    belief             — recombination of the above, file13's compound:
        (fork (compose (wall_at r c) (optimize (neg_dist gv) av)) (sync_to_world av))

`fork`/`sync`/`overlay` meet only through the generic pair interface
(`pair_gg` / `fn_p_g`), which is independently populated: pairs are *produced*
by fork AND by unfold_with_template; *consumed* by sync AND by overlay.  A busy
interface is the signature of a general calculus that belief merely traverses.

Run:
    python file14.py            # full run
    python file14.py --smoke    # tiny corpus, short timeouts
"""

import sys
import numpy as np

from ecd import Deltas, Delta, mat_key, solve_enumeration, ECD, normalize
from dsl import (
    fn, util, direction, fn_p_g,
    RIGHT, LEFT, UP, DOWN,
    fork, sync_to_world, overlay, then_sync,
    compose, step, optimize, neg_distance, wall_at,
    unfold, unfold_with_template, tr,
)
from file13 import make_belief_tasks, _physically_explainable

# ── Configuration (shared with file11/12/13) ─────────────────────────────────────
COMBOS = [(1, 2), (1, 5), (4, 2), (4, 5)]
SIZE   = 5
DIRS   = {'right': RIGHT, 'left': LEFT, 'up': UP, 'down': DOWN}


# ── Non-mental task generators ───────────────────────────────────────────────────
# Both generate through the same combinators/interpreters the searcher uses, so a
# solvability failure is always search, never encoding (file13's discipline).

def make_overlay_tasks(n, size=SIZE, vals=(1, 4), seed=0, max_T=6):
    """fork without sync: a value leaves a trail.

        (fork (step v d) overlay)

    Each frame overlays the grid with its one-step shift, so the output depends on
    BOTH the grid and the transform — no single non-branching fn (step/optimize)
    reproduces it, hence fork is *necessary*.  The commit is `overlay`, not `sync`.
    Pure motion blur / trail rendering: nothing mental.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    while len(tasks) < n:
        v = int(rng.choice(vals))
        dname = str(rng.choice(list(DIRS)))
        dr, dc = DIRS[dname]
        T = int(rng.integers(3, max_T))
        # keep the whole trail on-grid
        r_lo, r_hi = max(0, -dr * (T - 1)), min(size - 1, size - 1 - dr * (T - 1))
        c_lo, c_hi = max(0, -dc * (T - 1)), min(size - 1, size - 1 - dc * (T - 1))
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r = int(rng.integers(r_lo, r_hi + 1))
        c = int(rng.integers(c_lo, c_hi + 1))
        g = np.zeros((size, size), dtype=int)
        g[r, c] = v

        x = unfold(g, T, fork(step(v, DIRS[dname]), overlay))
        if _physically_explainable(x, g):       # fork must be required
            continue
        tasks.append((x, {'kind': 'overlay', 'val': v, 'dir': dname}))
    return tasks


def make_registration_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, n_distract=2):
    """sync without fork: snap a misplaced object onto an external template.

        (sync_to_world v)            applied per frame to (working, template)

    The pair is two *given* grids — working + template — paired by
    unfold_with_template, NOT by fork.  Distractor values already sit in their
    template cells, so step/optimize (which cannot read the template) can't solve
    it: sync is *necessary*, and the second grid is a spec, not a mind.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    while len(tasks) < n:
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        if len(cells) < n_distract + 2:
            continue
        perm = rng.permutation(vals).tolist()
        v, distract_vals = perm[0], perm[1:1 + n_distract]
        src, tgt = cells[0], cells[1]
        if src == tgt:
            continue

        working  = np.zeros((size, size), dtype=int)
        template = np.zeros((size, size), dtype=int)
        working[src]  = v                          # v is misplaced…
        template[tgt] = v                          # …it belongs here, per template
        for dv, dcell in zip(distract_vals, cells[2:2 + n_distract]):
            working[dcell] = template[dcell] = dv  # distractors already aligned

        x = unfold_with_template(working, template, 2, sync_to_world(v))
        if np.array_equal(x[0], x[-1]):            # v must actually move
            continue
        tasks.append((x, {'kind': 'registration', 'val': v, 'template': template}))
    return tasks


# ── DSL ──────────────────────────────────────────────────────────────────────────
# file13's primitives, plus `overlay` as a second inhabitant of fn_p_g and
# `then_sync` so multi-value registration is expressible.  Crucially fork/sync/
# overlay are all *given* core primitives — what the searcher discovers is which
# of them each task family reaches for.

def make_core_prims():
    return [
        # Pair interface: fork produces pairs; fn_p_g consumes them.
        Delta(fork,          fn,     [fn, fn_p_g],   repr='fork'),
        Delta(sync_to_world, fn_p_g, [int],          repr='sync_to_world'),
        Delta(overlay,       fn_p_g,                 repr='overlay'),
        Delta(then_sync,     fn_p_g, [fn_p_g, int],  repr='then_sync'),

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


def verify_ground_truth(D, tasks, kind):
    """Re-express each task's ground truth as a Delta tree (the searcher's own
    encoding) through the matching interpreter — catches encoding/typing bugs."""
    for x, m in tasks:
        if kind == 'overlay':
            tree = tr(D, f"(fork (step {m['val']} {m['dir']}) overlay)")
            out  = unfold(x[0], x.shape[0], tree())
        elif kind == 'registration':
            tree = tr(D, f"(sync_to_world {m['val']})")
            out  = unfold_with_template(x[0], m['template'], x.shape[0], tree())
        else:  # belief
            pr, pc = m['pw']
            tree = tr(D, f"(fork (compose (wall_at {pr} {pc}) "
                        f"(optimize (neg_dist {m['gv']}) {m['av']})) "
                        f"(sync_to_world {m['av']}))")
            out  = unfold(x[0], x.shape[0], tree())
        assert np.array_equal(out, x), f"ground truth failed for {kind}: {m}"
    print(f"  ground-truth check: {len(tasks)} {kind} tasks verified via Delta trees")


# ── fn_p_g search (registration runs through the template interpreter) ────────────

def _uniform_q(D):
    "type-conditioned uniform log-prob tensor, mirroring ECD's first-iteration Q"
    import math
    import torch as th
    q = th.zeros(len(D))
    for _tp, idxs in D.bytype.items():
        lp = -math.log(len(idxs))
        for i in idxs:
            q[i] = lp
    return q


def solve_registration(tasks, D, timeout=30):
    "Enumerate fn_p_g programs against (working, template) pairs via solve_enumeration."
    Xs        = [x for x, _ in tasks]
    templates = {mat_key(x): m['template'] for x, m in tasks}
    sols = solve_enumeration(Xs, D, _uniform_q(D), {}, timeout=timeout,
                             root_type=fn_p_g, templates=templates)
    return sols


# ── Reporting helpers ────────────────────────────────────────────────────────────

def _uses(sol_str):
    "which interface primitives a solution s-expression reaches for"
    return {p for p in ('fork', 'sync_to_world', 'overlay', 'then_sync',
                        'optimize', 'wall_at', 'step') if p in sol_str}


def _report(title, tasks, sols):
    print(f"\n  {title}")
    n = 0
    seen_shapes = set()
    for x, m in tasks:
        k = mat_key(x)
        sol = sols.get(k)
        if sol is None:
            continue
        n += 1
        s = str(normalize(sol)) if not isinstance(sol, str) else sol
        tag = {kk: vv for kk, vv in m.items() if kk not in ('kind', 'template')}
        if str(_uses(s)) not in seen_shapes:
            seen_shapes.add(str(_uses(s)))
            print(f"    {tag}")
            print(f"      {s}")
            print(f"      uses: {sorted(_uses(s))}")
    print(f"    solved {n}/{len(tasks)}")
    # union of primitives used across all solved tasks of this family
    used = set()
    for x, _ in tasks:
        sol = sols.get(mat_key(x))
        if sol is not None:
            s = str(normalize(sol)) if not isinstance(sol, str) else sol
            used |= _uses(s)
    return used


# ── Main ──────────────────────────────────────────────────────────────────────────

def main(smoke=False):
    n_ov   = 2 if smoke else 6
    n_reg  = 2 if smoke else 6
    n_bel  = 1 if smoke else 4   # per combo
    t_fn   = 20 if smoke else 120
    t_pair = 10 if smoke else 30
    iters  = 2  if smoke else 6

    print("Generating tasks…")
    overlay_tasks = make_overlay_tasks(n_ov, seed=0)
    reg_tasks     = make_registration_tasks(n_reg, seed=1)
    bel_tasks     = make_belief_tasks(n_bel, COMBOS, seed=2)
    print(f"  {len(overlay_tasks)} overlay, {len(reg_tasks)} registration, "
          f"{len(bel_tasks)} belief\n")

    D = Deltas(make_core_prims())
    print(f"DSL: {len(D)} primitives "
          f"(fork, sync_to_world, overlay, then_sync all given as core)")
    verify_ground_truth(D, overlay_tasks, 'overlay')
    verify_ground_truth(D, reg_tasks, 'registration')
    verify_ground_truth(D, bel_tasks, 'belief')

    # ── Phase A: learn fork from MINDS-FREE data ─────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE A — search the overlay tasks (no minds in the corpus)")
    print("=" * 72)
    ZA, _ = ECD([x for x, _ in overlay_tasks], Deltas(make_core_prims()),
                per_task_timeout=t_fn, max_iterations=iters, max_arity=5,
                stitch_iterations=2, root_type=fn, run_dream=False)
    used_A = _report("overlay solutions:", overlay_tasks, ZA)

    # ── Phase B: attest sync on a non-mental two-grid task ───────────────────────
    print("\n" + "=" * 72)
    print("PHASE B — search the registration tasks (sync, pair from two given grids)")
    print("=" * 72)
    ZB = solve_registration(reg_tasks, Deltas(make_core_prims()), timeout=t_pair)
    used_B = _report("registration solutions:", reg_tasks, ZB)

    # ── Phase C: belief is recombination of the already-attested parts ───────────
    print("\n" + "=" * 72)
    print("PHASE C — search the belief tasks (file13's compound)")
    print("=" * 72)
    ZC, _ = ECD([x for x, _ in bel_tasks], Deltas(make_core_prims()),
                per_task_timeout=t_fn, max_iterations=iters, max_arity=5,
                stitch_iterations=4, root_type=fn, run_dream=False)
    used_C = _report("belief solutions:", bel_tasks, ZC)

    # ── Verdict ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICT — did belief's parts earn their place independently?")
    print("=" * 72)
    print(f"  fork attested on non-mental overlay tasks : {'fork' in used_A}")
    print(f"  sync attested on non-mental registration  : {'sync_to_world' in used_B}")
    print(f"  belief reuses both fork AND sync          : "
          f"{'fork' in used_C and 'sync_to_world' in used_C}")
    print(f"\n  overlay used : {sorted(used_A)}")
    print(f"  registration : {sorted(used_B)}")
    print(f"  belief       : {sorted(used_C)}")
    if 'fork' in used_A and 'sync_to_world' in used_B and {'fork', 'sync_to_world'} <= used_C:
        print("\n  => fork and sync each do non-mental work; belief is their recombination,")
        print("     not a believe-primitive decomposed and rediscovered.")
    else:
        print("\n  => not fully demonstrated this run (try without --smoke, or raise timeouts).")


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)
