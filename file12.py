"""File 12: monomorphic probe of pair-state synthesis.

file11 gated whether a program uses a model channel (belief vs physics) but
kept the *existence* of the channel as a type-system commitment:
  unfold_state(g, T, sf)   -- state = (g, copy(g)) is hardcoded
  sf :: pair_gg -> pair_gg

Here the (init, step, render) triple lives in program space.  Two monomorphic
state shapes are exposed:

    mk_machine_g  :: fn, fn, fn               -> machine    (s = grid)
    mk_machine_gg :: fn_g_p, sfn, fn_p_g      -> machine    (s = pair_gg)

with init/render adapters:

    id_fn    :: fn       (grid -> grid identity)
    dup_g    :: fn_g_p   (grid -> pair_gg; model = copy(world))
    fst_gg   :: fn_p_g   (pair_gg -> grid; render world)
    snd_gg   :: fn_p_g   (pair_gg -> grid; render model)

and the same pair-state combinators as file11 (compose_s, on_world / on_model,
sync_w), now re-typed as sfn = pair_gg -> pair_gg with no special interpreter
support.  The interpreter shrinks to:

    unfold_m(g, T, m)   -- (kind, init, step, render) = m
                           s = init(g); render each frame of step^k(s)

Nothing privileges the pair_gg machine.  For grid-state tasks (physics,
desire) the cheapest program picks mk_machine_g with id_fn init/render; for
belief tasks the only viable program picks mk_machine_gg with dup_g/fst_gg
and routes behaviour through sync_w.

Expected costs (content-aware Q boosts av, gv visible in frame 0):

    physics : (mk_machine_g id_fn (step v d) id_fn)                          5 nodes
    desire  : (mk_machine_g id_fn (optimize (neg_dist gv) av) id_fn)         7 nodes
    belief  : (mk_machine_gg dup_g
                 (compose_s (on_model (compose (wall_at r c)
                                                (optimize (neg_dist gv) av)))
                            (sync_w av))
                 fst_gg)                                                    ~15 nodes

Belief is ~3 nodes (≈4 nats) more expensive than file11 — the question this
probe answers is whether enumeration can still reach that ladder when nothing
privileges the pair-state machine over the grid-state one.

Run:
    python file12.py            # full run
    python file12.py --smoke    # tiny corpus, 10s timeouts
"""

import sys
import re as _re
from collections import Counter
from copy import deepcopy

import numpy as np

from ecd import Deltas, Delta, ECD, normalize, mat_key
from dsl import (
    sfn, fn, util, direction,
    machine, fn_g_p, fn_p_g,
    RIGHT, LEFT, UP, DOWN,
    compose_s, on_world, on_model, sync_w, wall_at,
    compose, step, optimize, neg_distance, id_fn,
    mk_machine_g, mk_machine_gg, dup_g, fst_gg, snd_gg,
    unfold_m, tr,
)

# ── Configuration (same combos as file11) ──────────────────────────────────────
COMBOS = [(1, 2), (1, 5), (4, 2), (4, 5)]
SIZE   = 5
DIRS   = {'right': RIGHT, 'left': LEFT, 'up': UP, 'down': DOWN}


# ── Task generation ────────────────────────────────────────────────────────────
# Every task is generated through unfold_m, so the searcher and the generator
# share the same interpreter and any solvability failure is search, never
# encoding.

def _agent_pos(frame, av):
    pos = np.argwhere(frame == av)
    return (int(pos[0][0]), int(pos[0][1])) if len(pos) else None


def _physically_explainable(x, g):
    "True if x is reproduced by a single grid-state machine using step or optimize."
    T = x.shape[0]
    vals = [int(v) for v in np.unique(g) if v != 0]
    for v in vals:
        for d in DIRS.values():
            try:
                m = mk_machine_g(id_fn, step(v, d), id_fn)
                if np.array_equal(unfold_m(g, T, m), x):
                    return True
            except Exception:
                pass
        for u in vals:
            if u == v:
                continue
            try:
                m = mk_machine_g(id_fn, optimize(neg_distance(u), v), id_fn)
                if np.array_equal(unfold_m(g, T, m), x):
                    return True
            except Exception:
                pass
    return False


def make_physics_tasks(n, size=SIZE, vals=(1, 4), seed=0):
    "Ground truth: mk_machine_g id_fn (step v d) id_fn."
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
        m = mk_machine_g(id_fn, step(v, DIRS[dname]), id_fn)
        x = unfold_m(g, T, m)
        tasks.append((x, {'kind': 'physics', 'val': v, 'dir': dname}))
    return tasks


def make_desire_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0):
    "Ground truth: mk_machine_g id_fn (optimize (neg_dist gv) av) id_fn."
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
            m = mk_machine_g(id_fn, optimize(neg_distance(gv), av), id_fn)
            x = unfold_m(g, T, m)
            if x[-1][gr, gc] != av:
                continue
            tasks.append((x, {'kind': 'desire', 'av': av, 'gv': gv}))
            made += 1
    return tasks


def make_belief_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8):
    """False-belief: ground truth is a pair_gg machine with phantom wall on model.

      (mk_machine_gg dup_g
         (compose_s (on_model (compose (wall_at pr pc) (optimize (neg_dist gv) av)))
                    (sync_w av))
         fst_gg)
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

            direct_m = mk_machine_g(id_fn, optimize(neg_distance(gv), av), id_fn)
            direct   = unfold_m(g, max_T, direct_m)
            path  = [_agent_pos(direct[t], av) for t in range(max_T)]
            inter = [p for p in path if p and p != (ar, ac) and p != (gr, gc)]
            if not inter:
                continue
            pr, pc = inter[int(rng.integers(len(inter)))]

            gt_step = compose_s(
                on_model(compose(wall_at(pr, pc), optimize(neg_distance(gv), av))),
                sync_w(av))
            gt = mk_machine_gg(dup_g, gt_step, fst_gg)
            x_full = unfold_m(g, max_T, gt)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (gr, gc)), None)
            if t_arrive is None or t_arrive < 3:
                continue
            T = t_arrive + 1
            x = x_full[:T].copy()
            if np.array_equal(x, direct[:T]):
                continue
            if _physically_explainable(x, g):
                continue
            tasks.append((x, {'kind': 'belief', 'av': av, 'gv': gv, 'pw': (pr, pc)}))
            made += 1
    return tasks


# ── DSL ────────────────────────────────────────────────────────────────────────

def make_core_prims():
    return [
        # Machine layer: the (init, step, render) bundles
        Delta(mk_machine_g,  machine, [fn, fn, fn],          repr='mk_machine_g'),
        Delta(mk_machine_gg, machine, [fn_g_p, sfn, fn_p_g], repr='mk_machine_gg'),

        # Init and render adapters
        Delta(dup_g,  fn_g_p, repr='dup_g'),
        Delta(fst_gg, fn_p_g, repr='fst_gg'),
        Delta(snd_gg, fn_p_g, repr='snd_gg'),

        # Pair-state combinators (sfn = pair_gg -> pair_gg)
        Delta(compose_s, sfn, [sfn, sfn], repr='compose_s'),
        Delta(on_world,  sfn, [fn],       repr='on_world'),
        Delta(on_model,  sfn, [fn],       repr='on_model'),
        Delta(sync_w,    sfn, [int],      repr='sync_w'),

        # Grid-state primitives (fn = grid -> grid)
        Delta(compose,      fn,   [fn, fn],          repr='compose'),
        Delta(step,         fn,   [int, direction],  repr='step'),
        Delta(optimize,     fn,   [util, int],       repr='optimize'),
        Delta(wall_at,      fn,   [int, int],        repr='wall_at'),
        Delta(id_fn,        fn,                      repr='id_fn'),

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
    for x, m in tasks_meta:
        if m['kind'] == 'physics':
            prog = f"(mk_machine_g id_fn (step {m['val']} {m['dir']}) id_fn)"
        elif m['kind'] == 'desire':
            prog = (f"(mk_machine_g id_fn "
                    f"(optimize (neg_dist {m['gv']}) {m['av']}) id_fn)")
        else:
            pr, pc = m['pw']
            prog = (f"(mk_machine_gg dup_g "
                    f"(compose_s (on_model (compose (wall_at {pr} {pc}) "
                    f"(optimize (neg_dist {m['gv']}) {m['av']}))) "
                    f"(sync_w {m['av']})) "
                    f"fst_gg)")
        tree = tr(D, prog)
        out  = unfold_m(x[0], x.shape[0], tree())
        assert np.array_equal(out, x), f"ground truth failed for {m}: {prog}"
    print(f"ground-truth check: {len(tasks_meta)} tasks verified via Delta trees")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(smoke=False):
    n_phys = 2 if smoke else 6
    n_des  = 1 if smoke else 2   # per combo
    # belief is +~1.4 nats vs file11 — the explicit (mk_machine_gg dup_g _ fst_gg)
    # envelope adds ~e^1.4 ≈ 4x enumeration effort.  Longer per-task budget gives
    # belief a fighting chance in iter 1; once stitch extracts an envelope
    # abstraction (mk_machine_gg dup_g $0 fst_gg) belief cost should drop back
    # to file11 levels in subsequent iterations.
    n_bel  = 1 if smoke else 6   # per combo
    per_task_timeout = 10 if smoke else 1800
    max_iterations   = 2  if smoke else 10

    print("Generating tasks…")
    phys = make_physics_tasks(n_phys, seed=0)
    des  = make_desire_tasks(n_des, COMBOS, seed=1)
    bel  = make_belief_tasks(n_bel, COMBOS, seed=2)

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
    print("  expected physics solution (5 nodes):")
    print("    (mk_machine_g id_fn (step v d) id_fn)")
    print("  expected desire solution  (7 nodes):")
    print("    (mk_machine_g id_fn (optimize (neg_dist gv) av) id_fn)")
    print("  expected belief solution (~15 nodes):")
    print("    (mk_machine_gg dup_g")
    print("       (compose_s (on_model (compose (wall_at r c)")
    print("                                     (optimize (neg_dist gv) av)))")
    print("                  (sync_w av))")
    print("       fst_gg)\n")

    verify_ground_truth(D, all_tasks)

    print("\nRunning ECD…\n")
    Z, rewritten = ECD(
        Xs, D,
        per_task_timeout=per_task_timeout,
        max_iterations=max_iterations,
        max_arity=5,
        stitch_iterations=4,
        root_type=machine,
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n=== Results ===")
    for kind in ('physics', 'desire', 'belief'):
        ks = [x for x, m in all_tasks if m['kind'] == kind]
        n  = sum(1 for x in ks if mat_key(x) in Z and Z[mat_key(x)] is not None)
        print(f"  {kind:8s}: {n}/{len(ks)}")

    # The headline result of the probe: did each task class land in the
    # state-space the design predicted?  Physics + desire should pick
    # mk_machine_g; belief should pick mk_machine_gg.
    print("\n=== Solution machine kinds ===")
    kinds = Counter()
    for x, m in all_tasks:
        k = mat_key(x)
        if k in Z and Z[k] is not None:
            s = str(normalize(deepcopy(Z[k])))
            if   'mk_machine_gg' in s: kinds[(m['kind'], 'pair_gg')] += 1
            elif 'mk_machine_g'  in s: kinds[(m['kind'], 'grid'   )] += 1
            else:                       kinds[(m['kind'], 'other'  )] += 1
    for (task_kind, mach_kind), cnt in sorted(kinds.items()):
        print(f"  {task_kind:8s} via {mach_kind:8s}  ×{cnt}")

    print("\n=== Invented primitives ===")
    if not D.invented:
        print("  (none)")
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        print(f"  {d.repr}  [{argtypes}]  -> {d.type}")
        print(f"    body: {body_str}")
        if 'mk_machine_gg' in body_str and 'sync_w' in body_str:
            print(f"    *** AGENT MACHINE CONSTRUCTOR (pair-state, model channel) ***")
            if shared:
                shared_str = ', '.join(f'{v} (×{c})' for v, c in shared.items())
                print(f"        shared: {shared_str}")
        elif 'mk_machine_g' in body_str and 'mk_machine_gg' not in body_str:
            print(f"    *** OBJECT MACHINE CONSTRUCTOR (grid-state) ***")

    print("\n=== Belief solution shapes ===")
    skeletons = Counter()
    for x, m in all_tasks:
        if m['kind'] != 'belief':
            continue
        k = mat_key(x)
        if k in Z and Z[k] is not None:
            sol = normalize(deepcopy(Z[k]))
            skeletons[_re.sub(r'\b\d+\b', '_', str(sol))] += 1
    for shape, cnt in skeletons.most_common():
        print(f"  ×{cnt}  {shape}")

    print("\n=== Sample solutions ===")
    kind_seen = Counter()
    for x, m in all_tasks:
        if m['kind'] != 'belief' and kind_seen[m['kind']] >= 2:
            continue
        kind_seen[m['kind']] += 1
        k = mat_key(x)
        tag = {k2: v for k2, v in m.items() if k2 != 'kind'}
        if k in Z and Z[k] is not None:
            sol = normalize(deepcopy(Z[k]))
            rw  = rewritten.get(k, '')
            print(f"  [{m['kind']}] {tag}")
            print(f"    found:     {sol}")
            if rw:
                print(f"    rewritten: {rw}")
        else:
            print(f"  [{m['kind']}] {tag}  → unsolved")


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)
