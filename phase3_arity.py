"""File 17: arity generalization — the number of private channels is discovered.

file16's "cube" gives belief's primitives *role* symmetry: for every choice baked
into the two channels (direction / scope / z-order / projection / bifunctor /
pairing / utility / grid-edit) the complementary corner is in the DSL, and joint
MDL still selects exactly belief's corner — so the agency signature is discovered,
not gerrymandered.  But every cube combinator is hardwired to ARITY 2: pair_gg =
(grid, grid), `swap` exchanges "the two" channels, `dup` makes a binary product.
The cube holds the *number of private channels* fixed, so "why one world + one
model, not three?" is still answered by an interpreter commitment (the pair_gg
type), not by search.

This file adds the missing *arity* symmetry.  The fixed pair is replaced by a
single recursive grid-stack (`gstack`, dsl.py) with depth-polymorphic combinators
— `base`/`dup_top`/`blank_top`/`map_top`/`swap_top`/`zip_top`/`commit_top`/`peek`
plus the composers `compose_gs`/`pipe_gsg`.  One type, one combinator set, any
depth.  Belief's private channel is within-step (atomic `fork` is fn = grid->grid;
the model is rederived each frame), so "arity" is the arity of fork's product, and
`fork_stack_decomposed(derive, av) ≡ fork(derive, sync_to_world av)` at depth-1.

ARITY here means the number of grids a program must hold *simultaneously*, minus
one (the output): physics/desire transform in place (arity 0); belief and overlay
hold {world, model} at once (arity 1) — belief's is genuinely non-serializable,
since the sync_to_world commit needs world AND model together.

A subtlety the curriculum exposes: in the single-grid fork/compose calculus,
channel arity is already *unbounded sequentially* — nested forks stack up, but each
model is created and collapsed inside one fork, so they never coexist (max 2 grids
live).  And UNION-style combines (overlay) serialize: w ∪ shift_d1 ∪ shift_d2
(crossblur) looks like it needs three grids, but because each piece is just folded
in, the searcher reproduces it holding only two at a time (arity 1).  So no natural
task here genuinely requires arity ≥ 2 — that would need a non-serializable ternary
combine, which is exotic and unfindable.

The grid-stack's distinct contribution is making *simultaneous* arity an explicit,
unbounded free parameter (a height-3 stack program runs — see verify_ground_truth).
The verdict (A″) is therefore analytical: with arity ≥ 2 freely EXPRESSIBLE, joint
MDL never SELECTS more than one private channel — physics/desire pick 0, belief
picks exactly 1, and even ostensibly two-buffer tasks serialize back to 1.  Belief's
single private model is thus the *generic* non-zero arity (shared with non-mental
overlay/crossblur), discovered as the minimal sufficient channel count rather than
imposed by a pair_gg type.  What distinguishes belief is the commit DIRECTION (the
file16 cube axis), not the channel COUNT.

Run:
    python phase3_arity.py            # full run (generous timeouts; deep stack search)
    python phase3_arity.py --smoke    # tiny corpus, short timeouts
    python phase3_arity.py --no-dream # uniform-Q baseline (no recognition model)

Sleep also DREAMS: between wake-sleep rounds a recognition model is trained on
replays (solved programs) + fantasies (sampled programs), and the next rounds
enumerate under that learned Q instead of the uniform/content prior.
"""

import sys
import re as _re
import math
from collections import Counter
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch as th

from ecd import (
    Deltas, Delta, solve_enumeration, saturate_stitch, mat_key, normalize,
    _worker_init, _n_cpus_available, dream, dreamed_q,
)
from dsl import (
    fn, util, direction, cellvalue, fn_p_g, fn_g_s, fn_s_s, fn_s_g,
    RIGHT, LEFT, UP, DOWN,
    fork, sync_to_world, overlay, compose, step, optimize, neg_distance, wall_at,
    unfold, tr, simplify,
    # stack calculus (the arity generalization)
    base, dup_top, blank_top, map_top, swap_top, zip_top, commit_top, peek,
    compose_gs, pipe_gsg, fork_stack_decomposed,
)
from tasks_minds import (
    make_physics_tasks, make_desire_tasks, make_belief_tasks,
    _physically_explainable, COMBOS, SIZE, DIRS,
)
from tasks_world import make_overlay_tasks
from prims import make_stack_prims

# core parts whose presence in a solution we report (pre-stitch)
_INTERFACE = ('fork', 'sync_to_world', 'overlay', 'optimize', 'wall_at', 'step',
              'base', 'dup_top', 'blank_top', 'map_top', 'swap_top', 'zip_top',
              'commit_top', 'peek')
# stack ops that grow / shrink the live channel set (max simultaneous height = arity)
_PUSH = ('dup_top', 'blank_top')
_POP  = ('commit_top', 'zip_top')


# ── the depth-2 program (built from the same stack combinators the searcher uses) ──

def crossblur_prog(v, d1, d2):
    """fn: w |-> w ∪ shift_d1(w) ∪ shift_d2(w), via the grid-stack at depth-2.

        peek ∘ zip_top(overlay) ∘ zip_top(overlay) ∘ map_top(step d2) ∘ swap_top
             ∘ map_top(step d1) ∘ dup_top ∘ dup_top ∘ base

    Three grids are live at the peak ([s2, s1, w]); w is held in a second private
    channel precisely because shift_d2 needs the clean original.
    """
    prod = compose_gs(compose_gs(compose_gs(compose_gs(compose_gs(compose_gs(compose_gs(
        base, dup_top), dup_top), map_top(step(v, d1))), swap_top),
        map_top(step(v, d2))), zip_top(overlay)), zip_top(overlay))
    return pipe_gsg(prod, peek)


def _low_arity_explainable(x, g, v):
    "True if x is reproduced by a depth-≤1 program (physics, or a one-direction blur)."
    T = x.shape[0]
    if _physically_explainable(x, g):
        return True
    for d in DIRS.values():
        try:
            if np.array_equal(unfold(g, T, fork(step(v, d), overlay)), x):
                return True
        except Exception:
            pass
    return False


def make_crossblur_tasks(n, size=SIZE, vals=(1, 4), seed=0):
    """Two-direction motion trail: w ∪ shift_d1(w) ∪ shift_d2(w), iterated.

    Generated through the grid-stack at height 3 (so a solve failure is search, not
    encoding), and rejected only if reproducible by a depth-0 program or a single-
    direction blur — i.e. it is at least a genuine two-direction effect.  It is NOT
    required to need arity 2: being a union, it serialises, and the searcher is
    expected to solve it at arity 1.  It serves as the arity-2 EXPRESSIBILITY
    witness (the ground-truth height-3 stack program runs) against which the census
    shows MDL still selecting arity 1.
    """
    # non-opposite direction pairs, so the trail is a true 2-D spread
    pairs = [('right', 'down'), ('right', 'up'), ('left', 'down'), ('left', 'up')]
    rng = np.random.default_rng(seed)
    tasks = []
    attempts = 0
    while len(tasks) < n and attempts < 5000:
        attempts += 1
        v = int(rng.choice(vals))
        d1n, d2n = pairs[int(rng.integers(len(pairs)))]
        # start near the corner the trail spreads away from, so it stays mostly on-grid
        r = int(rng.integers(0, 2)); c = int(rng.integers(0, 2))
        T = int(rng.integers(3, 5))
        g = np.zeros((size, size), dtype=int)
        g[r, c] = v
        x = unfold(g, T, crossblur_prog(v, DIRS[d1n], DIRS[d2n]))
        if np.array_equal(x[0], x[-1]):          # must actually spread
            continue
        if _low_arity_explainable(x, g, v):      # must require depth-2
            continue
        tasks.append((x, {'kind': 'crossblur', 'val': v, 'd1': d1n, 'd2': d2n}))
    return tasks


# ── DSL: lean atomic core + the stack calculus (isolates the ARITY axis) ──────────
# Deliberately NOT the file16 cube (that isolates the role axis; the two extensions
# are orthogonal).  Atomic fork/sync stay — they are belief's cheap depth-1 path —
# so the search chooses depth, it isn't forced into the stack.

# the stack DSL itself lives in prims.make_stack_prims (imported above) — the one
# canonical home for primitive sets.


# ── Q tensors (mirror file16 so enumeration cost matches the curriculum) ──────────

def uniform_type_q(D):
    "type-conditioned uniform log-prob: logp[i] = -log(#symbols sharing i's type)"
    q = th.zeros(len(D))
    for _tp, idxs in D.bytype.items():
        lp = -math.log(len(idxs))
        for i in idxs:
            q[i] = lp
    return q


def content_q(D, x):
    "uniform type Q, with integer literals visible in frame 0 boosted to cost 0"
    q = uniform_type_q(D)
    visible = {int(v) for v in np.unique(x[0]) if v not in (0, 3)}
    for d in D.ds:
        # only cellvalue literals are content-priced; coord stays at uniform cost.
        if d.tailtypes is None and d.type == cellvalue and d.head in visible:
            q[D.index(d)] = 0.0
    return q


# ── parallel solve worker (module-level so it pickles); like ecd._solve_one_task
# but with a tunable maxdepth — the stack programs are deeper than the default 10. ──

def _solve_task_md(args):
    x, D, q, timeout, root_type, maxdepth = args
    res = solve_enumeration([x], D, q, {}, maxdepth=maxdepth,
                            timeout=timeout, root_type=root_type)
    return mat_key(x), res.get(mat_key(x))


# ── ground-truth check (every family + the depth-1 stack identity) ────────────────

def verify_ground_truth(D, tasks):
    for x, m in tasks:
        k = m['kind']
        if k == 'physics':
            tree = tr(D, f"(step {m['val']} {m['dir']})")
        elif k == 'desire':
            tree = tr(D, f"(optimize (neg_dist {m['gv']}) {m['av']})")
        elif k == 'overlay':
            tree = tr(D, f"(fork (step {m['val']} {m['dir']}) overlay)")
        elif k == 'crossblur':
            tree = tr(D, f"(pipe_gsg (compose_gs (compose_gs (compose_gs (compose_gs "
                         f"(compose_gs (compose_gs (compose_gs base dup_top) dup_top) "
                         f"(map_top (step {m['val']} {m['d1']}))) swap_top) "
                         f"(map_top (step {m['val']} {m['d2']}))) (zip_top overlay)) "
                         f"(zip_top overlay)) peek)")
        else:  # belief
            pr, pc = m['pw']
            tree = tr(D, f"(fork (compose (wall_at c{pr} c{pc}) "
                         f"(optimize (neg_dist {m['gv']}) {m['av']})) "
                         f"(sync_to_world {m['av']}))")
        out = unfold(x[0], x.shape[0], tree())
        assert np.array_equal(out, x), f"ground truth failed for {k}: {m}"
    print(f"  ground-truth check: {len(tasks)} tasks verified via Delta trees")

    # the decomposition identity: depth-1 stack belief == atomic fork+sync
    g = np.zeros((SIZE, SIZE), dtype=int); g[0, 0] = 1; g[3, 3] = 2
    derive = compose(wall_at(1, 1), optimize(neg_distance(2), 1))
    a = unfold(g, 6, fork(derive, sync_to_world(1)))
    s = unfold(g, 6, fork_stack_decomposed(derive, 1))
    assert np.array_equal(a, s), "fork_stack_decomposed identity broken"
    print("  identity check: fork_stack_decomposed ≡ fork(…, sync_to_world …) at depth-1")


# ── reporting helpers ─────────────────────────────────────────────────────────────

def _normstr(sol):
    return str(simplify(normalize(deepcopy(sol)))) if not isinstance(sol, str) else sol


def _uses(sol):
    s = _normstr(sol)
    return {p for p in _INTERFACE if _re.search(rf'\b{p}\b', s)}


def _arity(sol):
    """private-channel arity = (max grids held SIMULTANEOUSLY) - 1.

    This is the honest depth: not how many channels a program creates over its
    run, but how many it must hold at once.  Atomic `fork`s never coexist (each
    model is created and collapsed within one fork, even when nested or composed),
    so any fork-based program holds at most {world, model} = 2 grids → arity 1;
    physics/desire transform in place → arity 0.  Stack programs hold whatever
    height they reach: the textual token order of the left-nested compose_gs spine
    IS application order (base leftmost), so a single scan gives the peak height.
    """
    s = _normstr(sol)
    if _re.search(r'\bbase\b', s):                      # a grid-stack program
        h = mx = 0
        for t in _re.findall(r'[A-Za-z_]+', s):
            if t == 'base':       h = 1; mx = max(mx, h)
            elif t in _PUSH:      h += 1; mx = max(mx, h)
            elif t in _POP:       h -= 1
        return max(mx - 1, 0)
    return 1 if 'fork' in s else 0                       # atomic: 1 model, or none


def _shared_holes(body_str):
    c = Counter(_re.findall(r'\$\d+', body_str))
    return {v: n for v, n in c.items() if n > 1}


# ── Main ───────────────────────────────────────────────────────────────────────────

def main(smoke=False, dream_on=True):
    if smoke:
        n_phys, n_des, n_ov, n_bel, n_cb = 2, 1, 2, 1, 2
        t_fn, stitch_iters, maxdepth = 20, 3, 11
        rounds, dream_iters = 2, 120
    else:
        n_phys, n_des, n_ov, n_bel, n_cb = 4, 2, 4, 6, 4
        t_fn, stitch_iters, maxdepth = 300, 6, 14
        rounds, dream_iters = 3, 600

    print("Generating mixed corpus…")
    phys = make_physics_tasks(n_phys, seed=0)
    des  = make_desire_tasks(n_des, COMBOS, seed=1)
    ov   = make_overlay_tasks(n_ov, seed=3)
    bel  = make_belief_tasks(n_bel, COMBOS, seed=2)
    cb   = make_crossblur_tasks(n_cb, seed=5)

    fn_tasks = phys + des + ov + bel + cb
    seen, tasks = set(), []
    for x, m in fn_tasks:
        k = mat_key(x)
        if k in seen:
            continue
        seen.add(k)
        tasks.append((x, m))

    by_kind = Counter(m['kind'] for _, m in tasks)
    print(f"  {by_kind['physics']} physics, {by_kind['desire']} desire, "
          f"{by_kind['overlay']} overlay, {by_kind['belief']} belief, "
          f"{by_kind['crossblur']} crossblur — {len(tasks)} total\n")

    D = Deltas(make_stack_prims())
    print(f"DSL: {len(D)} primitives — lean atomic core + grid-stack calculus")
    print("  arity = max grids held at once − 1.  physics/desire: 0 | overlay/belief: 1")
    print("  the stack makes arity ≥ 2 EXPRESSIBLE (free parameter); the run asks whether")
    print("  MDL ever SELECTS it.\n")

    verify_ground_truth(D, tasks)

    # ── solve: wake-sleep over root_type=fn (all families are fn-rooted) ─────────────
    # Like phases 1/2 this is several ECD rounds: enumerate ↦ joint stitch ↦ dream.
    # After each stitch a recognition model is trained on the round's replays (solved
    # programs, rewritten through the learned abstractions) + fantasies, and the next
    # rounds enumerate under that learned Q rather than the uniform/content prior.  A
    # uniform mop-up after DREAM_USE_ROUNDS keeps the search complete (so a model that
    # mis-prioritises a deep stack program can never make it unreachable).
    print("\n" + "=" * 72)
    print(f"SOLVE — wake-sleep over root_type=fn, {rounds} rounds "
          f"(dreaming {'on' if dream_on else 'off'})")
    print("=" * 72)
    sols = {}
    nw = _n_cpus_available()
    DREAM_USE_ROUNDS = 2
    qmodel = None
    all_Xs = [x for x, _ in tasks]
    rewritten = {}
    print(f"  enumerating on {nw} workers (maxdepth={maxdepth}, timeout={t_fn}s)…", flush=True)
    for it in range(1, rounds + 1):
        unsolved = [x for x in all_Xs if mat_key(x) not in sols]
        if not unsolved:
            break
        n_before = len(sols)
        use_model = dream_on and qmodel is not None and it <= 1 + DREAM_USE_ROUNDS
        print(f"\n--- round {it}/{rounds}: {len(unsolved)} unsolved; |D|={len(D)} "
              f"({len(D.invented)} invented); Q={'dreamed' if use_model else 'uniform/content'} ---",
              flush=True)
        with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as pool:
            args = [(x, D, (dreamed_q(qmodel, D, x) if use_model else content_q(D, x)),
                     t_fn, fn, maxdepth) for x in unsolved]
            for k, sol in pool.map(_solve_task_md, args):
                if sol is not None:
                    sols[k] = sol
        print(f"    solved {len(sols)}/{len(tasks)} (+{len(sols) - n_before} this round)", flush=True)

        # joint stitch over ALL solutions; abstractions are registered in D for the
        # next round's enumeration (and reported as-is in (B) below).
        sol_keys = [k for k, v in sols.items() if v is not None]
        _trees, rewritten_strs = saturate_stitch(D, sols, iterations=stitch_iters, max_arity=5)
        rewritten = dict(zip(sol_keys, rewritten_strs))

        if len(sols) == len(tasks):
            print("    all tasks solved — wake-sleep converged.")
            break
        if len(sols) == n_before and it > 1:
            print("    no new tasks solved this round — stopping.")
            break

        # SLEEP-dream: train next round's recognition model on replays + fantasies.
        # Skip on the final round and past the use window (the model would go unused).
        if dream_on and it < rounds and it <= DREAM_USE_ROUNDS:
            replays = []
            for k in sol_keys:
                s = rewritten.get(k)
                if s:
                    try:
                        replays.append(tr(D, s))
                    except Exception:
                        pass
            if not replays:
                replays = [sols[k] for k in sol_keys if sols.get(k) is not None]
            print(f"    dreaming: training recognition Q on {len(replays)} replays "
                  f"+ fantasies ({dream_iters} steps)…", flush=True)
            qmodel = dream(D, replays, training_Xs=all_Xs, root_type=fn, n_iters=dream_iters)

    # ── (A″) ARITY CENSUS — which private-channel arity did MDL select per family? ────
    print("\n" + "=" * 72)
    print("(A″) ARITY CENSUS — private-channel arity SELECTED per family (lower = simpler)")
    print("=" * 72)
    arity_by_kind = {}
    uses_by_kind = {}
    solved = Counter(); total = Counter()
    for x, m in tasks:
        total[m['kind']] += 1
        sol = sols.get(mat_key(x))
        if sol is None:
            continue
        solved[m['kind']] += 1
        arity_by_kind.setdefault(m['kind'], Counter())[_arity(sol)] += 1
        uses_by_kind.setdefault(m['kind'], set()).update(_uses(sol))
    for kind in ('physics', 'desire', 'overlay', 'belief', 'crossblur'):
        ar = arity_by_kind.get(kind)
        astr = ', '.join(f'arity {a}×{n}' for a, n in sorted(ar.items())) if ar else '(none)'
        print(f"  {kind:10s} {solved[kind]}/{total[kind]} solved   {astr}")
        if uses_by_kind.get(kind):
            print(f"             uses: {sorted(uses_by_kind[kind])}")

    def _only_arity(kind, a):
        ar = arity_by_kind.get(kind)
        return bool(ar) and set(ar) == {a}

    # arity-2 EXPRESSIBILITY witness: the height-3 stack program reproduces a crossblur
    # task (ground-truth), so the calculus CAN hold a third grid — yet the searcher
    # solved the same family at arity-1 (it serializes).  Expressible, not selected.
    cb_meta = next((m for _, m in tasks if m['kind'] == 'crossblur'), None)
    if cb_meta is not None:
        stack_str = (f"(pipe_gsg (compose_gs (compose_gs (compose_gs (compose_gs "
                     f"(compose_gs (compose_gs (compose_gs base dup_top) dup_top) "
                     f"(map_top (step {cb_meta['val']} {cb_meta['d1']}))) swap_top) "
                     f"(map_top (step {cb_meta['val']} {cb_meta['d2']}))) (zip_top overlay)) "
                     f"(zip_top overlay)) peek)")
        expressible_arity = _arity(stack_str)
    else:
        expressible_arity = None

    max_selected = max((a for ar in arity_by_kind.values() for a in ar), default=0)
    physics_a0  = _only_arity('physics', 0)
    desire_a0   = _only_arity('desire', 0)
    belief_a1   = _only_arity('belief', 1)
    print(f"\n  physics/desire select arity 0 (no private channel)      : {physics_a0 and desire_a0}")
    print(f"  belief selects arity 1 (one private model)               : {belief_a1}")
    print(f"  max arity SELECTED by any family                         : {max_selected}")
    print(f"  max arity EXPRESSIBLE by the stack (crossblur ground-truth): {expressible_arity}")
    arity_ok = (physics_a0 and desire_a0 and belief_a1
                and max_selected == 1 and (expressible_arity or 0) >= 2)
    if arity_ok:
        print("  => the stack makes arity ≥ 2 a free parameter, yet MDL never selects")
        print("     more than ONE private channel: belief's single private model is the")
        print("     minimal sufficient arity, discovered — not a pair_gg stipulation.")
        print("     (belief shares arity-1 with non-mental overlay/crossblur; what makes")
        print("      it belief is the commit DIRECTION — the cube axis — not the COUNT.)")

    # ── (B) joint compression — the final library learned across the wake-sleep rounds ─
    print("\n" + "=" * 72)
    print(f"(B) JOINT COMPRESSION — final library over all "
          f"{sum(1 for v in sols.values() if v)} solutions (last stitch: iterations={stitch_iters})")
    print("=" * 72)

    print("\n  invented abstractions:")
    agent_constructor = None
    for d in D.invented:
        body = str(simplify(normalize(deepcopy(d))))
        shared = _shared_holes(body)
        argt = ', '.join(str(t) for t in (d.tailtypes or []))
        print(f"    {d.repr}  [{argt}] -> {d.type}")
        print(f"      body: {body}")
        if 'fork' in body and 'sync_to_world' in body and 'wall_at' in body:
            cand = (d, body, shared)
            if agent_constructor is None or len(shared) > len(agent_constructor[2]):
                agent_constructor = cand
            print(f"      *** AGENT TYPE CONSTRUCTOR (belief, depth-1) ***")
            if shared:
                print("          shared holes: "
                      + ', '.join(f'{v} (×{n})' for v, n in shared.items())
                      + "  — actor AND committer")

    # ── verdict ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  (A″) arity ≥2 expressible but never selected; belief = arity 1     : {arity_ok}")
    print(f"  (B)  arity-1 agent constructor extracted by joint stitch          : "
          f"{agent_constructor is not None}")
    if agent_constructor is not None and agent_constructor[2]:
        print(f"       shared-av signature survives                                : True")


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv, dream_on='--no-dream' not in sys.argv)
