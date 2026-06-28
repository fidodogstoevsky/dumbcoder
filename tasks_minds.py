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

This is a library module — the minds-task generators (physics / desire / belief /
witness-belief) imported by the phase drivers.  Run a phase via `python phase1.py`.
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
    compose, step, optimize, neg_distance, wall_at, clear_at,
    unfold, tr, simplify,
)

# ── Configuration ────────────────────────────────────────────────────────────
# (av, gv) combos.  Deliberately diverse: every value in 1,2,4,5,6,7,8,9 (3 = wall,
# 0 = empty are reserved) appears across the set, and several appear in BOTH the
# agent and goal role, so no single literal — and no role-position — dominates the
# corpus.  That is what pushes stitch to keep (gv,av) as HOLES in the seek/policy
# abstractions instead of baking them; with the coord/cellvalue type split (dsl.py)
# this added value-diversity costs nothing at the latent wall-coordinate slot.
COMBOS = [(1, 2), (4, 5), (6, 7), (8, 9), (2, 6), (5, 8), (7, 1), (9, 4)]
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


def _witness_belief_program(av, gv, aw, gw, pr, pc):
    """Per-frame transition for false-belief WITH a non-believing witness:

        (compose (fork (compose (wall_at r c) (optimize (neg_dist gv) av))
                       (sync_to_world av))
                 (optimize (neg_dist gw) aw))

    av acts on a PRIVATE walled copy (the belief); the witness aw seeks gw on the
    real, wall-free grid.  Composed so av's belief-move happens, then the witness
    moves on the committed world.
    """
    return compose(
        fork(compose(wall_at(pr, pc), optimize(neg_distance(gv), av)),
             sync_to_world(av)),
        optimize(neg_distance(gw), aw))


def _witness_rival_explainable(x, g, av, gv, aw, gw, pr, pc):
    """True if any frame-invariant non-mental program reproduces the witnessed
    scene — the transient-wall family (stamp wall / act / erase, in every order),
    the no-wall physics, and the av-only program.  When the witness *traverses*
    the phantom-wall cell these all fail (the unconditional per-frame wall stamp
    clobbers the witness), so a surviving scene is uniquely the private-belief one.
    """
    T = x.shape[0]
    oa = optimize(neg_distance(gv), av)
    ow = optimize(neg_distance(gw), aw)
    W, C = wall_at(pr, pc), clear_at(pr, pc)
    rivals = [
        compose(oa, ow),                                    # no wall (pure physics)
        compose(compose(W, oa), C),                         # transient wall, witness ignored
        compose(compose(compose(W, oa), C), ow),            # stamp/act/erase, then witness
        compose(compose(compose(ow, W), oa), C),            # witness, then stamp/act/erase
        compose(compose(compose(W, ow), oa), C),            # stamp, witness, av, erase
        compose(compose(compose(W, oa), ow), C),            # stamp, av, witness, erase
    ]
    for r in rivals:
        try:
            if np.array_equal(unfold(g, T, r), x):
                return True
        except Exception:
            pass
    return False


def make_witness_belief_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8):
    """False-belief hardened against the transient-wall rival by a witness agent.

    A second agent aw (seeking its own goal gw) traverses the phantom-wall cell on
    the real grid.  Because `unfold` iterates one fixed per-frame fn, any program
    that makes av detour by stamping a *real* wall must stamp it every frame and so
    clobbers the witness as it crosses — only a private-copy `fork` lets av see the
    wall while the witness passes through.  Scenes are kept only if a battery of
    transient/physical rivals all fail, so the private-belief program is the sole
    explanation (cf. why Sally-Anne needs a second observer).
    """
    rng = np.random.default_rng(seed)
    tasks = []
    pool = [1, 2, 4, 5]
    for av, gv in combos:
        rest = [v for v in pool if v not in (av, gv)]
        made, attempts = 0, 0
        while made < n_per_combo and attempts < 40000:
            attempts += 1
            perm = rng.permutation(rest)
            aw, gw = int(perm[0]), int(perm[1])
            allcells = [(r, c) for r in range(size) for c in range(size)]
            idx = rng.permutation(len(allcells))
            (ar, ac), (gr, gc), (wr, wc), (wgr, wgc) = [allcells[i] for i in idx[:4]]
            if abs(ar - gr) + abs(ac - gc) < 3:
                continue
            g = np.zeros((size, size), dtype=int)
            g[ar, ac] = av; g[gr, gc] = gv; g[wr, wc] = aw; g[wgr, wgc] = gw

            # clean single-agent trajectories (agents are transparent to each
            # other's BFS, which blocks only on walls=3) — sources of the wall cell.
            av_clean = unfold(g, max_T, optimize(neg_distance(gv), av))
            aw_clean = unfold(g, max_T, optimize(neg_distance(gw), aw))
            av_cells = {_agent_pos(av_clean[t], av) for t in range(max_T)}
            aw_cells = {_agent_pos(aw_clean[t], aw) for t in range(max_T)}
            occupied = {(ar, ac), (gr, gc), (wr, wc), (wgr, wgc)}
            cand = [p for p in (av_cells & aw_cells)
                    if p and p not in occupied and g[p[0], p[1]] == 0]
            if not cand:
                continue
            pr, pc = cand[int(rng.integers(len(cand)))]

            prog = _witness_belief_program(av, gv, aw, gw, pr, pc)
            x_full = unfold(g, max_T, prog)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (gr, gc)), None)
            if t_arrive is None or t_arrive < 3:
                continue
            T = t_arrive + 1
            x = x_full[:T].copy()
            if np.array_equal(x, av_clean[:T]):                 # av must really detour
                continue
            if not any(_agent_pos(x[t], aw) == (pr, pc) for t in range(T)):
                continue                                         # witness must cross the wall cell
            # all four values present and distinct up to (not incl.) av's arrival
            if not all((x[t] == v).sum() == 1
                       for t in range(T - 1) for v in (av, gv, aw, gw)):
                continue                                         # reject collisions/clobbers
            if _witness_rival_explainable(x, g, av, gv, aw, gw, pr, pc):
                continue                                         # private belief must be unique
            tasks.append((x, {'kind': 'belief', 'av': av, 'gv': gv,
                              'aw': aw, 'gw': gw, 'pw': (pr, pc)}))
            made += 1
    return tasks

