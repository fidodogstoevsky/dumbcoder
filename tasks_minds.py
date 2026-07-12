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
    fork, sync_to_world, sync_all, sync_except,
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


# ── Task family 1: goal-displacement false belief (Sally-Anne) ──────────────────
# False belief about an OBJECT'S LOCATION rather than about an obstacle.  The
# agent acts as if the goal sits one cell from where it really is; the true goal
# never moves in the world, so a stationary object is the witness that defeats the
# single-grid rival (a program that genuinely shoves the goal would render the
# goal drifting).  Crucially `move_goal_in_model` is NOT a new primitive — it is
# `(step gv d)`, an ordinary physics fn, sitting in fork's derive slot.  This is
# the same construction the wall-belief generator already rejects as a rival via
# `_displaced_goal_explainable`; here we PROMOTE it to its own belief family.

def _goal_displacement_program(av, gv, ds):
    """Per-frame transition for a displaced-goal false belief:

        (fork (compose (step gv d1) … (step gv dk) (optimize (neg_dist gv) av))
              (sync_to_world av))

    Each frame the agent privately shoves the goal along `ds` (a sequence of one
    or more direction vectors — a stale belief about where gv sits) on a copy of
    the world, seeks the displaced goal on that copy, and commits only its own
    move.  The true goal is never touched in the world.  step + optimize are
    ordinary primitives: the displacement is a COMPOUND, not a `move_goal`
    primitive — and varying its shape across the family (up vs down, one- vs
    two-cell shoves; see _GOAL_CONTENT_SPECS for why only vertical shoves are
    expressible) is what pushes stitch to keep the propositional content a HOLE
    in the belief abstraction instead of baking one particular shove into the
    body.
    """
    return fork(_seq(*([step(gv, d) for d in ds]
                       + [optimize(neg_distance(gv), av)])),
                sync_to_world(av))


def _wall_explainable(x, g, size=SIZE):
    """True if any phantom-WALL belief reproduces x (keeps the goal family distinct).

    Two rival shapes are excluded:
      * detour — stamp a wall and seek a visible value around it (the original check);
      * beacon — stamp a wall and seek THE WALL VALUE 3 itself: a private marker the
        agent simply walks toward.  A stationary displaced goal is extensionally a
        walk to a fixed empty cell, so a beacon one-past the believed cell (or
        anywhere on the greedy ray) reproduces most interior scenes — phase-1 runs
        exploited exactly this to rewrite Sally-Anne tasks through the wall-belief
        token (fn_6 with seek-target 3).  Seek targets therefore include 3, and the
        stamp cell ranges over ALL cells: stamping over the true goal is legal in the
        private model (only av is committed, so the clobber never renders).

    Scenes that survive this check force genuinely goal-shaped propositional content —
    no wall-content spelling of the belief exists for them.
    """
    T = x.shape[0]
    vals = [int(v) for v in np.unique(g) if v != 0]
    for av in vals:
        for gv in dict.fromkeys(vals + [3]):   # 3 = the wall value: the beacon target
            if av == gv:
                continue
            seek = optimize(neg_distance(gv), av)
            for pr in range(size):
                for pc in range(size):
                    prog = fork(compose(wall_at(pr, pc), seek), sync_to_world(av))
                    try:
                        if np.array_equal(unfold(g, T, prog), x):
                            return True
                    except Exception:
                        pass
    return False


# Content specs for the goal-displacement family: the believed goal is the true goal
# shoved along one of these direction-name sequences.  Deliberately varied in BOTH
# direction and subtree size — a two-step shove is (compose (step gv d) (step gv d)),
# a bigger content subtree than a single step — so no one shove recurs often enough
# for stitch to bake it into an abstraction body: the content must stay a hole.
#
# Only VERTICAL shoves appear, and that is extensional, not a choice: `optimize`'s
# greedy tie-break tries vertical neighbours first, so every seek path walks its
# vertical leg first and its horizontal leg last.  A horizontally-displaced believed
# cell therefore always sits on the direct-seek path (the scene is a truncated plain
# desire — rejected as its prefix), and an L-displaced cell is always approached
# horizontally, so a wall-beacon one cell past it reproduces the walk (rejected by
# `_wall_explainable`).  Verified empirically: 0/4000 candidate scenes survive for
# every horizontal or L-shaped spec, at every combo.
_GOAL_CONTENT_SPECS = [('up',), ('down',), ('up', 'up'), ('down', 'down')]


def make_goal_displacement_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8):
    """Sally-Anne: the agent walks to where it *believes* the goal is — displaced
    along a one- or two-step shove from its true position — while the true goal
    sits still.  Each combo cycles through a shuffled `_GOAL_CONTENT_SPECS`, so the
    family covers diverse propositional contents rather than one baked-in shove.

    Necessity (the scene survives only if all hold):
      * the agent settles exactly on the believed (displaced) cell, never on the
        true goal cell — so it is not plain desire (`optimize (neg_dist gv) av`),
        which settles ON the goal;
      * the true goal keeps its value & position in every frame — the stationary
        witness that rules out any program that *actually* moves the goal;
      * no single physical fn reproduces it (`_physically_explainable`);
      * no phantom wall reproduces it (`_wall_explainable`) — including the seek-
        the-wall BEACON rival, so wall-content spellings (fn_6 mimicry) are
        excluded and only genuinely goal-shaped content survives.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in combos:
        order = list(_GOAL_CONTENT_SPECS)
        rng.shuffle(order)
        made, attempts = 0, 0
        spec_i, spec_tries = 0, 0
        while made < n_per_combo and attempts < 40000:
            attempts += 1
            spec_tries += 1
            if spec_tries > 3000:      # this content shape won't generate here; move on
                spec_i, spec_tries = spec_i + 1, 0
            spec = order[spec_i % len(order)]
            vecs = [DIRS[dn] for dn in spec]
            ar, ac = int(rng.integers(size)), int(rng.integers(size))
            gr, gc = int(rng.integers(size)), int(rng.integers(size))
            if (ar, ac) == (gr, gc):
                continue
            # believed (displaced) goal cell: every intermediate shove cell must stay
            # in bounds and clear of the agent, or the model shove clobbers/vanishes
            br, bc, ok = gr, gc, True
            for dr, dc in vecs:
                br, bc = br + dr, bc + dc
                if not (0 <= br < size and 0 <= bc < size) or (br, bc) == (ar, ac):
                    ok = False
                    break
            if not ok:
                continue
            if (br, bc) == (gr, gc):
                continue
            if abs(ar - br) + abs(ac - bc) < 3:       # need a real trajectory to the belief
                continue
            g = np.zeros((size, size), dtype=int)
            g[ar, ac] = av
            g[gr, gc] = gv

            prog = _goal_displacement_program(av, gv, vecs)
            x_full = unfold(g, max_T, prog)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (br, bc)), None)
            if t_arrive is None or t_arrive < 3:
                continue
            T = t_arrive + 1
            x = x_full[:T].copy()
            # true goal must stay put (stationary witness) and never be clobbered
            if any(_agent_pos(x[t], gv) != (gr, gc) for t in range(T)):
                continue
            # agent must never step onto the true goal cell
            if any(_agent_pos(x[t], av) == (gr, gc) for t in range(T)):
                continue
            # must diverge from a plain goal-seek (else it is mere desire)
            direct = unfold(g, T, optimize(neg_distance(gv), av))
            if np.array_equal(x, direct):
                continue
            if _physically_explainable(x, g):
                continue
            if _wall_explainable(x, g, size):
                continue
            tasks.append((x, {'kind': 'belief', 'av': av, 'gv': gv,
                              'displaced_to': (br, bc), 'dirs': spec}))
            made += 1
            spec_i, spec_tries = spec_i + 1, 0
    return tasks


# ── Task family 2: two agents with contradictory false beliefs ──────────────────
# Two agents each detour around their OWN phantom wall on one shared world; the
# two walls never coexist in any rendered frame.  This is the witness trick made
# symmetric: each agent crosses the OTHER's phantom-wall cell, so any single
# per-frame world-stamp that creates one detour clobbers the agent standing on
# the other's cell.  No passive bystander needed — each agent is the other's
# witness, and two private models must be live at once.

def _dual_belief_program(av1, gv1, pw1, av2, gv2, pw2):
    """Per-frame transition for two contradictory false beliefs:

        (compose (fork (compose (wall_at r1 c1) (optimize (neg_dist gv1) av1))
                       (sync_to_world av1))
                 (fork (compose (wall_at r2 c2) (optimize (neg_dist gv2) av2))
                       (sync_to_world av2)))

    Agent1 acts on its own walled copy and commits only av1; then agent2 acts on a
    copy of the resulting world with ITS wall and commits only av2.  Neither wall
    ever appears in the world.
    """
    r1, c1 = pw1
    r2, c2 = pw2
    return compose(
        fork(compose(wall_at(r1, c1), optimize(neg_distance(gv1), av1)),
             sync_to_world(av1)),
        fork(compose(wall_at(r2, c2), optimize(neg_distance(gv2), av2)),
             sync_to_world(av2)))


def _seq(*fs):
    "left-fold compose: _seq(a, b, c)(x) = c(b(a(x)))"
    prog = fs[0]
    for f in fs[1:]:
        prog = compose(prog, f)
    return prog


def _dual_rival_explainable(x, g, av1, gv1, pw1, av2, gv2, pw2):
    """True if any frame-invariant single-grid program reproduces the scene.

    The discriminating rival is `_seq(W1, o1, C1, W2, o2, C2)` — a transient
    schedule where each wall is up only while its own agent moves.  It would match
    were it not that each agent OCCUPIES the other's wall cell: stamping a real
    wall there overwrites that agent (value -> 3), and the cleared cell renders it
    gone, so the witness is lost.  All wall-bearing rivals fail for the same
    reason; the no-wall rival fails because both agents detour.
    """
    T = x.shape[0]
    o1 = optimize(neg_distance(gv1), av1)
    o2 = optimize(neg_distance(gv2), av2)
    W1, C1 = wall_at(*pw1), clear_at(*pw1)
    W2, C2 = wall_at(*pw2), clear_at(*pw2)
    rivals = [
        _seq(o1, o2),                          # no walls (pure physics)
        _seq(W1, W2, o1, o2),                  # both walls permanent
        _seq(W1, W2, o1, o2, C1, C2),          # both walls transient, up for both moves
        _seq(W1, o1, C1, W2, o2, C2),          # each wall up only for its own agent
        _seq(W2, o2, C2, W1, o1, C1),          # reverse order
        _seq(W1, W2, o2, o1, C1, C2),          # acts reordered
    ]
    for r in rivals:
        try:
            if np.array_equal(unfold(g, T, r), x):
                return True
        except Exception:
            pass
    return False


def make_dual_belief_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8):
    """Two agents holding contradictory false beliefs, simultaneously.

    Each phantom wall is placed on its own agent's true-belief path (so it forces
    a detour) AND on a cell the *other* agent traverses (so it acts as the other's
    witness).  Scenes survive only if: both agents detour; each agent really
    occupies the other's wall cell in the realised trajectory; all four values stay
    present & distinct (no clobber); and the single-grid rival battery all fail.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    pool = [1, 2, 4, 5, 6, 7, 8, 9]
    for av1, gv1 in combos:
        rest = [v for v in pool if v not in (av1, gv1)]
        made, attempts = 0, 0
        while made < n_per_combo and attempts < 80000:
            attempts += 1
            perm = rng.permutation(rest)
            av2, gv2 = int(perm[0]), int(perm[1])
            allcells = [(r, c) for r in range(size) for c in range(size)]
            idx = rng.permutation(len(allcells))
            (a1r, a1c), (g1r, g1c), (a2r, a2c), (g2r, g2c) = [allcells[i] for i in idx[:4]]
            if abs(a1r - g1r) + abs(a1c - g1c) < 3:
                continue
            if abs(a2r - g2r) + abs(a2c - g2c) < 3:
                continue
            g = np.zeros((size, size), dtype=int)
            g[a1r, a1c] = av1; g[g1r, g1c] = gv1
            g[a2r, a2c] = av2; g[g2r, g2c] = gv2

            # clean single-agent paths (agents transparent to each other's BFS)
            c1 = unfold(g, max_T, optimize(neg_distance(gv1), av1))
            c2 = unfold(g, max_T, optimize(neg_distance(gv2), av2))
            cells1 = [_agent_pos(c1[t], av1) for t in range(max_T)]
            cells2 = [_agent_pos(c2[t], av2) for t in range(max_T)]
            occ = {(a1r, a1c), (g1r, g1c), (a2r, a2c), (g2r, g2c)}
            # wall on own path AND on the other's path; an empty cell
            cand1 = [p for p in cells1 if p and p in cells2 and p not in occ and g[p[0], p[1]] == 0]
            cand2 = [p for p in cells2 if p and p in cells1 and p not in occ and g[p[0], p[1]] == 0]
            if not cand1 or not cand2:
                continue
            pw1 = cand1[int(rng.integers(len(cand1)))]
            pw2_opts = [p for p in cand2 if p != pw1]
            if not pw2_opts:
                continue
            pw2 = pw2_opts[int(rng.integers(len(pw2_opts)))]

            prog = _dual_belief_program(av1, gv1, pw1, av2, gv2, pw2)
            x_full = unfold(g, max_T, prog)
            t1 = next((t for t in range(max_T) if _agent_pos(x_full[t], av1) == (g1r, g1c)), None)
            t2 = next((t for t in range(max_T) if _agent_pos(x_full[t], av2) == (g2r, g2c)), None)
            if t1 is None or t2 is None:
                continue
            T = max(t1, t2) + 1
            if T < 4 or T > max_T:
                continue
            x = x_full[:T].copy()
            # both must really detour
            if all(_agent_pos(x[t], av1) == cells1[t] for t in range(T)):
                continue
            if all(_agent_pos(x[t], av2) == cells2[t] for t in range(T)):
                continue
            # each agent must occupy the OTHER's wall cell (mutual witnessing)
            if not any(_agent_pos(x[t], av2) == pw1 for t in range(T)):
                continue
            if not any(_agent_pos(x[t], av1) == pw2 for t in range(T)):
                continue
            # all four present & distinct up to (not incl.) the last frame
            if not all((x[t] == v).sum() == 1
                       for t in range(T - 1) for v in (av1, gv1, av2, gv2)):
                continue
            if _dual_rival_explainable(x, g, av1, gv1, pw1, av2, gv2, pw2):
                continue
            tasks.append((x, {'kind': 'belief', 'av': av1, 'gv': gv1, 'pw': pw1,
                              'av2': av2, 'gv2': gv2, 'pw2': pw2}))
            made += 1
    return tasks


# ── non-mental rival SPELLINGS, as priceable strings ────────────────────────────
# The `_*_rival_explainable` batteries above build EXECUTABLE closures to reject scenes
# a non-mental program reproduces; task generation keeps only scenes where they all fail.
# For the MDL-margin experiment we need the same rivals as PRICEABLE s-expressions — the
# non-mental programs a skeptic would offer as "almost as short" as the belief compound.
# These builders mirror the closure batteries one-for-one (keep them in sync), emitting
# base-primitive strings in the same token conventions as `experiment.gt_program_str`
# (coords as c{r}/c{c}, `neg_dist`, directions by name).  Only NON-MENTAL rivals are
# listed: a fork/sync rival would itself be a belief reading, so it cannot bear on the
# "a non-mental program was almost as short" objection.

_DIR_NAME = {v: k for k, v in DIRS.items()}   # (dr,dc) vector -> 'right'/'left'/...


def belief_variant(m):
    "Fine label for a kind='belief' task (wall / witness / goal / dual / false-obstacle)."
    if 'pw2' in m:
        return 'belief_dual'
    if 'real_wall' in m:                 # false about BOTH obstacle and goal; also carries
        return 'belief_false_obstacle'   # displaced_to, so test it BEFORE the goal branch
    if 'displaced_to' in m:
        return 'belief_goal'
    if 'aw' in m:
        return 'belief_witness'
    return 'belief_wall'


def _seek(gv, av):   return f"(optimize (neg_dist {gv}) {av})"
def _wall(pr, pc):   return f"(wall_at c{pr} c{pc})"
def _clear(pr, pc):  return f"(clear_at c{pr} c{pc})"
def _stepstr(v, dn): return f"(step {v} {dn})"


def _seq_str(*ss):
    "left-fold compose to match tasks_minds._seq: _seq_str(a,b,c) = (compose (compose a b) c)"
    prog = ss[0]
    for s in ss[1:]:
        prog = f"(compose {prog} {s})"
    return prog


def belief_rival_specs(m):
    """Non-mental rival spellings for a belief task, as (label, s-expression) pairs.

    Wall/witness/dual rivals are the transient-wall family (stamp / act / erase, in the
    interpreter's one fixed per-frame fn) plus the no-wall physics — the exact programs
    the corresponding `_*_rival_explainable` battery enumerated.  Goal-displacement's
    rival genuinely shoves the goal in the WORLD (no private copy), the spelling the
    stationary-goal witness rules out.  All are priced against the found belief program.
    """
    var = belief_variant(m)
    if var == 'belief_dual':
        av1, gv1, pw1 = m['av'], m['gv'], m['pw']
        av2, gv2, pw2 = m['av2'], m['gv2'], m['pw2']
        o1, o2 = _seek(gv1, av1), _seek(gv2, av2)
        W1, C1 = _wall(*pw1), _clear(*pw1)
        W2, C2 = _wall(*pw2), _clear(*pw2)
        return [
            ('no walls (physics)',        _seq_str(o1, o2)),
            ('both walls permanent',      _seq_str(W1, W2, o1, o2)),
            ('both walls transient',      _seq_str(W1, W2, o1, o2, C1, C2)),
            ('each wall for own agent',   _seq_str(W1, o1, C1, W2, o2, C2)),
            ('reverse order',             _seq_str(W2, o2, C2, W1, o1, C1)),
            ('acts reordered',            _seq_str(W1, W2, o2, o1, C1, C2)),
        ]
    if var == 'belief_witness':
        av, gv, aw, gw, pw = m['av'], m['gv'], m['aw'], m['gw'], m['pw']
        oa, ow = _seek(gv, av), _seek(gw, aw)
        W, C = _wall(*pw), _clear(*pw)
        return [
            ('no wall (physics)',          _seq_str(oa, ow)),
            ('transient, witness ignored', _seq_str(W, oa, C)),
            ('stamp/act/erase then witness', _seq_str(W, oa, C, ow)),
            ('witness then stamp/act/erase', _seq_str(ow, W, oa, C)),
            ('stamp, witness, act, erase', _seq_str(W, ow, oa, C)),
            ('stamp, act, witness, erase', _seq_str(W, oa, ow, C)),
        ]
    if var == 'belief_goal':
        av, gv = m['av'], m['gv']
        shoves = [_stepstr(gv, dn) for dn in m['dirs']]
        return [
            ('pure desire (no belief)',   _seek(gv, av)),
            ('shove goal in world',       _seq_str(*shoves, _seek(gv, av))),
        ]
    if var == 'belief_false_obstacle':
        # Wrong about BOTH the obstacle and the goal, with a REAL wall in the world.
        # Every single-grid rival is expressiveness-excluded: the agent's realised path
        # was computed on a private copy where the real wall was ERASED and a phantom
        # wall stamped, so no world-level program (which must keep the real wall put)
        # reproduces that trajectory — none can be a behavioural competitor.  We spell
        # the closest analogues anyway so the margin table shows they don't reproduce.
        av, gv, dn = m['av'], m['gv'], m['dir']
        Wb, Cb = _wall(*m['pw']), _clear(*m['pw'])            # phantom (believed) wall
        return [
            ('no belief (seek past real wall)', _seek(gv, av)),
            ('shove goal in world',             _seq_str(_stepstr(gv, dn), _seek(gv, av))),
            ('transient phantom + shove goal',
             _seq_str(Wb, _stepstr(gv, dn), _seek(gv, av), Cb)),
        ]
    # belief_wall: transient real wall (the tightest single-grid analogue) + no-wall physics
    av, gv, pw = m['av'], m['gv'], m['pw']
    W, C = _wall(*pw), _clear(*pw)
    return [
        ('no wall (physics)',                _seek(gv, av)),
        ('transient wall (stamp/act/erase)', _seq_str(W, _seek(gv, av), C)),
    ]


# ── Task family: DUAL false belief (obstacle + goal) — forbids the degenerate commit ──
# The single-agent belief scenes above are, on the world, extensionally solved by a
# SCOPE complement (sync_all / sync_except gv) as well as by sync_to_world(av): when
# the only world-value that ends up moved is the agent, "move all-but-goal" == "move
# the agent" and the cheaper complement wins (see the (A) disclosure in experiment.py).
# That is why the invented constructor commits via sync_all and carries no shared av
# hole.  This family removes the degeneracy at its root: the agent is wrong about the
# obstacle's location AND the goal's location, and the obstacle is a REAL wall in the
# world (value 3, a committable world-value).  av's realised detour is therefore bent
# by two UNCOMMITTED relocations of world-values (the wall and the goal), so every
# scope complement must drag at least one of {gv, 3} to its model-position and fails.
# Only sync_to_world(av) — the single-value agency commit whose av is shared with the
# policy — leaves both put.  Solutions here must use the literal commit, so stitch can
# recover the agency signature as a shared hole.

def _false_obstacle_program(av, gv, real_w, bel_w, dg):
    """Per-frame transition for a dual false belief (wrong obstacle + wrong goal):

        (fork (_seq (clear_at Wr) (wall_at Wb) (step gv dg)
                    (optimize (neg_dist gv) av))
              (sync_to_world av))

    On its private copy the agent erases the real wall Wr, stamps the wall where it
    (falsely) believes it to be (Wb), shoves the goal one cell toward where it
    (falsely) believes it sits, then seeks that believed goal around that believed
    wall — committing only its own move.  The world's real wall and goal never move.
    """
    (wr, wc), (br, bc) = real_w, bel_w
    return fork(_seq(clear_at(wr, wc), wall_at(br, bc), step(gv, dg),
                     optimize(neg_distance(gv), av)),
                sync_to_world(av))


def _scope_complements_all_fail(x, g, derive, av, world_vals):
    """True iff sync_to_world(av) reproduces x from g under `derive` but NO scope
    complement does — i.e. the single-value agency commit is the unique commit.
    `world_vals` are the values sync_all / sync_except range over (nonzero cells,
    which here include the real wall 3)."""
    def repro(commit):
        try:
            return np.array_equal(unfold(g, x.shape[0], fork(derive, commit)), x)
        except Exception:
            return False
    if not repro(sync_to_world(av)):        # must be a valid literal-commit scene
        return False
    if repro(sync_all):                     # wholesale adoption must be WRONG
        return False
    for k in set(world_vals):               # every all-but-one commit must be WRONG
        if repro(sync_except(k)):
            return False
    return True


def make_false_obstacle_belief_tasks(n_per_combo, combos=COMBOS, size=SIZE,
                                     seed=0, max_T=8):
    """Single agent, false about BOTH the obstacle and the goal location.

    Forces the literal single-value commit sync_to_world(av): the agent detours
    around a wall it mis-locates while heading for a goal it mis-locates, and both
    the real wall (value 3) and the real goal stay put in the world.  A scene is
    kept only if sync_to_world(av) reproduces it AND every scope complement
    (sync_all, sync_except k for every world value k, incl. the wall) fails — so no
    'move everything but ⋯' commit can undercut the agency commit — and no single
    physical program explains it.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in combos:
        made, attempts = 0, 0
        while made < n_per_combo and attempts < 60000:
            attempts += 1
            ar, ac = int(rng.integers(size)), int(rng.integers(size))
            gr, gc = int(rng.integers(size)), int(rng.integers(size))
            if (ar, ac) == (gr, gc) or abs(ar - gr) + abs(ac - gc) < 3:
                continue
            # believed goal: the real goal shoved one cell in direction dg (on grid,
            # not onto the agent).
            dgname = str(rng.choice(list(DIRS)))
            dg = DIRS[dgname]
            bgr, bgc = gr + dg[0], gc + dg[1]
            if not (0 <= bgr < size and 0 <= bgc < size) or (bgr, bgc) == (ar, ac):
                continue

            g = np.zeros((size, size), dtype=int)
            g[ar, ac] = av
            g[gr, gc] = gv

            # believed-world true-belief path toward the DISPLACED goal — source of
            # the phantom-wall cell (place the believed wall on it to force a detour).
            bg = g.copy()
            bg[gr, gc] = 0
            bg[bgr, bgc] = gv
            direct = unfold(bg, max_T, optimize(neg_distance(gv), av))
            bpath = [_agent_pos(direct[t], av) for t in range(max_T)]
            inter = [p for p in bpath
                     if p and p not in ((ar, ac), (gr, gc), (bgr, bgc))]
            if not inter:
                continue
            br, bc = inter[int(rng.integers(len(inter)))]      # believed wall

            # real wall: an empty cell distinct from the believed wall and not on the
            # agent's realised path (checked below via the distinctness gate).
            free = [(r, c) for r in range(size) for c in range(size)
                    if g[r, c] == 0 and (r, c) != (br, bc)]
            if not free:
                continue
            wr, wc = free[int(rng.integers(len(free)))]
            g[wr, wc] = 3                                       # REAL wall in the world

            derive = _seq(clear_at(wr, wc), wall_at(br, bc), step(gv, dg),
                          optimize(neg_distance(gv), av))
            prog = fork(derive, sync_to_world(av))
            x_full = unfold(g, max_T, prog)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (bgr, bgc)), None)
            if t_arrive is None or t_arrive < 3:
                continue
            T = t_arrive + 1
            x = x_full[:T].copy()

            # the wall and goal must stay put in the world for every frame, and all
            # four world-values (av, gv, real wall, and — until arrival — the cell the
            # agent came from) stay present & distinct: no clobber, no accidental move.
            if any(x[t][gr, gc] != gv or x[t][wr, wc] != 3 for t in range(T)):
                continue
            if not all((x[t] == v).sum() == 1 for t in range(T) for v in (av, gv)):
                continue
            if not all((x[t] == 3).sum() == 1 for t in range(T)):
                continue
            # the agent must really detour (not the clean displaced-goal path)
            if [_agent_pos(x[t], av) for t in range(T)] == bpath[:T]:
                continue
            if _physically_explainable(x, g):                  # not a bare step/seek
                continue
            world_vals = [int(v) for v in np.unique(g) if v != 0]
            if not _scope_complements_all_fail(x, g, derive, av, world_vals):
                continue                                        # literal commit unique
            tasks.append((x, {'kind': 'belief', 'av': av, 'gv': gv,
                              'pw': (br, bc), 'real_wall': (wr, wc),
                              'displaced_to': (bgr, bgc), 'dir': dgname}))
            made += 1
    return tasks

