"""Task-generator library: one undifferentiated bag of tasks.

The interpreter threads a single grid and renders every frame:

    unfold(g, T, f)    f :: fn = grid -> grid

An agent's private model is never interpreter state and never a primitive. It is
introduced locally in program space by general, non-mental combinators and
collapsed back to one grid in the same step:

    fork(derive, commit) :: fn, fn_p_g -> fn      w |-> commit((w, derive(w)))
    sync_to_world(v)     :: int      -> fn_p_g     move v in world to its pos in derived

Neither primitive denotes a mental state on its own. Belief is the *composition* --
and that composition is what stitch is meant to extract as the agent-type
constructor. A `believe(...)` primitive would have been intensional by
construction; the whole point is that belief stays a compound of general parts.

The generators here are the whole corpus, distinguished only by each task's
individual `kind` -- never by any coarse world/mind or scaffold/non-scaffold
label. Physics and desire are the shallow rungs whose programs are literally
sub-terms of the belief compound (compositional structure, not a category
boundary); the fork/sync parts also earn independent extension on non-mental
tasks (overlay trails, registration, and one family per symmetric cube corner) so
the decomposition is honest rather than a `believe` primitive split into two gears
that only ever re-mesh. Which tasks end up *serving* as scaffold for discovering
belief is read off a run's results, not stipulated here.

Expected solutions (one shared DSL across the curriculum):

  physics  (3 nodes):  (step v d)
  desire   (4 nodes):  (optimize (neg_dist gv) av)
  belief  (11 nodes):  (fork (compose (wall_at r c) (optimize (neg_dist gv) av))
                             (sync_to_world av))

av appears twice -- in optimize (who acts on the model) and in sync_to_world
(whose move is committed to the world) -- the structural signature of agency. The
hoped-for stitch discovery is the agent-type constructor:

  fn_agent($r, $c, $gv, $av) =
    (fork (compose (wall_at $r $c) (optimize (neg_dist $gv) $av))
          (sync_to_world $av))

Run a phase via `python phase1.py`.
"""

import numpy as np

from dsl import (
    fn, direction, fn_p_g,
    RIGHT, LEFT, UP, DOWN,
    fork, sync_to_world, sync_all, sync_except,
    compose, step, optimize, neg_distance, distance, wall_at, clear_at, erase,
    unfold, unfold_with_template, tr,
    overlay, underlay, fst_gg, snd_gg, sync_to_model,   # non-mental fork/pair families
    dup, bimap, compose_gp, pipe_gpg,   # decomposed-fork plumbing (world-channel rival sweep)
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
    """True if x is reproduced by a single physical fn: step, seek (optimize
    neg_dist), or FLEE (optimize dist).  The flee arm matters because `optimize
    (distance u) v` walking away from a value is extensionally a plain physical
    move; on the all-vertical goal-content specs it can coincide with an up/down
    shove (fleeing the true goal ≡ walking to the displaced cell), and the Jul-12
    logs caught exactly such flee-content programs as bare non-mental solves."""
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
            for util_fn in (neg_distance, distance):
                if u == v and util_fn is neg_distance:
                    continue                      # seeking self is stationary
                try:
                    if np.array_equal(unfold(g, T, optimize(util_fn(u), v)), x):
                        return True
                except Exception:
                    pass
    return False


def _compose_base_fns(g):
    """The non-mental per-frame atoms a short compose chain can be built from, over
    a scene's own values: `step v d`, seek `optimize (neg_dist u) v`, and flee
    `optimize (dist u) v` (u ranging over the scene values, incl. u==v for the
    degenerate self-flee the Jul-12 R1 rival used).  These are exactly the physical
    primitives the searcher composes; a chain of them is a purely world-level
    (non-mental) program, so any scene it reproduces is NOT uniquely a belief."""
    vals = [int(v) for v in np.unique(g) if v != 0]
    fns = []
    for v in vals:
        for d in DIRS.values():
            fns.append(step(v, d))
        for u in vals:
            if u != v:
                fns.append(optimize(neg_distance(u), v))
            fns.append(optimize(distance(u), v))
    return fns


def _compose_rival_explainable(x, g, max_len=3):
    """True if any compose chain of length ≤ `max_len` over `_compose_base_fns`
    reproduces x — the pure-compose non-mental rival family.  Longer HPC search
    (t-fn 3600) reaches these 2–3 deep physical chains and used them to solve
    goal-displacement / witness / dual scenes without any fork (Jul-12 R1/R2);
    rejecting them at generation keeps a fork+sync belief the sole explanation.

    Cost is bounded (~b^max_len unfolds, b≈#base fns) and only paid on scenes that
    already passed the cheaper physical/wall gates, so few candidates reach here."""
    T = x.shape[0]
    base = _compose_base_fns(g)
    chains = list(base)                     # length-1 chains (redundant with the
    frontier = list(base)                   # physical check, but cheap and uniform)
    for _ in range(max_len - 1):
        nxt = []
        for pref in frontier:
            for f in base:
                nxt.append(compose(pref, f))
        chains.extend(nxt)
        frontier = nxt
    for prog in chains:
        try:
            if np.array_equal(unfold(g, T, prog), x):
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
    unless the wall causes a detour that costs *extra* steps over the free-space
    shortest path (a non-monotone route — a monotone shortest path, even one that
    differs from the reference optimize tie-break, is reproduced by a plain desire
    agent and so needs no wall) AND the trajectory is not explainable by any single
    physical program.
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
            if (T - 1) <= abs(ar - gr) + abs(ac - gc):
                # The detour must cost *extra* steps over the free-space shortest
                # path.  When the wall merely sits on one of several equally-short
                # routes, the "detour" is just another monotone shortest path — a
                # plain desire agent (optimize neg_dist) reproduces it with different
                # tie-breaking, so the phantom wall carries no explanatory load and
                # the scene is extensionally a goal task.  Requiring the trajectory
                # to be strictly longer than the Manhattan distance keeps only walls
                # that force a genuinely non-monotone detour.
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
            # nor any pure-compose chain (one physical fn per agent, either polarity):
            # length-2 suffices for the wall-free 2-agent rivals and keeps the 4-value
            # sweep tractable (a length-3 sweep is ~45× costlier here for no extra reach —
            # a genuine 3-deep detour needs a wall, which clobbers the crossing witness).
            if _compose_rival_explainable(x, g, max_len=2):
                continue
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
            # both the SEEK reading (walk toward gv around a wall) and the FLEE reading
            # (walk away from gv around a wall) — on the all-vertical content specs a
            # flee of the true goal coincides with an up/down shove, and the Jul-12 R3
            # rival was exactly a wall∘flee fork; rejecting it keeps the surviving scene
            # from having a wall-content spelling of any polarity.
            for policy in (optimize(neg_distance(gv), av), optimize(distance(gv), av)):
                for pr in range(size):
                    for pc in range(size):
                        prog = fork(compose(wall_at(pr, pc), policy), sync_to_world(av))
                        try:
                            if np.array_equal(unfold(g, T, prog), x):
                                return True
                        except Exception:
                            pass
    return False


def _goal_scope_certified(x, g, derive, av, gv):
    """Certify the goal-displacement commit field — everything that CAN discriminate.

    Full `_scope_complements_all_fail` certification is impossible for this family,
    intrinsically: a goal scene has exactly two nonzero values, so sync_except(gv)
    moves exactly {av} — it IS sync_to_world(av) on EVERY expressible scene, not a
    sampling accident (verified: 0% of candidate scenes separate them).  Prising the
    two apart needs a third world value the derive is FORCED to perturb, and forcing
    that requires blocking semantics — which is the false-obstacle family.  So the
    census's 'degenerate' label on goal solves is a provably benign spelling of the
    same agency commit, never a rival, and the forced-literal claim lives with fob.

    What CAN be certified per scene is that no OTHER scope commit reproduces it
    under the canonical derive: sync_all and sync_except(k) for every k != gv must
    all fail.  Given the stationary-goal filters this should reject nothing — it
    turns an implied invariant into a checked one.
    """
    def repro(commit):
        try:
            return np.array_equal(unfold(g, x.shape[0], fork(derive, commit)), x)
        except Exception:
            return False
    if not repro(sync_to_world(av)):        # must be a valid literal-commit scene
        return False
    if repro(sync_all):                     # wholesale adoption must be WRONG
        return False
    for k in (int(v) for v in np.unique(g) if v != 0 and v != gv):
        if repro(sync_except(k)):           # every discriminating complement must fail
            return False
    return True


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
        excluded and only genuinely goal-shaped content survives;
      * every scope complement that CAN discriminate fails (`_goal_scope_certified`):
        sync_all and sync_except(k != gv) never reproduce the scene.  sync_except(gv)
        is exempt because it is provably sync_to_world(av) on every two-value scene —
        the census reports those solves as 'degenerate' (a benign spelling of the
        same agency commit); the family that forces the literal spelling is fob.
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
            if _compose_rival_explainable(x, g):    # no pure-compose non-mental chain
                continue
            derive = _seq(*([step(gv, d) for d in vecs]
                            + [optimize(neg_distance(gv), av)]))
            if not _goal_scope_certified(x, g, derive, av, gv):
                continue                            # discriminating complements must fail
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
            if _compose_rival_explainable(x, g, max_len=2):   # one physical fn per agent, either polarity
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
    "left-fold compose to match tasks._seq: _seq_str(a,b,c) = (compose (compose a b) c)"
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


def _false_obstacle_rival_explainable(x, g, size=SIZE):
    """True if any NON-MENTAL transient-wall program, or any scope-complement fork
    over a goal/wall-preserving derive, reproduces the false-obstacle scene.

    The scene keeps a REAL wall (value 3) and the goal fixed in the world, so the
    intended solution is fork(derive, sync_to_world(av)) with a literal single-value
    commit.  Two rival families would undercut that agency commit and are rejected:

      * transient real wall — stamp a wall, seek/flee around it, then clear it (or
        erase all 3s): a purely world-level detour.  The stamp-and-clear-the-believed-
        cell variants are the confirmed Jul-12 live leak (C1–C3); the fob generator
        had NO battery for them (unlike witness/dual);
      * scope-complement fork — fork(derive', sync_all / sync_except k) where derive'
        moves only the agent in the model (goal & wall left put), which makes a
        wholesale / all-but-one commit coincide with the single-value agency commit.
        `_scope_complements_all_fail` guarantees the complements fail only for the
        CANONICAL derive; this sweep repairs that gap over alternative derives;
      * world-channel-transforming fork — in the decomposed DSL, `bimap` / `mapfst`
        let the WORLD (first) channel be transformed too, not just the model.  Then a
        scope commit can reproduce a real-wall scene even though `_scope_complements_
        all_fail` (which holds the world pristine) certifies it cannot: a seek in the
        world channel places the agent, and — because that derive never stamps the
        phantom wall — the real wall stays aligned, so sync_except(gv) / sync_all is a
        no-op on everything but the agent, i.e. extensionally sync_to_world(av).  This
        is the confirmed Jul-14 phase2 leak (fob solved via bimap+sync_except).  Sweep
        a bounded (world_op, model_derive) battery with a scope commit."""
    T = x.shape[0]
    agents  = [int(v) for v in np.unique(g) if v not in (0, 3)]
    targets = [int(v) for v in np.unique(g) if v != 0]     # incl. real wall 3 (beacon)
    world_vals = list(targets)
    cells = [(r, c) for r in range(size) for c in range(size)]
    _ident = lambda z: z
    for av in agents:
        for gv in targets:
            if av == gv:
                continue
            commits = [sync_all] + [sync_except(k) for k in set(world_vals)]
            for util_fn in (neg_distance, distance):
                seek = optimize(util_fn(gv), av)
                # 1) transient real-wall rivals: stamp / act / (clear cell | erase all 3s)
                for (pr, pc) in cells:
                    for prog in (compose(compose(wall_at(pr, pc), seek), clear_at(pr, pc)),
                                 compose(compose(wall_at(pr, pc), seek), erase(3))):
                        try:
                            if np.array_equal(unfold(g, T, prog), x):
                                return True
                        except Exception:
                            pass
                # 2) scope-complement forks over goal/wall-preserving derives
                derives = [seek] + [compose(wall_at(pr, pc), seek) for (pr, pc) in cells]
                for d in derives:
                    for commit in commits:
                        try:
                            if np.array_equal(unfold(g, T, fork(d, commit)), x):
                                return True
                        except Exception:
                            pass
            # 3) world-channel-transforming forks (decomposed DSL): a bounded battery of
            #    (world_op, model_derive) pairs with a scope commit.  world_op/model_op
            #    range over cross-seeks (both dirs, both utils), self-seeks, single steps,
            #    and identity; the model derive additionally allows a 2-op seek
            #    composition (mapsnd∘bimap in the found leak).  Kept bounded (~7k unfolds/
            #    scene) — this runs only after the cheaper gates above pass.
            seeks = [optimize(u(t), s) for u in (neg_distance, distance)
                                       for (t, s) in ((gv, av), (av, gv))]
            seeks += [optimize(distance(av), av), optimize(distance(gv), gv)]
            steps = [step(v, d) for v in (av, gv) for d in DIRS.values()]
            chan_ops  = seeks + steps + [_ident]
            model_ops = chan_ops + [compose(a, b) for a in seeks for b in seeks]
            for wop in chan_ops:
                for mop in model_ops:
                    prod = compose_gp(dup, bimap(wop, mop))
                    for commit in commits:
                        try:
                            if np.array_equal(unfold(g, T, pipe_gpg(prod, commit)), x):
                                return True
                        except Exception:
                            pass
    return False


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
            if _false_obstacle_rival_explainable(x, g, size):   # no transient-wall / alt-derive rival
                continue
            tasks.append((x, {'kind': 'belief', 'av': av, 'gv': gv,
                              'pw': (br, bc), 'real_wall': (wr, wc),
                              'displaced_to': (bgr, bgc), 'dir': dgname}))
            made += 1
    return tasks


# ── Obstacle family configuration (policy donor) ─────────────────────────────
# Obstacle is the policy DONOR: belief reuses the abstraction the joint stitch learns
# from the obstacle family — (compose (optimize (neg_dist gv) av) (wall_at r c)), the
# shared derive.  For stitch to keep (gv,av) as HOLES rather than bake a literal, the
# donor must span the SAME diverse value set as belief (all 8 usable cell ids; 0=empty
# and 3=wall are reserved — see prims._CELLVALUES).  So obstacle reuses belief's COMBOS
# defined above.  A narrow {1,4}×{2,5} subset under-powered the abstraction: stitch
# baked gv/av into the policy and belief could not reuse it across its 8 combos.


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


def _overlay_trail_explainable(x, g):
    """True if a fork(step v d, OVERLAY) trail reproduces x for any (v, d).

    The overlay and underlay trails are extensionally IDENTICAL for a lone mover
    (the union of trail cells is the same) and diverge only where the trail crosses
    a bystander of another value: overlay lets the shifted copy paint over it,
    underlay keeps the world's pixel.  A scene the overlay commit also reproduces
    therefore fails to make underlay necessary, so we reject it — the same 'reject
    any scene a cheaper rival explains' discipline as `_physically_explainable`.
    """
    T = x.shape[0]
    for v in _grid_vals(g):
        for d in DIRS.values():
            try:
                if np.array_equal(unfold(g, T, fork(step(v, d), overlay)), x):
                    return True
            except Exception:
                pass
    return False


def make_underlay_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, max_T=6):
    """z-order complement of overlay in a FORK context: a world-wins motion trail.

        (fork (step v d) underlay)

    Same fork-required motion blur as make_overlay_tasks, but the commit is
    `underlay` (the WORLD's nonzero cells win ties) instead of `overlay` (the
    derived copy wins).  For a lone mover the two are indistinguishable, so each
    scene puts an occluding bystander of a *different* value on the mover's path:
    when the trail reaches it, overlay overwrites the bystander and the trail runs
    on, while underlay preserves the bystander and the trail flows *under* it (and
    is blocked there).  We reject any scene the OVERLAY commit also reproduces
    (`_overlay_trail_explainable` — the crossing must actually happen within T
    frames) and any scene a bare physical fn explains (`_physically_explainable`),
    so both fork AND underlay are necessary.  Pure graphics (z-order); no mind.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        v, b = (int(u) for u in rng.permutation(vals)[:2])   # distinct mover & bystander
        dname = str(rng.choice(list(DIRS)))
        dr, dc = DIRS[dname]
        sign = dr + dc                              # ±1; exactly one of dr,dc is nonzero
        k = int(rng.integers(1, size - 1))          # mover→bystander gap along the path
        T = int(rng.integers(k + 2, max_T + 1))     # enough frames for the trail to reach it
        if T < 3:
            continue
        # place mover and bystander colinearly along d, both on-grid, bystander ahead
        lo, hi = (0, size - 1 - k) if sign > 0 else (k, size - 1)
        if lo > hi:
            continue
        m_idx = int(rng.integers(lo, hi + 1))
        b_idx = m_idx + sign * k
        L = int(rng.integers(0, size))              # fixed perpendicular line
        g = np.zeros((size, size), dtype=int)
        if dc != 0:                                 # horizontal travel: vary column
            g[L, m_idx], g[L, b_idx] = v, b
        else:                                       # vertical travel: vary row
            g[m_idx, L], g[b_idx, L] = v, b

        x = unfold(g, T, fork(step(v, DIRS[dname]), underlay))
        if np.array_equal(x[0], x[-1]):             # something must move
            continue
        if _physically_explainable(x, g):           # fork must be required
            continue
        if _overlay_trail_explainable(x, g):        # underlay must be required (crossing occurs)
            continue
        tasks.append((x, {'kind': 'underlay', 'val': v, 'dir': dname, 'bystander': b}))
    return tasks


def make_comet_tasks(n, size=SIZE, combos=COMBOS, seed=0):
    """fork without sync, VARYING THE DERIVE: a goal-seeker leaves a comet trail.

        (fork (optimize (neg_dist gv) av) overlay)

    Same overlay commit as make_overlay_tasks, but the derive is a goal-directed
    seek (desire's `optimize (neg_dist gv) av`) instead of a fixed `step v d`.  The
    agent av greedily approaches goal gv and overlay unions each step onto the
    trail, so av's whole L-shaped path is rendered — a comet trail.  The seek bends
    (the goal is off-axis), so no fixed-direction step-trail reproduces it
    (`_overlay_trail_explainable`) and no bare physical fn does either
    (`_physically_explainable`) — fork AND the seek-derive are both necessary.  This
    exhibits fork's derive slot as a *general* fn, not one wired to `step`: the same
    fork/overlay motion-blur wrapper carries desire's utility policy.  Nothing
    mental (utility-driven motion + graphics).

    Geometry: the goal sits strictly up-LEFT of the agent so the row-major-first av
    cell is always the moving FRONT (`optimize` advances `agents[0]`); other
    orientations make agents[0] the tail and the trail collapses instead of growing.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        av, gv = (int(u) for u in combos[int(rng.integers(len(combos)))])
        ar, ac = int(rng.integers(2, size)), int(rng.integers(2, size))
        gr, gc = int(rng.integers(0, ar)), int(rng.integers(0, ac))   # strictly up-left
        L = (ar - gr) + (ac - gc)
        if not (3 <= L <= 5):
            continue
        g = np.zeros((size, size), dtype=int)
        g[ar, ac], g[gr, gc] = av, gv
        T = L + 1
        x = unfold(g, T, fork(optimize(neg_distance(gv), av), overlay))
        if x[-1][gr, gc] != av:                       # the comet head must reach the goal
            continue
        if int((x[-1] == av).sum()) != L + 1:         # clean trail: one cell per step, no collapse
            continue
        if _physically_explainable(x, g):             # fork must be required
            continue
        if _overlay_trail_explainable(x, g):          # the seek-bend must be required (not a fixed step)
            continue
        tasks.append((x, {'kind': 'comet', 'av': av, 'gv': gv}))
    return tasks


def make_registration_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, n_distract=2):
    """sync without fork: snap ONE named object onto an external template.

        (sync_to_world v)            applied per frame to (working, template)

    The pair is two *given* grids — working + template — paired by
    unfold_with_template, NOT by fork.  Only the target v is registered; the
    n_distract>=2 other shared values are *also* misplaced but must stay put, so
    the output is neither the bare template (defeats `snd_gg`/`sync_all`, which the
    cube DSL would otherwise solve it with for free) nor reachable by leaving a
    single value (defeats `sync_except`).  step/optimize cannot read the template
    at all.  Hence a single `sync_to_world v` is the unique cheapest commit — sync
    is necessary, and the second grid is a spec, not a mind.
    """
    rng = np.random.default_rng(seed)
    n_distract = max(2, n_distract)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        need = 2 * (1 + n_distract)
        if len(cells) < need or len(vals) < 1 + n_distract:
            continue
        perm = rng.permutation(vals).tolist()
        v, distract_vals = perm[0], perm[1:1 + n_distract]

        working  = np.zeros((size, size), dtype=int)
        template = np.zeros((size, size), dtype=int)
        working[cells[0]]  = v                     # v is misplaced…
        template[cells[1]] = v                     # …it belongs here, per template
        ci, ok = 2, True
        for dv in distract_vals:                   # distractors misplaced AND retained
            dsrc, dtgt = cells[ci], cells[ci + 1]
            ci += 2
            if dsrc == dtgt:
                ok = False
                break
            working[dsrc]  = dv
            template[dtgt] = dv
        if not ok:
            continue

        x = unfold_with_template(working, template, 2, sync_to_world(v))
        if np.array_equal(x[0], x[-1]):            # v must actually move
            continue
        if not _unique_pair_corner(working, template, 'sync_to_world', x[-1]):
            continue
        tasks.append((x, {'kind': 'registration', 'val': v, 'template': template}))
    return tasks


# ── one non-mental task per symmetric corner (file16's "cube") ────────────────────
# The cube (make_symmetric_prims) hands the searcher the *complement* of every
# choice baked into belief's corner.  A complement that no task ever needs is an
# inert distractor: the cube census can only show "belief avoids the complements"
# if the complements are genuinely *useful elsewhere*.  So each corner gets the
# minds-free task the dsl comment names it for, generated through the same
# interpreter the searcher uses, with a necessity check that rejects any scene a
# rival corner solves just as cheaply.  Two interpreters, matching the two root
# types already in play:
#
#   fn      (unfold)               : flee (distance), deletion (clear_at),
#                                    denoise (erase)
#   fn_p_g  (unfold_with_template) : perception (sync_to_model), multi-registration
#                                    (sync_all), registration-except (sync_except),
#                                    inpainting (underlay), readout (snd_gg)
#
# fst_gg (the kept projection corner) and via_swap (a decomposed-only wiring
# witness, == sync_to_model on a swapped pair) get no standalone task: the former
# is the trivial "keep the world" already implicit everywhere, the latter is a
# re-expression of the perception corner, not an independent operation.

def _reproduces(g, x, f):
    "True if unfold(g, T, f) == x (T = x's frame count); swallows interpreter errors."
    try:
        return np.array_equal(unfold(g, x.shape[0], f), x)
    except Exception:
        return False


def _grid_vals(g):
    return [int(v) for v in np.unique(g) if v != 0]


def make_flee_tasks(n, size=SIZE, vals=(1, 4), seed=0, max_T=6):
    """utility complement (distance): an agent flees the nearest hazard.

        (optimize (distance hv) av)

    av greedily maximises BFS distance from hazard hv — predator/prey, hazard
    avoidance.  We keep the trajectory through the frame where av runs out of room
    and *stays put*: a fixed-direction `step` would keep moving (or leave the grid)
    there, and every `neg_dist` seeker is attracted toward a value, not repelled,
    so `_physically_explainable` rejects any scene a non-fleeing program reproduces.
    Hence `distance` is required.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        perm = rng.permutation(vals).tolist()
        av, hv = int(perm[0]), int(perm[1])
        hr, hc = int(rng.integers(1, size - 1)), int(rng.integers(1, size - 1))
        ar, ac = int(rng.integers(size)), int(rng.integers(size))
        if (ar, ac) == (hr, hc) or abs(ar - hr) + abs(ac - hc) > 2:
            continue
        g = np.zeros((size, size), dtype=int)
        g[hr, hc] = hv
        g[ar, ac] = av
        traj = unfold(g, max_T, optimize(distance(hv), av))
        pos = [_agent_pos(traj[t], av) for t in range(max_T)]
        T = next((t + 1 for t in range(1, max_T) if pos[t] == pos[t - 1]), max_T)
        if T < 3:                                   # need at least two real moves
            continue
        x = traj[:T].copy()
        if np.array_equal(x[0], x[-1]):
            continue
        if _physically_explainable(x, g):           # not a seek or a straight step
            continue
        tasks.append((x, {'kind': 'flee', 'av': av, 'hv': hv}))
    return tasks


def _step_or_erase_reproduces(g, x):
    """True if any `step v d` (incl. v=0, shifting the background) or `erase v`
    reproduces x — the cheap grid->grid rivals to a single-cell `clear_at`."""
    for v in range(6):
        if _reproduces(g, x, erase(v)):
            return True
        for d in DIRS.values():
            if _reproduces(g, x, step(v, d)):
                return True
    return False


def make_deletion_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0):
    """grid-edit complement (clear_at): punch ONE hole in a solid object.

        (clear_at r c)

    A solid 3x3 block of value v is drawn and its strictly-interior cell is
    blanked.  Because that cell's four neighbours are all non-zero, no `step 0 d`
    (shifting the background) can slide a zero into it, and `step v d` moves the
    whole block — so the sneaky "delete by moving zeros/cells" rivals all fail.
    `erase v` wipes the whole block (too much).  Hence a single `clear_at` is the
    unique cheapest solution.  Targeted deletion / object editing; nothing mental.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        v = int(rng.choice(vals))
        r0 = int(rng.integers(0, size - 2))         # 3x3 block top-left
        c0 = int(rng.integers(0, size - 2))
        g = np.zeros((size, size), dtype=int)
        g[r0:r0 + 3, c0:c0 + 3] = v
        tr, tc = r0 + 1, c0 + 1                       # the protected interior cell
        x = unfold(g, 2, clear_at(tr, tc))
        if np.array_equal(x[0], x[-1]):
            continue
        if _step_or_erase_reproduces(g, x):
            continue
        if _physically_explainable(x, g):
            continue
        tasks.append((x, {'kind': 'deletion', 'val': v, 'cell': (tr, tc)}))
    return tasks


def make_denoise_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, n_noise=3, n_signal=2):
    """grid-edit complement (erase): drop EVERY cell of the noise value.

        (erase nv)

    A signal value sv is kept; the noise value nv is scattered over n_noise>=2
    cells that all vanish at once.  No single `clear_at` can remove >=2 cells, and
    `step`/`optimize` relocate rather than delete — so whole-value `erase` is
    required.  Pure denoising: nothing mental.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        perm = rng.permutation(vals).tolist()
        sv, nv = int(perm[0]), int(perm[1])
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        if len(cells) < n_signal + n_noise:
            continue
        sig, noise = cells[:n_signal], cells[n_signal:n_signal + n_noise]
        g = np.zeros((size, size), dtype=int)
        for (r, c) in sig:
            g[r, c] = sv
        for (r, c) in noise:
            g[r, c] = nv
        x = unfold(g, 2, erase(nv))
        if np.array_equal(x[0], x[-1]):
            continue
        if _physically_explainable(x, g):
            continue
        if any(_reproduces(g, x, clear_at(r, c)) for (r, c) in noise):
            continue
        tasks.append((x, {'kind': 'denoise', 'noise': nv, 'signal': sv}))
    return tasks


def make_obstacle_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8):
    """grid-edit complement (wall_at): a real obstacle appears, the agent detours.

        (compose (wall_at pr pc) (optimize (neg_dist gv) av))

    The PHYSICAL counterpart of belief: a wall (value 3, impassable) is stamped into
    the WORLD on the agent's direct path, and the agent navigates around it to the
    goal — no private model, no fork, no sync.  This gives the grid-edit "add" corner
    its own minds-free home (symmetric with clear_at→deletion and erase→denoise, the
    "remove" corners), so wall_at is no longer belief-exclusive.

    Two payoffs.  (1) The fragment `(compose (wall_at) (optimize (neg_dist)))` — which
    is exactly belief's policy/derive — now occurs in a SOLVED non-mental task, and an
    obstacle *family* makes it recur, so the joint stitch can abstract it.  That lowers
    belief's first-solve cost (the derive collapses to one node) and, more importantly,
    means the only thing left unique to belief is the fork∧sync agency wrapper, not
    wall_at itself — a stronger "discovered, not gerrymandered" claim.  (2) The library
    that does so is justified by the obstacle family on its own, independent of belief.

    Distinct from belief: here the wall is VISIBLE in every frame (rendered into the
    world); belief's wall lives only in the private model and never shows in the world,
    so the two trajectories differ.  The wall is placed on the direct path and scenes
    are rejected unless it actually forces a detour (else wall_at would be decorative).
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

            # wall-free trajectory: the source of on-path obstacle candidates
            direct = unfold(g, max_T, optimize(neg_distance(gv), av))
            dpath  = [_agent_pos(direct[t], av) for t in range(max_T)]
            inter  = [p for p in dpath if p and p != (ar, ac) and p != (gr, gc)]
            if not inter:
                continue
            pr, pc = inter[int(rng.integers(len(inter)))]

            derive = compose(wall_at(pr, pc), optimize(neg_distance(gv), av))
            x_full = unfold(g, max_T, derive)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (gr, gc)), None)
            if t_arrive is None or t_arrive < 3:
                continue
            T = t_arrive + 1
            x = x_full[:T].copy()
            # the wall must force a detour: the agent's path with the wall must differ
            # from the wall-free path (else wall_at is decorative, not required).
            if [_agent_pos(x[t], av) for t in range(T)] == dpath[:T]:
                continue
            if _physically_explainable(x, g):     # no bare step/optimize rival
                continue
            tasks.append((x, {'kind': 'obstacle', 'av': av, 'gv': gv, 'pw': (pr, pc)}))
            made += 1
    return tasks


def make_relocation_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0):
    """grid-edit compound (clear_at ▸ wall_at): a REAL wall jumps to a new cell.

        (compose (clear_at r1 c1) (wall_at r2 c2))

    A wall (value 3) vanishes from one cell and appears at another in a single
    frame, then the scene is static; bystander values stay put throughout.  This
    is the "move" grid-edit corner, the compound of the "remove" (deletion) and
    "add" (obstacle) corners — a visible, non-mental relocation.

    Purpose (mirrors the obstacle family's role for the wall policy): the
    fragment `(compose (clear_at …) (wall_at …))` — or its erase spelling — is
    exactly the derive PREFIX of the false-obstacle belief family (clear the
    real wall, stamp the phantom), which prices ~14 nats beyond the search
    frontier when every node is paid at primitive cost.  A relocation family
    makes that compound recur in SOLVED non-mental tasks, so the joint stitch
    can abstract it into one cheap token and pull false-obstacle belief inside
    the enumerator's reach — the same curriculum trick that lowered belief's
    first-solve via the obstacle corner.

    Necessity: the jump is ≥2 cells (a 1-cell jump is a plain `step 3 d`), and
    T=3 with an idempotent transition (frames 1 and 2 identical) kills every
    step/optimize drift rival, which would keep moving; `erase 3` alone removes
    the wall but conjures none; `wall_at` alone leaves two walls.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        (r1, c1), (r2, c2), b1, b2 = cells[0], cells[1], cells[2], cells[3]
        if abs(r1 - r2) + abs(c1 - c2) < 2:
            continue
        bv1, bv2 = (int(v) for v in rng.choice(vals, size=2))
        g = np.zeros((size, size), dtype=int)
        g[r1, c1] = 3
        g[b1] = bv1
        g[b2] = bv2
        x = unfold(g, 3, compose(clear_at(r1, c1), wall_at(r2, c2)))
        if np.array_equal(x[0], x[-1]):
            continue
        if _step_or_erase_reproduces(g, x):
            continue
        if _physically_explainable(x, g):
            continue
        tasks.append((x, {'kind': 'relocate', 'from': (r1, c1), 'to': (r2, c2)}))
    return tasks


# fn_p_g corners share the registration interpreter (unfold_with_template); each
# is a single pair->grid commit over (working, template).  A scene is kept only if
# the intended corner is the ONLY atomic commit that reproduces it — compositional
# rivals (then_sync chains) are strictly longer, so they never undercut a 1-node
# corner and are not checked.

# atomic fn_p_g corners, grouped by node cost: nullary commits are one node, the
# value-parameterised ones are two (node + int).  The searcher breaks ties by
# length, so a corner is "required" when it is the UNIQUE CHEAPEST commit that
# reproduces the scene — a pricier rival never undercuts it, an equal-cost one does.
_PAIR_NULLARY = {'overlay': overlay, 'underlay': underlay,
                 'fst_gg': fst_gg, 'snd_gg': snd_gg, 'sync_all': sync_all}
_PAIR_INT     = {'sync_to_world': sync_to_world, 'sync_to_model': sync_to_model,
                 'sync_except': sync_except}


def _unique_pair_corner(working, template, want, want_out):
    "True iff `want` is the unique CHEAPEST atomic fn_p_g corner producing want_out."
    p = (working.copy(), template.copy())
    vals = sorted(set(_grid_vals(working)) | set(_grid_vals(template)))
    want_cost = 1 if want in _PAIR_NULLARY else 2
    for nm, f in _PAIR_NULLARY.items():          # cost 1 — undercuts/ties anything
        if nm == want:
            continue
        try:
            if np.array_equal(f(p), want_out):
                return False
        except Exception:
            pass
    if want_cost >= 2:                            # cost-2 rivals only matter to cost-2 wants
        for nm, ctor in _PAIR_INT.items():
            if nm == want:
                continue
            for v in vals:
                try:
                    if np.array_equal(ctor(v)(p), want_out):
                        return False
                except Exception:
                    pass
    return True


def make_perception_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0):
    """direction complement (sync_to_model): record a world observation into the map.

        (sync_to_model v)            applied to (working, template)

    Reads v's coordinate off the WORLD (working), writes it into the MODEL
    (template), and returns the MODEL — a sensation recorded into a private map,
    not an action on the world.  Working and template carry *different* distractors
    so the two sync directions are distinguishable; the output keeps the template's
    frame with v relocated to where the world sees it, which only `sync_to_model`
    yields.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        perm = rng.permutation(vals).tolist()
        v, dw, dm = int(perm[0]), int(perm[1]), int(perm[2])
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        P, Q, cw, cm = cells[0], cells[1], cells[2], cells[3]
        if P == Q:
            continue
        working  = np.zeros((size, size), dtype=int)
        template = np.zeros((size, size), dtype=int)
        working[P]   = v                 # world sees v here…
        template[Q]  = v                 # …the map still has it there (stale)
        working[cw]  = dw                # distractors differ between the channels so
        template[cm] = dm                # keep-world and keep-model are distinguishable
        x = unfold_with_template(working, template, 2, sync_to_model(v))
        if np.array_equal(x[0], x[-1]):
            continue
        if not _unique_pair_corner(working, template, 'sync_to_model', x[-1]):
            continue
        tasks.append((x, {'kind': 'perception', 'val': v, 'template': template}))
    return tasks


def make_multi_registration_tasks(n, size=SIZE, vals=(1, 2, 4, 5), seed=0, k=2):
    """scope complement (sync_all): snap EVERY misplaced object to the template.

        sync_all                     applied to (working, template)

    k>=2 shared values are each misplaced; wholesale state adoption moves them all
    at once.  A `sync_to_world v` fixes only one, and the only nullary commits that
    could tie `sync_all` (snd_gg/fst_gg/blends) are broken by a working-only
    `static` value: it is unshared, so sync_all leaves it where the world has it,
    making the output neither the bare template nor the bare world.  So `sync_all`
    is the unique cheapest solution.  Multi-object registration; no mind.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        chosen = rng.permutation(vals).tolist()
        movers, static = chosen[:k], chosen[k]      # last is world-only (unshared)
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        if len(cells) < 2 * len(movers) + 1:
            continue
        working  = np.zeros((size, size), dtype=int)
        template = np.zeros((size, size), dtype=int)
        ci, ok = 0, True
        for u in movers:
            src, tgt = cells[ci], cells[ci + 1]
            ci += 2
            if src == tgt:
                ok = False
                break
            working[src]  = u
            template[tgt] = u
        if not ok:
            continue
        working[cells[ci]] = static                 # present in world only → stays put
        x = unfold_with_template(working, template, 2, sync_all)
        if np.array_equal(x[0], x[-1]):
            continue
        if not _unique_pair_corner(working, template, 'sync_all', x[-1]):
            continue
        tasks.append((x, {'kind': 'multi_reg', 'vals': movers, 'template': template}))
    return tasks


def make_registration_except_tasks(n, size=SIZE, vals=(1, 2, 4, 5), seed=0, k=2):
    """scope complement (sync_except): register everything but one anchor.

        (sync_except a)              applied to (working, template)

    The anchor `a` and k>=2 other values are all misplaced; every value but `a`
    snaps to the template while `a` is held at its world position.  `sync_all`
    moves `a` too, and with >=2 non-anchor movers no single `sync_to_world`
    suffices.  The subtle rival is `sync_to_model a`, which equals `sync_except a`
    whenever the two channels carry the same value-set — broken here by a
    world-only `static` value: `sync_except` returns the world (keeping it) while
    `sync_to_model` returns the model (without it).  So `sync_except a` is the
    unique cheapest solution (set-complement registration).
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        chosen = rng.permutation(vals).tolist()
        if len(chosen) < k + 2:
            continue
        movers, static = chosen[:k + 1], chosen[k + 1]   # movers[0] = anchor
        anchor = movers[0]
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        if len(cells) < 2 * len(movers) + 1:
            continue
        working  = np.zeros((size, size), dtype=int)
        template = np.zeros((size, size), dtype=int)
        ci, ok = 0, True
        for u in movers:                 # every value (incl. anchor) is misplaced
            src, tgt = cells[ci], cells[ci + 1]
            ci += 2
            if src == tgt:
                ok = False
                break
            working[src]  = u
            template[tgt] = u
        if not ok:
            continue
        working[cells[ci]] = static      # world-only: breaks the sync_to_model tie
        x = unfold_with_template(working, template, 2, sync_except(anchor))
        if np.array_equal(x[0], x[-1]):
            continue
        if not _unique_pair_corner(working, template, 'sync_except', x[-1]):
            continue
        tasks.append((x, {'kind': 'reg_except', 'anchor': anchor,
                          'template': template}))
    return tasks


def make_inpainting_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0):
    """z-order complement (underlay): fill holes from the template, keep your own pixels.

        underlay                     applied to (working, template)

    The template is a reference image; the working grid is the same image with a
    hole punched (zeros) AND one pixel painted a different value.  `underlay` lets
    the working pixels win and the template fill only the holes — so the painted
    pixel survives and the hole is reconstructed.  `overlay` would let the template
    overwrite the painted pixel, `fst_gg` leaves the hole, `snd_gg` drops the
    painted pixel — hence `underlay` is required.  Inpainting; nothing mental.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        ref_val, paint = int(rng.choice(vals)), int(rng.choice(vals))
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        block = cells[:5]                    # the reference shape (>=2 cells)
        template = np.zeros((size, size), dtype=int)
        for (r, c) in block:
            template[r, c] = ref_val
        hole, witness = block[0], block[1]
        working = template.copy()
        working[hole] = 0                    # punch a hole the template must fill
        if paint == ref_val:
            paint = ref_val + 1
        working[witness] = paint             # a pixel that disagrees with the template
        x = unfold_with_template(working, template, 2, underlay)
        if np.array_equal(x[0], x[-1]):
            continue
        if not _unique_pair_corner(working, template, 'underlay', x[-1]):
            continue
        tasks.append((x, {'kind': 'inpaint', 'template': template}))
    return tasks


def make_readout_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0):
    """projection complement (snd_gg): report the stored map, ignore the query.

        snd_gg                       applied to (working, template)

    The output is the template regardless of the working grid — a pure channel
    projection (recall the model, discard the world).  Working and template share
    no cells, so `fst_gg` (keep world), the blends, and every sync all diverge from
    a verbatim template; only `snd_gg` reproduces it.
    """
    rng = np.random.default_rng(seed)
    tasks, attempts = [], 0
    while len(tasks) < n and attempts < 8000:
        attempts += 1
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        wcells, tcells = cells[:3], cells[3:6]
        working  = np.zeros((size, size), dtype=int)
        template = np.zeros((size, size), dtype=int)
        for (r, c) in wcells:
            working[r, c] = int(rng.choice(vals))
        for (r, c) in tcells:
            template[r, c] = int(rng.choice(vals))
        if not _grid_vals(working) or not _grid_vals(template):
            continue
        x = unfold_with_template(working, template, 2, snd_gg)
        if np.array_equal(x[0], x[-1]):
            continue
        if not _unique_pair_corner(working, template, 'snd_gg', x[-1]):
            continue
        tasks.append((x, {'kind': 'readout', 'template': template}))
    return tasks


