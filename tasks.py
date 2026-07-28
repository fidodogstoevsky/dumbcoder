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

The two families the belief claim rests on are both false beliefs about an OBJECT,
so they need no wall and no blocking semantics:

  Sally-Anne     (fork (compose (step gv d) (optimize (neg_dist gv) av))
                       (sync_to_world av))
  two observers  (compose (optimize (neg_dist gv) ao)
                          (fork (compose (step gv d) (optimize (neg_dist gv) av))
                                (sync_to_world av)))

The second is the first with a bystander who is right about the world: two agents,
one goal, one grid, walking to two different cells.  The library has to invent the
agent constructor and then apply it to av and NOT to ao — mental state attributed
selectively, to exactly the agent whose behaviour cannot be read off the world.

av appears twice -- in optimize (who acts on the model) and in sync_to_world
(whose move is committed to the world) -- the structural signature of agency. The
hoped-for stitch discovery is the agent-type constructor:

  fn_agent($r, $c, $gv, $av) =
    (fork (compose (wall_at $r $c) (optimize (neg_dist $gv) $av))
          (sync_to_world $av))

Run a phase via `python phase1.py`.
"""

from collections import defaultdict

import numpy as np

from scenes import Scenes, solves
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

# How many scenes make up one task.  A task is a SET of trajectories sharing one
# latent program (see scenes.py): the program's literals are held fixed across the
# set and everything the program does not name — where the entities start, how long
# the trajectory runs — varies.  One scene under-determines its program, which is
# what the old per-scene rival batteries were trying (and failing) to patch: a
# literal `(step av down)` reproduced one phantom-wall detour exactly, so search
# dropped the wall.  It cannot reproduce four detours from four start positions,
# and a wall at the wrong coordinate dies on scene 2.
K_SCENES = 4


# ── Task generation ────────────────────────────────────────────────────────────
# Every task is generated through the same `unfold` the searcher uses, so any
# solvability failure is search, never encoding.

def _collect(propose, n, k, rng, attempts, certify=None, target_tries=0):
    """Build `n` k-scene tasks by pooling single-scene proposals under their latent.

    `propose(rng)` samples ONE scene and returns `(latent, scene, meta)`, or None if
    the sample was rejected.  `latent` is the tuple of literals the family's ground-
    truth program actually names — (av, gv, wall row, wall col) for phantom-wall
    belief, () for a nullary corner like `snd_gg`.  Scenes are bucketed by latent,
    and a bucket that reaches k becomes a task: by construction all k scenes were
    generated by the same program, so that program solves the task, and any rival
    must now fit all k.

    Blind pooling is enough where valid scenes are common and latents few, but it
    collapses on the tightly constrained families: false-obstacle belief pins five
    literals, so a bucket almost never fills by chance.  Those families pass `target_tries` and accept a `target=`
    latent in `propose`; the driver then switches to seek-then-fill — sample freely
    until some latent proves viable, then pin it and spend up to `target_tries`
    proposals looking for other placements that realise the SAME program.

    Pooling (rather than "pick a latent, then find k scenes for it") is what makes
    this cheap to retrofit: each family keeps the per-scene geometry it already had,
    and the only thing it must newly declare is which of its parameters the program
    names.

    A latent is NOT retired after it yields a task.  Several families deliberately
    repeat a latent — desire builds n_per_combo tasks per (av, gv) pair — and those
    repeat counts are what make a program recur often enough for the joint stitch to
    abstract it.  Changing them would change the MDL dynamics, so a spent latent just
    starts a fresh pool; every scene is used at most once, so no two tasks are equal.

    `certify(scenes, meta)` optionally vets the finished set (used where a family
    must show that its intended primitive is the unique cheapest one over a CLOSED
    set of rivals, e.g. the scope commits).  A task-level check is the right scope:
    a rival only undercuts the intended program if it reproduces every scene.
    """
    pools, used, tasks = defaultdict(list), defaultdict(set), []

    def take(got):
        "Pool one proposal; True if its latent's bucket is now full."
        if got is None:
            return False
        latent, scene, meta = got
        tag = (scene.shape, scene.tobytes())
        if tag in used[latent]:            # a repeated trajectory teaches nothing
            return False
        used[latent].add(tag)
        pools[latent].append((scene, meta))
        return len(pools[latent]) >= k

    for _ in range(attempts):
        if len(tasks) >= n:
            break
        got = propose(rng)
        if got is None:
            continue
        latent = got[0]
        full = take(got)
        if not full and target_tries:
            for _ in range(target_tries):
                if take(propose(rng, target=latent)):
                    full = True
                    break
        if not full:
            continue
        pool  = pools[latent]
        scs   = [s for s, _ in pool]
        metas = [mt for _, mt in pool]
        pools[latent] = []                 # next k scenes start the next task
        m = dict(metas[0])
        m['per_scene'] = metas          # fields that vary scene to scene (templates,
                                        # the displaced goal cell) live here; the
                                        # top-level copy is scene 0's, and every
                                        # LATENT field agrees across the set anyway.
        if certify is not None and not certify(scs, m):
            continue
        tasks.append((Scenes(scs), m))
    return tasks

def _agent_pos(frame, av):
    pos = np.argwhere(frame == av)
    return (int(pos[0][0]), int(pos[0][1])) if len(pos) else None


# ── scene-level variation: what the k scenes must NOT hold fixed ──────────────
# The scene set pins a program's literals down extensionally only to the extent
# that everything the program does NOT name actually varies.  Four scenes that
# happen to share an agent start, or all run to the same length, re-open exactly
# the hole the format closes: a rival keyed on that accident fits the whole set.
# Pooling never guarantees this on its own (it only rejects a repeated
# trajectory), so the families that carry the belief claim assert it.

def _varies(scenes, of, least=2):
    "True iff `of(scene)` takes at least `least` distinct values across the set."
    return len({of(s) for s in scenes}) >= least


def _placements_vary(scenes, av, gv, least=2):
    """The three unnamed degrees of freedom of a seek-style scene really vary:
    where the agent starts, where the goal sits, and how long the walk runs.

    A task whose k scenes agree on any of these is rejected — not because some
    particular rival exploits it, but because the scene set is then k copies of
    one datum with respect to that parameter, and the format's whole force is
    that the latents are the ONLY thing the scenes share."""
    return (_varies(scenes, lambda s: _agent_pos(s[0], av), least)
            and _varies(scenes, lambda s: _agent_pos(s[0], gv), least)
            and _varies(scenes, lambda s: int(s.shape[0]), least))


def _distractor_values(av, gv, pool=(1, 2, 4, 5, 6, 7, 8, 9)):
    "cell values free to name an inert bystander on an (av, gv) scene (3 = wall)."
    return [v for v in pool if v not in (av, gv)]


def _add_distractors(rng, g, exclude, values):
    """Scatter the task's bystander `values` on empty cells of `g` (in place).

    Distractors are scene noise no program in the corpus names, and
    `optimize`/`neg_distance` block on walls (3) alone, so a bystander cannot bend a
    trajectory — it can only be walked over, which the caller rejects.  What they buy
    is a scene whose value set is bigger than the program's literals, so a rival that
    aims at a third value (a "beacon" seek that happens to sit on the route) has that
    value put somewhere ELSE in the next scene and dies there.

    `values` is a per-TASK set, fixed across the k scenes, and that is a deliberate
    cost decision, not a shortcut.  `experiment.content_q` prices a cellvalue terminal
    at 0 nats when its head is visible anywhere in the task, so every distinct
    bystander value the scene set mentions is another free literal at each of the
    program's four cellvalue slots.  Drawing fresh values per SCENE put ~7 free values
    on a task that names 2, and measurably pushed the family out of budget (a
    one-shove task the searcher solved in 24 s at 2 free values was still unsolved at
    90 s).  Fixing the set per task keeps the free band at 3 while leaving the
    bystanders where they do their work: a beacon rival must now fit the same value at
    four different cells.  Their CELLS, and which of them appear, still vary per scene.

    Returns {value: (r, c)}, or None if there was no room; empty `values` disables
    them (the A/B baseline).
    """
    if not len(values):
        return {}
    size = g.shape[0]
    free = [(r, c) for r in range(size) for c in range(size)
            if g[r, c] == 0 and (r, c) not in exclude]
    n = 1 + int(rng.integers(len(values)))        # 1..len(values) of them this scene
    if len(free) < n:
        return None
    cells = [free[i] for i in rng.permutation(len(free))[:n]]
    vals  = [values[i] for i in rng.permutation(len(values))[:n]]
    out = {}
    for v, (r, c) in zip(vals, cells):
        g[r, c] = v
        out[int(v)] = (int(r), int(c))
    return out


def _bystanders_still(x, cells):
    "True iff every distractor keeps its value and its cell in every frame."
    return all(int((x[t] == v).sum()) == 1 and int(x[t][r, c]) == v
               for v, (r, c) in cells.items() for t in range(x.shape[0]))


# ── on the rival batteries that used to live here ─────────────────────────────
# Six hand-maintained functions once sat between `_agent_pos` and the generators:
# `_physically_explainable`, `_compose_base_fns`/`_compose_rival_explainable`, and
# `_displaced_goal_explainable` (plus the per-family `_witness_` and
# `_false_obstacle_` and `_wall_` variants below).  Each enumerated a list of
# non-mental programs and rejected any scene one of them reproduced, so that the
# intended program stayed the only explanation.
#
# They are gone.  Each was a patch on a specific hole in a representation with
# infinitely many holes: one trajectory simply does not determine its program, so
# every long run found a new coincidence (a fixed shove that happened to trace one
# detour, a wall-beacon that happened to sit on the greedy ray) and the answer was
# always another battery entry.  The k-scene task format is the general cure — a
# rival must now reproduce every scene of the set, and a coincidence that survives
# four independent placements of the same latent program is not a coincidence.
#
# What is NOT deleted: the closed-set uniqueness certifications
# (`_scope_complements_all_fail`, `_goal_scope_certified`, `_unique_pair_corner`).
# Those are a different animal — they range over a FINITE, fully enumerated set of
# rivals (the scope commits; the atomic fn_p_g corners) and some of what they
# exclude is an identity rather than a coincidence: sync_except(gv) IS
# sync_to_world(av) on any two-value scene, so no number of scenes separates them.
# They are lifted to task level instead (a rival counts only if it fits all k).

def make_physics_tasks(n, size=SIZE, vals=(1, 4), seed=0, k=K_SCENES):
    "Ground truth: (step v d).  Latent: the moved value and the direction."
    rng = np.random.default_rng(seed)

    def propose(rng):
        v     = int(rng.choice(vals))
        dname = str(rng.choice(list(DIRS)))
        dr, dc = DIRS[dname]
        T = int(rng.integers(4, 6))
        r_lo, r_hi = max(0, -dr * (T - 1)), min(size - 1, size - 1 - dr * (T - 1))
        c_lo, c_hi = max(0, -dc * (T - 1)), min(size - 1, size - 1 - dc * (T - 1))
        if r_lo > r_hi or c_lo > c_hi:
            return None
        r, c = int(rng.integers(r_lo, r_hi + 1)), int(rng.integers(c_lo, c_hi + 1))
        g = np.zeros((size, size), dtype=int)
        g[r, c] = v
        x = unfold(g, T, step(v, DIRS[dname]))
        return (v, dname), x, {'kind': 'physics', 'val': v, 'dir': dname}

    return _collect(propose, n, k, rng, attempts=4000)


def make_desire_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, k=K_SCENES):
    "Ground truth: (optimize (neg_dist gv) av).  Latent: the (agent, goal) values."
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in combos:
        def propose(rng, av=av, gv=gv):
            ar, ac = int(rng.integers(size)), int(rng.integers(size))
            gr, gc = int(rng.integers(size)), int(rng.integers(size))
            if ar == gr or ac == gc:
                return None
            L = abs(ar - gr) + abs(ac - gc)
            if not (3 <= L <= 5):
                return None
            g = np.zeros((size, size), dtype=int)
            g[ar, ac] = av
            g[gr, gc] = gv
            T = L + 1
            x = unfold(g, T, optimize(neg_distance(gv), av))
            if x[-1][gr, gc] != av:
                return None
            return (av, gv), x, {'kind': 'desire', 'av': av, 'gv': gv}

        tasks += _collect(propose, n_per_combo, k, rng, attempts=4000)
    return tasks


def make_belief_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8,
                      k=K_SCENES):
    """False-belief: the agent detours around a wall that exists only in its model.

      (fork (compose (wall_at pr pc) (optimize (neg_dist gv) av)) (sync_to_world av))

    Latent: (av, gv, pr, pc) — the two values the policy names and the phantom
    wall's coordinate.  The k scenes of a task therefore hold the wall FIXED and
    vary where the agent and goal start, which is exactly what makes the wall
    identifiable.  This is the family the single-scene format failed on: with one
    trajectory, a literal shove in fork's derive slot traced that detour and search
    dropped the obstacle entirely (obstacle_belief_misrepresents_obstacle came out
    False in all four runs).  A shove cannot bend four different routes, and a wall
    one cell off dies on whichever scene routes through it.

    Per scene the wall must still be load-bearing: placed on the true-belief BFS
    path, and required to cost EXTRA steps over the free-space shortest path.  A
    merely monotone "detour" is another shortest path that plain desire reproduces
    with different tie-breaking, so the wall would carry no explanatory load — that
    check is about the scene being well-posed, not about any particular rival, so it
    stays.

    A BYSTANDER sits on the phantom-wall cell.  k scenes pin down *where* the wall
    is (a wall one cell off dies on whichever scene routes through it) but they
    cannot on their own distinguish a wall the agent only believes in from a real
    wall stamped and then taken back down before the frame is rendered — the
    transient-wall rival

        (compose (compose (wall_at pr pc) (optimize (neg_dist gv) av)) (erase 3))

    reproduces every scene honestly, because on a bare grid there is nothing at
    (pr,pc) for the stamp to disturb.  That is a structural rival, not a per-scene
    coincidence, so no amount of k closes it; it was merely priced out while phase 1
    handed `fork` over atomically, and it took all 24 belief_wall solves in both
    phase-2 runs once fork had to be spelled out.  The cure is to give the world
    something at that cell to lose: an inert value `bv` (not av, not gv, never 3, so
    it neither blocks BFS nor is mistaken for a wall) that stays put for the whole
    trajectory.  `wall_at (pr,pc)` overwrites it and no undo can put it back — the
    DSL's only writers are wall_at (3) and clear_at (0) — so a world-level stamp is
    now visible in the rendered frame.  Only a program that stamps on a PRIVATE copy
    and commits just the agent leaves it standing.  This is `belief_witness`'s
    argument with a static object in place of a crossing agent, which is why it
    also closes the projection and z-order commits (snd_gg / overlay / underlay all
    carry the model's wall, or lose bv, into the world).

    `bv` is drawn deterministically from the latent, not sampled: `_collect` pools
    scenes by latent, so a per-scene draw would put different bystanders in one
    task's scenes and turn a value the program never names into a spurious degree of
    freedom (each distinct distractor value is otherwise a free literal under
    content_q).  It stays out of the latent tuple proper because no program names it.

    What the bystander does NOT close is the scope complements: `av` is the only
    value this family's derive moves, so sync_all and sync_except(gv) are
    extensionally sync_to_world(av) here no matter what else is on the grid.  That
    degeneracy is the false-obstacle family's job (it moves the goal in the model
    too), and is reported as a degeneracy rather than a rival.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in combos:
        def propose(rng, av=av, gv=gv):
            ar, ac = int(rng.integers(size)), int(rng.integers(size))
            gr, gc = int(rng.integers(size)), int(rng.integers(size))
            if (ar, ac) == (gr, gc) or abs(ar - gr) + abs(ac - gc) < 3:
                return None
            g = np.zeros((size, size), dtype=int)
            g[ar, ac] = av
            g[gr, gc] = gv

            # true-belief trajectory: source of on-path phantom-wall candidates.
            # Computed before the bystander is placed, which is harmless: bv is not
            # 3, so it blocks neither BFS nor `optimize`, and the path is the same
            # either way (the detour check below re-runs it on the final grid).
            direct = unfold(g, max_T, optimize(neg_distance(gv), av))
            path  = [_agent_pos(direct[t], av) for t in range(max_T)]
            inter = [p for p in path if p and p != (ar, ac) and p != (gr, gc)]
            if not inter:
                return None
            pr, pc = inter[int(rng.integers(len(inter)))]

            bv = _bystander_value(av, gv, (pr, pc))
            if g[pr, pc] != 0:
                return None
            g[pr, pc] = bv                        # the thing a world-level stamp eats

            direct = unfold(g, max_T, optimize(neg_distance(gv), av))
            gt = fork(compose(wall_at(pr, pc), optimize(neg_distance(gv), av)),
                      sync_to_world(av))
            x_full = unfold(g, max_T, gt)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (gr, gc)), None)
            if t_arrive is None or t_arrive < 3:
                return None
            T = t_arrive + 1
            x = x_full[:T].copy()
            if np.array_equal(x, direct[:T]):     # phantom wall must cause a detour
                return None
            if (T - 1) <= abs(ar - gr) + abs(ac - gc):
                return None                       # …and one that costs extra steps
            # the bystander must survive intact: the agent's model has a wall on its
            # cell, so `optimize` never routes through it and sync_to_world(av) never
            # lands av there.  gv is deliberately NOT checked — this family's
            # trajectory ends with av arriving ON the goal cell and overwriting it.
            # And no frame may read as containing a wall: the phantom is private, so
            # a 3 in the world would hand the transient rival its footing back.
            if not all((x[t] == bv).sum() == 1 and (x[t] == av).sum() == 1
                       for t in range(T)):
                return None
            if any((x[t] == 3).any() for t in range(T)):
                return None
            return ((av, gv, pr, pc), x,
                    {'kind': 'belief', 'av': av, 'gv': gv, 'pw': (pr, pc), 'bv': bv})

        # A wall cell has to recur across independently sampled placements for a
        # bucket to fill, so this family needs a far bigger proposal budget than the
        # single-scene generator did.
        tasks += _collect(propose, n_per_combo, k, rng, attempts=200000,
                          certify=_no_transient_wall)
    return tasks


def _bystander_value(av, gv, cell, size=SIZE):
    """The inert value that guards a phantom-wall cell — a function of the LATENT.

    Never av or gv (they are what the policy names) and never 3 (that is a wall).
    Keyed on the wall cell so the marker is not the same value in every task, but
    deterministic so all k scenes of one task agree — see the pooling note in
    `make_belief_tasks`."""
    pool = [v for v in (1, 2, 4, 5, 6, 7, 8, 9) if v not in (av, gv)]
    return pool[(cell[0] * size + cell[1]) % len(pool)]


def _no_transient_wall(scenes, m):
    """CLOSED-SET certification: no stamp-act-undo schedule reproduces every scene.

    The rival the bystander is there to kill is a world-level wall that is taken
    back down before the frame is rendered.  Under `unfold`'s one fixed per-frame
    fn there are only so many ways to write that with one stamp and one undo, and
    the set is finite because the grid is: for EVERY cell q, stamp a wall at q, run
    the policy, and remove it — either with `clear_at q` (the targeted undo) or
    `erase 3` (the wholesale one).  An undo at a cell other than q leaves the wall
    standing and is caught by the same sweep at q.  Also rules out the no-wall
    physics, so the detour is not merely a tie-break.

    Checked at task level: a rival only undercuts the intended program if it
    reproduces all k scenes.  Without the bystander every task in the family fails
    this at its own wall cell (and only there — k already pins the cell down)."""
    av, gv = m['av'], m['gv']
    seek = optimize(neg_distance(gv), av)

    def repro(prog):
        for s in scenes:
            try:
                if not np.array_equal(unfold(s[0], s.shape[0], prog), s):
                    return False
            except Exception:
                return False
        return True

    if repro(seek):                                   # plain desire, no wall at all
        return False
    size = scenes[0].shape[-1]
    for r in range(scenes[0].shape[-2]):
        for c in range(size):
            stamped = compose(wall_at(r, c), seek)
            if repro(compose(stamped, clear_at(r, c))) or repro(compose(stamped, erase(3))):
                return False
    return True


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


def make_witness_belief_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8,
                              k=K_SCENES):
    """False-belief hardened against the transient-wall rival by a witness agent.

    A second agent aw (seeking its own goal gw) traverses the phantom-wall cell on
    the real grid.  Because `unfold` iterates one fixed per-frame fn, any program
    that makes av detour by stamping a *real* wall must stamp it every frame and so
    clobbers the witness as it crosses — only a private-copy `fork` lets av see the
    wall while the witness passes through.  That argument is structural, which is
    why this family needs no rival battery: the crossing witness does the excluding.

    Latent: (av, gv, aw, gw, pr, pc) — both agent/goal pairs and the phantom wall.
    The k scenes hold all six fixed and vary the four placements, so the witness has
    to cross the same cell from k different routes.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    pool = [1, 2, 4, 5]
    for av, gv in combos:
        rest = [v for v in pool if v not in (av, gv)]

        def propose(rng, av=av, gv=gv, rest=rest):
            perm = rng.permutation(rest)
            aw, gw = int(perm[0]), int(perm[1])
            allcells = [(r, c) for r in range(size) for c in range(size)]
            idx = rng.permutation(len(allcells))
            (ar, ac), (gr, gc), (wr, wc), (wgr, wgc) = [allcells[i] for i in idx[:4]]
            if abs(ar - gr) + abs(ac - gc) < 3:
                return None
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
                return None
            pr, pc = cand[int(rng.integers(len(cand)))]

            prog = _witness_belief_program(av, gv, aw, gw, pr, pc)
            x_full = unfold(g, max_T, prog)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (gr, gc)), None)
            if t_arrive is None or t_arrive < 3:
                return None
            T = t_arrive + 1
            x = x_full[:T].copy()
            if np.array_equal(x, av_clean[:T]):                 # av must really detour
                return None
            if not any(_agent_pos(x[t], aw) == (pr, pc) for t in range(T)):
                return None                                      # witness must cross the wall cell
            # all four values present and distinct up to (not incl.) av's arrival
            if not all((x[t] == v).sum() == 1
                       for t in range(T - 1) for v in (av, gv, aw, gw)):
                return None                                      # reject collisions/clobbers
            return ((av, gv, aw, gw, pr, pc), x,
                    {'kind': 'belief', 'av': av, 'gv': gv,
                     'aw': aw, 'gw': gw, 'pw': (pr, pc)})

        tasks += _collect(propose, n_per_combo, k, rng, attempts=400000)
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


def _goal_scope_certified(scenes, derive_of, av, gv, assemble=fork):
    """Certify the goal-displacement commit field — everything that CAN discriminate.

    Full `_scope_complements_all_fail` certification is impossible for this family,
    intrinsically: the derive perturbs exactly one world value (the goal), so
    sync_except(gv) moves everything the derive left alone — and a value the derive
    left alone is already at its model position, so moving it is a no-op.
    sync_except(gv) IS sync_to_world(av) on EVERY expressible scene of this family,
    not a sampling accident (verified: 0% of candidate scenes separate them), and
    the distractors do not change that: an inert bystander sits at the same cell in
    both channels.  Prising the two apart needs a third world value the derive is
    FORCED to perturb, and forcing that requires blocking semantics — which is the
    false-obstacle family.  So the census's 'degenerate' label on goal solves is a
    provably benign spelling of the same agency commit, never a rival, and the
    forced-literal claim lives with fob.

    `assemble(derive, commit)` builds the family's whole per-frame program around
    the commit under test — plain `fork` for Sally-Anne, and the observer's bare
    seek composed in front of the fork for the two-observer family.

    What CAN be certified is that no OTHER scope commit reproduces the task under
    the canonical derive: sync_all and sync_except(k) for every k != gv must fail.
    This is a CLOSED set of rivals — the whole scope family, enumerated — which is
    why it survives the deletion of the open-ended coincidence batteries, and why
    more scenes could not replace it: sync_except(gv) does not merely happen to
    coincide with sync_to_world(av) here, it IS that function on any two-value scene.

    Checked at TASK level: a commit only undercuts the agency commit if it
    reproduces every scene, and the derive is per-scene (`derive_of(scene)`) because
    the shove is the same but the seek runs from a different placement.
    """
    def repro_all(commit):
        return all(np.array_equal(
                       unfold(s[0], s.shape[0], assemble(derive_of(s), commit)), s)
                   for s in scenes)

    def repro_any(commit):
        for s in scenes:
            try:
                if np.array_equal(unfold(s[0], s.shape[0],
                                         assemble(derive_of(s), commit)), s):
                    return True
            except Exception:
                pass
        return False

    try:
        if not repro_all(sync_to_world(av)):   # must be a valid literal-commit task
            return False
    except Exception:
        return False
    if repro_any(sync_all):                    # wholesale adoption must be WRONG
        return False
    ks = {int(v) for s in scenes for v in np.unique(s[0]) if v != 0 and v != gv}
    for kk in ks:
        if repro_any(sync_except(kk)):         # discriminating complements must fail
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


def make_goal_displacement_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0,
                                 max_T=8, k=K_SCENES, n_distract=1):
    """Sally-Anne: the agent walks to where it *believes* the goal is — displaced
    along a one- or two-step shove from its true position — while the true goal
    sits still.  Each combo cycles through a shuffled `_GOAL_CONTENT_SPECS`, so the
    family covers diverse propositional contents rather than one baked-in shove.

    This is the trunk of the belief curriculum: the one family that solves in every
    run (48/48 across all four), the one the behavioural probe discriminates on
    (`viz/behavioral_probe.py` — the belief compound walks to the believed cell, the
    shortest non-mental rival walks to the true one), and the one whose false belief
    is about an object rather than about the geometry, so it needs no wall and no
    blocking semantics.  The wall-based families are hardening around it, not the
    other way round.

    Latent: (av, gv, shove) — the two values and the propositional content.  Held
    fixed across the k scenes; EVERYTHING else varies, and the format only bites to
    the extent that it does:
      * the agent's start cell and the goal's cell (so the believed cell — an empty
        cell one or two steps off the goal — sits somewhere different in every
        scene: a phantom wall, or a seek-the-wall beacon, can be fitted to any ONE
        walk toward a fixed empty cell, but not to four that move with the goal);
      * the trajectory length (so nothing can be read off a shared arrival time);
      * the distractor objects — inert bystanders drawn from `n_distract` values the
        combo does not use (default 1), sitting at a different cell in every scene.
        They put more in the scene than the program names, so a beacon-style rival (a
        seek aimed at a third value that happens to sit on the route) has to find that
        same value at four different cells.  The value SET is per task, not per scene,
        and it is deliberately small: see `_add_distractors` — every distinct value
        the set mentions is a free literal under `content_q`, so bystanders are bought
        with search budget.  Measured at 90 s against the smoke library: 2 free values
        (none) solved a one-shove task in 24 s, 3 (`n_distract=1`) in 77 s, 4 not at
        all.  One bystander is the setting that pays for itself.
    `_placements_vary` asserts the first two actually came out varied rather than
    pooling four near-copies.

    Necessity (per scene, all structural — the task must be well-posed):
      * the agent settles exactly on the believed (displaced) cell, never on the
        true goal cell — so it is not plain desire, which settles ON the goal;
      * the true goal keeps its value & position in every frame — the stationary
        witness that rules out any program that *actually* moves the goal;
      * so does every distractor: a bystander the agent walked over would be a
        scene the program does not explain (nothing in the program names it).
    And per task: every scope complement that CAN discriminate fails
    (`_goal_scope_certified`) — sync_all and sync_except(k != gv) never reproduce
    it.  sync_except(gv) is exempt because it is provably sync_to_world(av) on every
    scene of this family (the derive perturbs only gv); the census reports those
    solves as 'degenerate' (a benign spelling of the same agency commit), and the
    family that forces the literal spelling is fob.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in combos:
        dvals = _distractor_values(av, gv)
        order = list(_GOAL_CONTENT_SPECS)
        rng.shuffle(order)
        made = 0
        # Walk the content shapes in turn rather than sampling them, so the family
        # keeps covering diverse propositional contents: with pooling, a purely
        # random spec would let whichever shape generates most easily take every
        # slot, and stitch would bake that one shove into the abstraction body.
        for spec in (order * ((n_per_combo // len(order)) + 1))[:]:
            if made >= n_per_combo:
                break
            vecs = [DIRS[dn] for dn in spec]
            derive = _seq(*([step(gv, d) for d in vecs]
                            + [optimize(neg_distance(gv), av)]))
            # this task's bystander values, drawn once and shared by its k scenes
            dset = [int(v) for v in rng.permutation(dvals)[:n_distract]]

            def propose(rng, av=av, gv=gv, spec=spec, vecs=vecs, dset=dset):
                ar, ac = int(rng.integers(size)), int(rng.integers(size))
                gr, gc = int(rng.integers(size)), int(rng.integers(size))
                if (ar, ac) == (gr, gc):
                    return None
                # believed (displaced) goal cell: every intermediate shove cell must
                # stay in bounds and clear of the agent, or the model shove
                # clobbers/vanishes
                br, bc, ok = gr, gc, True
                shove_cells = []
                for dr, dc in vecs:
                    br, bc = br + dr, bc + dc
                    shove_cells.append((br, bc))
                    if not (0 <= br < size and 0 <= bc < size) or (br, bc) == (ar, ac):
                        ok = False
                        break
                if not ok or (br, bc) == (gr, gc):
                    return None
                if abs(ar - br) + abs(ac - bc) < 3:   # need a real trajectory to the belief
                    return None
                g = np.zeros((size, size), dtype=int)
                g[ar, ac] = av
                g[gr, gc] = gv
                # inert bystanders: never on the believed cell (the agent settles
                # there) and never on a shove cell (the model's `step gv` would
                # overwrite them in the private copy).
                dist_cells = _add_distractors(
                    rng, g, {(ar, ac), (gr, gc), *shove_cells}, dset)
                if dist_cells is None:
                    return None

                prog = _goal_displacement_program(av, gv, vecs)
                x_full = unfold(g, max_T, prog)
                t_arrive = next((t for t in range(max_T)
                                 if _agent_pos(x_full[t], av) == (br, bc)), None)
                if t_arrive is None or t_arrive < 3:
                    return None
                T = t_arrive + 1
                x = x_full[:T].copy()
                # true goal must stay put (stationary witness) and never be clobbered
                if any(_agent_pos(x[t], gv) != (gr, gc) for t in range(T)):
                    return None
                # agent must never step onto the true goal cell
                if any(_agent_pos(x[t], av) == (gr, gc) for t in range(T)):
                    return None
                # …nor over a bystander: the program names none of them, so a scene
                # where one vanishes is a scene the program does not explain
                if not _bystanders_still(x, dist_cells):
                    return None
                # must diverge from a plain goal-seek (else it is mere desire)
                if np.array_equal(x, unfold(g, T, optimize(neg_distance(gv), av))):
                    return None
                return ((av, gv, spec), x,
                        {'kind': 'belief', 'av': av, 'gv': gv,
                         'displaced_to': (br, bc), 'dirs': spec,
                         'distractors': dist_cells})

            def certify(scs, m, derive=derive, av=av, gv=gv):
                # the bystanders must move around too — a value pinned to one cell in
                # all k scenes is exactly the beacon a rival could seek.  (Vacuous
                # when n_distract=0, the A/B baseline: there is nothing to pin.)
                placements = {tuple(sorted(sm['distractors'].items()))
                              for sm in m['per_scene']}
                return ((n_distract == 0 or len(placements) >= 2)
                        and _placements_vary(scs, av, gv)
                        and _goal_scope_certified(scs, lambda _s: derive, av, gv))

            got = _collect(propose, 1, k, rng, attempts=60000, certify=certify)
            tasks += got
            made += len(got)
    return tasks


# ── Task family 2: two observers, one world ────────────────────────────────────
# Two agents, one goal, one grid.  `ao` seeks the goal where it actually is; `av`
# seeks it one step off, on a private copy.  The two walk to DIFFERENT cells out of
# an identical world — and since `unfold` iterates ONE per-frame fn over ONE grid,
# no single-grid program can send two agents with the same goal value to two
# different cells.  Whatever explains the divergence has to be attached to the
# agent, not to the world: a private model.
#
# This family replaced, and then DELETED, the contradictory-beliefs family
# (`belief_dual` — two agents each detouring around their own phantom wall).  Dual
# asked for two phantom walls at once: six literals in the derive, two nested forks,
# ~12 candidate scenes per 20 000 tries, and 0/24 solved in every run including the
# 3600 s-per-task one.  Two observers buys the same argument for one direction
# literal — one fork, over a derive the Sally-Anne family already makes the searcher
# pay for, plus a bare seek — so nothing is left for dual to contribute.
#
# It is also the stronger claim.  Dual gave BOTH agents a fork, so the library's
# constructor applied to everything that moved.  Here the library must invent the
# agent constructor and then apply it to exactly the one agent whose behaviour
# requires it, leaving the other bare — selective attribution of mental state, which
# is what a theory of mind is for.

def _two_observer_program(ao, av, gv, d):
    """Per-frame transition for two observers of one world:

        (compose (optimize (neg_dist gv) ao)
                 (fork (compose (step gv d) (optimize (neg_dist gv) av))
                       (sync_to_world av)))

    `ao` (the observer) moves on the real grid toward the real goal — plain desire,
    no private copy, no fork.  Then `av` shoves the goal by one step on a copy of
    the resulting world, seeks it there, and commits only its own move.  One goal
    value, one grid, two destinations.
    """
    return compose(optimize(neg_distance(gv), ao),
                   fork(compose(step(gv, d), optimize(neg_distance(gv), av)),
                        sync_to_world(av)))


# Single-step shoves only — the cheap content (one direction literal, no nested
# compose), which is the point of this family.  Vertical for the
# reason `_GOAL_CONTENT_SPECS` documents: `optimize`'s tie-break walks the vertical
# leg first, so a horizontally displaced believed cell always lies on the direct
# seek path and the scene degenerates into a truncated plain desire.
_OBSERVER_CONTENT_SPECS = [('up',), ('down',)]


def make_two_observer_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8,
                            k=K_SCENES):
    """Two agents, one goal, one world: one true belief, one false.

    Latent: (ao, av, gv, shove) — the observer's value, the believer's value, the
    shared goal, and the propositional content.  The k scenes hold those fixed and
    vary both agents' start cells, the goal's cell and the trajectory length
    (`_placements_vary`), so the believed cell moves with the goal across the set
    and no fixed coordinate can stand in for it.

    Necessity (per scene, all structural):
      * the believer settles on the believed cell and never touches the true goal —
        so its walk is not plain desire;
      * the observer's realised walk is EXACTLY its own wall-free greedy seek to the
        true goal — so the scene attributes nothing to it: whatever the searcher
        wraps around the believer must not be wrapped around the observer, or it
        would predict the wrong path for it;
      * the two agents end on different cells (that is the family's whole content);
      * the goal keeps its value and cell in every frame — the stationary witness
        that rules out any program that really shoves the goal (which would, in a
        single-grid world, drag the observer along with it);
      * all three values present exactly once in every frame — no agent walks
        through the other.

    Per task, `_goal_scope_certified` runs the same closed-set commit check as
    Sally-Anne, with the observer's seek composed in front of the fork: sync_all and
    every discriminating sync_except fail, so no wholesale-adoption commit undercuts
    the single-value agency commit.  sync_except(gv) remains the same provably
    benign spelling of sync_to_world(av) it is there (the derive perturbs only gv;
    the observer sits at the same cell in both channels).
    """
    rng = np.random.default_rng(seed)
    tasks = []
    pool = [1, 2, 4, 5, 6, 7, 8, 9]
    for av, gv in combos:
        rest = [v for v in pool if v not in (av, gv)]
        order = list(_OBSERVER_CONTENT_SPECS)
        rng.shuffle(order)
        made = 0
        for spec in (order * ((n_per_combo // len(order)) + 1))[:]:
            if made >= n_per_combo:
                break
            (dn,) = spec
            d = DIRS[dn]
            derive = compose(step(gv, d), optimize(neg_distance(gv), av))

            def propose(rng, target=None, av=av, gv=gv, rest=rest, spec=spec, d=d):
                ao = (int(rng.choice(rest)) if target is None else target[0])
                allcells = [(r, c) for r in range(size) for c in range(size)]
                idx = rng.permutation(len(allcells))
                (ar, ac), (gr, gc), (or_, oc) = [allcells[i] for i in idx[:3]]
                br, bc = gr + d[0], gc + d[1]              # the believed goal cell
                if not (0 <= br < size and 0 <= bc < size):
                    return None
                if (br, bc) in {(ar, ac), (gr, gc), (or_, oc)}:
                    return None
                if abs(ar - br) + abs(ac - bc) < 3:        # a real walk to the belief
                    return None
                # The observer must still be walking when the believer arrives: if it
                # reached the goal it would stand on the goal cell and clobber the
                # witness (and, from the next frame, the believer's shove would have
                # nothing to shove).
                d_obs = abs(or_ - gr) + abs(oc - gc)
                if d_obs < 3:
                    return None

                g = np.zeros((size, size), dtype=int)
                g[ar, ac] = av
                g[gr, gc] = gv
                g[or_, oc] = ao

                prog = _two_observer_program(ao, av, gv, d)
                x_full = unfold(g, max_T, prog)
                t_arrive = next((t for t in range(max_T)
                                 if _agent_pos(x_full[t], av) == (br, bc)), None)
                if t_arrive is None or t_arrive < 3:
                    return None
                T = t_arrive + 1
                if d_obs < T:                              # observer would arrive in frame
                    return None
                x = x_full[:T].copy()

                # the goal is the stationary witness; the believer never reaches it
                if any(_agent_pos(x[t], gv) != (gr, gc) for t in range(T)):
                    return None
                if any(_agent_pos(x[t], av) == (gr, gc) for t in range(T)):
                    return None
                # no clobbering: three values, one cell each, every frame
                if not all(int((x[t] == v).sum()) == 1
                           for t in range(T) for v in (av, gv, ao)):
                    return None
                # the believer must diverge from its own true-belief seek…
                clean_av = unfold(g, T, optimize(neg_distance(gv), av))
                if all(_agent_pos(x[t], av) == _agent_pos(clean_av[t], av)
                       for t in range(T)):
                    return None
                # …and the observer must NOT: its walk is the bare seek, exactly
                clean_ao = unfold(g, T, optimize(neg_distance(gv), ao))
                if any(_agent_pos(x[t], ao) != _agent_pos(clean_ao[t], ao)
                       for t in range(T)):
                    return None
                # and the two really part company
                if _agent_pos(x[-1], ao) == _agent_pos(x[-1], av):
                    return None
                return ((ao, av, gv, spec), x,
                        {'kind': 'belief', 'av': av, 'gv': gv, 'observer': ao,
                         'displaced_to': (br, bc), 'dirs': spec})

            def certify(scs, m, derive=derive, av=av, gv=gv):
                ao = m['observer']

                def assemble(dv, commit):
                    return compose(optimize(neg_distance(gv), ao), fork(dv, commit))

                return (_placements_vary(scs, av, gv)
                        and _goal_scope_certified(scs, lambda _s: derive, av, gv,
                                                  assemble=assemble))

            got = _collect(propose, 1, k, rng, attempts=60000, certify=certify,
                           target_tries=4000)
            tasks += got
            made += len(got)
    return tasks


# ── shared geometry + plumbing helpers ─────────────────────────────────────────
# `_clean_path_table` was written for the deleted contradictory-beliefs generator
# (see the two-observer note above).  It survives because the false-obstacle
# generator inverts it the same way (`_fob_fits`): a phantom wall is placeable at a
# cell only for placements whose clean path covers it.

_PATH_CELLS = {}


def _clean_path_table(size, max_T):
    """(agent cell, goal cell) -> the set of cells the agent's wall-free BFS walk
    visits.  Pure geometry: `optimize` blocks only on walls, and there are none on a
    clean grid, so the walk depends on the two cells and not on which values name
    them — one table serves every combo.  Cached; 625 unfolds on a 5x5.

    Used by the targeted samplers.  A phantom wall is only placeable at cell p for a
    placement whose clean path covers p, so inverting the table turns "sample four
    cells and hope the wall lands right" into a draw from a pre-filtered list.
    """
    key = (size, max_T)
    if key in _PATH_CELLS:
        return _PATH_CELLS[key]
    cells = [(r, c) for r in range(size) for c in range(size)]
    table = {}
    for a in cells:
        for gcell in cells:
            if a == gcell or abs(a[0] - gcell[0]) + abs(a[1] - gcell[1]) < 3:
                continue
            g = np.zeros((size, size), dtype=int)
            g[a] = 1
            g[gcell] = 2
            traj = unfold(g, max_T, optimize(neg_distance(2), 1))
            table[(a, gcell)] = frozenset(
                p for p in (_agent_pos(traj[t], 1) for t in range(max_T)) if p)
    _PATH_CELLS[key] = table
    return table


def _seq(*fs):
    "left-fold compose: _seq(a, b, c)(x) = c(b(a(x)))"
    prog = fs[0]
    for f in fs[1:]:
        prog = compose(prog, f)
    return prog


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
_OPPOSITE = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


def belief_variant(m):
    "Fine label for a kind='belief' task (goal / observers / wall / witness / false-obstacle)."
    if 'real_wall' in m:                 # false about BOTH obstacle and goal; also carries
        return 'belief_false_obstacle'   # displaced_to, so test it BEFORE the goal branch
    if 'observer' in m:                  # carries displaced_to too (its believer is a
        return 'belief_observers'        # Sally-Anne agent), so likewise test it first
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

    Wall/witness rivals are the transient-wall family (stamp / act / erase, in the
    interpreter's one fixed per-frame fn) plus the no-wall physics — the exact programs
    the corresponding `_*_rival_explainable` battery enumerated.  The displaced-goal
    families (Sally-Anne, two-observer) get the same three shapes with the goal in place
    of the wall: no belief at all, a permanent shove in the WORLD (which the stationary
    goal rules out), and the transient shove that puts the goal back before the frame is
    rendered (which the arrival cell rules out).  All are priced against the found
    belief program.
    """
    var = belief_variant(m)
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
        undo   = [_stepstr(gv, _OPPOSITE[dn]) for dn in reversed(m['dirs'])]
        return [
            ('pure desire (no belief)',   _seek(gv, av)),
            ('shove goal in world',       _seq_str(*shoves, _seek(gv, av))),
            # the displaced-goal analogue of the transient WALL: shove the goal, act,
            # put it back, so every rendered frame shows the goal where it belongs.
            # It dies on the arrival frame — the agent settles on the believed cell,
            # which is exactly where the transiently shoved goal is standing, so the
            # restore has nothing left to move and the goal vanishes.
            ('transient shove (shove/act/restore)',
             _seq_str(*shoves, _seek(gv, av), *undo)),
        ]
    if var == 'belief_observers':
        av, gv, ao = m['av'], m['gv'], m['observer']
        (dn,) = m['dirs']
        shove, undo = _stepstr(gv, dn), _stepstr(gv, _OPPOSITE[dn])
        oseek, bseek = _seek(gv, ao), _seek(gv, av)
        return [
            ('both agents seek the goal',  _seq_str(oseek, bseek)),
            ('shove goal in world',        _seq_str(oseek, shove, bseek)),
            ('transient shove (shove/act/restore)',
             _seq_str(oseek, shove, bseek, undo)),
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
    # belief_wall: transient real wall (the tightest single-grid analogue) + no-wall
    # physics.  Both are now expressiveness-excluded rather than behavioural
    # competitors: a bystander sits on the phantom-wall cell, so the stamp destroys it
    # and the transient rival renders a scene the world never showed (it still gets the
    # agent's PATH right, which is why the margin table reports it separately).  Spelled
    # out anyway, exactly as the false-obstacle rivals are, so the table shows the
    # closest non-mental analogues failing rather than silently omitting them.
    av, gv, pw = m['av'], m['gv'], m['pw']
    W, C = _wall(*pw), _clear(*pw)
    return [
        ('no wall (physics)',                _seek(gv, av)),
        ('transient wall (stamp/act/erase)', _seq_str(W, _seek(gv, av), C)),
        ('transient wall (stamp/act/erase 3)',
         _seq_str(W, _seek(gv, av), '(erase 3)')),
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

def _scope_complements_all_fail(scenes, derive_of, av):
    """True iff sync_to_world(av) reproduces every scene under the family's derive
    but NO scope complement reproduces any of them — i.e. the single-value agency
    commit is the unique commit for this task.

    A CLOSED set of rivals (the whole scope family: sync_all, and sync_except over
    every world value, the real wall 3 included), which is why it outlives the
    deleted coincidence batteries.  Checked at task level, with the derive taken per
    scene since it names that scene's own real-wall cell.
    """
    def repro(commit, s):
        try:
            return np.array_equal(
                unfold(s[0], s.shape[0], fork(derive_of(s), commit)), s)
        except Exception:
            return False

    if not all(repro(sync_to_world(av), s) for s in scenes):
        return False                        # must be a valid literal-commit task
    if any(repro(sync_all, s) for s in scenes):
        return False                        # wholesale adoption must be WRONG
    world_vals = {int(v) for s in scenes for v in np.unique(s[0]) if v != 0}
    for kk in world_vals:                   # every all-but-one commit must be WRONG
        if any(repro(sync_except(kk), s) for s in scenes):
            return False
    return True


_FOB_FITS = {}


def _fob_fits(dg, bw, size, max_T):
    """(agent, goal) placements whose believed-world walk passes through the believed
    wall `bw`, for a goal shoved by `dg`.  The targeted counterpart of the blind
    sampler, cached per (dg, bw): fob pins five literals, so hitting them by chance
    is hopeless."""
    key = (dg, bw, size, max_T)
    if key in _FOB_FITS:
        return _FOB_FITS[key]
    table = _clean_path_table(size, max_T)
    out = []
    for (a, bg), cs in table.items():
        gcell = (bg[0] - dg[0], bg[1] - dg[1])
        if not (0 <= gcell[0] < size and 0 <= gcell[1] < size):
            continue
        if gcell == a or bw in (a, gcell, bg) or bw not in cs:
            continue
        out.append((a, gcell, bg))
    _FOB_FITS[key] = out
    return out


def make_false_obstacle_belief_tasks(n_per_combo, combos=COMBOS, size=SIZE,
                                     seed=0, max_T=8, k=K_SCENES):
    """Single agent, false about BOTH the obstacle and the goal location.

    Forces the literal single-value commit sync_to_world(av): the agent hallucinates
    an obstacle that is not there while heading for a goal it mis-locates, and both
    the real wall (value 3) and the real goal stay put in the world.  A task is kept
    only if sync_to_world(av) reproduces every scene AND every scope complement
    (sync_all, sync_except k for every world value k, incl. the wall) fails on all of
    them — so no 'move everything but ⋯' commit can undercut the agency commit.

    Latent: (av, gv, believed wall, real wall, shove direction) — five literals, the
    most of any family, since the derive names both walls and the displacement.  The
    k scenes hold all five fixed and move the agent and goal, which is what retires
    the transient-wall sweep the old `_false_obstacle_rival_explainable` ran: a
    stamp-and-clear schedule fitted to one realised path does not survive the same
    two wall cells being approached from four directions.

    The derive does NOT erase the real wall.  It used to open with
    `clear_at(real wall)` — the agent believing the obstacle to be at (br,bc)
    *instead of* (wr,wc) — but the real wall is drawn from the free cells, so it
    almost never obstructs the believed-world walk and that token was measurably
    vacuous: dropping it still reproduced every scene of every task in the family.
    That mattered, because certification is DERIVE-RELATIVE.  Whether a scope
    complement undercuts the agency commit depends on what the derive leaves lying
    in the model, so certifying against a derive with a redundant token in it
    certifies the wrong program: the searcher prices the SHORTER derive, and against
    that one `sync_except(gv)` reproduced a quarter of the family — which is how a
    `sync_all` commit turned up on a real-wall scene in the jul-26 phase-2 runs and
    tripped the "should be impossible" warning downstream.  Certifying the derive
    the searcher will actually reach for closes it at generation time.  The family's
    claim is unchanged: the agent is still wrong about the obstacle (it detours
    around a wall that is not there) and wrong about the goal, and both mistakes are
    UNCOMMITTED, which is what every scope complement must drag and fail on.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in combos:
        def propose(rng, target=None, av=av, gv=gv):
            if target is None:
                ar, ac = int(rng.integers(size)), int(rng.integers(size))
                gr, gc = int(rng.integers(size)), int(rng.integers(size))
                if (ar, ac) == (gr, gc) or abs(ar - gr) + abs(ac - gc) < 3:
                    return None
                # believed goal: the real goal shoved one cell in direction dg (on
                # grid, not onto the agent).
                dgname = str(rng.choice(list(DIRS)))
                dg = DIRS[dgname]
                bgr, bgc = gr + dg[0], gc + dg[1]
                if not (0 <= bgr < size and 0 <= bgc < size) or (bgr, bgc) == (ar, ac):
                    return None
            else:
                _, _, tbw, trw, dgname = target
                dg = DIRS[dgname]
                fits = _fob_fits(dg, tbw, size, max_T)
                fits = [f for f in fits if trw not in (f[0], f[1], f[2])]
                if not fits:
                    return None
                (ar, ac), (gr, gc), (bgr, bgc) = fits[int(rng.integers(len(fits)))]

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
            if target is None:
                inter = [p for p in bpath
                         if p and p not in ((ar, ac), (gr, gc), (bgr, bgc))]
                if not inter:
                    return None
                br, bc = inter[int(rng.integers(len(inter)))]      # believed wall
            else:
                br, bc = target[2]
                if (br, bc) not in bpath:
                    return None

            # real wall: an empty cell distinct from the believed wall and not on the
            # agent's realised path (checked below via the distinctness gate).
            if target is None:
                free = [(r, c) for r in range(size) for c in range(size)
                        if g[r, c] == 0 and (r, c) != (br, bc)]
                if not free:
                    return None
                wr, wc = free[int(rng.integers(len(free)))]
            else:
                wr, wc = target[3]
                if g[wr, wc] != 0 or (wr, wc) == (br, bc):
                    return None
            g[wr, wc] = 3                                       # REAL wall in the world

            # Bystander on the BELIEVED-wall cell, for the same reason belief_wall has
            # one (see make_belief_tasks).  The real wall makes `erase 3` fail here —
            # it would delete a wall the world keeps — but `clear_at (br,bc)` leaves the
            # real wall alone and takes only the phantom back down, so the transient
            # rival still reproduced 9/24 of this family until the phantom cell had
            # something on it to lose.  The believed wall is stamped on the PRIVATE copy,
            # so sync_to_world(av) leaves bv standing in the world.
            bv = _bystander_value(av, gv, (br, bc))
            if g[br, bc] != 0:
                return None
            g[br, bc] = bv

            derive = _seq(wall_at(br, bc), step(gv, dg),
                          optimize(neg_distance(gv), av))
            prog = fork(derive, sync_to_world(av))
            x_full = unfold(g, max_T, prog)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (bgr, bgc)), None)
            if t_arrive is None or t_arrive < 3:
                return None
            T = t_arrive + 1
            x = x_full[:T].copy()

            # the wall and goal must stay put in the world for every frame, and all
            # four world-values (av, gv, real wall, and — until arrival — the cell the
            # agent came from) stay present & distinct: no clobber, no accidental move.
            if any(x[t][gr, gc] != gv or x[t][wr, wc] != 3 for t in range(T)):
                return None
            # …and the bystander survives intact on the believed-wall cell: the model
            # has a wall there, so the agent never routes through it and the commit
            # never lands av on it.
            if not all((x[t] == v).sum() == 1 for t in range(T) for v in (av, gv, bv)):
                return None
            if not all((x[t] == 3).sum() == 1 for t in range(T)):
                return None
            # the agent must really detour (not the clean displaced-goal path)
            if [_agent_pos(x[t], av) for t in range(T)] == bpath[:T]:
                return None
            return ((av, gv, (br, bc), (wr, wc), dgname), x,
                    {'kind': 'belief', 'av': av, 'gv': gv,
                     'pw': (br, bc), 'real_wall': (wr, wc), 'bv': bv,
                     'displaced_to': (bgr, bgc), 'dir': dgname})

        def certify(scs, m, av=av, gv=gv):
            # the derive certified here is EXACTLY the one `propose` ran — see the
            # derive-relativity note in the docstring.
            (br, bc), dgname = m['pw'], m['dir']
            derive = _seq(wall_at(br, bc), step(gv, DIRS[dgname]),
                          optimize(neg_distance(gv), av))
            return _scope_complements_all_fail(scs, lambda _s: derive, av)

        tasks += _collect(propose, n_per_combo, k, rng, attempts=30000,
                          target_tries=4000, certify=certify)
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

def make_overlay_tasks(n, size=SIZE, vals=(1, 4), seed=0, max_T=6, k=K_SCENES):
    """fork without sync: a value leaves a trail.

        (fork (step v d) overlay)

    Each frame overlays the grid with its one-step shift, so the output depends on
    BOTH the grid and the transform — no single non-branching fn (step/optimize)
    reproduces it, hence fork is *necessary*.  The commit is `overlay`, not `sync`.
    Pure motion blur / trail rendering: nothing mental.
    """
    rng = np.random.default_rng(seed)

    def propose(rng):
        v = int(rng.choice(vals))
        dname = str(rng.choice(list(DIRS)))
        dr, dc = DIRS[dname]
        T = int(rng.integers(3, max_T))
        # keep the whole trail on-grid
        r_lo, r_hi = max(0, -dr * (T - 1)), min(size - 1, size - 1 - dr * (T - 1))
        c_lo, c_hi = max(0, -dc * (T - 1)), min(size - 1, size - 1 - dc * (T - 1))
        if r_lo > r_hi or c_lo > c_hi:
            return None
        r = int(rng.integers(r_lo, r_hi + 1))
        c = int(rng.integers(c_lo, c_hi + 1))
        g = np.zeros((size, size), dtype=int)
        g[r, c] = v
        x = unfold(g, T, fork(step(v, DIRS[dname]), overlay))
        return (v, dname), x, {'kind': 'overlay', 'val': v, 'dir': dname}

    return _collect(propose, n, k, rng, attempts=4000)


def _overlay_trail_explainable(scenes):
    """True if a fork(step v d, OVERLAY) trail reproduces x for any (v, d).

    The overlay and underlay trails are extensionally IDENTICAL for a lone mover
    (the union of trail cells is the same) and diverge only where the trail crosses
    a bystander of another value: overlay lets the shifted copy paint over it,
    underlay keeps the world's pixel.  A task the overlay commit also reproduces
    therefore fails to make underlay necessary, so we reject it — the corner-
    uniqueness discipline, over the closed set of z-order commits.
    """
    for v in {u for s in scenes for u in _grid_vals(s[0])}:
        for d in DIRS.values():
            if _repro_any(scenes, fork(step(v, d), overlay)):
                return True
    return False


def make_underlay_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, max_T=6, k=K_SCENES):
    """z-order complement of overlay in a FORK context: a world-wins motion trail.

        (fork (step v d) underlay)

    Same fork-required motion blur as make_overlay_tasks, but the commit is
    `underlay` (the WORLD's nonzero cells win ties) instead of `overlay` (the
    derived copy wins).  For a lone mover the two are indistinguishable, so each
    scene puts an occluding bystander of a *different* value on the mover's path:
    when the trail reaches it, overlay overwrites the bystander and the trail runs
    on, while underlay preserves the bystander and the trail flows *under* it (and
    is blocked there).  We reject any task the OVERLAY commit also reproduces
    (`_overlay_trail_explainable` — the crossing must actually happen within T
    frames), so underlay is necessary; fork is necessary because the output depends
    on both channels.  Pure graphics (z-order); no mind.
    """
    rng = np.random.default_rng(seed)

    def propose(rng, target=None):
        if target is None:
            v, b = (int(u) for u in rng.permutation(vals)[:2])  # distinct mover & bystander
        else:
            v, _, b = target
        dname = str(rng.choice(list(DIRS))) if target is None else target[1]
        dr, dc = DIRS[dname]
        sign = dr + dc                              # ±1; exactly one of dr,dc is nonzero
        gap = int(rng.integers(1, size - 1))        # mover→bystander gap along the path
        T = int(rng.integers(gap + 2, max_T + 1))   # enough frames for the trail to reach it
        if T < 3:
            return None
        # place mover and bystander colinearly along d, both on-grid, bystander ahead
        lo, hi = (0, size - 1 - gap) if sign > 0 else (gap, size - 1)
        if lo > hi:
            return None
        m_idx = int(rng.integers(lo, hi + 1))
        b_idx = m_idx + sign * gap
        L = int(rng.integers(0, size))              # fixed perpendicular line
        g = np.zeros((size, size), dtype=int)
        if dc != 0:                                 # horizontal travel: vary column
            g[L, m_idx], g[L, b_idx] = v, b
        else:                                       # vertical travel: vary row
            g[m_idx, L], g[b_idx, L] = v, b

        x = unfold(g, T, fork(step(v, DIRS[dname]), underlay))
        if np.array_equal(x[0], x[-1]):             # something must move
            return None
        return (v, dname, b), x, {'kind': 'underlay', 'val': v, 'dir': dname,
                                  'bystander': b}

    def certify(scs, m):
        # underlay must be REQUIRED: the trail has to cross the bystander somewhere,
        # or the overlay corner reproduces the task and the z-order choice is idle.
        return not _overlay_trail_explainable(scs)

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000,
                    certify=certify)


def make_comet_tasks(n, size=SIZE, combos=COMBOS, seed=0, k=K_SCENES):
    """fork without sync, VARYING THE DERIVE: a goal-seeker leaves a comet trail.

        (fork (optimize (neg_dist gv) av) overlay)

    Same overlay commit as make_overlay_tasks, but the derive is a goal-directed
    seek (desire's `optimize (neg_dist gv) av`) instead of a fixed `step v d`.  The
    agent av greedily approaches goal gv and overlay unions each step onto the
    trail, so av's whole L-shaped path is rendered — a comet trail.  The seek bends
    (the goal is off-axis), so no fixed-direction step-trail reproduces it
    (`_overlay_trail_explainable`) — fork AND the seek-derive are both necessary.  This
    exhibits fork's derive slot as a *general* fn, not one wired to `step`: the same
    fork/overlay motion-blur wrapper carries desire's utility policy.  Nothing
    mental (utility-driven motion + graphics).

    Geometry: the goal sits strictly up-LEFT of the agent so the row-major-first av
    cell is always the moving FRONT (`optimize` advances `agents[0]`); other
    orientations make agents[0] the tail and the trail collapses instead of growing.
    """
    rng = np.random.default_rng(seed)

    def propose(rng, target=None):
        if target is None:
            av, gv = (int(u) for u in combos[int(rng.integers(len(combos)))])
        else:
            av, gv = target
        ar, ac = int(rng.integers(2, size)), int(rng.integers(2, size))
        gr, gc = int(rng.integers(0, ar)), int(rng.integers(0, ac))   # strictly up-left
        L = (ar - gr) + (ac - gc)
        if not (3 <= L <= 5):
            return None
        g = np.zeros((size, size), dtype=int)
        g[ar, ac], g[gr, gc] = av, gv
        T = L + 1
        x = unfold(g, T, fork(optimize(neg_distance(gv), av), overlay))
        if x[-1][gr, gc] != av:                       # the comet head must reach the goal
            return None
        if int((x[-1] == av).sum()) != L + 1:         # clean trail: one cell per step, no collapse
            return None
        return (av, gv), x, {'kind': 'comet', 'av': av, 'gv': gv}

    def certify(scs, m):
        # the seek-bend must be required: no fixed-direction step-trail may do it
        return not _overlay_trail_explainable(scs)

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000,
                    certify=certify)


def make_registration_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, n_distract=2,
                            k=K_SCENES):
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

    def propose(rng, target=None):
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        need = 2 * (1 + n_distract)
        if len(cells) < need or len(vals) < 1 + n_distract:
            return None
        perm = rng.permutation(vals).tolist()
        if target is not None:                     # pin the registered value
            v = target[0]
            perm = [v] + [u for u in perm if u != v]
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
            return None
        got = _pair_propose(rng, ('sync_to_world', sync_to_world(v)),
                            (working, template), {'kind': 'registration', 'val': v})
        return None if got is None else ((v,), got[0], got[1])

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000)


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


def _repro_any(scenes, f):
    """True if `f` reproduces ANY scene of the task.

    The task-level form of the corner-uniqueness checks: a rival corner only
    undercuts the intended one if it reproduces the whole task, so rejecting on
    "reproduces some scene" is the conservative reading and the one kept here — the
    corner families are one node deep, and a tie on even a single scene is worth
    avoiding when a fresh sample is nearly free.
    """
    return any(_reproduces(s[0], s, f) for s in scenes)


def _grid_vals(g):
    return [int(v) for v in np.unique(g) if v != 0]


def make_flee_tasks(n, size=SIZE, vals=(1, 4), seed=0, max_T=6, k=K_SCENES):
    """utility complement (distance): an agent flees the nearest hazard.

        (optimize (distance hv) av)

    av greedily maximises BFS distance from hazard hv — predator/prey, hazard
    avoidance.  We keep the trajectory through the frame where av runs out of room
    and *stays put*: a fixed-direction `step` would keep moving (or leave the grid)
    there, and every `neg_dist` seeker is attracted toward a value, not repelled.
    No such program survives k scenes with different hazard/agent placements, so
    `distance` is required.
    """
    rng = np.random.default_rng(seed)

    def propose(rng, target=None):
        if target is None:
            perm = rng.permutation(vals).tolist()
            av, hv = int(perm[0]), int(perm[1])
        else:
            av, hv = target
        hr, hc = int(rng.integers(1, size - 1)), int(rng.integers(1, size - 1))
        ar, ac = int(rng.integers(size)), int(rng.integers(size))
        if (ar, ac) == (hr, hc) or abs(ar - hr) + abs(ac - hc) > 2:
            return None
        g = np.zeros((size, size), dtype=int)
        g[hr, hc] = hv
        g[ar, ac] = av
        traj = unfold(g, max_T, optimize(distance(hv), av))
        pos = [_agent_pos(traj[t], av) for t in range(max_T)]
        T = next((t + 1 for t in range(1, max_T) if pos[t] == pos[t - 1]), max_T)
        if T < 3:                                   # need at least two real moves
            return None
        x = traj[:T].copy()
        if np.array_equal(x[0], x[-1]):
            return None
        return (av, hv), x, {'kind': 'flee', 'av': av, 'hv': hv}

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000)


def _step_or_erase_reproduces(scenes):
    """True if any `step v d` (incl. v=0, shifting the background) or `erase v`
    reproduces the task — the cheap grid->grid rivals to a single-cell `clear_at`.
    A closed set (six values x four directions, plus the erases), kept for the same
    reason as the other corner-uniqueness checks."""
    for v in range(6):
        if _repro_any(scenes, erase(v)):
            return True
        for d in DIRS.values():
            if _repro_any(scenes, step(v, d)):
                return True
    return False


def make_deletion_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, k=K_SCENES):
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

    def propose(rng, target=None):
        v = int(rng.choice(vals))
        if target is None:
            r0 = int(rng.integers(0, size - 2))     # 3x3 block top-left
            c0 = int(rng.integers(0, size - 2))
        else:
            r0, c0 = target[0] - 1, target[1] - 1   # the block that owns this cell
        g = np.zeros((size, size), dtype=int)
        g[r0:r0 + 3, c0:c0 + 3] = v
        tr, tc = r0 + 1, c0 + 1                     # the protected interior cell
        # Bystanders off the block.  The cell fixes the block, and the block IS the
        # grid, so without them a `clear_at r c` latent admits only as many scenes as
        # there are block values — too few to fill a set of k.  They also give the
        # closed-set rival check something to bite on: a `step`/`erase` that would
        # have matched the bare block has to move them too.
        free = [(r, c) for r in range(size) for c in range(size) if g[r, c] == 0]
        rng.shuffle(free)
        for cell in free[:int(rng.integers(1, 3))]:
            g[cell] = int(rng.choice([u for u in vals if u != v] or vals))
        x = unfold(g, 2, clear_at(tr, tc))
        if np.array_equal(x[0], x[-1]):
            return None
        return (tr, tc), x, {'kind': 'deletion', 'val': v, 'cell': (tr, tc)}

    def certify(scs, m):
        return not _step_or_erase_reproduces(scs)

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000,
                    certify=certify)


def make_denoise_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, n_noise=3,
                       n_signal=2, k=K_SCENES):
    """grid-edit complement (erase): drop EVERY cell of the noise value.

        (erase nv)

    A signal value sv is kept; the noise value nv is scattered over n_noise>=2
    cells that all vanish at once.  No single `clear_at` can remove >=2 cells, and
    `step`/`optimize` relocate rather than delete — so whole-value `erase` is
    required.  Pure denoising: nothing mental.
    """
    rng = np.random.default_rng(seed)

    def propose(rng, target=None):
        if target is None:
            perm = rng.permutation(vals).tolist()
            sv, nv = int(perm[0]), int(perm[1])
        else:
            nv = target[0]
            sv = int(rng.choice([u for u in vals if u != nv]))
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        if len(cells) < n_signal + n_noise:
            return None
        sig, noise = cells[:n_signal], cells[n_signal:n_signal + n_noise]
        g = np.zeros((size, size), dtype=int)
        for (r, c) in sig:
            g[r, c] = sv
        for (r, c) in noise:
            g[r, c] = nv
        x = unfold(g, 2, erase(nv))
        if np.array_equal(x[0], x[-1]):
            return None
        # no single clear_at may do the job: whole-value erase must be required
        if any(_reproduces(g, x, clear_at(r, c)) for (r, c) in noise):
            return None
        return (nv,), x, {'kind': 'denoise', 'noise': nv, 'signal': sv}

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000)


def make_obstacle_tasks(n_per_combo, combos=COMBOS, size=SIZE, seed=0, max_T=8,
                        k=K_SCENES):
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
        def propose(rng, target=None, av=av, gv=gv):
            ar, ac = int(rng.integers(size)), int(rng.integers(size))
            gr, gc = int(rng.integers(size)), int(rng.integers(size))
            if (ar, ac) == (gr, gc) or abs(ar - gr) + abs(ac - gc) < 3:
                return None
            g = np.zeros((size, size), dtype=int)
            g[ar, ac] = av
            g[gr, gc] = gv

            # wall-free trajectory: the source of on-path obstacle candidates
            direct = unfold(g, max_T, optimize(neg_distance(gv), av))
            dpath  = [_agent_pos(direct[t], av) for t in range(max_T)]
            inter  = [p for p in dpath if p and p != (ar, ac) and p != (gr, gc)]
            if not inter:
                return None
            if target is None:
                pr, pc = inter[int(rng.integers(len(inter)))]
            else:
                pr, pc = target[2]
                if (pr, pc) not in inter:
                    return None

            derive = compose(wall_at(pr, pc), optimize(neg_distance(gv), av))
            x_full = unfold(g, max_T, derive)
            t_arrive = next((t for t in range(max_T)
                             if _agent_pos(x_full[t], av) == (gr, gc)), None)
            if t_arrive is None or t_arrive < 3:
                return None
            T = t_arrive + 1
            x = x_full[:T].copy()
            # the wall must force a detour: the agent's path with the wall must differ
            # from the wall-free path (else wall_at is decorative, not required).
            if [_agent_pos(x[t], av) for t in range(T)] == dpath[:T]:
                return None
            return ((av, gv, (pr, pc)), x,
                    {'kind': 'obstacle', 'av': av, 'gv': gv, 'pw': (pr, pc)})

        tasks += _collect(propose, n_per_combo, k, rng, attempts=20000,
                          target_tries=4000)
    return tasks


def make_relocation_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, k=K_SCENES):
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

    def propose(rng, target=None):
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        if target is None:
            (r1, c1), (r2, c2) = cells[0], cells[1]
            if abs(r1 - r2) + abs(c1 - c2) < 2:
                return None
        else:
            (r1, c1), (r2, c2) = target
        # bystanders: the two wall cells are the latent, so the scenes of one task
        # differ only in what else is on the grid (and it must all stay put).
        rest = [p for p in cells if p not in ((r1, c1), (r2, c2))]
        b1, b2 = rest[0], rest[1]
        bv1, bv2 = (int(v) for v in rng.choice(vals, size=2))
        g = np.zeros((size, size), dtype=int)
        g[r1, c1] = 3
        g[b1] = bv1
        g[b2] = bv2
        x = unfold(g, 3, compose(clear_at(r1, c1), wall_at(r2, c2)))
        if np.array_equal(x[0], x[-1]):
            return None
        return ((r1, c1), (r2, c2)), x, {'kind': 'relocate', 'from': (r1, c1),
                                         'to': (r2, c2)}

    def certify(scs, m):
        return not _step_or_erase_reproduces(scs)

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000,
                    certify=certify)


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


def _pair_propose(rng, want, build, meta):
    """Shared tail of the fn_p_g corner generators: render one (working, template)
    pair through `unfold_with_template` and keep it only if `want` is the unique
    cheapest atomic commit for it.

    `_unique_pair_corner` stays per scene rather than moving to task level, unlike
    the other closed-set checks.  These programs are ONE node: there is no room for
    a rival to be wrong on scene 2 and still lose on length, so a tie on any single
    scene is already a defeat.  Templates vary per scene, which is why each family's
    meta carries its own — the interpreter needs the right one for each.
    """
    working, template = build
    x = unfold_with_template(working, template, 2, want[1])
    if np.array_equal(x[0], x[-1]):
        return None
    if not _unique_pair_corner(working, template, want[0], x[-1]):
        return None
    return x, dict(meta, template=template)


def make_perception_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, k=K_SCENES):
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

    def propose(rng, target=None):
        perm = rng.permutation(vals).tolist()
        v = int(perm[0]) if target is None else target[0]
        others = [u for u in vals if u != v]
        dw, dm = (int(u) for u in rng.permutation(others)[:2])
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        P, Q, cw, cm = cells[0], cells[1], cells[2], cells[3]
        if P == Q:
            return None
        working  = np.zeros((size, size), dtype=int)
        template = np.zeros((size, size), dtype=int)
        working[P]   = v                 # world sees v here…
        template[Q]  = v                 # …the map still has it there (stale)
        working[cw]  = dw                # distractors differ between the channels so
        template[cm] = dm                # keep-world and keep-model are distinguishable
        got = _pair_propose(rng, ('sync_to_model', sync_to_model(v)),
                            (working, template), {'kind': 'perception', 'val': v})
        return None if got is None else ((v,), got[0], got[1])

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000)


def make_multi_registration_tasks(n, size=SIZE, vals=(1, 2, 4, 5), seed=0,
                                  n_movers=2, k=K_SCENES):
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

    def propose(rng, target=None):
        chosen = rng.permutation(vals).tolist()
        movers, static = chosen[:n_movers], chosen[n_movers]  # last is world-only
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        if len(cells) < 2 * len(movers) + 1:
            return None
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
            return None
        working[cells[ci]] = static                 # present in world only → stays put
        got = _pair_propose(rng, ('sync_all', sync_all), (working, template),
                            {'kind': 'multi_reg', 'vals': movers})
        # `sync_all` is nullary: the program names nothing, so every scene of this
        # family shares one latent and the movers may differ scene to scene.
        return None if got is None else ((), got[0], got[1])

    return _collect(propose, n, k, rng, attempts=20000)


def make_registration_except_tasks(n, size=SIZE, vals=(1, 2, 4, 5), seed=0,
                                   n_movers=2, k=K_SCENES):
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

    def propose(rng, target=None):
        chosen = rng.permutation(vals).tolist()
        if len(chosen) < n_movers + 2:
            return None
        if target is not None:                       # pin the anchor
            a = target[0]
            chosen = [a] + [u for u in chosen if u != a]
        movers, static = chosen[:n_movers + 1], chosen[n_movers + 1]  # movers[0]=anchor
        anchor = movers[0]
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        if len(cells) < 2 * len(movers) + 1:
            return None
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
            return None
        working[cells[ci]] = static      # world-only: breaks the sync_to_model tie
        got = _pair_propose(rng, ('sync_except', sync_except(anchor)),
                            (working, template),
                            {'kind': 'reg_except', 'anchor': anchor})
        return None if got is None else ((anchor,), got[0], got[1])

    return _collect(propose, n, k, rng, attempts=20000, target_tries=2000)


def make_inpainting_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, k=K_SCENES):
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

    def propose(rng, target=None):
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
        got = _pair_propose(rng, ('underlay', underlay), (working, template),
                            {'kind': 'inpaint'})
        return None if got is None else ((), got[0], got[1])   # `underlay` is nullary

    return _collect(propose, n, k, rng, attempts=20000)


def make_readout_tasks(n, size=SIZE, vals=(1, 2, 4), seed=0, k=K_SCENES):
    """projection complement (snd_gg): report the stored map, ignore the query.

        snd_gg                       applied to (working, template)

    The output is the template regardless of the working grid — a pure channel
    projection (recall the model, discard the world).  Working and template share
    no cells, so `fst_gg` (keep world), the blends, and every sync all diverge from
    a verbatim template; only `snd_gg` reproduces it.
    """
    rng = np.random.default_rng(seed)

    def propose(rng, target=None):
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
            return None
        got = _pair_propose(rng, ('snd_gg', snd_gg), (working, template),
                            {'kind': 'readout'})
        return None if got is None else ((), got[0], got[1])   # `snd_gg` is nullary

    return _collect(propose, n, k, rng, attempts=20000)


