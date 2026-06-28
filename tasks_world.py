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

This is a library module — the non-mental task generators (overlay, registration,
and one family per symmetric cube corner) imported by the phase drivers.  Run a
phase via `python phase1.py`.
"""

import sys
import numpy as np

from dsl import (
    RIGHT, LEFT, UP, DOWN,
    unfold, unfold_with_template, tr,
    fork, sync_to_world, overlay, step, optimize,
    compose, wall_at, neg_distance,
    # symmetric complements referenced when building corner ground truth
    fst_gg, snd_gg, sync_to_model, sync_all, sync_except, underlay,
    distance, clear_at, erase,
)
from tasks_minds import (
    make_belief_tasks, _physically_explainable, _agent_pos, COMBOS,
)

# ── Configuration (shared with file11/12/13) ─────────────────────────────────────
# Obstacle is the policy DONOR: belief reuses the abstraction the joint stitch learns
# from the obstacle family — (compose (optimize (neg_dist gv) av) (wall_at r c)), the
# shared derive.  For stitch to keep (gv,av) as HOLES rather than bake a literal, the
# donor must span the SAME diverse value set as belief (all 8 usable cell ids; 0=empty
# and 3=wall are reserved — see prims._CELLVALUES).  So obstacle reuses belief's COMBOS
# imported above.  A narrow {1,4}×{2,5} subset under-powered the abstraction: stitch
# baked gv/av into the policy and belief could not reuse it across its 8 combos.
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


