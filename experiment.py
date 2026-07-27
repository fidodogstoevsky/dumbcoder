"""experiment.py: the unified one-corpus test — belief as an MDL win, not a silo.

The other task families answered the "gerrymandered decomposition" charge
(fork/sync are a
`believe` primitive split into two gears that only ever re-mesh into belief) by
giving the gears work unrelated to belief: fork without sync (overlay/motion-blur) and
sync without fork (registration/coordinate-join).  But it proved this across
THREE isolated searches — three separate `ECD` calls, three separate `Deltas`,
three separate stitch passes.  That shows each gear is *reachable* somewhere; it
does not show that, under ONE compression objective seeing ALL the evidence at
once, belief still emerges as the MDL-optimal recombination while its parts stay
general.  Gerrymandering is an MDL claim, and MDL is only tested by joint
compression.

This file runs ONE library and ONE joint stitch over one corpus:

    physics       (step v d)                              fn
    desire        (optimize (neg_dist gv) av)             fn
    overlay       (fork (step v d) overlay)               fn   — fork WITHOUT sync
    comet         (fork (optimize (neg_dist gv) av)
                       overlay)                            fn   — fork WITHOUT sync (seek derive)
    registration  (sync_to_world v) on (work, template)   fn_p_g — sync WITHOUT fork
    belief        (fork (compose (wall_at r c)
                          (optimize (neg_dist gv) av))
                       (sync_to_world av))                 fn   — fork WITH sync

Enumeration is necessarily per-root-type: registration asks the searcher for an
`fn_p_g` (a pair->grid commit, paired against a *given* template by
`unfold_with_template`), while every other family asks for an `fn` (grid->grid,
threaded by `unfold`).  Those are different typed queries; a single budget walk
cannot produce both.  What IS unified — and what the charge is actually about —
is the library and the compression: all five families' solutions pool into ONE
`sols`, compressed by ONE `saturate_stitch`.

The verdict has two independent halves:

  (A) USAGE CENSUS (from normalized solutions, stitch-independent):
      fork appears in overlay AND belief; sync_to_world in registration AND
      belief; wall_at in belief ALONE.  The parts are general; the intensional
      pattern is unique to belief.

  (B) JOINT COMPRESSION (from the single stitch over the pooled corpus):
      the MDL-optimal library carves out the agent constructor
      (fork + sync_to_world + wall_at + optimize, av shared x2) for the belief
      tasks, WHILE the same objective leaves fork/sync bare (or in a different,
      non-agency abstraction) in the overlay/registration solutions.

If both halves hold in one run, belief is a *discovered* recombination of
independently-used parts — not a believe-primitive decomposed and rediscovered,
and not a silo artefact.

Shared harness for phases 1 and 2; `run_phase(decomposed=)` is the single entry
point, invoked by the thin wrappers `phase1.py` (atomic) and `phase2.py` (decomposed).

Run:
    python phase1.py            # phase 1: atomic fork/sync
    python phase2.py            # phase 2: decomposed fork/sync
    python phase1.py --smoke    # tiny corpus, short timeouts
"""

import sys
import os
import math
import re as _re
from collections import Counter
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch as th

import tasks as tasks_mod
from scenes import Scenes, as_scenes, solves
from ecd import (
    Deltas, solve_enumeration, saturate_stitch, mat_key, mat_key_id, normalize,
    _solve_one_task, _worker_init, _n_cpus_available,
    dream, dreamed_q, content_boost, uniform_type_q,
)
from dsl import (
    fn, fn_p_g, cellvalue, coord, Delta,
    unfold, unfold_with_template, tr, simplify, length,
    # used by check_decomposition_identities (phase 2)
    compose, wall_at, clear_at, optimize, neg_distance, step,
    fork, sync_to_world, fork_decomposed, register, locate, place,
)

# Reuse the generators and DSL already vetted in the curriculum, so the experiment
# driver adds only the *unification*, never a new encoding to second-guess.
from tasks import (
    make_physics_tasks, make_desire_tasks, make_belief_tasks,
    make_witness_belief_tasks,
    make_goal_displacement_tasks, make_two_observer_tasks,
    make_false_obstacle_belief_tasks,
    belief_variant,
    COMBOS, SIZE, DIRS,
    make_overlay_tasks, make_comet_tasks, make_registration_tasks,
    # one task per symmetric corner
    make_flee_tasks, make_deletion_tasks, make_denoise_tasks, make_obstacle_tasks,
    make_relocation_tasks, make_underlay_tasks,
    make_perception_tasks, make_multi_registration_tasks,
    make_registration_except_tasks, make_inpainting_tasks, make_readout_tasks,
)
from prims import make_core_prims, make_symmetric_prims

# corpus families by interpreter / root type.  The first block is fixed; the cube
# corner families (second block) are appended only in --cube runs, where their
# primitives exist.  Reporting loops iterate these so new families show up
# everywhere without per-call edits.
_FN_KINDS   = ['physics', 'desire', 'overlay', 'comet', 'belief',
               'flee', 'deletion', 'denoise', 'obstacle', 'relocate', 'underlay']
_PAIR_KINDS = ['registration', 'perception', 'multi_reg', 'reg_except',
               'inpaint', 'readout']
_CUBE_KINDS = ['flee', 'deletion', 'denoise', 'obstacle', 'relocate', 'underlay',
               'perception', 'multi_reg', 'reg_except', 'inpaint', 'readout']
_ALL_KINDS  = ['physics', 'desire', 'overlay', 'comet', 'registration', 'belief',
               'flee', 'deletion', 'denoise', 'obstacle', 'relocate', 'underlay',
               'perception', 'multi_reg', 'reg_except', 'inpaint', 'readout']

# Per-family generator seeds.  Kept in one table (rather than as literals at the
# call sites) so the provenance header records the seeds the run ACTUALLY used —
# a seed can't be changed here without the artifacts saying so.  Keys are the
# generator's family, not always the task `kind` ('belief' covers witness/plain,
# whose variants get their own).  'belief_extra' is a second, distinct-seeded batch
# of plain wall-belief tasks (emitted as kind='belief' like any other) that enriches
# the corpus so witness-belief is reachable in budget; it carries no special label.
TASK_SEEDS = {
    'physics': 0, 'desire': 1, 'belief': 2, 'overlay': 3, 'registration': 4,
    'comet': 5,
    'flee': 10, 'deletion': 11, 'denoise': 12,
    'perception': 13, 'multi_reg': 14, 'reg_except': 15, 'inpaint': 16,
    'readout': 17, 'obstacle': 18, 'underlay': 19,
    'belief_extra': 22, 'belief_goal': 23, 'belief_observers': 27,
    'belief_fob': 25, 'relocate': 26,
}
# 24 was `belief_dual`, deleted in favour of the two-observer family (tasks.py);
# left unassigned so a run artifact's seed table stays unambiguous.

# interface primitives whose presence in a solution we care about.  pipe_gpg/
# compose_gp/dup/mapsnd are fork's decomposition and register is sync's: in a
# decomposed (phase 2) run atomic `fork`/`sync_to_world` are gone and the same
# structure shows up as (pipe_gpg (compose_gp dup (mapsnd …)) (register …)).
_INTERFACE = ('fork', 'sync_to_world', 'overlay', 'then_sync',
              'wall_at', 'optimize', 'step', 'neg_dist', 'compose',
              'pipe_gpg', 'compose_gp', 'dup', 'mapsnd', 'register')


def _uses_sync(use_set):
    "True if a family's used-token set realises the agency commit, atomically OR decomposed."
    return 'sync_to_world' in use_set or 'register' in use_set


def _uses_fork(use_set):
    "True if a family's used-token set realises fork — atomically OR decomposed."
    return 'fork' in use_set or {'pipe_gpg', 'compose_gp', 'dup'} <= set(use_set)


def _has_fork(body_str):
    "True if an abstraction body realises fork — atomically OR decomposed."
    return 'fork' in body_str or ('pipe_gpg' in body_str
                                  and 'compose_gp' in body_str and 'dup' in body_str)


def _belief_ctor_in_D(D):
    """Return (repr, wall_based) of the first invented abstraction exhibiting the
    AGENCY signature — fork ∧ a sync-family commit — else (None, False).  The commit
    may be literal (sync_to_world / register) or a degenerate scope complement
    (sync_except / sync_all), which on a single-mover scene equals sync_to_world(av).
    `wall_based` flags whether that same body also carries wall_at (the headline
    plain-belief constructor); witness-belief's constructor is wall-FREE, so keying the
    detector to wall_at (as an earlier version did) misses it entirely.  Used by the
    wake-sleep loop to print, per round, whether the constructor the recognition model
    would need to steer toward actually exists in the library yet."""
    _commit = ('sync_to_world', 'register', 'sync_except', 'sync_all')
    for d in D.invented:
        body = str(simplify(normalize(deepcopy(d))))
        if _has_fork(body) and any(c in body for c in _commit):
            return d.repr, ('wall_at' in body)
    return None, False


# the symmetric "corners" populated only in --cube runs: each is the complement
# of a core primitive along one axis (direction / scope / z-order / projection /
# utility / grid-edit / bifunctor / pairing).  The cube test asks whether belief
# still avoids ALL of these while the other families happily reach for them.
_CORNERS = ('sync_to_world', 'sync_to_model', 'sync_all', 'sync_except',
            'overlay', 'underlay', 'fst_gg', 'snd_gg', 'via_swap',
            'neg_dist', 'distance', 'wall_at', 'clear_at', 'erase',
            # bifunctor axis (decomposed runs): belief's derive runs on the model
            # channel (mapsnd); the complements act on the world channel or swap
            # the channels (mapfst / swap / bimap).
            'mapsnd', 'mapfst', 'swap', 'bimap',
            # pairing axis (decomposed runs): belief forks via the diagonal (dup);
            # the complement opens a fresh scratch channel instead (pair_blank).
            'dup', 'pair_blank',
            # sync decomposition (phase 2): belief's agency commit sync_to_world
            # becomes register(locate av)(place av) — its own corner, not a complement.
            'register', 'locate', 'place')


# ── Q tensors (mirror ECD.task_Q so enumeration cost matches the rest of the curriculum) ──
# uniform_type_q and content_boost are imported from ecd rather than re-derived here:
# dreamed_q builds on the same two, and a local copy is exactly how the coord half of
# the content boost came to exist in only one of the two Qs.


def content_q(D, x):
    """uniform type Q, with the task's own visible terminals boosted to cost 0.

    The boost itself lives in ecd.content_boost, shared with dreamed_q so the two Qs
    price content identically — see that function for which terminals and why."""
    return content_boost(D, uniform_type_q(D), x)


# ── ground-truth check (one driver, all five families) ────────────────────────────

def _forks(D, derive_str, commit_str):
    """Ground-truth string for fork(derive, commit), written so it type-checks in
    whichever DSL is active: atomic `(fork …)` if fork is a primitive, else its
    decomposition `(pipe_gpg (compose_gp dup (mapsnd derive)) commit)`."""
    if any(d.repr == 'fork' for d in D.ds):
        return f"(fork {derive_str} {commit_str})"
    return f"(pipe_gpg (compose_gp dup (mapsnd {derive_str})) {commit_str})"


def _sync(D, v):
    """Ground-truth string for sync_to_world(v): atomic if available, else its
    decomposition `(register (locate v) (place v))` (phase 2)."""
    if any(d.repr == 'sync_to_world' for d in D.ds):
        return f"(sync_to_world {v})"
    return f"(register (locate {v}) (place {v}))"


def _sync_model(D, v):
    """Ground-truth string for sync_to_model(v): atomic if available, else its
    decomposition `(via_swap (register (locate v) (place v)))` (phase 2)."""
    if any(d.repr == 'sync_to_model' for d in D.ds):
        return f"(sync_to_model {v})"
    return f"(via_swap (register (locate {v}) (place {v})))"


def _belief_gt_str(D, m):
    """Canonical AGENCY-commit ground-truth program string for any belief variant.

    Every variant is `fork(compose(derive, optimize(neg_dist gv) av), sync_to_world av)`
    with the derive distinguishing the family (wall_at / step), and the
    commit ALWAYS the single-value agency commit sync_to_world(av) (its decomposition
    in phase 2).  Shared between verify_ground_truth and the verdict's degeneracy
    recognition, so both build the same canonical program."""
    if 'real_wall' in m:                             # false about BOTH obstacle and goal:
        # a REAL wall stays in the world, so on this scene NO scope complement reproduces
        # x under this derive (verified at construction by _scope_complements_all_fail) —
        # the commit can only be the literal sync_to_world(av).  Also carries displaced_to,
        # so test it BEFORE the goal branch.  derive = _seq(stamp phantom, shove goal,
        # seek): the real wall is NOT erased in the model (the agent hallucinates an extra
        # obstacle rather than mislocating this one).  That opening `clear_at(real wall)`
        # was vacuous — the real wall is drawn from the free cells and rarely obstructs —
        # and since the scope certification is derive-relative, certifying a derive the
        # searcher would never price is what let sync_except(gv) back in.  Keep this in
        # step with tasks.make_false_obstacle_belief_tasks.
        pr, pc = m['pw']
        derive = (f"(compose (compose (wall_at c{pr} c{pc}) "
                  f"(step {m['gv']} {m['dir']})) (optimize (neg_dist {m['gv']}) {m['av']}))")
        return _forks(D, derive, _sync(D, m['av']))
    if 'displaced_to' in m:                          # false belief about the goal's location
        # left-fold the shove sequence + seek, mirroring tasks._seq
        parts = ([f"(step {m['gv']} {dn})" for dn in m['dirs']]
                 + [f"(optimize (neg_dist {m['gv']}) {m['av']})"])
        derive = parts[0]
        for p in parts[1:]:
            derive = f"(compose {derive} {p})"
        belief = _forks(D, derive, _sync(D, m['av']))
        if 'observer' in m:
            # two observers of one world: the bystander's BARE seek runs first, on the
            # real grid, and only the believer is wrapped in the fork.  The asymmetry
            # is the point — the constructor is applied to one agent, not to both.
            return (f"(compose (optimize (neg_dist {m['gv']}) {m['observer']}) "
                    f"{belief})")
        return belief
    pr, pc = m['pw']                                 # phantom wall, optionally + witness
    derive = (f"(compose (wall_at c{pr} c{pc}) "
              f"(optimize (neg_dist {m['gv']}) {m['av']}))")
    belief = _forks(D, derive, _sync(D, m['av']))
    if 'aw' in m:
        belief = (f"(compose {belief} (optimize (neg_dist {m['gw']}) {m['aw']}))")
    return belief


# template-rooted (fn_p_g) families render through unfold_with_template; every other
# family threads a single grid through plain unfold.
_TEMPLATE_KINDS = frozenset({'registration', 'perception', 'multi_reg', 'reg_except',
                             'inpaint', 'readout'})


def gt_program_str(D, m):
    """Canonical ground-truth s-expression for any family, in whichever DSL is active
    (fork/sync atomic in phase 1, decomposed in phase 2).  Single source of truth for
    the corpus: `verify_ground_truth` renders it to confirm it reproduces the scene, and
    the MDL-margin experiment stitches these exact strings to build the final library and
    price the found (mental) program.  The searcher recovers these programs — the verify
    assert is what licenses reusing them as the 'found program' without re-enumerating."""
    k = m['kind']
    if k == 'physics':
        return f"(step {m['val']} {m['dir']})"
    if k == 'desire':
        return f"(optimize (neg_dist {m['gv']}) {m['av']})"
    if k == 'overlay':
        return _forks(D, f"(step {m['val']} {m['dir']})", "overlay")
    if k == 'comet':
        return _forks(D, f"(optimize (neg_dist {m['gv']}) {m['av']})", "overlay")
    if k == 'underlay':
        return _forks(D, f"(step {m['val']} {m['dir']})", "underlay")
    if k == 'registration':
        return _sync(D, m['val'])
    # ── symmetric-corner families (cube runs only) ───────────────────────────
    if k == 'flee':
        return f"(optimize (distance {m['hv']}) {m['av']})"
    if k == 'deletion':
        r, c = m['cell']
        return f"(clear_at c{r} c{c})"
    if k == 'denoise':
        return f"(erase {m['noise']})"
    if k == 'relocate':
        # spelled clear_at ▸ wall_at (not erase ▸ wall_at) to share the exact
        # derive prefix of the false-obstacle belief gt, so gt-mode stitch joins them
        (r1, c1), (r2, c2) = m['from'], m['to']
        return f"(compose (clear_at c{r1} c{c1}) (wall_at c{r2} c{c2}))"
    if k == 'obstacle':
        pr, pc = m['pw']
        return (f"(compose (wall_at c{pr} c{pc}) "
                f"(optimize (neg_dist {m['gv']}) {m['av']}))")
    if k == 'perception':
        return _sync_model(D, m['val'])
    if k == 'multi_reg':
        return "sync_all"
    if k == 'reg_except':
        return f"(sync_except {m['anchor']})"
    if k == 'inpaint':
        return "underlay"
    if k == 'readout':
        return "snd_gg"
    # belief (wall / witness / goal-displacement / two-observer) — all agency-committed
    return _belief_gt_str(D, m)


def templates_of(m):
    """The per-scene template channels of an fn_p_g task, in scene order.

    Each scene of a template family carries its own (working, template) pair, so the
    interpreter needs the right one per scene; the task's top-level `template` is
    scene 0's.  Returns None for the fn-rooted families."""
    if m['kind'] not in _TEMPLATE_KINDS:
        return None
    return [sm['template'] for sm in m['per_scene']]


def verify_ground_truth(D, tasks):
    n_scenes = 0
    for x, m in tasks:
        tree = tr(D, gt_program_str(D, m))
        tmpl = templates_of(m)
        assert solves(tree(), x, tmpl), f"ground truth failed for {m['kind']}: {m}"
        n_scenes += x.k
    print(f"  ground-truth check: {len(tasks)} tasks / {n_scenes} scenes "
          f"verified via Delta trees")


# ── reporting helpers ─────────────────────────────────────────────────────────────

def _core_uses(tree):
    "core primitives reached for in a solution, AFTER expanding to primitives."
    s = str(simplify(normalize(deepcopy(tree))))
    return {p for p in _INTERFACE if p in s}


def _shared_holes(body_str):
    "map $i -> count for holes that occur more than once (the agency signature)."
    c = Counter(_re.findall(r'\$\d+', body_str))
    return {v: n for v, n in c.items() if n > 1}


_SCOPE_COMPLEMENTS = {'sync_except', 'sync_all'}


def _fork_derive(node):
    """The derive subtree of a fork / decomposed fork, or None if `node`'s produce
    shape isn't a recognised model-channel derive.  The commit is always tails[1].

    Handles the atomic `fork` and the decomposed `pipe_gpg(produce, commit)` where
    `produce` is a (possibly nested) compose_gp chain whose last endo is `mapsnd` or
    `bimap` (the model-channel map).  A NON-mapsnd/bimap tail (or none) means the
    produce shape isn't recognised — return None, and callers conservatively decline
    to rewrite rather than guess."""
    if not isinstance(node, Delta):
        return None
    if not (node.tails and len(node.tails) == 2):
        return None
    if node.repr == 'fork':
        return node.tails[0]
    if node.repr == 'pipe_gpg' and node.tails[0].repr == 'compose_gp':
        produce = node.tails[0]
        if not (produce.tails and len(produce.tails) == 2):
            return None
        endo = produce.tails[1]
        # mapsnd(derive): model-channel map — its arg is the derive.
        if endo.repr == 'mapsnd' and endo.tails:
            return endo.tails[0]
        # bimap(world_op, model_op): the model derive is the SECOND arg (the first
        # transforms the world channel).  The world-op is incidental to the commit's
        # actor, which _actor reads from the derive's own seek.
        if endo.repr == 'bimap' and endo.tails and len(endo.tails) == 2:
            return endo.tails[1]
    return None


def _is_fork_node(node):
    "structural test: a fork / decomposed fork, whose commit is tails[1]."
    return (isinstance(node, Delta) and node.repr in ('fork', 'pipe_gpg')
            and node.tails is not None and len(node.tails) == 2)


def _world_level_commits(tree):
    """The commit subtrees that write to the WORLD — the agency commits.

    A commit is the fn_p_g (pair -> grid) argument of a fork / decomposed fork, always
    tails[1].  Only forks reachable from the root WITHOUT passing through another
    fork's derive are world-level: a fork nested inside a derive runs on the agent's
    private model channel and its commit writes into that model, never into the world,
    so it says nothing about how the solution commits.  This positional scoping is the
    whole point — a string test for `sync_except` cannot tell the two apart, which is
    exactly how 12 literal-committing false-obstacle solves were read as degenerate.

    Template-rooted families (registration / perception / inpaint / readout /
    multi_reg / reg_except) ARE a bare commit: the program is itself fn_p_g and the
    model channel comes from the task template, so the root is the commit.

    No fn_p_g constructor takes an `fn`, so no fork can hide inside a commit; it is
    therefore sound to stop at each fork without descending into either tail."""
    if isinstance(tree, Delta) and tree.type == fn_p_g:
        return [tree]
    out = []

    def walk(node):
        if not isinstance(node, Delta):
            return
        if _is_fork_node(node):
            out.append(node.tails[1])
            return                      # derive (tails[0]) is model-level: out of scope
        for t in (node.tails or []):
            walk(t)

    walk(tree)
    return out


def _commit_axis_corners(D):
    """The corners that live on the COMMIT axis: those typed fn_p_g, i.e. the nodes
    that can occupy a fork's commit slot, plus register's own arguments (locate/place),
    which exist only inside a register commit.  Read off the ACTIVE DSL rather than
    hardcoded, so phase 1 (atomic sync_to_world / sync_to_model) and phase 2
    (register / via_swap) each get the right set."""
    axis = {d.repr for d in D.ds if d.type == fn_p_g}
    return (axis | {'locate', 'place'}) & set(_CORNERS)


def _corner_uses(D, tree):
    """which symmetric corners a solution reaches for, AFTER expanding to primitives.

    Corners on the COMMIT axis are judged AT the world-level commit position, not by
    presence anywhere in the string: the cube's commit claim is about what a solution
    COMMITS with, and a scope complement inside a derive commits into the agent's
    private model, not the world (see _world_level_commits).  Corners on the other
    axes (grid-edit wall_at/clear_at/erase, utility neg_dist/distance, bifunctor
    mapsnd/mapfst/swap/bimap, pairing dup/pair_blank) are structural, have no commit
    position to speak of, and stay judged over the whole program — the cube is a
    multi-axis claim and only its commit axis is positional."""
    t = simplify(normalize(deepcopy(tree)))
    s_all = str(t)
    axis = _commit_axis_corners(D)
    s_commit = ' '.join(str(c) for c in _world_level_commits(t))
    return {p for p in _CORNERS
            if _re.search(rf'\b{p}\b', s_commit if p in axis else s_all)}


def _swap_scope_commit(D, tree, m):
    """Rewrite `tree` (a normalized belief solution) in place so every fork /
    decomposed-fork scope-complement commit (sync_all / sync_except) becomes the
    literal agency commit sync_to_world(av) — av being *that fork's own* actor (the
    seek's target).  Returns the rewritten tree if at least one scope commit was
    swapped, else None (nothing to test).

    This drives the SOLUTION-relative degeneracy test: a scope commit is the agency
    commit on a scene iff swapping it for sync_to_world(av) in the solution's own
    derive still reproduces the scene.

    Only WORLD-LEVEL commits are swapped (see _world_level_commits).  A scope commit
    nested inside a derive is the agent's own model-internal bookkeeping; rewriting it
    would test a claim about the private model, not about agency, and it reproduces the
    scene either way — which is precisely how it used to manufacture 'degenerate'."""
    swapped = [False]

    def _actor(derive):
        "actor value of this fork = the seek's target (optimize's cellvalue arg)."
        found = []
        def scan(n):
            if not isinstance(n, Delta):
                return
            if n.repr == 'optimize' and n.tails and len(n.tails) == 2:
                found.append(n.tails[1])
            for t in (n.tails or []):
                scan(t)
        scan(derive)
        for cand in found:
            try:
                return int(cand.head)
            except (TypeError, ValueError):
                continue
        return m.get('av')

    def walk(node):
        if not isinstance(node, Delta):
            return
        if _is_fork_node(node):
            if node.tails[1].repr in _SCOPE_COMPLEMENTS:
                derive = _fork_derive(node)     # None ⇒ unrecognised shape: don't guess
                if derive is not None:
                    av = _actor(derive)
                    if av is not None:
                        node.tails[1] = tr(D, _sync(D, av))
                        swapped[0] = True
            return                              # model-level forks below: out of scope
        for t in (node.tails or []):
            walk(t)

    walk(tree)
    return tree if swapped[0] else None


def _belief_commit_form(D, sol, x, m):
    """How a belief solution realises the single-value agency commit.

      'literal'    — commits via sync_to_world / register: the clean shared-av signature.
      'degenerate' — commits (also/only) via a SCOPE complement (sync_except gv / sync_all)
                     that, ON THIS SCENE, is extensionally the agency commit: swapping it
                     for sync_to_world(av) in the solution's OWN derive still reproduces x.
                     When the agent is the only committed model-mover, 'move everything but
                     the goal' == 'move the agent'; same fork-structured belief, agency hole
                     expressed on gv rather than av.  A minimal-scene degeneracy of the
                     commit, NOT a rival (non-belief) theory — see the disclosure note.
      'complement' — a scope commit whose swap for sync_to_world(av) does NOT reproduce x:
                     the commit is carrying non-agency work (a derive that leaves gv/3
                     unmoved in the model, so only wholesale adoption fits).  A genuine
                     non-belief rival; the verdict counts it as a FAILURE.
      None         — neither: not an agency commit (e.g. unsolved / non-fork).

    The degeneracy guard is SOLUTION-relative.  An earlier version tested the canonical
    agency program for this task, which on a false-obstacle scene always reproduces x —
    so it auto-classified every fork+scope solution 'degenerate' (vacuously), and could
    never surface a real complement rival.

    It is also POSITIONAL: the form is read off the world-level commits only (see
    _world_level_commits), never off the solution string.  A string test conflates 'the
    solution commits with a scope complement' with 'a scope complement occurs somewhere
    in the solution', and those come apart exactly when the derive itself forks — the
    agent modelling a model.  That is not hypothetical: every one of the 24 phase-2
    false-obstacle solves commits literally via register(locate av)(place av), yet the
    12 whose derive contains a nested (sync_except k) fork were read as 'degenerate',
    inverting the phase-1/phase-2 comparison.  Nothing about the scene changed — a
    model-internal commit cannot reproduce a real wall in the world, and the generator's
    _scope_complements_all_fail certification only ever constrained WORLD-level commits."""
    tree = simplify(normalize(deepcopy(sol)))
    commit_reprs = [c.repr for c in _world_level_commits(tree)]
    has_literal = any(r in ('sync_to_world', 'register') for r in commit_reprs)
    has_scope = any(r in _SCOPE_COMPLEMENTS for r in commit_reprs)
    # the fork requirement stays: a bare scope commit (no fork, e.g. the multi_reg /
    # reg_except template shape) is not an agency commit at all — it is None, not a rival.
    if has_scope and _has_fork(str(tree)):
        swapped = _swap_scope_commit(D, simplify(normalize(deepcopy(sol))), m)
        if swapped is not None:
            try:
                if solves(swapped(), x, templates_of(m)):
                    return 'degenerate'
            except Exception:
                pass
        return 'complement'
    if has_literal:
        return 'literal'
    return None


def _canon_belief_uses(uses_or_corners, form):
    """rewrite a degenerate scope-complement belief commit to the canonical agency commit.

    Only fires on form=='degenerate', which is now a world-level judgement: a solution
    that merely *contains* a scope complement inside its derive is 'literal' and keeps
    its corners untouched (its nested sync_except is not a commit and, being off the
    world-level commit position, _corner_uses never picked it up in the first place)."""
    if form == 'degenerate':
        return (set(uses_or_corners) - _SCOPE_COMPLEMENTS) | {'sync_to_world'}
    return set(uses_or_corners)


# ── task visualisation ──────────────────────────────────────────────────────────

def _grid_rows(g):
    "row-strings for one grid: '.'=empty(0), '#'=wall(3), digit otherwise."
    return [' '.join('.' if v == 0 else '#' if v == 3 else str(int(v))
                     for v in row) for row in np.asarray(g)]


def _side_by_side(grids, labels, gap='   ', indent='    '):
    "render same-height grids left-to-right under their column labels."
    blocks = [_grid_rows(g) for g in grids]
    h = max((len(b) for b in blocks), default=0)
    widths = [max([len(lab)] + [len(r) for r in b]) for b, lab in zip(blocks, labels)]
    out = [gap.join(lab.ljust(w) for lab, w in zip(labels, widths))]
    for r in range(h):
        out.append(gap.join((b[r] if r < len(b) else '').ljust(w)
                            for b, w in zip(blocks, widths)))
    return '\n'.join(indent + line for line in out)


# Display order, trunk first: the two displaced-goal families carry the belief claim
# (Sally-Anne solves in every run; two-observer adds the selective attribution), the
# wall-based ones harden it.
_BELIEF_VARIANTS = ['belief_goal', 'belief_observers', 'belief_wall',
                    'belief_witness', 'belief_false_obstacle']


def _sample_kind(m):
    """Finer label for sampling only — the belief variants all share kind='belief'
    (so they feed one unified verdict), but should display as separate panels."""
    return m['kind'] if m['kind'] != 'belief' else belief_variant(m)


def _select_samples(tasks, max_frames=6):
    """One example per kind (first seen, in _ALL_KINDS order) as labelled panels.

    Each sample is {kind, tag, T, panels:[(label, grid), …]}.  fn families show
    successive `unfold` frames t0…; fn_p_g families show world | template | result,
    surfacing the otherwise-invisible constant template channel.  The rendered
    grids are exactly what the searcher sees — belief's phantom wall lives only in
    the private model, so it never appears here.  The single kind='belief' is split
    into its variants (wall / witness / goal-displacement / two-observer) for display."""
    seen = {}
    for x, m in tasks:
        seen.setdefault(_sample_kind(m), (x, m))
    order = []
    for kind in _ALL_KINDS:
        if kind == 'belief':
            order += [k for k in _BELIEF_VARIANTS if k in seen]
        elif kind in seen:
            order.append(kind)
    out = []
    for kind in order:
        x, m = seen[kind]
        tag = ', '.join(f"{k}={v}" for k, v in m.items()
                        if k not in ('kind', 'template', 'per_scene'))
        scene = x.rep                        # one representative scene of the set
        if 'template' in m:                  # fn_p_g: world | template | result(s)
            extra = list(scene[1:max_frames - 1])
            panels = [('world', scene[0]), ('template', m['template'])]
            panels += [(f't{t}', g) for t, g in enumerate(extra, start=1)]
        else:                                # fn: successive frames of the unfold
            panels = [(f't{t}', scene[t])
                      for t in range(min(len(scene), max_frames))]
        out.append({'kind': kind, 'tag': tag, 'T': int(scene.shape[0]),
                    'k': x.k, 'panels': panels})
    return out


def print_task_samples(tasks, max_frames=6):
    "Text dump of one example trajectory per family (see _select_samples)."
    print("\n" + "=" * 72)
    print("TASK SAMPLES — one example trajectory per family")
    print("=" * 72)
    for s in _select_samples(tasks, max_frames):
        print(f"\n  [{s['kind']}]  {{{s['tag']}}}   T={s['T']}")
        labels = [lab for lab, _ in s['panels']]
        grids  = [g for _, g in s['panels']]
        print(_side_by_side(grids, labels))


def export_task_samples(tasks, path='task_samples.json', max_frames=6):
    "Write one example per family to JSON for viz.typ (see _select_samples)."
    import json
    size = int(np.asarray(tasks[0][0]).shape[-1]) if tasks else 0
    data = {'size': size, 'samples': [
        {'kind': s['kind'], 'tag': s['tag'], 'T': s['T'],
         'panels': [{'label': lab, 'grid': np.asarray(g).astype(int).tolist()}
                    for lab, g in s['panels']]}
        for s in _select_samples(tasks, max_frames)]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=1)
    print(f"  wrote {len(data['samples'])} task samples to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────────

def report_abstraction_generality(D, all_tasks, rewritten):
    """(B′) Does the stitched seek/policy carry HOLES for (gv, av), or bake them in?

    Belief is only shallow — and therefore reachable by a modest budget — if the
    `policy = compose(wall_at, seek)` it plugs into the agency wrapper is ONE reusable
    abstraction every belief task fills with its own (gv, av), rather than a stack of
    near-duplicates that each bake one combo.  The robust test: is any single
    seek/policy abstraction reused across >=2 DISTINCT (gv, av) combos?  That can only
    happen if the combo positions are holes.  If every such abstraction maps to exactly
    one combo, stitch baked the literals and belief's skeleton is NOT actually shallow
    for the rest — no budget fixes that, the corpus must be diversified.
    """
    print("\n" + "=" * 72)
    print("(B′) ABSTRACTION GENERALITY — does seek/policy carry holes for (gv, av)?")
    print("=" * 72)

    absts = {d.repr: str(simplify(normalize(deepcopy(d)))) for d in D.invented}
    sp = {n: b for n, b in absts.items() if 'optimize' in b or 'neg_dist' in b}
    if not sp:
        print("  no seek/policy abstraction was invented this run (belief's derive never")
        print("  compressed — solve desire+obstacle first, or raise stitch_iters).")
        return

    def _absts_in(s):
        return [n for n in sp if _re.search(rf'\b{n}\b', s)]
    def _kind(b):
        return 'policy' if 'wall_at' in b else 'seek'
    def _holes(b):
        return len(set(_re.findall(r'\$\d+', b)))

    # per donor/consumer family: which seek/policy abstraction each (gv,av) combo reuses
    for kind in ('desire', 'obstacle', 'belief'):
        rows, combos = {}, set()
        for x, m in all_tasks:
            if m['kind'] != kind or 'gv' not in m or 'av' not in m:
                continue
            s = rewritten.get(mat_key(x))
            if not s:
                continue
            combo = (m['gv'], m['av'])
            combos.add(combo)
            for n in _absts_in(s):
                rows.setdefault(n, set()).add(combo)
        if not combos:
            continue
        print(f"\n  {kind}: {len(combos)} distinct (gv,av) combos solved & rewritten")
        for n in sorted(rows, key=lambda k: -len(rows[k])):
            print(f"    {n} [{_kind(sp[n])}]  reused by {len(rows[n])}/{len(combos)} combos  "
                  f"holes={_holes(sp[n])}")
            print(f"        body: {sp[n]}")
        if not rows:
            print("    (no seek/policy abstraction reached these rewritten programs)")

    # widest combo-span achieved by one abstraction, restricted to policy (has wall_at)
    # or to any seek/policy, within a family.  Policy is belief's actual derive, so the
    # OBSTACLE policy span is the leading indicator (it's solvable from scratch); the
    # belief span confirms it once belief itself is solved.
    def _span(kind, policy_only):
        span = {}
        for x, m in all_tasks:
            if m['kind'] != kind or 'gv' not in m:
                continue
            s = rewritten.get(mat_key(x))
            if not s:
                continue
            for n in _absts_in(s):
                if policy_only and 'wall_at' not in sp[n]:
                    continue
                span.setdefault(n, set()).add((m['gv'], m['av']))
        return max((len(v) for v in span.values()), default=0)

    belief_combos = {(m['gv'], m['av']) for _, m in all_tasks
                     if m['kind'] == 'belief' and 'gv' in m}
    obstacle_policy = _span('obstacle', policy_only=True)
    belief_span = _span('belief', policy_only=False)
    print(f"\n  distinct belief (gv,av) combos in corpus            : {len(belief_combos)}")
    print(f"  widest combo-span of one OBSTACLE policy (leading)  : {obstacle_policy}")
    print(f"  widest combo-span of one belief seek/policy (confirm): {belief_span}")
    if obstacle_policy < 2 and len(belief_combos) >= 2:
        why = ("no wall_at abstraction was formed at all" if obstacle_policy == 0
               else "the policy bakes its (gv,av) literals (~one abstraction per combo)")
        print(f"  => obstacle's policy did NOT generalize ({why}): belief's derive will not be")
        print("     a single cheap token, so belief is NOT shallow even with the wrapper.")
        print("     Budget alone won't reach the off-combo beliefs — diversify the (gv,av) corpus")
        print("     so no single literal dominates and stitch keeps the (gv,av) holes.")
    elif belief_span == 0:
        print("  => no belief solved yet, but obstacle's policy DOES generalize: once one belief")
        print("     is reached (budget) the wrapper should let the rest reuse that policy cheaply.")
    elif belief_span >= 2:
        print("  => policy GENERALIZES into belief: one abstraction serves multiple combos via")
        print("     holes — belief's skeleton is genuinely shallow.")
    else:
        print("  => belief reuses a per-combo specialization — diversify (gv,av) to force holes.")


def _dl(D, Q, prog):
    """Description length (nats) of a program string/tree under library D and prior Q.
    Same convention as mdl_margin._dl / phase3_arity._dl, so the numbers this file
    reports and the margins mdl_margin.py prices are on one scale."""
    t = tr(D, prog) if isinstance(prog, str) else prog
    return float(-D.logp(Q, t))


def report_dl_census(D, all_tasks, sols, rewritten, ctor_name=None):
    """(B″) What the joint stitch BOUGHT each family, in nats.

    The verdict is otherwise all booleans; the prose wants magnitudes.  Price every
    solved task's found program twice under the FINAL library: spelled out in base
    primitives ('before' — the library-free description enumeration actually returned)
    and in its library-rewritten form ('after' — what it costs once the abstractions
    exist).  The gap is the compression the stitch bought that family; for belief it is
    what the agent constructor is worth, and the other (non-belief) families are the
    baseline that says whether belief's saving is special or just what abstraction does
    to everything.

    Both sides are priced under the SAME final library and uniform type prior, which is
    also what mdl_margin.py prices belief-vs-rival margins with — so `dl_lib` here is
    the same quantity as its `dl_found_lib`.

    Returns the census dict (persisted in the verdict artifact under 'dl').
    """
    print("\n" + "=" * 72)
    print("(B″) DL CENSUS — what the library buys each family, in nats")
    print("=" * 72)

    Q = uniform_type_q(D)
    per_kind, per_variant = {}, {}
    n_belief_ctor = 0
    for x, m in all_tasks:
        k = mat_key(x)
        base, lib = sols.get(k), rewritten.get(k)
        if base is None or not lib:
            continue
        try:
            dl_base, dl_lib = _dl(D, Q, base), _dl(D, Q, lib)
        except Exception:
            # a rewritten form that won't parse under D is a stitch/library mismatch,
            # not a magnitude to report — skip rather than price the wrong program.
            continue
        rec = per_kind.setdefault(m['kind'], {'dl_base': [], 'dl_lib': []})
        rec['dl_base'].append(dl_base)
        rec['dl_lib'].append(dl_lib)
        if m['kind'] == 'belief':
            v = per_variant.setdefault(belief_variant(m), {'dl_base': [], 'dl_lib': []})
            v['dl_base'].append(dl_base)
            v['dl_lib'].append(dl_lib)
            if ctor_name and _re.search(rf'\b{ctor_name}\b', lib):
                n_belief_ctor += 1

    def _summarize(rec):
        b, l = np.array(rec['dl_base']), np.array(rec['dl_lib'])
        return {'n': len(b),
                'dl_base_median': float(np.median(b)), 'dl_lib_median': float(np.median(l)),
                'saved_median': float(np.median(b - l)), 'saved_mean': float(np.mean(b - l)),
                'saved_total': float(np.sum(b - l))}

    dl_by_kind = {k: _summarize(r) for k, r in per_kind.items()}
    dl_by_variant = {v: _summarize(r) for v, r in per_variant.items()}

    print(f"\n  {'family':13s} {'solves':>6s} {'before':>9s} {'after':>9s} {'saved':>9s}"
          f"   (median nats per task; saved > 0 = the library is shorter)")
    print("  " + "-" * 66)
    for kind in _ALL_KINDS:
        s = dl_by_kind.get(kind)
        if s is None:
            continue
        mark = '  <- belief' if kind == 'belief' else ''
        print(f"  {kind:13s} {s['n']:6d} {s['dl_base_median']:9.2f} {s['dl_lib_median']:9.2f} "
              f"{s['saved_median']:9.2f}{mark}")
    if dl_by_variant:
        print("\n  belief by variant:")
        for var in _BELIEF_VARIANTS:
            s = dl_by_variant.get(var)
            if s is None:
                continue
            print(f"    {var:22s} {s['n']:3d} solves  {s['dl_base_median']:8.2f} -> "
                  f"{s['dl_lib_median']:8.2f}  (saved {s['saved_median']:6.2f})")

    bel = dl_by_kind.get('belief')
    # the baseline that gives belief's number meaning: what the same stitch saved the
    # families that are NOT belief.  If belief's saving is merely typical, the
    # constructor is not carrying the compression the (B) claim rests on.
    nonmental = [s['saved_median'] for k, s in dl_by_kind.items()
                 if k != 'belief']
    nonmental_median = float(np.median(nonmental)) if nonmental else None
    if bel:
        print(f"\n  belief: {bel['dl_base_median']:.2f} nats before the library -> "
              f"{bel['dl_lib_median']:.2f} after (median saving {bel['saved_median']:.2f}; "
              f"{bel['saved_total']:.2f} over {bel['n']} solves)")
        if nonmental_median is not None:
            print(f"  other families' median saving, for scale: "
                  f"{nonmental_median:.2f} nats/task")
        if ctor_name:
            print(f"  belief solves whose library form invokes {ctor_name} (the constructor): "
                  f"{n_belief_ctor}/{bel['n']}")
    return {'by_kind': dl_by_kind, 'belief_by_variant': dl_by_variant,
            'nonmental_saved_median': nonmental_median,
            'n_belief_using_ctor': n_belief_ctor}


def load_mdl_margin(decomposed, smoke, run_path=None):
    """The MDL margin (nats) belief holds over its non-belief rivals, read back from the
    artifact `mdl_margin.py` writes.  That experiment prices belief against the
    transient-wall/pure-physics rivals under THIS phase's library, but it runs as a
    separate pass, so its magnitudes never reached the verdict — the run said "belief is
    the MDL win" in booleans while the nats lived in another file.  Echoing it here (and
    mdl_margin.py back-filling the verdict artifact after it runs) closes that loop.

    Returns (summary, note): summary is the margin artifact's summary block, or None
    with `note` saying why it is unavailable — absent, another phase, or STALE (priced
    from a run older than this one, so its nats describe a library this run replaced).
    """
    import os, json
    path = f"mdl_margins{'.decomposed' if decomposed else ''}.json"
    if run_path is None:
        run_path = f"phase{2 if decomposed else 1}_run{'.smoke' if smoke else ''}.json"
    try:
        with open(path) as f:
            art = json.load(f)
    except (FileNotFoundError, ValueError):
        return None, f"no {path} yet — run `python mdl_margin.py{' --decomposed' if decomposed else ''}"\
                     f"{' --smoke' if smoke else ''}` to price belief against its rivals"
    if bool(art.get('decomposed')) != bool(decomposed) or bool(art.get('smoke')) != bool(smoke):
        return None, f"{path} is a different phase/mode run — not echoed"
    summary = art.get('summary')
    if summary is None:
        return None, f"{path} predates the summary block — re-run mdl_margin.py"
    try:
        if os.path.getmtime(path) < os.path.getmtime(run_path):
            return summary, (f"STALE: {path} was priced from a run older than this one; "
                             f"re-run mdl_margin.py to price THIS library")
    except OSError:
        pass
    return summary, None


def report_mdl_margin(decomposed, smoke):
    """Echo the belief-vs-rival MDL margins into the verdict, so the run reports the
    magnitude of its own central claim rather than only that it held."""
    summary, note = load_mdl_margin(decomposed, smoke)
    if summary is None:
        print(f"  (B) MDL margin over non-belief rivals: unavailable — {note}")
        return None, note
    if note:
        print(f"  (B) MDL margin over non-belief rivals  [{note}]")
    else:
        print(f"  (B) MDL margin over non-belief rivals (nats, under this run's library):")
    for var in _BELIEF_VARIANTS:
        s = (summary.get('by_variant') or {}).get(var)
        if s is None:
            continue
        if s.get('n_competitor_pairs'):
            print(f"        {var:22s} median {s['margin_lib_median']:+6.2f}, "
                  f"min {s['margin_lib_min']:+6.2f}  over {s['n_competitor_pairs']} "
                  f"competitor rival(s)")
        else:
            print(f"        {var:22s} no behavioural competitor (expressiveness-excluded)")
    n = summary.get('n_competitor_pairs') or 0
    if n:
        print(f"        => {summary.get('pct_competitors_longer', 0):.0f}% of {n} competitor "
              f"pairs are LONGER than the mental reading "
              f"({summary.get('n_competitors_shorter', 0)} shorter)")
    return summary, note


def _belief_progs(m):
    """(atomic, decomposed) program pair for a belief task, built variant-by-variant
    to mirror `_belief_gt_str`.  Each variant is fork(derive, commit) with the SAME
    derive but commit atomic (sync_to_world av) vs decomposed (register(locate, place)),
    and the fork atomic vs fork_decomposed — so the two programs realise the identical
    machine iff the decomposition identities hold.  Returns None for a non-belief task."""
    if m['kind'] != 'belief':
        return None

    def one(derive, av):
        return (fork(derive, sync_to_world(av)),
                fork_decomposed(derive, register(locate(av), place(av))))

    if 'real_wall' in m:                             # false about BOTH obstacle and goal
        # mirror _belief_gt_str's real_wall branch: stamp phantom ▸ shove goal ▸ seek
        # (the real wall is left standing in the model — see the note there).  Carries
        # displaced_to too, so test it BEFORE the goal branch.
        derive = compose(compose(wall_at(*m['pw']), step(m['gv'], DIRS[m['dir']])),
                         optimize(neg_distance(m['gv']), m['av']))
        return one(derive, m['av'])
    if 'displaced_to' in m:                          # goal-displacement: false belief about the goal
        derive = step(m['gv'], DIRS[m['dirs'][0]])
        for dn in m['dirs'][1:]:
            derive = compose(derive, step(m['gv'], DIRS[dn]))
        derive = compose(derive, optimize(neg_distance(m['gv']), m['av']))
        orig, deco = one(derive, m['av'])
        if 'observer' in m:      # two observers: the bystander's bare seek runs first
            seek = optimize(neg_distance(m['gv']), m['observer'])
            orig, deco = compose(seek, orig), compose(seek, deco)
        return orig, deco
    derive = compose(wall_at(*m['pw']), optimize(neg_distance(m['gv']), m['av']))
    orig, deco = one(derive, m['av'])
    if 'aw' in m:   # witness-belief: the witness's direct seek follows the belief move
        seek = optimize(neg_distance(m['gw']), m['aw'])
        orig, deco = compose(orig, seek), compose(deco, seek)
    return orig, deco


def check_decomposition_identities(tasks):
    "phase 2 self-check: the decomposed combinators are the same machine as fork/sync."
    n = 0
    for x, m in tasks:
        progs = _belief_progs(m)
        if progs is None:
            continue
        prog_orig, prog_deco = progs
        assert solves(prog_orig, x), f"orig fork != task for {m}"
        assert solves(prog_deco, x), f"decomposed != task for {m}"
        n += 1
    print(f"decomposition identity: fork ≡ commit∘mapsnd(derive)∘dup and "
          f"sync ≡ register(locate, place) verified on {n} belief tasks "
          f"(every scene)")


# Round-1 fn-timeout cap (seconds): the curriculum budget for the first enumeration
# round.  Set to the calibrated slowest-solve of a task GIVEN its abstractions are
# present (~1200s from the Jul-12 timing reports); round 1 has no invented tokens yet,
# so a longer budget only buys deep extensional rivals, not the belief compound.
ROUND1_T_FN_CAP = 1200.0


def run_phase(decomposed=False, smoke=False, samples=False, ecd_iters=None, t_fn=None,
              t_fn_round1=None, dream_on=True, plain_belief=False, curriculum=True,
              k=None):
    """One phase of the curriculum (phase 1 = atomic, phase 2 = decomposed).

    Both phases run the full symmetric cube over the one undifferentiated corpus
    (`cube` is always on); the only knob is whether fork/sync are atomic or spelled
    out.  Phase 2 additionally proves the decomposition is numerically identical to
    the atomic machine before searching.

    A phase is several full ECD (wake-sleep) rounds: each round enumerates the still-
    unsolved tasks against the current library, then runs ONE joint stitch over every
    solution; the abstractions it discovers are added to the library so the next round
    reaches programs that were out of budget before.  This is what lets belief — deep
    from primitives — become reachable once its parts have been compressed.  Override
    the round count with `ecd_iters` and the per-task fn timeout with `t_fn` (belief is
    the long pole, so HPC runs want a generous `t_fn`).

    The sleep phase also DREAMS (`dream_on`, default True): after each stitch a
    recognition model is trained on the round's replays (solved fn programs) and
    fantasies (programs sampled from the library), and the next rounds enumerate the fn
    tasks under that learned matrix-conditioned Q rather than the uniform/content prior.
    Pass `dream_on=False` (CLI `--no-dream`) to recover the uniform-Q baseline.

    `k` (CLI `--k`) is the number of scenes per task — see scenes.py.  Each task is a
    SET of trajectories sharing one latent program, and a program solves it only by
    reproducing all of them; that is what identifies the program's literals (belief's
    phantom-wall coordinate above all) instead of leaving them to be explained away by
    whatever coincidence fits a single trajectory.  Defaults to tasks.K_SCENES.
    """
    cube = True
    k = tasks_mod.K_SCENES if k is None else int(k)
    if smoke:
        n_phys, n_des, n_ov, n_reg, n_bel, n_corner = 2, 1, 2, 2, 1, 2
        n_comet = 2
        n_belvar = 1
        n_goal = 2
        n_obstacle = 2
        n_relocate = 4
        _t_fn, t_reg, stitch_iters, _ecd_iters = 15, 8, 3, 2
    else:
        n_phys, n_des, n_ov, n_reg, n_bel, n_corner = 4, 2, 4, 4, 6, 4
        n_comet = 4
        n_belvar = 3
        # goal-displacement gets ~2× the other belief variants' mass: its content
        # subtree is the one that VARIES (12 shove shapes, see _GOAL_CONTENT_SPECS),
        # and stitch only keeps the content slot as a hole if the varied family is
        # heavy enough to out-compress a per-shove specialization.
        n_goal = 6
        # obstacle is deliberately the DENSEST corner family: its solution IS belief's
        # wall policy (compose (wall_at) (optimize (neg_dist))), and that 4-hole compound
        # must recur enough to hold a top-`stitch_iters` abstraction slot even after
        # belief's fork-wrapped rivals (fn_10/fn_11) enter the joint stitch.  Round-1
        # stitch already forms the standalone wall policy (fn_5); giving obstacle ~3× the
        # per-combo mass of the other corners keeps it alive so wall-based belief can
        # reuse it as a single cheap token instead of re-deriving it inside the fork.
        n_obstacle = 6
        n_relocate = 16
        _t_fn, t_reg, stitch_iters, _ecd_iters = 180, 30, 6, 4
    t_fn = _t_fn if t_fn is None else t_fn
    ecd_iters = _ecd_iters if ecd_iters is None else ecd_iters
    # Per-round fn timeout schedule (DreamCoder-style curriculum): round 1 is capped
    # near the calibrated slowest-solve so a deep transient-wall rival (the
    # ~349k program the Jul-12 3600s runs found) cannot land BEFORE the wall/fork policy
    # tokens exist; dreamed rounds 2+ get the full budget.  A no-op when t_fn ≤ the cap
    # (smoke / short runs), so it only bites the long runs it is meant to protect.
    t_fn_round1 = min(t_fn, ROUND1_T_FN_CAP) if t_fn_round1 is None else t_fn_round1
    dream_iters = 120 if smoke else 600   # recognition-model training steps per round

    # Every knob this run resolved to, in one dict: printed as the log's provenance
    # header and embedded in all three artifacts, so a figure caption can cite
    # "run of commit abc123, t_fn=180" instead of an unreproducible number.
    knobs = dict(
        decomposed=bool(decomposed), smoke=bool(smoke), cube=bool(cube),
        t_fn=t_fn, t_fn_round1=t_fn_round1, t_reg=t_reg,
        ecd_iters=ecd_iters, stitch_iters=stitch_iters,
        dream_on=bool(dream_on), dream_iters=dream_iters,
        plain_belief=bool(plain_belief), curriculum=bool(curriculum),
        k=k,
        n_phys=n_phys, n_des=n_des, n_ov=n_ov, n_comet=n_comet, n_reg=n_reg,
        n_bel=n_bel, n_belvar=n_belvar, n_goal=n_goal, n_corner=n_corner,
        n_obstacle=n_obstacle, n_relocate=n_relocate,
    )
    prov = build_provenance(knobs)
    print_provenance(prov)

    print("Generating corpus…")
    phys = make_physics_tasks(n_phys, seed=TASK_SEEDS['physics'], k=k)
    des  = make_desire_tasks(n_des, COMBOS, seed=TASK_SEEDS['desire'], k=k)
    ov   = make_overlay_tasks(n_ov, seed=TASK_SEEDS['overlay'], k=k)
    # comet: fork WITHOUT sync like overlay, but the derive is desire's seek policy
    # instead of a fixed step — shows fork's derive slot is a general fn (step OR
    # optimize), the dual of the underlay family varying the commit slot.
    comet = make_comet_tasks(n_comet, seed=TASK_SEEDS['comet'], k=k)
    reg  = make_registration_tasks(n_reg, seed=TASK_SEEDS['registration'], k=k)
    # In a --cube run the DSL contains clear_at and erase, which let a non-belief
    # "transient wall" (stamp / act / undo) reproduce single-agent belief.  Witness
    # belief excludes that structurally — the witness crosses the phantom cell on the
    # real grid — which is why it is the default here.  Single-agent belief now
    # excludes it too, by the static counterpart of the same argument (a bystander on
    # the phantom-wall cell that a world-level stamp destroys; see
    # tasks.make_belief_tasks), so `--plain-belief` is a depth/budget diagnostic
    # rather than a weakening of the uniqueness claim: plain belief is ~8 nodes vs
    # witness ~12+, and it is the shallow probe for what is blocking a first solve.
    use_witness = cube and not plain_belief
    if plain_belief and cube:
        print("  [--plain-belief] using single-agent belief in a cube run — a DEPTH "
              "diagnostic; transient-wall rivals stay excluded (bystander).")
    bel  = (make_witness_belief_tasks(n_bel, COMBOS, seed=TASK_SEEDS['belief'], k=k) if use_witness
            else make_belief_tasks(n_bel, COMBOS, seed=TASK_SEEDS['belief'], k=k))
    # Two further belief families (kind='belief'), so the unified verdict tests
    # whether ONE fork(policy, sync_to_world av) agent constructor generalizes
    # across belief about an obstacle, about an object's location (goal-displacement),
    # and — with two agents in one world — across an agent it must NOT be applied to.
    # See tasks.py.  (`obs` replaces the retired contradictory-beliefs family, which
    # cost six latent literals and solved 0/24 in every run.)
    gdb  = make_goal_displacement_tasks(n_goal, COMBOS, seed=TASK_SEEDS['belief_goal'], k=k)
    obs  = make_two_observer_tasks(n_belvar, COMBOS,
                                   seed=TASK_SEEDS['belief_observers'], k=k)
    # False-obstacle belief (kind=belief): wrong about BOTH the obstacle and the goal,
    # with a REAL wall (value 3) left in the world.  Its construction forbids the
    # scope-complement degeneracy — no sync_all / sync_except k reproduces the scene, so
    # any solution MUST commit via the literal sync_to_world(av).  This is what lets the
    # (A) disclosure say the agency commit was FORCED and found, not merely argued
    # extensionally equivalent to it.  Needs cube primitives (clear_at, wall value 3).
    fob  = (make_false_obstacle_belief_tasks(n_belvar, COMBOS, seed=TASK_SEEDS['belief_fob'], k=k)
            if cube else [])
    print(f"  belief variants: +{len(gdb)} goal-displacement, +{len(obs)} two-observer, "
          f"+{len(fob)} false-obstacle (all kind=belief; false-obstacle forbids the "
          f"scope-complement commit)")

    # ── extra plain-belief tasks (on by default; --no-curriculum to disable) ──────
    # Witness-belief is deep — compose(fork(policy, sync), seek) — so its FIRST solve
    # is out of budget even once the policy is a cheap token (the fork∧sync block is
    # still searched from scratch).  Plain single-agent belief is the SAME inner block
    # without the outer witness seek, and it is shallow (the --plain-belief diagnostic
    # solves it at tiny t_fn).  A second, distinct-seeded batch of plain wall-belief
    # tasks makes fork(policy, sync) recur across more (gv,av) combos, so the joint
    # stitch abstracts it into one token — and then witness-belief = compose(<that
    # token>, seek) is shallow enough to reach, exactly as obstacle's policy lowered
    # plain belief's first-solve.
    #
    # These are just wall-belief tasks (kind='belief', variant belief_wall) — no
    # special label, no separate bucket.  Whether they *served* as the scaffold that
    # unlocked belief is read off the run's results, not stipulated here.
    #
    # Sound because the transient-wall rival does not REPRODUCE these scenes (the
    # bystander on the phantom-wall cell is destroyed by a world-level stamp), so the
    # searcher cannot return it whatever it costs.  It used to rest on pricing instead
    # — once the policy is a token, fork(policy, sync) is cheaper than the rival, which
    # additionally pays clear_at + two coords — and that argument is exactly what
    # phase 2 falsified: with fork spelled out as dup/mapsnd/compose_gp/pipe_gpg the
    # rival was the cheaper program, `(erase 3)` undercut `(clear_at r c)` by a token,
    # and all 24 belief_wall solves in both jul-26 phase-2 runs were the non-mental
    # world-edit.  A cost argument holds only under the pricing that happens to be in
    # force; the scene has to do the excluding.
    belief_extra = []
    if curriculum and use_witness:
        n_extra = max(1, n_bel // 2)
        belief_extra = make_belief_tasks(n_extra, COMBOS, seed=TASK_SEEDS['belief_extra'], k=k)
        print(f"  [curriculum] +{len(belief_extra)} extra plain wall-belief tasks "
              f"(kind=belief) to seed the fork(policy, sync) abstraction.")

    # One task per symmetric corner, so every complement the cube adds
    # is *useful somewhere* — otherwise "belief avoids the complements" is vacuous
    # (an unused complement is trivially avoided).  These only make sense when the
    # corner primitives exist, i.e. in a --cube run.  fn-rooted corners join the
    # `unfold` search; pair-rooted ones join the `unfold_with_template` search.
    fn_corner, pair_corner = [], []
    if cube:
        # obstacle is the wall_at corner: a plain detour task whose solution is
        # belief's policy `(compose (wall_at) (optimize (neg_dist)))`.  A *family* of
        # them (per-combo, like belief) makes that derive recur so the joint stitch
        # abstracts it — which is what lowers belief's first-solve cost and leaves only
        # the fork∧sync agency wrapper unique to belief.
        # underlay is the z-order complement of overlay in a FORK context: a
        # world-wins motion trail crossing an occluding bystander, so the cube's
        # underlay corner is exercised by a *fork* task (like overlay), not only by
        # the template-rooted inpainting task — the fork-producer side of the pair
        # interface now populates both z-order corners.
        # relocate is the "move" grid-edit corner (clear_at ▸ wall_at) and, like
        # obstacle, doubles as curriculum: its solution is the derive PREFIX of the
        # false-obstacle belief (clear real wall ▸ stamp phantom), which is ~14 nats
        # past the search frontier at primitive prices.  It gets extra mass
        # (n_relocate) for the same reason obstacle does: the compound must recur
        # enough to hold a stitch slot so false-obstacle can buy it as one token.
        fn_corner = (make_flee_tasks(n_corner, seed=TASK_SEEDS['flee'], k=k)
                     + make_deletion_tasks(n_corner, seed=TASK_SEEDS['deletion'], k=k)
                     + make_denoise_tasks(n_corner, seed=TASK_SEEDS['denoise'], k=k)
                     + make_underlay_tasks(n_corner, seed=TASK_SEEDS['underlay'], k=k)
                     + make_obstacle_tasks(n_obstacle, seed=TASK_SEEDS['obstacle'], k=k)
                     + make_relocation_tasks(n_relocate, seed=TASK_SEEDS['relocate'], k=k))
        pair_corner = (make_perception_tasks(n_corner, seed=TASK_SEEDS['perception'], k=k)
                       + make_multi_registration_tasks(n_corner, seed=TASK_SEEDS['multi_reg'], k=k)
                       + make_registration_except_tasks(n_corner, seed=TASK_SEEDS['reg_except'], k=k)
                       + make_inpainting_tasks(n_corner, seed=TASK_SEEDS['inpaint'], k=k)
                       + make_readout_tasks(n_corner, seed=TASK_SEEDS['readout'], k=k))

    # fn-rooted families share the `unfold` interpreter; pair families are fn_p_g.
    fn_tasks = phys + des + ov + comet + bel + gdb + obs + fob + belief_extra + fn_corner
    reg_tasks = reg + pair_corner

    # dedupe across the whole corpus (identical mats would skew stitch counts)
    seen, fn_tasks_d = set(), []
    for x, m in fn_tasks:
        k = mat_key(x)
        if k in seen:
            continue
        seen.add(k)
        fn_tasks_d.append((x, m))
    fn_tasks = fn_tasks_d

    by_kind = Counter(m['kind'] for _, m in fn_tasks + reg_tasks)
    print(f"  {by_kind['physics']} physics, {by_kind['desire']} desire, "
          f"{by_kind['overlay']} overlay, {by_kind['comet']} comet, "
          f"{by_kind['registration']} registration, "
          f"{by_kind['belief']} belief — {len(fn_tasks) + len(reg_tasks)} total")
    if cube:
        print("  corner families: "
              + ', '.join(f"{by_kind[k]} {k}" for k in _CUBE_KINDS))
    print()

    if samples:
        print_task_samples(fn_tasks + reg_tasks)
        export_task_samples(fn_tasks + reg_tasks)   # data for viz.typ

    if cube:
        D = Deltas(make_symmetric_prims(decomposed=decomposed))
        print(f"DSL: {len(D)} primitives — CUBE run "
              f"(core + symmetric complements{', decomposed plumbing' if decomposed else ''})")
        print(f"  added corners: {sorted(set(_CORNERS) - {'sync_to_world','overlay','fst_gg','neg_dist','wall_at'})}")
    else:
        D = Deltas(make_core_prims())
        print(f"DSL: {len(D)} primitives "
              f"(fork, sync_to_world, overlay, then_sync given as core)")
    verify_ground_truth(D, fn_tasks + reg_tasks)
    if decomposed:
        check_decomposition_identities(fn_tasks + reg_tasks)

    # ── wake-sleep: several full ECD rounds (enumerate ↦ joint stitch ↦ re-enumerate) ──
    # Enumeration is per-root-type (fn for 4+ families, fn_p_g for registration), but
    # the library and the stitch are shared.  Each round enumerates only the still-
    # unsolved tasks against the current library (core + abstractions invented so far),
    # then runs ONE joint stitch over every solution pooled across both root types.
    # The abstractions saturate_stitch discovers are added to D, so the next round
    # reaches programs that were out of budget before — this is how belief (deep from
    # primitives) becomes reachable once its parts have been compressed into reuse.
    sols = {}
    solve_round = {}          # mat_key -> the ECD round it was first solved (for corpus_dl.py)
    constructor_round = None  # the ECD round whose joint stitch first produced the agency
                              # constructor (item 3: the agency-signature round, for the verdict)
    # per-round per-task fn timings (mat_key, round, solved?, seconds), so solve_dynamics.py
    # can chart the cumulative-solve S-curve and the per-task solve-time collapse (the
    # ~t_fn miss in round r ↦ seconds in round r+1) that is otherwise only in the log text.
    timing_log = []
    templates = {mat_key(x): templates_of(m) for x, m in reg_tasks}
    all_tasks = fn_tasks + reg_tasks
    n_total = len(all_tasks)
    rewritten = {}
    nw = _n_cpus_available()

    # Dreaming: after each round's stitch, train a recognition model on this round's
    # replays (solved fn programs, rewritten through the learned abstractions) plus
    # fantasies (programs sampled from the library) — see ecd.dream.  The next rounds
    # enumerate the fn tasks under that learned, matrix-conditioned Q instead of the
    # uniform/content prior.  As a completeness mop-up (mirroring ECD's post-iter-3
    # fallback) the search reverts to the uniform/content Q after DREAM_USE_ROUNDS so
    # a model that mis-prioritises belief can never make it unreachable.  Registration
    # (fn_p_g) stays on the uniform Q: dream / MatRecognitionModel model the single
    # world trajectory, not the paired template channel.
    DREAM_USE_ROUNDS = 2
    qmodel = None
    fn_Xs = [x for x, _ in fn_tasks]
    fn_keys = {mat_key(x) for x, _ in fn_tasks}

    print("\n" + "=" * 72)
    print(f"WAKE-SLEEP — up to {ecd_iters} ECD rounds (enumerate ↦ joint stitch), {nw} workers")
    print(f"  fn Q: {'dreamed recognition model (replays + fantasies)' if dream_on else 'uniform/content (no dreaming)'}"
          f"; registration Q: uniform")
    print("=" * 72)

    # No curriculum gate: every unsolved fn task is attempted every round, on a uniform
    # per-task timeout (--t-fn).  Belief programs are deep from the primitives, so they
    # simply MISS (cheaply, within the timeout) until the abstractions they reuse have
    # been compressed into the library by stitch; the round one solves is decided purely
    # by the MDL dynamics, with nothing in the scheduler knowing belief is special.  Keep
    # --t-fn small enough that early-round deep misses don't starve the shallow tasks of
    # workers, but ≥ the solve time of the slowest task GIVEN its abstractions are present
    # (see the per-task timing report below, which calibrates exactly this).
    kind_by_key  = {mat_key(x): m['kind'] for x, m in fn_tasks}
    kind_of_all  = {mat_key(x): m['kind'] for x, m in all_tasks}
    # fine per-variant label (belief_wall / belief_witness / …) for per-solve logging
    # and the round-attribution artifact; coarse kind for every non-belief family.
    fine_kind_of = {mat_key(x): _sample_kind(m) for x, m in all_tasks}

    for it in range(1, ecd_iters + 1):
        round_t_fn = t_fn_round1 if it == 1 else t_fn   # curriculum budget schedule
        unsolved_fn  = [x for x, _ in fn_tasks if mat_key(x) not in sols]
        unsolved_reg = [x for x, _ in reg_tasks if mat_key(x) not in sols]
        n_before = len(sols)
        use_model = dream_on and qmodel is not None and it <= 1 + DREAM_USE_ROUNDS
        # Dreamed Q is applied ONLY to families with at least one solved instance — a
        # replay the recognition model was actually trained on.  A zero-replay family
        # (belief, until its first solve) is invisible to the model, which can then only
        # mis-price it: it floods the early budget windows with the non-belief primitives
        # it HAS seen and delays belief's own primitives past the timeout — dreamed_q's
        # uniform mixture bounds that delay (log(1/DREAM_PRIOR_W) nats per node) but does
        # not remove it.  Those families stay on the proven uniform/content
        # baseline and earn the dreamed Q only once solved (they then contribute replays).
        replay_kinds = {kind_by_key[k] for k in sols
                        if sols.get(k) is not None and k in kind_by_key}
        print(f"\n--- round {it}/{ecd_iters}: {len(unsolved_fn)} fn + {len(unsolved_reg)} "
              f"fn_p_g unsolved; |D|={len(D)} ({len(D.invented)} invented); "
              f"fn Q={'dreamed (replay families only)' if use_model else 'uniform/content'}; "
              f"t_fn={round_t_fn:.0f}s{' (round-1 cap)' if it == 1 and round_t_fn < t_fn else ''} ---",
              flush=True)
        # INSTRUMENTATION: does the fn_9-style belief constructor exist in the library
        # at the START of this round (i.e. is there anything for the recognition model
        # to steer toward), and how many belief tasks are already solved & thus eligible
        # to become belief replays in this round's dream set?
        _ctor, _ctor_wall = _belief_ctor_in_D(D)
        _bel_solved = sum(1 for k, kd in kind_by_key.items()
                          if kd == 'belief' and sols.get(k) is not None)
        _bel_total  = sum(1 for kd in kind_by_key.values() if kd == 'belief')
        print(f"    [instr] agency constructor in D: "
              f"{(_ctor + (' (wall-based)' if _ctor_wall else ' (wall-free)')) if _ctor else 'ABSENT'}; "
              f"belief solved (replay-eligible): {_bel_solved}/{_bel_total}", flush=True)

        if unsolved_fn:
            with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as pool:
                args = [(x, D,
                         (dreamed_q(qmodel, D, x)
                          if (use_model and kind_by_key.get(mat_key(x)) in replay_kinds)
                          else content_q(D, x)),
                         dict(sols), round_t_fn, 0, fn) for x in unsolved_fn]
                # per-task timing: collected so t_fn can be calibrated from a real run —
                # the max SOLVE time bounds how low the uniform timeout can safely go.
                timings = {}
                for k, sol, elapsed in pool.map(_solve_one_task, args):
                    timings[k] = (sol is not None, elapsed)
                    timing_log.append((k, it, sol is not None, float(elapsed)))
                    if sol is not None:
                        sols[k] = sol
                        # item 1: name the family and the exact (base-primitive) program
                        # at the moment it is solved — the raw fact the thesis writes from.
                        print(f"      + solved [{fine_kind_of.get(k, '?')}] in {elapsed:.1f}s: "
                              f"{str(simplify(normalize(deepcopy(sol))))}", flush=True)
                solved_t = [e for hit, e in timings.values() if hit]
                miss_t   = [e for hit, e in timings.values() if not hit]
                max_solve = max(solved_t) if solved_t else 0.0
                print(f"    per-task time (s): {len(solved_t)} solved"
                      + (f" [slowest solve {max_solve:.1f}]" if solved_t else ""), flush=True)
                # (per-task OK/MISS dump removed — it clogged slurm output; the full
                # per-attempt table is persisted in timing_log/the run artifact and can
                # be reconstructed offline if t_fn ever needs recalibrating.)
                if miss_t and solved_t:
                    print(f"    (misses burned up to {max(miss_t):.1f}s each; the uniform "
                          f"--t-fn can drop toward slowest-solve={max_solve:.1f}s to cut "
                          f"wasted early-round time without losing a solve)", flush=True)
        if unsolved_reg:
            reg_before = set(sols)
            solve_enumeration(unsolved_reg, D, uniform_type_q(D), sols,
                              timeout=t_reg, root_type=fn_p_g, templates=templates)
            # solve_enumeration mutates sols in place and returns no per-task info, so
            # diff the keys to report the newly-solved fn_p_g families (item 1).
            for k in set(sols) - reg_before:
                if sols[k] is not None:
                    print(f"      + solved [{fine_kind_of.get(k, '?')}]: "
                          f"{str(simplify(normalize(deepcopy(sols[k]))))}", flush=True)

        n_solved = len(sols)
        # stamp the round each newly-solved task first appeared (corpus_dl.py replays the
        # cumulative sols solved through round r to reconstruct that round's library).
        for k in sols:
            if sols[k] is not None:
                solve_round.setdefault(k, it)
        print(f"    solved {n_solved}/{n_total} (+{n_solved - n_before} this round)", flush=True)

        # joint stitch over ALL solutions; abstractions are registered in D and so
        # become available to the next round's enumeration.
        sol_keys = [k for k, v in sols.items() if v is not None]
        _trees, rewritten_strs = saturate_stitch(D, sols, iterations=stitch_iters, max_arity=5)
        rewritten = dict(zip(sol_keys, rewritten_strs))
        # item 3: stamp the round whose stitch first carved out the agency constructor
        # (fork ∧ a sync-family commit) — the agency-signature round the trajectory figure
        # and the verdict both refer to.
        if constructor_round is None and _belief_ctor_in_D(D)[0] is not None:
            constructor_round = it

        if n_solved == n_total:
            print("    all tasks solved — wake-sleep converged.")
            break
        if n_solved == n_before and it > 1:
            # A stall on the DREAMED Q is not terminal: the post-DREAM_USE_ROUNDS
            # uniform/content fallback still has to fire, since a mis-prioritising model
            # can stall a round on programs the uniform Q would reach.  Break only once a
            # round on the uniform/content Q also adds nothing — otherwise the fallback
            # could never run in a stalled run (the dreamed stall would end the loop first).
            if not use_model:
                print("    no new tasks solved this round (uniform/content Q) — wake-sleep "
                      "stalled (raise t_fn / ecd_iters, or the abstraction belief needs "
                      "hasn't formed).")
                break
            print("    no new tasks solved this round (dreamed Q) — not terminal; continuing "
                  "to the uniform/content fallback before declaring a stall.", flush=True)

        # SLEEP-dream: train next round's recognition model on this round's replays
        # (fn solutions, preferring the abstraction-rewritten forms so the model also
        # learns to predict the invented constructors) + sampled fantasies.  Skip on
        # the final round (it < ecd_iters) and past the use window — a model trained
        # then would never be enumerated against.
        if dream_on and it < ecd_iters and it <= DREAM_USE_ROUNDS:
            replays = []
            for k in fn_keys:
                s = rewritten.get(k)
                if s:
                    try:
                        replays.append(tr(D, s))
                    except Exception:
                        continue
            if not replays:   # fall back to the raw (pre-stitch) fn solutions
                replays = [sols[k] for k in fn_keys if sols.get(k) is not None]
            print(f"    dreaming: training recognition Q on {len(replays)} replays "
                  f"+ fantasies ({dream_iters} steps)…", flush=True)
            qmodel = dream(D, replays, training_Xs=fn_Xs, root_type=fn, n_iters=dream_iters)

    # Persist this run's searched sols + library-rewritten programs so the MDL-margin
    # figure can price belief against its rivals under the library SEARCH found here,
    # not a ground-truth reconstruction (python mdl_margin.py --run <this file>).
    save_run_artifact(D, all_tasks, sols, rewritten, decomposed, smoke, prov=prov)
    # Per-round trajectory artifact: base-prim sols keyed by mat_key_id, tagged with the
    # round each was first solved, so `python corpus_dl.py --run <this file>` can rebuild
    # the corpus description-length curve (DreamCoder-style) round by round.
    save_trajectory_artifact(D, all_tasks, sols, solve_round, decomposed, smoke,
                             stitch_iters, ecd_iters, timing_log=timing_log, prov=prov)

    # ── (A) usage census: stitch-independent evidence about the bare parts ───────────
    print("\n" + "=" * 72)
    print("(A) USAGE CENSUS — which core parts each family reaches for (pre-stitch)")
    print("=" * 72)
    # Per-belief-task commit form (literal sync_to_world / degenerate scope-complement),
    # so the census, the cube corners, and the disclosure all share one classification.
    belief_form = {}
    for x, m in all_tasks:
        if m['kind'] != 'belief':
            continue
        sol = sols.get(mat_key(x))
        if sol is not None:
            belief_form[mat_key(x)] = _belief_commit_form(D, sol, x, m)
    n_belief_degenerate = sum(1 for f in belief_form.values() if f == 'degenerate')
    n_belief_literal    = sum(1 for f in belief_form.values() if f == 'literal')
    # a scope commit that ISN'T the agency commit on its own scene (swap-to-literal
    # fails to reproduce x): a genuine non-belief rival, counted as a verdict failure.
    n_belief_complement = sum(1 for f in belief_form.values() if f == 'complement')

    uses_by_kind = {}
    solved_by_kind = Counter()
    total_by_kind = Counter()
    for x, m in all_tasks:
        total_by_kind[m['kind']] += 1
        sol = sols.get(mat_key(x))
        if sol is None:
            continue
        solved_by_kind[m['kind']] += 1
        uses = _core_uses(sol)
        if m['kind'] == 'belief':
            # a degenerate scope-complement commit IS the agency commit on this scene
            uses = _canon_belief_uses(uses, belief_form.get(mat_key(x)))
        uses_by_kind.setdefault(m['kind'], set()).update(uses)
    for kind in _ALL_KINDS:
        if total_by_kind[kind] == 0:
            continue
        n, tot = solved_by_kind[kind], total_by_kind[kind]
        u = sorted(uses_by_kind.get(kind, set()))
        print(f"  {kind:13s} {n}/{tot} solved   uses: {u}")

    # belief-variant sub-census (item 5): the five variants pool into kind='belief' for
    # the unified verdict, but the thesis results table wants them split back out — per
    # variant, solved/total and how the agency commit was realised (literal sync_to_world
    # vs the degenerate scope complement).  Reuses _sample_kind and the belief_form above.
    belief_variants = {}
    for x, m in all_tasks:
        if m['kind'] != 'belief':
            continue
        rec = belief_variants.setdefault(_sample_kind(m),
                                         {'solved': 0, 'total': 0,
                                          'literal': 0, 'degenerate': 0, 'complement': 0})
        rec['total'] += 1
        if sols.get(mat_key(x)) is not None:
            rec['solved'] += 1
            form = belief_form.get(mat_key(x))
            if form in ('literal', 'degenerate', 'complement'):
                rec[form] += 1
    if belief_variants:
        print("\n  belief variants (pooled as kind='belief' above; agency commit form):")
        for v in _BELIEF_VARIANTS:
            r = belief_variants.get(v)
            if r is None:
                continue
            comp = f", {r['complement']} complement" if r['complement'] else ""
            print(f"    {v:22s} {r['solved']}/{r['total']} solved   "
                  f"commit: {r['literal']} literal, {r['degenerate']} degenerate{comp}")

    # ── SELECTIVE ATTRIBUTION (the two-observer family's own claim) ─────────────────
    # Those scenes hold two agents on one grid: the believer `av`, and an observer that
    # is simply right about the world.  Solving the task is not by itself the result —
    # the result is that the agent constructor lands on `av` and NOT on the observer.
    # So read the solution's WORLD-LEVEL commits (the same positional scoping the commit
    # census uses — a commit inside a derive writes to a private model, not the world)
    # and ask which agent value they name.  A solution that committed for both agents
    # would be attributing a private model to a bystander whose walk is plain desire.
    #
    # The commit is read AFTER canonicalising a degenerate scope commit, exactly as the
    # cube census does via _canon_belief_uses.  A `(sync_except gv)` commit names the
    # GOAL, so a raw string test finds neither agent in it and scores the task zero for
    # both — which is what collapsed this figure to 2/24 and 0/24 in the jul-26 phase-2
    # runs, where 22 and 24 of the solves respectively committed via a scope complement.
    # Those solves do put the observer outside the fork and fork only for the believer;
    # the metric simply could not see it.  `_swap_scope_commit` already recovers the
    # committed value from the fork's own derive, so re-reading the swapped tree asks
    # the question the figure is actually about: WHICH agent got a private model.
    selective = {'n': 0, 'solved': 0, 'believer_only': 0, 'observer_committed': 0}
    for x, m in all_tasks:
        if m['kind'] != 'belief' or belief_variant(m) != 'belief_observers':
            continue
        selective['n'] += 1
        sol = sols.get(mat_key(x))
        if sol is None:
            continue
        selective['solved'] += 1
        tree = simplify(normalize(deepcopy(sol)))
        if belief_form.get(mat_key(x)) == 'degenerate':
            swapped = _swap_scope_commit(D, simplify(normalize(deepcopy(sol))), m)
            if swapped is not None:
                tree = swapped
        commits = ' '.join(str(c) for c in _world_level_commits(tree))
        names_believer = _re.search(rf'\b{m["av"]}\b', commits) is not None
        names_observer = _re.search(rf'\b{m["observer"]}\b', commits) is not None
        selective['observer_committed'] += int(names_observer)
        selective['believer_only'] += int(names_believer and not names_observer)
    if selective['solved']:
        print(f"\n  two observers, one world — SELECTIVE attribution:")
        print(f"  {selective['believer_only']}/{selective['solved']} solved scenes commit "
              f"for the believer and NOT for the observer")
        print(f"  ({selective['observer_committed']} attribute a private model to the "
              f"bystander, whose walk is plain desire)")
        print(f"  (degenerate scope commits are canonicalised to the agency commit "
              f"first — see (A))")

    # fork is general if a non-belief family reaches for it — with EITHER derive:
    # overlay's fixed `step` or comet's `optimize` seek (fork's derive slot is a
    # general fn, not wired to step).
    fork_general = (_uses_fork(uses_by_kind.get('overlay', set()))
                    or _uses_fork(uses_by_kind.get('comet', set())))
    sync_general = _uses_sync(uses_by_kind.get('registration', set()))
    wall_general = 'wall_at' in uses_by_kind.get('obstacle', set())
    belief_uses_both = (_uses_fork(uses_by_kind.get('belief', set()))
                        and _uses_sync(uses_by_kind.get('belief', set())))
    # Every PART is now general (fork←overlay, sync←registration, wall_at←obstacle,
    # optimize←desire); what stays unique to belief is the AGENCY COMPOSITION —
    # fork and sync co-occurring (acting through a private model).  That is the
    # claim the cube run actually defends.
    def _uses_agency(s):
        return _uses_fork(s) and _uses_sync(s)
    agency_unique = (
        _uses_agency(uses_by_kind.get('belief', set())) and
        not any(_uses_agency(uses_by_kind.get(k, set()))
                for k in _ALL_KINDS if k != 'belief')
    )
    # Wall-based uniqueness is a claim about the families whose derive actually
    # stamps a wall — plain-belief and false-obstacle.  The displaced-goal families
    # (goal-displacement, two-observer) represent the false belief as a moved GOAL
    # inside the private model (pure fork+sync, no wall), and witness-belief may
    # solve either way; the earlier `!= belief_goal` key wrongly pulled them into the
    # all() and made it unsatisfiable the moment any such task solved.  Restrict to
    # the wall-based variants only.
    _WALL_BASED_VARIANTS = {'belief_wall', 'belief_false_obstacle'}
    # the mirror set, used by the cube census below: false belief about an OBJECT
    _DISPLACED_GOAL_VARIANTS = {'belief_goal', 'belief_observers'}
    belief_is_wall_based = all(
        'wall_at' in _core_uses(sols[mat_key(x)])
        for x, m in all_tasks
        if m['kind'] == 'belief' and belief_variant(m) in _WALL_BASED_VARIANTS
        and sols.get(mat_key(x)) is not None
    )
    print(f"\n  fork used outside belief (overlay/comet)   : {fork_general}")
    print(f"  sync used outside belief (registration)    : {sync_general}")
    print(f"  wall_at used outside belief (obstacle)     : {wall_general}")
    print(f"  belief reuses BOTH fork and sync           : {belief_uses_both}")
    print(f"  fork∧sync agency is unique to belief       : {agency_unique}")
    print(f"  wall-belief solutions are wall-based       : {belief_is_wall_based}"
          f"   (plain/false-obstacle only; goal/observers are displaced-goal, not wall)")
    if n_belief_degenerate:
        tot_bel = n_belief_literal + n_belief_degenerate
        print(f"\n  DISCLOSURE — agency commit on minimal scenes")
        print(f"  {n_belief_degenerate}/{tot_bel} belief solutions commit via a SCOPE complement "
              f"(sync_except gv / sync_all) rather")
        print(f"  than sync_to_world(av).  On a scene whose only committed model-mover is the")
        print(f"  agent, 'move everything but the goal' == 'move the agent' — the two commits are")
        print(f"  extensionally identical (verified for every belief family).  It is the SAME")
        print(f"  fork-structured belief with the agency hole expressed on the goal rather than")
        print(f"  the actor, NOT a non-belief rival; counted as the agency commit above.")
        if belief_variants.get('belief_goal', {}).get('degenerate', 0):
            print(f"  For goal-displacement this is INTRINSIC, not sampled: on a two-value scene")
            print(f"  sync_except(gv) moves exactly {{av}}, so it IS sync_to_world(av) on every")
            print(f"  expressible scene.  Certified at construction (_goal_scope_certified): every")
            print(f"  OTHER scope complement fails per scene; the family whose scenes FORCE the")
            print(f"  literal spelling is false-obstacle, below.")

    # ── the degeneracy REMOVED: false-obstacle scenes force the literal commit ────────
    # On a false-obstacle scene a REAL wall (value 3) stays in the world, so — verified
    # per task at construction (_scope_complements_all_fail) — NO scope complement
    # (sync_all, sync_except k for every world value k, incl. the wall) reproduces x.
    # The commit therefore CANNOT be degenerate: any solution the search returns commits
    # via the literal sync_to_world(av).  This is the disclosure's answer, not an
    # argument: where the complement is genuinely available (minimal scenes) it is
    # extensionally the agency commit; where it is excluded (here) the literal commit is
    # the one that gets found.
    #
    # Both sides of this are WORLD-level claims, and must be compared as such.
    # _scope_complements_all_fail rules out scope commits that write to the world; it says
    # nothing about a fork *inside* the derive, whose commit only ever writes to the
    # agent's private model and cannot disturb the real wall.  Reading the commit form off
    # the solution STRING conflated the two and fired this warning on 12 phase-2 solves
    # that in fact committed literally via register(locate av)(place av) — the census now
    # reads the world-level commits (_world_level_commits), so the warning below means
    # what it says.
    #
    # It is also a DERIVE-RELATIVE claim, and the print below now says so.  What
    # _scope_complements_all_fail certifies is that no scope complement reproduces the
    # scenes WHEN PAIRED WITH THIS FAMILY'S DERIVE; a scope commit is extensionally
    # unconstrained at the commit position (it can relocate any set of world values), so
    # what rules it out is always the model configuration some particular derive leaves
    # behind.  There is no derive-independent version of this guarantee to certify.  The
    # generator therefore certifies exactly the derive it ran (tasks.py), which is what
    # closed the jul-26 gap: the old derive opened with a vacuous `clear_at(real wall)`,
    # the searcher priced the shorter derive without it, and against THAT one
    # sync_except(gv) reproduced a quarter of the family.  A solve that still lands here
    # is a genuine finding — a shorter derive the certification did not anticipate — so
    # print the offending solution rather than asserting it cannot happen.
    fob_forms = [belief_form[mat_key(x)] for x, m in all_tasks
                 if m['kind'] == 'belief' and 'real_wall' in m
                 and mat_key(x) in belief_form]
    n_fob = sum(1 for x, m in all_tasks if m['kind'] == 'belief' and 'real_wall' in m)
    n_fob_lit = n_fob_deg = n_fob_comp = 0
    if n_fob:
        n_fob_lit = sum(1 for f in fob_forms if f == 'literal')
        n_fob_deg = sum(1 for f in fob_forms if f == 'degenerate')
        n_fob_comp = sum(1 for f in fob_forms if f == 'complement')
        print(f"\n  FORCED LITERAL COMMIT — false-obstacle family (degeneracy excluded by construction)")
        print(f"  {len(fob_forms)}/{n_fob} false-obstacle tasks solved; commit forms: "
              f"{n_fob_lit} literal, {n_fob_deg} degenerate, {n_fob_comp} complement.")
        if not (n_fob_deg or n_fob_comp):
            print(f"  Every solved one commits via the literal sync_to_world(av): no scope complement")
            print(f"  reproduces these scenes when paired with the derive this family certifies")
            print(f"  (real wall and goal stay put), so the agency commit is FORCED and found —")
            print(f"  not merely argued extensionally equal to a cheaper complement.")
        else:
            print(f"  {n_fob_deg} degenerate + {n_fob_comp} complement: a scope commit reached the "
                  f"world-level commit position")
            print(f"  on a real-wall scene.  _scope_complements_all_fail certifies the scope "
                  f"complements against")
            print(f"  the derive the generator RAN, and a scope commit is extensionally "
                  f"unconstrained on its own, so")
            print(f"  this means the search found a derive the certification did not price. "
                  f"The solutions:")
            for x, m in all_tasks:
                if m['kind'] != 'belief' or 'real_wall' not in m:
                    continue
                form = belief_form.get(mat_key(x))
                if form not in ('degenerate', 'complement'):
                    continue
                print(f"    [{form}] av={m['av']} gv={m['gv']} pw={m['pw']} "
                      f"real_wall={m['real_wall']}")
                print(f"      {simplify(normalize(deepcopy(sols[mat_key(x)])))}")

    # ── (A′) cube census: with the full symmetric field present, which corner did each family pick? ──
    cube_ok = None
    # defaults so the verdict artifact can read these even in a (hypothetical) non-cube run
    corners_by_kind = {}
    belief_keeps_corner = belief_avoids_complements = any_complement_used = None
    used_complements, unused_complements = set(), []
    if cube:
        print("\n" + "=" * 72)
        print("(A′) CUBE CENSUS — which symmetric corner each family selected")
        print("     (commit-axis corners judged AT the world-level commit; the other")
        print("      axes — grid-edit / utility / bifunctor / pairing — over the program)")
        print("=" * 72)
        corners_by_kind = {}
        for x, m in all_tasks:
            if m['kind'] == 'belief' and belief_variant(m) in _DISPLACED_GOAL_VARIANTS:
                # ONLY the displaced-goal families are excluded: their derive moves the
                # goal, it does not stamp the wall corner (see (A)'s belief_is_wall_based
                # exclusion above).  false-obstacle carries displaced_to too but IS
                # wall-based, so keying on belief_variant keeps its literal wall solves
                # in the census.
                continue
            sol = sols.get(mat_key(x))
            if sol is None:
                continue
            cu = _corner_uses(D, sol)
            if m['kind'] == 'belief':
                # a degenerate scope-complement commit is the agency corner on this scene,
                # so it must not register as belief "reaching for" a complement (see (A))
                cu = _canon_belief_uses(cu, belief_form.get(mat_key(x)))
            corners_by_kind.setdefault(m['kind'], Counter()).update(cu)
        for kind in _ALL_KINDS:
            if total_by_kind[kind] == 0:
                continue
            cs = corners_by_kind.get(kind)
            items = ', '.join(f'{c}×{n}' for c, n in cs.most_common()) if cs else '(none)'
            print(f"  {kind:13s} {items}")

        belief_corners = corners_by_kind.get('belief', Counter())
        # belief must keep exactly the agency corner and avoid every complement.  The
        # commit-axis members (sync_to_model, sync_all, sync_except, underlay, snd_gg,
        # via_swap) are counted only when belief COMMITS with them; the off-axis members
        # are counted wherever they occur, including inside a derive — an agent that
        # erases the world's walls in its private model has still reached for `erase`.
        complements = {'sync_to_model', 'sync_all', 'sync_except', 'underlay',
                       'snd_gg', 'via_swap', 'distance', 'clear_at', 'erase',
                       # bifunctor / pairing complements (decomposed runs): belief
                       # uses mapsnd + dup, so the wrong-channel / fresh-channel
                       # corners are the ones it must avoid.
                       'mapfst', 'swap', 'bimap', 'pair_blank'}
        # agency commit is sync_to_world atomically (phase 1) or its decomposition
        # register(locate)(place) (phase 2); either counts as keeping the corner.
        # Both are now read at the world-level commit position, so this asks that belief
        # COMMITS with the agency corner, not merely that the token occurs somewhere.
        belief_keeps_corner = (('sync_to_world' in belief_corners
                                or 'register' in belief_corners)
                               and 'wall_at' in belief_corners)
        # `distance` is neg_dist's symmetric complement, but belief composes it as
        # its OWN world-model seek metric — (optimize (distance k) k) is a policy step
        # inside the agent's model (see fn_9/fn_11), not a rival non-agency commit.
        # It has no dedicated family of its own either, so it is an agency-internal
        # corner, not one belief must avoid; subtract it from the avoidance test (it
        # is still reported below under field-liveness).
        _belief_seek_corners = {'distance'}
        belief_avoids_complements = not (
            set(belief_corners) & (complements - _belief_seek_corners))
        # each complement should be the corner its own family reaches for — a live,
        # fully-exercised field, not a handful of inert distractors.
        used_complements = set().union(*(
            set(corners_by_kind.get(k, Counter())) & complements
            for k in _ALL_KINDS if k != 'belief'
        )) if corners_by_kind else set()
        any_complement_used = bool(used_complements)
        # don't flag a complement as "unreached" if it has no dedicated family
        # of its own (the bifunctor/pairing corners) or isn't even a primitive in the
        # active DSL (e.g. sync_to_model, decomposed away to via_swap in phase 2).
        present = {d.repr for d in D.ds}
        no_home = {'mapfst', 'swap', 'bimap', 'pair_blank'}
        unused_complements = sorted(
            (complements - no_home - (complements - present)) - used_complements)
        cube_ok = belief_keeps_corner and belief_avoids_complements
        _agency = 'register' if decomposed else 'sync_to_world'
        print(f"\n  belief keeps the agency corner ({_agency} + wall_at){'':<{14 - len(_agency)}}: {belief_keeps_corner}")
        print(f"  belief avoids every symmetric complement                 : {belief_avoids_complements}")
        if n_belief_degenerate:
            print(f"    (note: {n_belief_degenerate} belief solve(s) committed via a scope complement "
                  f"extensionally")
            print(f"     equal to {_agency}(av) — the degenerate agency commit, not a complement; see (A))")
        print(f"  some complement is used elsewhere (field is live)        : {any_complement_used}")
        print(f"  complements claimed by another family                    : {sorted(used_complements)}")
        if unused_complements:
            print(f"  complements no family reached for (search/timeout?)      : {unused_complements}")
        if cube_ok:
            print("  => over the full symmetric field, MDL still selects the one asymmetric")
            print("     corner for belief — the agency signature is discovered, not gerrymandered.")

    # ── (B) joint compression: the final library learned across the ECD rounds ───────
    print("\n" + "=" * 72)
    print(f"(B) JOINT COMPRESSION — final library over all {sum(1 for v in sols.values() if v)} "
          f"solutions (last stitch: iterations={stitch_iters})")
    print("=" * 72)

    print("\n  invented abstractions:")
    agent_constructor = None        # literal sync_to_world / register commit (preferred)
    agent_constructor_degen = None  # degenerate scope-complement commit (fallback)
    for d in D.invented:
        body = str(simplify(normalize(deepcopy(d))))
        shared = _shared_holes(body)
        argt = ', '.join(str(t) for t in (d.tailtypes or []))
        # stable semantic label (item 6) so the thesis/viz refer to abstractions by role
        # ('agent_constructor', 'wall_policy', …) rather than the volatile fn_N index.
        print(f"    {d.repr} «{_abstraction_role(body)}»  [{argt}] -> {d.type}")
        print(f"      body: {body}")
        _has_sync = 'sync_to_world' in body or 'register' in body  # atomic | decomposed
        _has_seek = 'optimize' in body or 'neg_dist' in body        # world-model policy
        _has_scope = bool(_SCOPE_COMPLEMENTS & {p for p in _CORNERS
                                                if _re.search(rf'\b{p}\b', body)})
        if _has_fork(body) and _has_sync and _has_seek:
            # The agent constructor is fork(world-model policy) ▸ literal agency commit
            # sync_to_world(av) / register(av).  It is NOT required to stamp a wall: the
            # canonical constructor is WALL-FREE (fork + seek + sync with a shared av
            # hole) — false-obstacle is just one belief scene that additionally stamps a
            # wall.  Requiring wall_at here wrongly hid the clean wall-free constructor
            # (e.g. (fork (compose $2 (optimize (neg_dist $1) $0)) (sync_to_world $0))),
            # penalising exactly the run with the all-literal agency signature.
            # Several matches can coexist: the general constructor AND stitch's
            # own specializations of it (e.g. fn_3 = (fn_0 1 2), which bakes av in
            # as a literal so its shared hole collapses).  Keep the one that best
            # exhibits the agency signature — most shared holes — not the last seen.
            cand = (d, body, shared)
            if agent_constructor is None or len(shared) > len(agent_constructor[2]):
                agent_constructor = cand
            print(f"      *** AGENT TYPE CONSTRUCTOR (belief) ***")
            if shared:
                print(f"          shared holes: "
                      + ', '.join(f'{v} (x{n})' for v, n in shared.items())
                      + "  — actor AND committer")
        elif _has_fork(body) and _has_scope and ('optimize' in body or 'neg_dist' in body):
            # Degenerate-form constructor: commits via a scope complement that is the
            # single-value agency commit on minimal scenes (see (A) disclosure).  Only a
            # FALLBACK — if stitch also invented a literal sync_to_world constructor that
            # one is preferred for the verdict.
            cand = (d, body, shared)
            if agent_constructor_degen is None or len(shared) > len(agent_constructor_degen[2]):
                agent_constructor_degen = cand
            print(f"      *** AGENT TYPE CONSTRUCTOR (belief — degenerate scope-complement commit) ***")
            if shared:
                print(f"          shared holes: "
                      + ', '.join(f'{v} (x{n})' for v, n in shared.items())
                      + "  — actor AND committer (commit expressed on the goal)")
        elif _has_fork(body) and 'overlay' in body:
            print(f"      (non-belief: fork + overlay — motion blur)")
        elif 'wall_at' in body and ('optimize' in body or 'neg_dist' in body):
            print(f"      (obstacle/belief policy: stamp wall ▸ navigate — the shared derive)")
        elif 'optimize' in body or 'neg_dist' in body:
            print(f"      (desire fragment)")

    # Prefer a literal sync_to_world constructor; fall back to the degenerate
    # scope-complement form only if stitch invented no literal one (see (A) disclosure).
    if agent_constructor is None and agent_constructor_degen is not None:
        agent_constructor = agent_constructor_degen
        print(f"\n  (no literal sync_to_world constructor invented; using the degenerate "
              f"scope-complement\n   form {agent_constructor[0].repr} as the agent constructor "
              f"— extensionally the agency commit)")

    # which abstraction each family's rewritten program reaches for
    abst_names = [d.repr for d in D.invented]
    def _absts_in(s):
        return sorted(a for a in abst_names if _re.search(rf'\b{a}\b', s))

    print("\n  abstraction usage by family (from stitch's rewritten programs):")
    fam_absts = {}
    for x, m in all_tasks:
        s = rewritten.get(mat_key(x))
        if s is None:
            continue
        fam_absts.setdefault(m['kind'], Counter()).update(_absts_in(s) or ['(bare prims)'])
    for kind in _ALL_KINDS:
        if kind in fam_absts:
            items = ', '.join(f'{a}×{n}' for a, n in fam_absts[kind].most_common())
            print(f"    {kind:13s} {items}")

    # ── (B′) does the donated seek/policy actually generalize across (gv,av)? ─────────
    report_abstraction_generality(D, all_tasks, rewritten)

    # ── (B″) the magnitudes behind (B): belief's DL before vs after the library ───────
    dl_census = report_dl_census(D, all_tasks, sols, rewritten,
                                 ctor_name=(agent_constructor[0].repr if agent_constructor
                                            else None))

    # ── verdict ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    constructor_found = agent_constructor is not None
    constructor_shared = bool(agent_constructor and agent_constructor[2])
    # the agency signature is a COUNT, not a flag: how many holes the constructor passes
    # to both the actor and the committer slots, and how many times each recurs.  A
    # constructor with one hole shared twice ($0 = av, actor AND committer) is the
    # signature; one with none is a fork that merely happens to contain a sync.
    ctor_shared_holes = dict(agent_constructor[2]) if agent_constructor else {}
    n_shared_holes = len(ctor_shared_holes)
    max_hole_uses = max(ctor_shared_holes.values(), default=0)
    # the constructor must be a BELIEF abstraction, and the non-belief families must
    # NOT have been swept into it (else it isn't really belief-specific).
    ctor_name = agent_constructor[0].repr if agent_constructor else None
    belief_uses_ctor = bool(ctor_name) and any(
        ctor_name in _absts_in(rewritten.get(mat_key(x), ''))
        for x, m in all_tasks if m['kind'] == 'belief'
    )
    nonmental_free_of_ctor = bool(ctor_name) and not any(
        ctor_name in _absts_in(rewritten.get(mat_key(x), ''))
        for x, m in all_tasks if m['kind'] != 'belief'
    )
    # no belief solution may commit via a scope complement that ISN'T its own agency
    # commit (a 'complement' rival — see _belief_commit_form).  A single one breaks the
    # claim that fork∧sync is doing agency, so the verdict conjoins its absence.
    no_belief_complement = (n_belief_complement == 0)

    print(f"  (A) parts are general    : fork∉belief-only={fork_general}, "
          f"sync∉belief-only={sync_general}, wall∉belief-only={wall_general}")
    print(f"  (A) only agency is unique: fork∧sync unique to belief={agency_unique}, "
          f"belief recombines={belief_uses_both}, wall-based={belief_is_wall_based}")
    print(f"  (A) no scope-complement rival among belief solves : {no_belief_complement}"
          f"   ({n_belief_complement} complement, {n_belief_degenerate} degenerate, "
          f"{n_belief_literal} literal)")
    print(f"  (B) constructor invented : {constructor_found}"
          + (f" ({ctor_name}, arity {len(agent_constructor[0].tailtypes or [])}; "
             f"{n_shared_holes} shared agency hole(s)"
             + (": " + ', '.join(f'{v}×{n}' for v, n in ctor_shared_holes.items())
                if ctor_shared_holes else "") + ")"
             if agent_constructor else ""))
    print(f"  (B) constructor is belief-specific: used by belief={belief_uses_ctor}, "
          f"absent from other families={nonmental_free_of_ctor}")
    if dl_census.get('by_kind', {}).get('belief'):
        _b = dl_census['by_kind']['belief']
        _nm = dl_census.get('nonmental_saved_median')
        print(f"  (B) belief DL before -> after the library: {_b['dl_base_median']:.2f} -> "
              f"{_b['dl_lib_median']:.2f} nats (median saving {_b['saved_median']:.2f} over "
              f"{_b['n']} solves;")
        print(f"      {dl_census['n_belief_using_ctor']}/{_b['n']} invoke the constructor"
              + (f"; other families save {_nm:.2f} nats/task for scale)"
                 if _nm is not None else ")"))
    # the central magnitude — priced by mdl_margin.py under this same library, echoed
    # here so the run reports how MUCH shorter the mental reading is, not only that the
    # booleans held.  See load_mdl_margin re: staleness.
    margin_summary, margin_note = report_mdl_margin(decomposed, smoke)
    if cube:
        print(f"  (A′) cube: belief picks the lone asymmetric corner over the full field: {cube_ok}")

    # every criterion the headline verdict conjoins; failures names the ones that broke,
    # so a partial run localises itself without the reader scanning the booleans above.
    checks = {
        'fork_general': fork_general, 'sync_general': sync_general,
        'wall_general': wall_general, 'belief_uses_both': belief_uses_both,
        'agency_unique': agency_unique, 'belief_is_wall_based': belief_is_wall_based,
        'no_belief_complement': no_belief_complement,
        'constructor_found': constructor_found, 'constructor_shared': constructor_shared,
        'belief_uses_ctor': belief_uses_ctor, 'nonmental_free_of_ctor': nonmental_free_of_ctor,
        'cube_ok': (cube_ok is not False),
    }
    failures = [name for name, v in checks.items() if not v]
    ok = not failures
    if ok:
        print("\n  => In ONE library and ONE MDL compression over the whole corpus, belief is")
        print("     the discovered recombination of parts that each do work unrelated to belief,")
        print("     and the same objective that builds the agent constructor leaves fork/sync")
        print("     bare elsewhere.  Not gerrymandered, not a silo artefact.")
    else:
        print("\n  => not fully demonstrated this run (raise timeouts / n_bel / stitch_iters,")
        print("     or drop --smoke).")
        print(f"     failed criteria: {failures}")

    # ── (C) persist the verdict as structured data (item 2) ───────────────────────────
    # Every boolean and census count the (A)/(A′)/(B)/VERDICT sections print — plus the
    # round attribution (item 3) — so the thesis text and the viz cite exact values from
    # one file instead of grepping the slurm log.  See save_verdict_artifact.
    first_solve_round, first_solve_round_fine = {}, {}
    for k, r in solve_round.items():
        kd, fk = kind_of_all.get(k), fine_kind_of.get(k)
        if kd is not None:
            first_solve_round[kd] = min(int(r), first_solve_round.get(kd, int(r)))
        if fk is not None:
            first_solve_round_fine[fk] = min(int(r), first_solve_round_fine.get(fk, int(r)))
    invented = []
    for d in D.invented:
        body = str(simplify(normalize(deepcopy(d))))
        invented.append({'repr': d.repr, 'type': str(d.type),
                         'tailtypes': [str(t) for t in (d.tailtypes or [])],
                         'body': body, 'shared_holes': _shared_holes(body),
                         'role': _abstraction_role(body)})
    verdict = {
        'decomposed': bool(decomposed), 'smoke': bool(smoke),
        'n_solved': sum(1 for v in sols.values() if v is not None), 'n_total': n_total,
        'stitch_iters': stitch_iters, 'ecd_iters': ecd_iters,
        't_fn': t_fn, 't_fn_round1': t_fn_round1,   # curriculum budget schedule (c/target-2)
        # round attribution (item 3)
        'constructor_round': constructor_round,
        'first_solve_round_by_kind': first_solve_round,
        'first_solve_round_by_variant': first_solve_round_fine,
        # (A) usage census
        'solved_by_kind': solved_by_kind, 'total_by_kind': total_by_kind,
        'uses_by_kind': uses_by_kind,
        'belief_commit': {'literal': n_belief_literal, 'degenerate': n_belief_degenerate,
                          'complement': n_belief_complement},
        'belief_variants': belief_variants,
        'selective_attribution': selective,
        'false_obstacle': {'n': n_fob, 'literal': n_fob_lit, 'degenerate': n_fob_deg,
                          'complement': n_fob_comp},
        'A': {'fork_general': fork_general, 'sync_general': sync_general,
              'wall_general': wall_general, 'belief_uses_both': belief_uses_both,
              'agency_unique': agency_unique, 'belief_is_wall_based': belief_is_wall_based,
              'no_belief_complement': no_belief_complement},
        # (A′) cube census
        'cube': {'enabled': bool(cube), 'cube_ok': cube_ok,
                 'corners_by_kind': corners_by_kind,
                 'belief_keeps_corner': belief_keeps_corner,
                 'belief_avoids_complements': belief_avoids_complements,
                 'any_complement_used': any_complement_used,
                 'used_complements': used_complements,
                 'unused_complements': unused_complements},
        # (B) joint compression
        'invented': invented,
        'agent_constructor': (agent_constructor[0].repr if agent_constructor else None),
        'agent_constructor_shared_holes': ctor_shared_holes,
        'agent_constructor_n_shared_holes': n_shared_holes,
        'agent_constructor_max_hole_uses': max_hole_uses,
        'abstraction_usage_by_kind': fam_absts,
        # (B″) magnitudes: DL before (base prims) vs after (library) per family/variant,
        # and the belief-vs-rival margins mdl_margin.py priced under this library — the
        # nats the prose quotes, in the same file as the booleans they support.
        'dl': dl_census,
        'mdl_margin': margin_summary,
        'mdl_margin_note': margin_note,
        # verdict
        'constructor_found': constructor_found, 'constructor_shared': constructor_shared,
        'belief_uses_ctor': belief_uses_ctor, 'nonmental_free_of_ctor': nonmental_free_of_ctor,
        'failures': failures, 'ok': ok,
    }
    save_verdict_artifact(verdict, decomposed, smoke, prov=prov)


def _abstraction_role(body):
    """Semantic role of an invented abstraction, mirroring the (B) print branches so the
    verdict artifact can name abstractions ('agent_constructor', 'wall_policy', …) rather
    than leave the thesis/viz to re-classify fn_0/fn_3/… by hand."""
    has_sync  = 'sync_to_world' in body or 'register' in body
    has_scope = bool(_SCOPE_COMPLEMENTS & {p for p in _CORNERS
                                           if _re.search(rf'\b{p}\b', body)})
    if _has_fork(body) and has_sync and 'wall_at' in body:
        return 'agent_constructor'
    if _has_fork(body) and has_sync:
        # fork ∧ a literal sync-family commit but no phantom wall — the wall-FREE agency
        # constructor (witness / goal-displacement belief).  The wall-based (B) headline
        # detector skips it; label it so the thesis doesn't read it as a desire fragment.
        return 'agent_constructor_wallfree'
    if _has_fork(body) and has_scope and ('optimize' in body or 'neg_dist' in body):
        return 'agent_constructor_degenerate'
    if _has_fork(body) and 'overlay' in body:
        return 'motion_blur'
    if 'wall_at' in body and ('optimize' in body or 'neg_dist' in body):
        return 'wall_policy'
    if 'optimize' in body or 'neg_dist' in body:
        return 'desire_fragment'
    return 'other'


def _coerce(o):
    "recursively make a verdict value JSON-serialisable (Counters/sets/numpy scalars)."
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, Counter):
        return {str(k): int(v) for k, v in o.items()}
    if isinstance(o, dict):
        return {str(k): _coerce(v) for k, v in o.items()}
    if isinstance(o, set):
        return sorted(_coerce(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_coerce(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def git_provenance():
    """The repo state this run was launched from: commit, branch, and whether the tree
    was dirty.  `dirty` is the honest part — a figure caption citing a commit is only
    reproducible if no uncommitted edits were in play, so we record that rather than
    quietly implying the commit is the whole story.  Degrades to nulls outside a git
    checkout (e.g. an HPC scratch copy) instead of failing the run."""
    import subprocess
    def _git(*args):
        try:
            out = subprocess.run(('git',) + args, cwd=os.path.dirname(os.path.abspath(__file__)),
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None
    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain')
    return {
        'commit':       commit,
        'commit_short': commit[:7] if commit else None,
        'branch':       _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty':        None if status is None else bool(status),
    }


def build_provenance(knobs):
    """One provenance block, embedded verbatim in every artifact a run writes.

    The log and the artifacts already record `decomposed`/`smoke`, but that is not
    enough to reproduce a number: the same phase at t_fn=15 and t_fn=3600 are different
    experiments (see the Jul-12 runs, where budget alone changed which rivals landed).
    So we record the full knob set — every generator seed and family size, the search
    budgets, and the git commit — and derive a one-line `repro` string a figure caption
    can quote directly.

    `knobs` is the resolved settings dict built at the top of run_phase (resolved, i.e.
    after CLI overrides and the smoke/full defaults, so it says what the run DID, not
    what was asked for)."""
    import datetime
    import platform
    git = git_provenance()
    prov = {
        'git':       git,
        'timestamp': datetime.datetime.now(datetime.timezone.utc)
                             .isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'host':      platform.node(),
        'python':    platform.python_version(),
        'argv':      ' '.join(sys.argv),
        'knobs':     dict(knobs),
        'seeds':     dict(TASK_SEEDS),
    }
    commit = git['commit_short'] or 'nogit'
    if git['dirty']:
        commit += '+dirty'
    phase = 2 if knobs['decomposed'] else 1
    prov['repro'] = (
        f"commit {commit} | phase{phase} "
        f"({'decomposed' if knobs['decomposed'] else 'atomic'}"
        f"{', smoke' if knobs['smoke'] else ''}) | "
        f"t_fn={_g(knobs['t_fn'])} t_fn_round1={_g(knobs['t_fn_round1'])} "
        f"t_reg={_g(knobs['t_reg'])} ecd_iters={knobs['ecd_iters']} "
        f"stitch_iters={knobs['stitch_iters']} "
        f"dream={'on' if knobs['dream_on'] else 'off'} "
        f"curriculum={'on' if knobs['curriculum'] else 'off'}"
        f"{' plain_belief' if knobs['plain_belief'] else ''} | "
        f"n_bel={knobs['n_bel']} n_goal={knobs['n_goal']} "
        f"n_obstacle={knobs['n_obstacle']} n_relocate={knobs['n_relocate']} | "
        f"{prov['timestamp']}"
    )
    return prov


def provenance_line(art, path=None):
    """The repro line for an artifact loaded off disk, for the figure scripts that consume
    one (mdl_margin, corpus_dl, solve_dynamics, behavioral_probe).  They print it, and
    carry the block into their own artifact, so a figure's numbers stay traceable to the
    run that produced them.  Artifacts written before the header existed have no
    provenance — say so rather than implying an unknown provenance is a clean one."""
    prov = (art or {}).get('provenance')
    if not prov:
        return (f"provenance: NONE recorded in {path or 'this artifact'} — predates the "
                f"provenance header; rerun the phase to make its figures citable.")
    return prov.get('repro', 'provenance: recorded but no repro line')


def _g(v):
    "compact number formatting for the repro line: 180.0 -> 180, 2.5 -> 2.5"
    return f"{v:g}" if isinstance(v, float) else str(v)


def print_provenance(prov):
    "Header at the top of the log, so a run's stdout alone identifies the run."
    print("=" * 72)
    print("PROVENANCE — quote this line in any caption reporting these numbers")
    print("=" * 72)
    print(f"  {prov['repro']}")
    if prov['git']['dirty']:
        print("  WARNING: working tree was DIRTY at launch — the commit above does not "
              "fully\n           determine this run.  Commit before a run whose numbers "
              "are cited.")
    elif prov['git']['commit'] is None:
        print("  WARNING: not a git checkout — no commit recorded for this run.")
    print(f"  host {prov['host']} | python {prov['python']} | argv: {prov['argv']}")
    print()


def save_verdict_artifact(verdict, decomposed, smoke, path=None, prov=None):
    """Persist the structured verdict (all (A)/(A′)/(B)/VERDICT booleans + census counts +
    round attribution) so the thesis text and the figures read exact values from one file
    instead of parsing stdout.  Consumed alongside phase{n}_run.json / phase{n}_traj.json.

    `prov` is the run's provenance block (see build_provenance): the git commit and the
    full knob set that produced these booleans."""
    import json
    if path is None:
        path = f"phase{2 if decomposed else 1}_verdict{'.smoke' if smoke else ''}.json"
    out = dict(_coerce(verdict))
    if prov is not None:
        out['provenance'] = prov
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote verdict artifact to {path}")
    return path


def save_run_artifact(D, all_tasks, sols, rewritten, decomposed, smoke,
                      path=None, prov=None):
    """Persist a phase run's search output so `mdl_margin.py` can price belief against
    its rivals under the library THIS RUN actually found, rather than re-deriving one
    from ground truth.

    We store, per solved task (keyed by a stable `mat_key_id`):
      * `sols`      — the searched program, fully expanded to base primitives
                      (`simplify(normalize(...))`, exactly the form fed to the joint
                      stitch), in solve order.  Re-stitching these reproduces this run's
                      final library deterministically.
      * `rewritten` — that program rewritten through the final library (the searched
                      program's library form — mdl_margin's "found_lib").
    Plus `kinds` (for diagnostics) and the phase flags so the consumer rebuilds the
    matching base DSL.  See mdl_margin.run(run_path=...).

    `prov` is the run's provenance block (see build_provenance), so a margin figure
    drawn from this file can name the commit and budgets that produced it.
    """
    import json
    if path is None:
        path = f"phase{2 if decomposed else 1}_run{'.smoke' if smoke else ''}.json"
    key2id = {mat_key(x): mat_key_id(x) for x, _ in all_tasks}
    kind_of = {mat_key(x): m['kind'] for x, m in all_tasks}
    sols_ser, kinds_ser = {}, {}
    for k, sol in sols.items():            # solve order → stitch input order
        if sol is None or k not in key2id:
            continue
        kid = key2id[k]
        sols_ser[kid] = str(simplify(normalize(sol)))
        kinds_ser[kid] = kind_of[k]
    rew_ser = {key2id[k]: s for k, s in rewritten.items()
               if k in key2id and s}
    out = {
        'provenance': prov,
        'decomposed': bool(decomposed),
        'smoke': bool(smoke),
        'library': [d.repr for d in D.invented],
        'n_solved': len(sols_ser),
        'sols': sols_ser,
        'rewritten': rew_ser,
        'kinds': kinds_ser,
    }
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote run artifact ({len(sols_ser)} solved programs) to {path}")
    return path


def save_trajectory_artifact(D, all_tasks, sols, solve_round, decomposed, smoke,
                             stitch_iters, ecd_iters, path=None, timing_log=None,
                             prov=None):
    """Persist the per-round SEARCH TRAJECTORY so `corpus_dl.py` can rebuild the corpus
    description-length curve (DreamCoder-style: total DL of every solution under the
    current library, plus the library's own cost, per ECD round).

    We store, per solved task (keyed by a stable `mat_key_id`):
      * `sol`    — the searched program expanded to base primitives (the exact form fed
                   to the joint stitch), so corpus_dl can re-stitch the cumulative
                   solved-through-round-r subset and reproduce that round's library.
      * `kind`   — the task family (physics / desire / belief / world makers), for the
                   per-family facet.
      * `round`  — the ECD round the task was first solved (1-indexed).  A task solved in
                   round r is present in the corpus from round r onward.

    Because `saturate_stitch` resets and re-discovers the library from the fully-expanded
    sols each call (see ecd.saturate_stitch), re-stitching the cumulative round-r subset
    reproduces round r's library deterministically — the same reproducibility mdl_margin
    relies on for the final round.  `stitch_iters` is recorded so corpus_dl compresses
    with the same budget.

    `timing_log` (optional) is the flat list of every fn enumeration attempt
    (mat_key, round, solved?, seconds) gathered in the wake-sleep loop.  We serialise it
    as `timings` so `solve_dynamics.py` can draw the cumulative-solve S-curve and the
    per-task solve-time collapse (a belief task burns ~t_fn missing in the round before
    its abstraction lands, then solves in seconds the next round) directly from the run —
    the numbers that are otherwise only printed to the log.  fn_p_g (registration) tasks
    go through `solve_enumeration`, which does not return per-task times, so they are
    absent here; belief — the family the collapse is about — is an fn task and is present.
    """
    import json
    if path is None:
        path = f"phase{2 if decomposed else 1}_traj{'.smoke' if smoke else ''}.json"
    key2id  = {mat_key(x): mat_key_id(x) for x, _ in all_tasks}
    kind_of = {mat_key(x): m['kind'] for x, m in all_tasks}
    tasks_ser = {}
    for k, sol in sols.items():
        if sol is None or k not in key2id:
            continue
        kid = key2id[k]
        tasks_ser[kid] = {
            'sol':   str(simplify(normalize(sol))),
            'kind':  kind_of[k],
            'round': int(solve_round.get(k, ecd_iters)),
        }
    timings_ser = [
        {'id': key2id[k], 'kind': kind_of.get(k, '?'),
         'round': int(rnd), 'solved': bool(hit), 'elapsed': float(sec)}
        for (k, rnd, hit, sec) in (timing_log or []) if k in key2id
    ]
    out = {
        'provenance':   prov,
        'decomposed':   bool(decomposed),
        'smoke':        bool(smoke),
        'stitch_iters': int(stitch_iters),
        'ecd_iters':    int(ecd_iters),
        'n_rounds':     max([t['round'] for t in tasks_ser.values()], default=0),
        'n_solved':     len(tasks_ser),
        'tasks':        tasks_ser,
        'timings':      timings_ser,
    }
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote trajectory artifact ({len(tasks_ser)} solved programs, "
          f"{out['n_rounds']} rounds) to {path}")
    return path


def cli_kwargs(argv):
    "shared CLI parsing for the phase wrappers: --smoke --samples --ecd-iters N --t-fn N --t-fn-round1 N --k N --no-dream --plain-belief --no-curriculum"
    def _opt(flag, cast):
        if flag in argv:
            return cast(argv[argv.index(flag) + 1])
        return None
    return dict(smoke='--smoke' in argv,
                samples='--samples' in argv,
                ecd_iters=_opt('--ecd-iters', int),
                t_fn=_opt('--t-fn', float),
                t_fn_round1=_opt('--t-fn-round1', float),
                k=_opt('--k', int),
                dream_on='--no-dream' not in argv,
                plain_belief='--plain-belief' in argv,
                curriculum='--no-curriculum' not in argv)


if __name__ == '__main__':
    run_phase(decomposed='--decomposed' in sys.argv, **cli_kwargs(sys.argv))
