"""experiment.py: the unified mixed-corpus test — belief as an MDL win, not a silo.

The non-mental task families answered the "gerrymandered decomposition" charge
(fork/sync are a
`believe` primitive split into two gears that only ever re-mesh into belief) by
giving the gears non-mental work: fork without sync (overlay/motion-blur) and
sync without fork (registration/coordinate-join).  But it proved this across
THREE isolated searches — three separate `ECD` calls, three separate `Deltas`,
three separate stitch passes.  That shows each gear is *reachable* somewhere; it
does not show that, under ONE compression objective seeing ALL the evidence at
once, belief still emerges as the MDL-optimal recombination while its parts stay
general.  Gerrymandering is an MDL claim, and MDL is only tested by joint
compression.

This file runs ONE library and ONE joint stitch over a mixed corpus:

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
      non-mental abstraction) in the overlay/registration solutions.

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
import math
import re as _re
from collections import Counter
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch as th

from ecd import (
    Deltas, solve_enumeration, saturate_stitch, mat_key, mat_key_id, normalize,
    _solve_one_task, _worker_init, _n_cpus_available,
    dream, dreamed_q,
)
from dsl import (
    fn, fn_p_g, cellvalue, coord,
    unfold, unfold_with_template, tr, simplify, length,
    # used by check_decomposition_identities (phase 2)
    compose, wall_at, clear_at, optimize, neg_distance, step,
    fork, sync_to_world, fork_decomposed, register, locate, place,
)

# Reuse the generators and DSL already vetted in the curriculum, so the experiment
# driver adds only the *unification*, never a new encoding to second-guess.
from tasks_minds import (
    make_physics_tasks, make_desire_tasks, make_belief_tasks,
    make_witness_belief_tasks,
    make_goal_displacement_tasks, make_dual_belief_tasks,
    make_false_obstacle_belief_tasks,
    COMBOS, SIZE, DIRS,
)
from tasks_world import (
    make_overlay_tasks, make_comet_tasks, make_registration_tasks,
    # one minds-free task per symmetric corner
    make_flee_tasks, make_deletion_tasks, make_denoise_tasks, make_obstacle_tasks,
    make_underlay_tasks,
    make_perception_tasks, make_multi_registration_tasks,
    make_registration_except_tasks, make_inpainting_tasks, make_readout_tasks,
)
from prims import make_core_prims, make_symmetric_prims

# corpus families by interpreter / root type.  The first block is fixed; the cube
# corner families (second block) are appended only in --cube runs, where their
# primitives exist.  Reporting loops iterate these so new families show up
# everywhere without per-call edits.
_FN_KINDS   = ['physics', 'desire', 'overlay', 'comet', 'belief',
               'flee', 'deletion', 'denoise', 'obstacle', 'underlay']
_PAIR_KINDS = ['registration', 'perception', 'multi_reg', 'reg_except',
               'inpaint', 'readout']
_CUBE_KINDS = ['flee', 'deletion', 'denoise', 'obstacle', 'underlay', 'perception',
               'multi_reg', 'reg_except', 'inpaint', 'readout']
_ALL_KINDS  = ['physics', 'desire', 'overlay', 'comet', 'registration', 'belief',
               'flee', 'deletion', 'denoise', 'obstacle', 'underlay',
               'perception', 'multi_reg', 'reg_except', 'inpaint', 'readout']

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
# still avoids ALL of these while the non-mental families happily reach for them.
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
    # A wall (value 3) anywhere in the trajectory is a REAL, visible obstacle (the
    # obstacle family), not belief's phantom wall — so its row/col are not a latent.
    # Boost the matching coord terminals so obstacle need not brute-force all SIZE×SIZE
    # wall positions (~5k programs → ~tens).  We scan ALL frames, not just frame 0: the
    # wall is stamped during the trajectory (unfold starts on a wall-free grid), so it
    # is absent from x[0].  Belief never renders its wall into the world frames, so this
    # never fires there — belief's wall coordinate stays a uniform-cost latent, as it
    # must, and the cheap-obstacle path can never leak into the belief search.
    wall_coords = {int(k) for trc in np.argwhere(x == 3) for k in trc[1:]}
    for d in D.ds:
        if d.tailtypes is None and d.type == cellvalue and d.head in visible:
            q[D.index(d)] = 0.0
        elif d.tailtypes is None and d.type == coord and d.head in wall_coords:
            q[D.index(d)] = 0.0
    return q


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
    with the derive distinguishing the family (wall_at / step / two forks), and the
    commit ALWAYS the single-value agency commit sync_to_world(av) (its decomposition
    in phase 2).  Shared between verify_ground_truth and the verdict's degeneracy
    recognition, so both build the same canonical program."""
    if 'pw2' in m:                                   # two contradictory false beliefs
        d1 = (f"(compose (wall_at c{m['pw'][0]} c{m['pw'][1]}) "
              f"(optimize (neg_dist {m['gv']}) {m['av']}))")
        d2 = (f"(compose (wall_at c{m['pw2'][0]} c{m['pw2'][1]}) "
              f"(optimize (neg_dist {m['gv2']}) {m['av2']}))")
        return f"(compose {_forks(D, d1, _sync(D, m['av']))} {_forks(D, d2, _sync(D, m['av2']))})"
    if 'real_wall' in m:                             # false about BOTH obstacle and goal:
        # a REAL wall stays in the world, so on this scene NO scope complement reproduces
        # x (verified at construction by _scope_complements_all_fail) — the commit can only
        # be the literal sync_to_world(av).  Also carries displaced_to, so test it BEFORE
        # the goal branch.  derive = _seq(clear real wall, stamp phantom, shove goal, seek).
        wr, wc = m['real_wall']
        pr, pc = m['pw']
        derive = (f"(compose (compose (compose (clear_at c{wr} c{wc}) (wall_at c{pr} c{pc})) "
                  f"(step {m['gv']} {m['dir']})) (optimize (neg_dist {m['gv']}) {m['av']}))")
        return _forks(D, derive, _sync(D, m['av']))
    if 'displaced_to' in m:                          # false belief about the goal's location
        # left-fold the shove sequence + seek, mirroring tasks_minds._seq
        parts = ([f"(step {m['gv']} {dn})" for dn in m['dirs']]
                 + [f"(optimize (neg_dist {m['gv']}) {m['av']})"])
        derive = parts[0]
        for p in parts[1:]:
            derive = f"(compose {derive} {p})"
        return _forks(D, derive, _sync(D, m['av']))
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
    # belief (wall / witness / goal-displacement / dual) — all agency-committed
    return _belief_gt_str(D, m)


def verify_ground_truth(D, tasks):
    for x, m in tasks:
        tree = tr(D, gt_program_str(D, m))
        if m['kind'] in _TEMPLATE_KINDS:
            out = unfold_with_template(x[0], m['template'], x.shape[0], tree())
        else:
            out = unfold(x[0], x.shape[0], tree())
        assert np.array_equal(out, x), f"ground truth failed for {m['kind']}: {m}"
    print(f"  ground-truth check: {len(tasks)} tasks verified via Delta trees")


# ── reporting helpers ─────────────────────────────────────────────────────────────

def _core_uses(tree):
    "core primitives reached for in a solution, AFTER expanding to primitives."
    s = str(simplify(normalize(deepcopy(tree))))
    return {p for p in _INTERFACE if p in s}


def _shared_holes(body_str):
    "map $i -> count for holes that occur more than once (the agency signature)."
    c = Counter(_re.findall(r'\$\d+', body_str))
    return {v: n for v, n in c.items() if n > 1}


def _corner_uses(tree):
    "which symmetric corners a solution reaches for, AFTER expanding to primitives."
    s = str(simplify(normalize(deepcopy(tree))))
    return {p for p in _CORNERS if _re.search(rf'\b{p}\b', s)}


_SCOPE_COMPLEMENTS = {'sync_except', 'sync_all'}


def _belief_commit_form(D, sol, x, m):
    """How a belief solution realises the single-value agency commit.

      'literal'    — commits via sync_to_world / register: the clean shared-av signature.
      'degenerate' — commits (also/only) via a SCOPE complement (sync_except gv / sync_all)
                     that is extensionally equal to sync_to_world(av) on this scene.  When
                     the agent is the only committed model-mover, 'move everything but the
                     goal' == 'move the agent'; same fork-structured belief, agency hole
                     expressed on gv rather than av.  Verified for every belief family
                     (plain/witness/goal-displacement/dual): sync_except(gv) and
                     sync_to_world(av) reproduce x identically.  A minimal-scene degeneracy
                     of the commit, NOT a rival (non-mental) theory — see the disclosure note.
      None         — neither: not an agency commit (e.g. unsolved / non-fork).

    Guarded by an extensional check: we only call a scope-complement commit degenerate
    when the canonical agency program for this exact task also reproduces x."""
    s = str(simplify(normalize(deepcopy(sol))))
    has_literal = 'sync_to_world' in s or 'register' in s
    has_scope = bool(_SCOPE_COMPLEMENTS & {p for p in _CORNERS if _re.search(rf'\b{p}\b', s)})
    if has_scope and _has_fork(s):
        try:
            canon = tr(D, _belief_gt_str(D, m))
            if np.array_equal(unfold(x[0], x.shape[0], canon()), x):
                return 'degenerate'
        except Exception:
            pass
    if has_literal:
        return 'literal'
    return None


def _canon_belief_uses(uses_or_corners, form):
    "rewrite a degenerate scope-complement belief commit to the canonical agency commit."
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


_BELIEF_VARIANTS = ['belief_wall', 'belief_witness', 'belief_goal', 'belief_dual',
                    'belief_false_obstacle']


def _sample_kind(m):
    """Finer label for sampling only — the belief variants all share kind='belief'
    (so they feed one unified verdict), but should display as separate panels."""
    if m['kind'] != 'belief':
        return m['kind']
    if 'pw2' in m:
        return 'belief_dual'
    if 'real_wall' in m:                 # false about both obstacle and goal (before goal)
        return 'belief_false_obstacle'
    if 'displaced_to' in m:
        return 'belief_goal'
    if 'aw' in m:
        return 'belief_witness'
    return 'belief_wall'


def _select_samples(tasks, max_frames=6):
    """One example per kind (first seen, in _ALL_KINDS order) as labelled panels.

    Each sample is {kind, tag, T, panels:[(label, grid), …]}.  fn families show
    successive `unfold` frames t0…; fn_p_g families show world | template | result,
    surfacing the otherwise-invisible constant template channel.  The rendered
    grids are exactly what the searcher sees — belief's phantom wall lives only in
    the private model, so it never appears here.  The single kind='belief' is split
    into its variants (wall / witness / goal-displacement / dual) for display."""
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
                        if k not in ('kind', 'template'))
        if 'template' in m:                  # fn_p_g: world | template | result(s)
            extra = list(x[1:max_frames - 1])
            panels = [('world', x[0]), ('template', m['template'])]
            panels += [(f't{t}', g) for t, g in enumerate(extra, start=1)]
        else:                                # fn: successive frames of the unfold
            panels = [(f't{t}', x[t]) for t in range(min(len(x), max_frames))]
        out.append({'kind': kind, 'tag': tag, 'T': int(x.shape[0]),
                    'panels': panels})
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

    if 'pw2' in m:                                   # dual: two contradictory beliefs
        d1 = compose(wall_at(*m['pw']),  optimize(neg_distance(m['gv']),  m['av']))
        d2 = compose(wall_at(*m['pw2']), optimize(neg_distance(m['gv2']), m['av2']))
        o1, e1 = one(d1, m['av'])
        o2, e2 = one(d2, m['av2'])
        return compose(o1, o2), compose(e1, e2)
    if 'real_wall' in m:                             # false about BOTH obstacle and goal
        # mirror _belief_gt_str's real_wall branch: clear real wall ▸ stamp phantom ▸
        # shove goal ▸ seek.  Carries displaced_to too, so test it BEFORE the goal branch.
        derive = compose(compose(compose(clear_at(*m['real_wall']), wall_at(*m['pw'])),
                                 step(m['gv'], DIRS[m['dir']])),
                         optimize(neg_distance(m['gv']), m['av']))
        return one(derive, m['av'])
    if 'displaced_to' in m:                          # goal-displacement: false belief about the goal
        derive = step(m['gv'], DIRS[m['dirs'][0]])
        for dn in m['dirs'][1:]:
            derive = compose(derive, step(m['gv'], DIRS[dn]))
        derive = compose(derive, optimize(neg_distance(m['gv']), m['av']))
        return one(derive, m['av'])
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
        orig = unfold(x[0], x.shape[0], prog_orig)
        deco = unfold(x[0], x.shape[0], prog_deco)
        assert np.array_equal(orig, x), f"orig fork != task for {m}"
        assert np.array_equal(deco, x), f"decomposed != task for {m}"
        n += 1
    print(f"decomposition identity: fork ≡ commit∘mapsnd(derive)∘dup and "
          f"sync ≡ register(locate, place) verified on {n} belief tasks")


def run_phase(decomposed=False, smoke=False, samples=False, ecd_iters=None, t_fn=None,
              dream_on=True, plain_belief=False, curriculum=True):
    """One phase of the curriculum (phase 1 = atomic, phase 2 = decomposed).

    Both phases run the full symmetric cube over the mixed minds/minds-free corpus
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
    """
    cube = True
    if smoke:
        n_phys, n_des, n_ov, n_reg, n_bel, n_corner = 2, 1, 2, 2, 1, 2
        n_comet = 2
        n_belvar = 1
        n_goal = 2
        n_obstacle = 2
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
        _t_fn, t_reg, stitch_iters, _ecd_iters = 180, 30, 6, 4
    t_fn = _t_fn if t_fn is None else t_fn
    ecd_iters = _ecd_iters if ecd_iters is None else ecd_iters
    dream_iters = 120 if smoke else 600   # recognition-model training steps per round

    print("Generating mixed corpus…")
    phys = make_physics_tasks(n_phys, seed=0)
    des  = make_desire_tasks(n_des, COMBOS, seed=1)
    ov   = make_overlay_tasks(n_ov, seed=3)
    # comet: fork WITHOUT sync like overlay, but the derive is desire's seek policy
    # instead of a fixed step — shows fork's derive slot is a general fn (step OR
    # optimize), the dual of the underlay family varying the commit slot.
    comet = make_comet_tasks(n_comet, seed=5)
    reg  = make_registration_tasks(n_reg, seed=4)
    # In a --cube run the DSL contains clear_at, which lets a non-mental
    # "transient wall" (stamp / act / erase) reproduce single-agent belief.  Use
    # witness-belief tasks there so the private-copy fork is the unique explanation.
    # `--plain-belief` overrides this to the shallower single-agent belief even in a
    # cube run: a DIAGNOSTIC to separate search budget from structure (plain belief
    # is ~8 nodes vs. witness ~12+).  It WEAKENS the uniqueness claim (transient-wall
    # rivals are no longer excluded), so it is for isolating the first-solve blocker,
    # not for headline results.
    use_witness = cube and not plain_belief
    if plain_belief and cube:
        print("  [--plain-belief] using single-agent belief in a cube run — DIAGNOSTIC "
              "only; transient-wall rivals are NOT excluded.")
    bel  = (make_witness_belief_tasks(n_bel, COMBOS, seed=2) if use_witness
            else make_belief_tasks(n_bel, COMBOS, seed=2))
    # Two further belief families (kind='belief'), so the unified verdict tests
    # whether ONE fork(policy, sync_to_world av) agent constructor generalizes
    # across belief about an obstacle, about an object's location (goal-displacement),
    # and across two contradictory beliefs at once (dual).  See tasks_minds.py.
    gdb  = make_goal_displacement_tasks(n_goal, COMBOS, seed=23)
    dual = make_dual_belief_tasks(n_belvar, COMBOS, seed=24)
    # False-obstacle belief (kind=belief): wrong about BOTH the obstacle and the goal,
    # with a REAL wall (value 3) left in the world.  Its construction forbids the
    # scope-complement degeneracy — no sync_all / sync_except k reproduces the scene, so
    # any solution MUST commit via the literal sync_to_world(av).  This is what lets the
    # (A) disclosure say the agency commit was FORCED and found, not merely argued
    # extensionally equivalent to it.  Needs cube primitives (clear_at, wall value 3).
    fob  = make_false_obstacle_belief_tasks(n_belvar, COMBOS, seed=25) if cube else []
    print(f"  belief variants: +{len(gdb)} goal-displacement, +{len(dual)} dual, "
          f"+{len(fob)} false-obstacle (all kind=belief; false-obstacle forbids the "
          f"scope-complement commit)")

    # ── curriculum scaffold (on by default; --no-curriculum to disable) ───────────
    # Witness-belief is deep — compose(fork(policy, sync), seek) — so its FIRST solve
    # is out of budget even once the policy is a cheap token (the fork∧sync block is
    # still searched from scratch).  Plain single-agent belief is the SAME inner block
    # without the outer witness seek, and it is shallow (the --plain-belief diagnostic
    # solves it at tiny t_fn).  We add it purely as a teacher: once it solves, the joint
    # stitch sees fork(policy, sync) recur across the (gv,av) combos and abstracts it
    # into one token — and then witness-belief = compose(<that token>, seek) is shallow
    # enough to reach, exactly as obstacle's policy lowered plain belief's first-solve.
    #
    # Tagged 'belief_scaffold' (a kind absent from the reporting lists at module top) so
    # it feeds the stitch but never counts toward the headline witness-belief verdict.
    # Sound because once the policy is a token, fork(policy, sync) is strictly cheaper
    # than the transient-wall rival (which additionally pays clear_at + two coords), so
    # the searcher returns the genuine compound — see the --plain-belief census, which
    # solves via fork/sync_to_world, not clear_at.  The headline claim still rests only
    # on the witness tasks, where the transient-wall rival is excluded by construction.
    scaffold = []
    if curriculum and use_witness:
        n_scaffold = max(1, n_bel // 2)
        scaffold = make_belief_tasks(n_scaffold, COMBOS, seed=22)
        for _, m in scaffold:
            m['kind'] = 'belief_scaffold'
        print(f"  [curriculum] +{len(scaffold)} plain-belief scaffold tasks "
              f"(kind=belief_scaffold) to seed the fork(policy, sync) abstraction; "
              f"excluded from the witness-belief verdict.")

    # One minds-free task per symmetric corner, so every complement the cube adds
    # is *useful somewhere* — otherwise "belief avoids the complements" is vacuous
    # (an unused distractor is trivially avoided).  These only make sense when the
    # corner primitives exist, i.e. in a --cube run.  fn-rooted corners join the
    # `unfold` search; pair-rooted ones join the `unfold_with_template` search.
    fn_corner, pair_corner = [], []
    if cube:
        # obstacle is the wall_at corner: a non-mental detour task whose solution is
        # belief's policy `(compose (wall_at) (optimize (neg_dist)))`.  A *family* of
        # them (per-combo, like belief) makes that derive recur so the joint stitch
        # abstracts it — which is what lowers belief's first-solve cost and leaves only
        # the fork∧sync agency wrapper unique to belief.
        # underlay is the z-order complement of overlay in a FORK context: a
        # world-wins motion trail crossing an occluding bystander, so the cube's
        # underlay corner is exercised by a *fork* task (like overlay), not only by
        # the template-rooted inpainting task — the fork-producer side of the pair
        # interface now populates both z-order corners.
        fn_corner = (make_flee_tasks(n_corner, seed=10)
                     + make_deletion_tasks(n_corner, seed=11)
                     + make_denoise_tasks(n_corner, seed=12)
                     + make_underlay_tasks(n_corner, seed=19)
                     + make_obstacle_tasks(n_obstacle, seed=18))
        pair_corner = (make_perception_tasks(n_corner, seed=13)
                       + make_multi_registration_tasks(n_corner, seed=14)
                       + make_registration_except_tasks(n_corner, seed=15)
                       + make_inpainting_tasks(n_corner, seed=16)
                       + make_readout_tasks(n_corner, seed=17))

    # fn-rooted families share the `unfold` interpreter; pair families are fn_p_g.
    fn_tasks = phys + des + ov + comet + bel + gdb + dual + fob + scaffold + fn_corner
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
    # per-round per-task fn timings (mat_key, round, solved?, seconds), so solve_dynamics.py
    # can chart the cumulative-solve S-curve and the per-task solve-time collapse (the
    # ~t_fn miss in round r ↦ seconds in round r+1) that is otherwise only in the log text.
    timing_log = []
    templates = {mat_key(x): m['template'] for x, m in reg_tasks}
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
    kind_by_key = {mat_key(x): m['kind'] for x, m in fn_tasks}

    for it in range(1, ecd_iters + 1):
        unsolved_fn  = [x for x, _ in fn_tasks if mat_key(x) not in sols]
        unsolved_reg = [x for x, _ in reg_tasks if mat_key(x) not in sols]
        n_before = len(sols)
        use_model = dream_on and qmodel is not None and it <= 1 + DREAM_USE_ROUNDS
        # Dreamed Q is applied ONLY to families with at least one solved instance — a
        # replay the recognition model was actually trained on.  A zero-replay family
        # (belief, until its first solve) is invisible to the model, which can then only
        # mis-price it: it floods the early budget windows with the non-belief primitives
        # it HAS seen (each pushed below uniform) and delays belief's own uniform-cost
        # primitives past the timeout — the uniform floor keeps them reachable in
        # principle, not within t_fn.  Those families stay on the proven uniform/content
        # baseline and earn the dreamed Q only once solved (they then contribute replays).
        replay_kinds = {kind_by_key[k] for k in sols
                        if sols.get(k) is not None and k in kind_by_key}
        print(f"\n--- round {it}/{ecd_iters}: {len(unsolved_fn)} fn + {len(unsolved_reg)} "
              f"fn_p_g unsolved; |D|={len(D)} ({len(D.invented)} invented); "
              f"fn Q={'dreamed (replay families only)' if use_model else 'uniform/content'} ---", flush=True)
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
                         dict(sols), t_fn, 0, fn) for x in unsolved_fn]
                # per-task timing: collected so t_fn can be calibrated from a real run —
                # the max SOLVE time bounds how low the uniform timeout can safely go.
                timings = {}
                for k, sol, elapsed in pool.map(_solve_one_task, args):
                    timings[k] = (sol is not None, elapsed)
                    timing_log.append((k, it, sol is not None, float(elapsed)))
                    if sol is not None:
                        sols[k] = sol
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
            solve_enumeration(unsolved_reg, D, uniform_type_q(D), sols,
                              timeout=t_reg, root_type=fn_p_g, templates=templates)

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
    save_run_artifact(D, all_tasks, sols, rewritten, decomposed, smoke)
    # Per-round trajectory artifact: base-prim sols keyed by mat_key_id, tagged with the
    # round each was first solved, so `python corpus_dl.py --run <this file>` can rebuild
    # the corpus description-length curve (DreamCoder-style) round by round.
    save_trajectory_artifact(D, all_tasks, sols, solve_round, decomposed, smoke,
                             stitch_iters, ecd_iters, timing_log=timing_log)

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
    # Wall-based uniqueness is a claim about the WALL belief families only (plain /
    # witness / dual); the goal-displacement family is deliberately NOT wall-based
    # (its derive is `step`, a displaced goal), so it is excluded from this check.
    belief_is_wall_based = all(
        'wall_at' in _core_uses(sols[mat_key(x)])
        for x, m in all_tasks
        if m['kind'] == 'belief' and 'displaced_to' not in m
        and sols.get(mat_key(x)) is not None
    )
    print(f"\n  fork used outside belief (overlay/comet)   : {fork_general}")
    print(f"  sync used outside belief (registration)    : {sync_general}")
    print(f"  wall_at used outside belief (obstacle)     : {wall_general}")
    print(f"  belief reuses BOTH fork and sync           : {belief_uses_both}")
    print(f"  fork∧sync agency is unique to belief       : {agency_unique}")
    print(f"  wall-belief solutions are wall-based       : {belief_is_wall_based}"
          f"   (no displaced-goal rival survived; goal-displacement family excluded)")
    if n_belief_degenerate:
        tot_bel = n_belief_literal + n_belief_degenerate
        print(f"\n  DISCLOSURE — agency commit on minimal scenes")
        print(f"  {n_belief_degenerate}/{tot_bel} belief solutions commit via a SCOPE complement "
              f"(sync_except gv / sync_all) rather")
        print(f"  than sync_to_world(av).  On a scene whose only committed model-mover is the")
        print(f"  agent, 'move everything but the goal' == 'move the agent' — the two commits are")
        print(f"  extensionally identical (verified for every belief family).  It is the SAME")
        print(f"  fork-structured belief with the agency hole expressed on the goal rather than")
        print(f"  the actor, NOT a non-mental rival; counted as the agency commit above.")

    # ── the degeneracy REMOVED: false-obstacle scenes force the literal commit ────────
    # On a false-obstacle scene a REAL wall (value 3) stays in the world, so — verified
    # per task at construction (_scope_complements_all_fail) — NO scope complement
    # (sync_all, sync_except k for every world value k, incl. the wall) reproduces x.
    # The commit therefore CANNOT be degenerate: any solution the search returns commits
    # via the literal sync_to_world(av).  This is the disclosure's answer, not an
    # argument: where the complement is genuinely available (minimal scenes) it is
    # extensionally the agency commit; where it is excluded (here) the literal commit is
    # the one that gets found.
    fob_forms = [belief_form[mat_key(x)] for x, m in all_tasks
                 if m['kind'] == 'belief' and 'real_wall' in m
                 and mat_key(x) in belief_form]
    n_fob = sum(1 for x, m in all_tasks if m['kind'] == 'belief' and 'real_wall' in m)
    if n_fob:
        n_fob_lit = sum(1 for f in fob_forms if f == 'literal')
        n_fob_deg = sum(1 for f in fob_forms if f == 'degenerate')
        print(f"\n  FORCED LITERAL COMMIT — false-obstacle family (degeneracy excluded by construction)")
        print(f"  {len(fob_forms)}/{n_fob} false-obstacle tasks solved; commit forms: "
              f"{n_fob_lit} literal, {n_fob_deg} degenerate.")
        print(f"  Every solved one commits via the literal sync_to_world(av): no scope complement")
        print(f"  reproduces these scenes (real wall stays put), so the agency commit is FORCED")
        print(f"  and found — not merely argued extensionally equal to a cheaper complement.")
        if n_fob_deg:
            print(f"  WARNING: {n_fob_deg} classified degenerate — should be impossible here; "
                  f"check _scope_complements_all_fail / _belief_commit_form.")

    # ── (A′) cube census: with the full symmetric field present, which corner did each family pick? ──
    cube_ok = None
    if cube:
        print("\n" + "=" * 72)
        print("(A′) CUBE CENSUS — which symmetric corner each family selected")
        print("=" * 72)
        corners_by_kind = {}
        for x, m in all_tasks:
            if m['kind'] == 'belief' and 'displaced_to' in m:
                # goal-displacement's derive is a displaced goal, not the wall
                # corner (see (A)'s belief_is_wall_based exclusion above).
                continue
            sol = sols.get(mat_key(x))
            if sol is None:
                continue
            cu = _corner_uses(sol)
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
        # belief must keep exactly the agency corner and avoid every complement
        complements = {'sync_to_model', 'sync_all', 'sync_except', 'underlay',
                       'snd_gg', 'via_swap', 'distance', 'clear_at', 'erase',
                       # bifunctor / pairing complements (decomposed runs): belief
                       # uses mapsnd + dup, so the wrong-channel / fresh-channel
                       # corners are the ones it must avoid.
                       'mapfst', 'swap', 'bimap', 'pair_blank'}
        # agency commit is sync_to_world atomically (phase 1) or its decomposition
        # register(locate)(place) (phase 2); either counts as keeping the corner.
        belief_keeps_corner = (('sync_to_world' in belief_corners
                                or 'register' in belief_corners)
                               and 'wall_at' in belief_corners)
        belief_avoids_complements = not (set(belief_corners) & complements)
        # each complement should be the corner its own family reaches for — a live,
        # fully-exercised field, not a handful of inert distractors.
        used_complements = set().union(*(
            set(corners_by_kind.get(k, Counter())) & complements
            for k in _ALL_KINDS if k != 'belief'
        )) if corners_by_kind else set()
        any_complement_used = bool(used_complements)
        # don't flag a complement as "unreached" if it has no dedicated minds-free
        # family (the bifunctor/pairing corners) or isn't even a primitive in the
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
        print(f"  complements claimed by a non-mental family               : {sorted(used_complements)}")
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
        print(f"    {d.repr}  [{argt}] -> {d.type}")
        print(f"      body: {body}")
        _has_sync = 'sync_to_world' in body or 'register' in body  # atomic | decomposed
        _has_scope = bool(_SCOPE_COMPLEMENTS & {p for p in _CORNERS
                                                if _re.search(rf'\b{p}\b', body)})
        if _has_fork(body) and _has_sync and 'wall_at' in body:
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
            print(f"      (non-mental: fork + overlay — motion blur)")
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

    # ── verdict ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    constructor_found = agent_constructor is not None
    constructor_shared = bool(agent_constructor and agent_constructor[2])
    # the constructor must be a BELIEF abstraction, and the non-mental families must
    # NOT have been swept into it (else it isn't really belief-specific).
    ctor_name = agent_constructor[0].repr if agent_constructor else None
    belief_uses_ctor = bool(ctor_name) and any(
        ctor_name in _absts_in(rewritten.get(mat_key(x), ''))
        for x, m in all_tasks if m['kind'] == 'belief'
    )
    nonmental_free_of_ctor = bool(ctor_name) and not any(
        ctor_name in _absts_in(rewritten.get(mat_key(x), ''))
        for x, m in all_tasks if m['kind'] not in ('belief', 'belief_scaffold')
    )

    print(f"  (A) parts are general    : fork∉belief-only={fork_general}, "
          f"sync∉belief-only={sync_general}, wall∉belief-only={wall_general}")
    print(f"  (A) only agency is unique: fork∧sync unique to belief={agency_unique}, "
          f"belief recombines={belief_uses_both}, wall-based={belief_is_wall_based}")
    print(f"  (B) constructor invented : {constructor_found} "
          f"(shared agency hole: {constructor_shared})")
    print(f"  (B) constructor is belief-specific: used by belief={belief_uses_ctor}, "
          f"absent from non-mental={nonmental_free_of_ctor}")
    if cube:
        print(f"  (A′) cube: belief picks the lone asymmetric corner over the full field: {cube_ok}")

    ok = (fork_general and sync_general and wall_general and belief_uses_both
          and agency_unique and belief_is_wall_based and constructor_found
          and constructor_shared and belief_uses_ctor and nonmental_free_of_ctor
          and (cube_ok is not False))
    if ok:
        print("\n  => In ONE library and ONE MDL compression over minds-free AND minds tasks,")
        print("     belief is the discovered recombination of parts that each do non-mental")
        print("     work, and the same objective that builds the agent constructor leaves")
        print("     fork/sync bare elsewhere.  Not gerrymandered, not a silo artefact.")
    else:
        print("\n  => not fully demonstrated this run (raise timeouts / n_bel / stitch_iters,")
        print("     or drop --smoke).  Each False above localises what failed.")


def save_run_artifact(D, all_tasks, sols, rewritten, decomposed, smoke,
                      path=None):
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
                             stitch_iters, ecd_iters, path=None, timing_log=None):
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
    "shared CLI parsing for the phase wrappers: --smoke --samples --ecd-iters N --t-fn N --no-dream --plain-belief --no-curriculum"
    def _opt(flag, cast):
        if flag in argv:
            return cast(argv[argv.index(flag) + 1])
        return None
    return dict(smoke='--smoke' in argv,
                samples='--samples' in argv,
                ecd_iters=_opt('--ecd-iters', int),
                t_fn=_opt('--t-fn', float),
                dream_on='--no-dream' not in argv,
                plain_belief='--plain-belief' in argv,
                curriculum='--no-curriculum' not in argv)


if __name__ == '__main__':
    run_phase(decomposed='--decomposed' in sys.argv, **cli_kwargs(sys.argv))
