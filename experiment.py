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
    Deltas, solve_enumeration, saturate_stitch, mat_key, normalize,
    _solve_one_task, _worker_init, _n_cpus_available,
    dream, dreamed_q,
)
from dsl import (
    fn, fn_p_g, cellvalue,
    unfold, unfold_with_template, tr, simplify, length,
    # used by check_decomposition_identities (phase 2)
    compose, wall_at, optimize, neg_distance,
    fork, sync_to_world, fork_decomposed, register, locate, place,
)

# Reuse the generators and DSL already vetted in the curriculum, so the experiment
# driver adds only the *unification*, never a new encoding to second-guess.
from tasks_minds import (
    make_physics_tasks, make_desire_tasks, make_belief_tasks,
    make_witness_belief_tasks,
    COMBOS, SIZE, DIRS,
)
from tasks_world import (
    make_overlay_tasks, make_registration_tasks,
    # one minds-free task per symmetric corner
    make_flee_tasks, make_deletion_tasks, make_denoise_tasks, make_obstacle_tasks,
    make_perception_tasks, make_multi_registration_tasks,
    make_registration_except_tasks, make_inpainting_tasks, make_readout_tasks,
)
from prims import make_core_prims, make_symmetric_prims

# corpus families by interpreter / root type.  The first block is fixed; the cube
# corner families (second block) are appended only in --cube runs, where their
# primitives exist.  Reporting loops iterate these so new families show up
# everywhere without per-call edits.
_FN_KINDS   = ['physics', 'desire', 'overlay', 'belief',
               'flee', 'deletion', 'denoise', 'obstacle']
_PAIR_KINDS = ['registration', 'perception', 'multi_reg', 'reg_except',
               'inpaint', 'readout']
_CUBE_KINDS = ['flee', 'deletion', 'denoise', 'obstacle', 'perception', 'multi_reg',
               'reg_except', 'inpaint', 'readout']
_ALL_KINDS  = ['physics', 'desire', 'overlay', 'registration', 'belief',
               'flee', 'deletion', 'denoise', 'obstacle', 'perception', 'multi_reg',
               'reg_except', 'inpaint', 'readout']

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
    for d in D.ds:
        # only cellvalue literals are content-priced; coord (wall position) is a
        # latent absent from the grid, so it must keep its uniform type cost.
        if d.tailtypes is None and d.type == cellvalue and d.head in visible:
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


def verify_ground_truth(D, tasks):
    for x, m in tasks:
        k = m['kind']
        if k == 'physics':
            tree = tr(D, f"(step {m['val']} {m['dir']})")
            out  = unfold(x[0], x.shape[0], tree())
        elif k == 'desire':
            tree = tr(D, f"(optimize (neg_dist {m['gv']}) {m['av']})")
            out  = unfold(x[0], x.shape[0], tree())
        elif k == 'overlay':
            tree = tr(D, _forks(D, f"(step {m['val']} {m['dir']})", "overlay"))
            out  = unfold(x[0], x.shape[0], tree())
        elif k == 'registration':
            tree = tr(D, _sync(D, m['val']))
            out  = unfold_with_template(x[0], m['template'], x.shape[0], tree())
        # ── symmetric-corner families (cube runs only) ───────────────────────────
        elif k == 'flee':
            tree = tr(D, f"(optimize (distance {m['hv']}) {m['av']})")
            out  = unfold(x[0], x.shape[0], tree())
        elif k == 'deletion':
            r, c = m['cell']
            tree = tr(D, f"(clear_at c{r} c{c})")
            out  = unfold(x[0], x.shape[0], tree())
        elif k == 'denoise':
            tree = tr(D, f"(erase {m['noise']})")
            out  = unfold(x[0], x.shape[0], tree())
        elif k == 'obstacle':
            pr, pc = m['pw']
            tree = tr(D, f"(compose (wall_at c{pr} c{pc}) "
                        f"(optimize (neg_dist {m['gv']}) {m['av']}))")
            out  = unfold(x[0], x.shape[0], tree())
        elif k == 'perception':
            tree = tr(D, _sync_model(D, m['val']))
            out  = unfold_with_template(x[0], m['template'], x.shape[0], tree())
        elif k == 'multi_reg':
            tree = tr(D, "sync_all")
            out  = unfold_with_template(x[0], m['template'], x.shape[0], tree())
        elif k == 'reg_except':
            tree = tr(D, f"(sync_except {m['anchor']})")
            out  = unfold_with_template(x[0], m['template'], x.shape[0], tree())
        elif k == 'inpaint':
            tree = tr(D, "underlay")
            out  = unfold_with_template(x[0], m['template'], x.shape[0], tree())
        elif k == 'readout':
            tree = tr(D, "snd_gg")
            out  = unfold_with_template(x[0], m['template'], x.shape[0], tree())
        else:  # belief
            pr, pc = m['pw']
            derive = (f"(compose (wall_at c{pr} c{pc}) "
                      f"(optimize (neg_dist {m['gv']}) {m['av']}))")
            belief = _forks(D, derive, _sync(D, m['av']))
            if 'aw' in m:   # witness-belief: compose the witness's direct seek after the belief move
                belief = (f"(compose {belief} "
                          f"(optimize (neg_dist {m['gw']}) {m['aw']}))")
            tree = tr(D, belief)
            out  = unfold(x[0], x.shape[0], tree())
        assert np.array_equal(out, x), f"ground truth failed for {k}: {m}"
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


def _select_samples(tasks, max_frames=6):
    """One example per kind (first seen, in _ALL_KINDS order) as labelled panels.

    Each sample is {kind, tag, T, panels:[(label, grid), …]}.  fn families show
    successive `unfold` frames t0…; fn_p_g families show world | template | result,
    surfacing the otherwise-invisible constant template channel.  The rendered
    grids are exactly what the searcher sees — belief's phantom wall lives only in
    the private model, so it never appears here."""
    seen = {}
    for x, m in tasks:
        seen.setdefault(m['kind'], (x, m))
    out = []
    for kind in _ALL_KINDS:
        if kind not in seen:
            continue
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


def check_decomposition_identities(tasks):
    "phase 2 self-check: the decomposed combinators are the same machine as fork/sync."
    n = 0
    for x, m in tasks:
        if m['kind'] != 'belief':
            continue
        av, gv, (pr, pc) = m['av'], m['gv'], m['pw']
        derive = compose(wall_at(pr, pc), optimize(neg_distance(gv), av))
        prog_orig = fork(derive, sync_to_world(av))
        prog_deco = fork_decomposed(derive, register(locate(av), place(av)))
        if 'aw' in m:   # witness-belief: the witness's direct seek follows the belief move
            seek = optimize(neg_distance(m['gw']), m['aw'])
            prog_orig = compose(prog_orig, seek)
            prog_deco = compose(prog_deco, seek)
        orig = unfold(x[0], x.shape[0], prog_orig)
        deco = unfold(x[0], x.shape[0], prog_deco)
        assert np.array_equal(orig, x), f"orig fork != task for {m}"
        assert np.array_equal(deco, x), f"decomposed != task for {m}"
        n += 1
    print(f"decomposition identity: fork ≡ commit∘mapsnd(derive)∘dup and "
          f"sync ≡ register(locate, place) verified on {n} belief tasks")


def run_phase(decomposed=False, smoke=False, samples=False, ecd_iters=None, t_fn=None,
              dream_on=True):
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
        _t_fn, t_reg, stitch_iters, _ecd_iters = 15, 8, 3, 2
    else:
        n_phys, n_des, n_ov, n_reg, n_bel, n_corner = 4, 2, 4, 4, 6, 4
        _t_fn, t_reg, stitch_iters, _ecd_iters = 180, 30, 6, 4
    t_fn = _t_fn if t_fn is None else t_fn
    ecd_iters = _ecd_iters if ecd_iters is None else ecd_iters
    dream_iters = 120 if smoke else 600   # recognition-model training steps per round

    print("Generating mixed corpus…")
    phys = make_physics_tasks(n_phys, seed=0)
    des  = make_desire_tasks(n_des, COMBOS, seed=1)
    ov   = make_overlay_tasks(n_ov, seed=3)
    reg  = make_registration_tasks(n_reg, seed=4)
    # In a --cube run the DSL contains clear_at, which lets a non-mental
    # "transient wall" (stamp / act / erase) reproduce single-agent belief.  Use
    # witness-belief tasks there so the private-copy fork is the unique explanation.
    bel  = (make_witness_belief_tasks(n_bel, COMBOS, seed=2) if cube
            else make_belief_tasks(n_bel, COMBOS, seed=2))

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
        fn_corner = (make_flee_tasks(n_corner, seed=10)
                     + make_deletion_tasks(n_corner, seed=11)
                     + make_denoise_tasks(n_corner, seed=12)
                     + make_obstacle_tasks(max(1, n_corner // 2), seed=18))
        pair_corner = (make_perception_tasks(n_corner, seed=13)
                       + make_multi_registration_tasks(n_corner, seed=14)
                       + make_registration_except_tasks(n_corner, seed=15)
                       + make_inpainting_tasks(n_corner, seed=16)
                       + make_readout_tasks(n_corner, seed=17))

    # fn-rooted families share the `unfold` interpreter; pair families are fn_p_g.
    fn_tasks = phys + des + ov + bel + fn_corner
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
          f"{by_kind['overlay']} overlay, {by_kind['registration']} registration, "
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

    for it in range(1, ecd_iters + 1):
        unsolved_fn  = [x for x, _ in fn_tasks  if mat_key(x) not in sols]
        unsolved_reg = [x for x, _ in reg_tasks if mat_key(x) not in sols]
        n_before = len(sols)
        use_model = dream_on and qmodel is not None and it <= 1 + DREAM_USE_ROUNDS
        print(f"\n--- round {it}/{ecd_iters}: {len(unsolved_fn)} fn + {len(unsolved_reg)} "
              f"fn_p_g unsolved; |D|={len(D)} ({len(D.invented)} invented); "
              f"fn Q={'dreamed' if use_model else 'uniform/content'} ---", flush=True)

        if unsolved_fn:
            with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as pool:
                args = [(x, D, (dreamed_q(qmodel, D, x) if use_model else content_q(D, x)),
                         dict(sols), t_fn, 0, fn) for x in unsolved_fn]
                for k, sol in pool.map(_solve_one_task, args):
                    if sol is not None:
                        sols[k] = sol
        if unsolved_reg:
            solve_enumeration(unsolved_reg, D, uniform_type_q(D), sols,
                              timeout=t_reg, root_type=fn_p_g, templates=templates)

        n_solved = len(sols)
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
            print("    no new tasks solved this round — wake-sleep stalled "
                  "(raise t_fn / ecd_iters, or the abstraction belief needs hasn't formed).")
            break

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
                        pass
            if not replays:   # fall back to the raw (pre-stitch) fn solutions
                replays = [sols[k] for k in fn_keys if sols.get(k) is not None]
            print(f"    dreaming: training recognition Q on {len(replays)} replays "
                  f"+ fantasies ({dream_iters} steps)…", flush=True)
            qmodel = dream(D, replays, training_Xs=fn_Xs, root_type=fn, n_iters=dream_iters)

    # ── (A) usage census: stitch-independent evidence about the bare parts ───────────
    print("\n" + "=" * 72)
    print("(A) USAGE CENSUS — which core parts each family reaches for (pre-stitch)")
    print("=" * 72)
    uses_by_kind = {}
    solved_by_kind = Counter()
    total_by_kind = Counter()
    for x, m in all_tasks:
        total_by_kind[m['kind']] += 1
        sol = sols.get(mat_key(x))
        if sol is None:
            continue
        solved_by_kind[m['kind']] += 1
        uses_by_kind.setdefault(m['kind'], set()).update(_core_uses(sol))
    for kind in _ALL_KINDS:
        if total_by_kind[kind] == 0:
            continue
        n, tot = solved_by_kind[kind], total_by_kind[kind]
        u = sorted(uses_by_kind.get(kind, set()))
        print(f"  {kind:13s} {n}/{tot} solved   uses: {u}")

    fork_general = _uses_fork(uses_by_kind.get('overlay', set()))
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
    belief_is_wall_based = all(
        'wall_at' in _core_uses(sols[mat_key(x)])
        for x, m in all_tasks
        if m['kind'] == 'belief' and sols.get(mat_key(x)) is not None
    )
    print(f"\n  fork used outside belief (overlay)         : {fork_general}")
    print(f"  sync used outside belief (registration)    : {sync_general}")
    print(f"  wall_at used outside belief (obstacle)     : {wall_general}")
    print(f"  belief reuses BOTH fork and sync           : {belief_uses_both}")
    print(f"  fork∧sync agency is unique to belief       : {agency_unique}")
    print(f"  every belief solution is wall-based        : {belief_is_wall_based}"
          f"   (no displaced-goal rival survived)")

    # ── (A′) cube census: with the full symmetric field present, which corner did each family pick? ──
    cube_ok = None
    if cube:
        print("\n" + "=" * 72)
        print("(A′) CUBE CENSUS — which symmetric corner each family selected")
        print("=" * 72)
        corners_by_kind = {}
        for x, m in all_tasks:
            sol = sols.get(mat_key(x))
            if sol is None:
                continue
            corners_by_kind.setdefault(m['kind'], Counter()).update(_corner_uses(sol))
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
    agent_constructor = None
    for d in D.invented:
        body = str(simplify(normalize(deepcopy(d))))
        shared = _shared_holes(body)
        argt = ', '.join(str(t) for t in (d.tailtypes or []))
        print(f"    {d.repr}  [{argt}] -> {d.type}")
        print(f"      body: {body}")
        _has_sync = 'sync_to_world' in body or 'register' in body  # atomic | decomposed
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
        elif _has_fork(body) and 'overlay' in body:
            print(f"      (non-mental: fork + overlay — motion blur)")
        elif 'wall_at' in body and ('optimize' in body or 'neg_dist' in body):
            print(f"      (obstacle/belief policy: stamp wall ▸ navigate — the shared derive)")
        elif 'optimize' in body or 'neg_dist' in body:
            print(f"      (desire fragment)")

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
        for x, m in all_tasks if m['kind'] != 'belief'
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


def cli_kwargs(argv):
    "shared CLI parsing for the phase wrappers: --smoke --samples --ecd-iters N --t-fn N --no-dream"
    def _opt(flag, cast):
        if flag in argv:
            return cast(argv[argv.index(flag) + 1])
        return None
    return dict(smoke='--smoke' in argv,
                samples='--samples' in argv,
                ecd_iters=_opt('--ecd-iters', int),
                t_fn=_opt('--t-fn', float),
                dream_on='--no-dream' not in argv)


if __name__ == '__main__':
    run_phase(decomposed='--decomposed' in sys.argv, **cli_kwargs(sys.argv))
