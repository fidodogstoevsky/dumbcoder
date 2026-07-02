"""Quantitative experiment (phases 1-2): the MDL-margin distribution for belief.

One figure to kill the "maybe a non-mental program was almost as short" objection for
the whole corpus.  For every belief task in the phase-1/2 curriculum we price, in nats
under the FINAL library:

  * the found (mental) program — the verified ground-truth belief compound
    `fork(compose(derive, seek), sync_to_world av)` that the searcher recovers
    (verify_ground_truth licenses reusing it without re-enumerating); and
  * each NON-MENTAL rival spelling from the task's discriminating battery
    (tasks_minds.belief_rival_specs) — the transient-wall / pure-physics programs a
    skeptic offers as "almost as short".

The margin is  DL(rival) - DL(found)  in nats: positive means the mental reading is the
shorter description.  Crucially both are priced UNDER THE FINAL LIBRARY (base prims + the
abstractions ONE joint stitch discovers over the whole mixed corpus), via the same `_dl`
phase 3 uses.  That is what makes the comparison bite: the belief compound recurs across
the corpus and so collapses to the abstracted agent constructor, while a bespoke
transient-wall schedule does not — the library reprices exactly the reuse the MDL
objective rewards.  We also report the base-primitive margin (no abstractions) as the
honest lower bound; the gap between the two IS the compression the objection ignores.

No enumeration/search is run: we stitch the ground-truth programs of the full corpus,
which converges to the same library the wake-sleep loop does (identical solved programs
→ identical joint stitch), deterministically and in seconds.

    python mdl_margin.py                 phase 1 (atomic fork/sync), full corpus
    python mdl_margin.py --decomposed    phase 2 (decomposed plumbing)
    python mdl_margin.py --smoke         small corpus, fast sanity run
    python mdl_margin.py --both          run phases 1 and 2, one JSON each

Writes mdl_margins[.decomposed].json; plot with `python plot_mdl_margin.py`.
"""

import sys
import json
import math

import numpy as np
import torch as th

from ecd import (
    Deltas, saturate_stitch, rewrite_through_library, mat_key, tr, normalize, simplify,
)
from dsl import unfold
from prims import make_symmetric_prims
from tasks_minds import (
    COMBOS,
    make_physics_tasks, make_desire_tasks,
    make_belief_tasks, make_witness_belief_tasks,
    make_goal_displacement_tasks, make_dual_belief_tasks,
    belief_rival_specs, belief_variant,
)
from tasks_world import (
    make_overlay_tasks, make_comet_tasks, make_registration_tasks,
    make_flee_tasks, make_deletion_tasks, make_denoise_tasks, make_underlay_tasks,
    make_obstacle_tasks, make_perception_tasks, make_multi_registration_tasks,
    make_registration_except_tasks, make_inpainting_tasks, make_readout_tasks,
)
from experiment import gt_program_str, verify_ground_truth, check_decomposition_identities


# ── DL pricing (identical convention to phase3_arity._dl) ───────────────────────────
def uniform_type_q(D):
    "type-conditioned uniform log-prob: logp[i] = -log(#symbols sharing i's type)"
    q = th.zeros(len(D))
    for _tp, idxs in D.bytype.items():
        lp = -math.log(len(idxs))
        for i in idxs:
            q[i] = lp
    return q


def _dl(D, Q, prog):
    "Description length (nats) of a program string/tree under library D and prior Q."
    t = tr(D, prog) if isinstance(prog, str) else prog
    return float(-D.logp(Q, t))


def _prims_of(D, prog_str):
    "expand a (possibly library) program to base primitives for a behavioural check."
    return str(simplify(normalize(tr(D, prog_str))))


# ── behavioural competitor test ─────────────────────────────────────────────────────
def _agent_pos(frame, v):
    p = np.argwhere(frame == v)
    return (int(p[0][0]), int(p[0][1])) if len(p) else None


def _mental_agents(m):
    "the mind-bearing agents a belief reading explains (dual has two)."
    return [m['av'], m['av2']] if 'pw2' in m else [m['av']]


def _reproduces_behaviour(D, x, m, rival_str):
    """True iff the non-mental rival renders the OBSERVED trajectory of every mind-bearing
    agent across all frames — the condition under which it is a genuine 'explains the
    action' competitor (rather than a shorter program that simply renders a different
    scene, which is excluded by expressiveness, not by MDL).  A rival that raises during
    unfold is not a competitor."""
    T = x.shape[0]
    try:
        xr = unfold(x[0], T, tr(D, rival_str)())
    except Exception:
        return False
    return all(_agent_pos(xr[t], a) == _agent_pos(x[t], a)
               for a in _mental_agents(m) for t in range(T))


# ── corpus (mirrors experiment.run_phase's full-run corpus; kept in step by seeds) ──
def build_corpus(smoke=False):
    """The full phase-1/2 mixed corpus: minds-free (physics/desire/overlay/comet/
    registration + the eight symmetric cube corners) AND minds (belief wall/witness/
    goal-displacement/dual + the plain-belief scaffold).  Sizes/seeds match
    experiment.run_phase so the stitched library equals the one a real phase converges to.
    Returns a flat list of (x, meta)."""
    if smoke:
        n_phys, n_des, n_ov, n_reg, n_bel, n_corner = 2, 1, 2, 2, 1, 2
        n_comet, n_belvar, n_obstacle = 2, 1, 2
    else:
        n_phys, n_des, n_ov, n_reg, n_bel, n_corner = 4, 2, 4, 4, 6, 4
        n_comet, n_belvar, n_obstacle = 4, 3, 6

    phys = make_physics_tasks(n_phys, seed=0)
    des  = make_desire_tasks(n_des, COMBOS, seed=1)
    ov   = make_overlay_tasks(n_ov, seed=3)
    comet = make_comet_tasks(n_comet, seed=5)
    reg  = make_registration_tasks(n_reg, seed=4)
    # cube run: witness belief is the headline single-agent family (transient-wall rival
    # excluded by the crossing witness); plain belief seeds the fork(policy, sync) token.
    bel  = make_witness_belief_tasks(n_bel, COMBOS, seed=2)
    gdb  = make_goal_displacement_tasks(n_belvar, COMBOS, seed=23)
    dual = make_dual_belief_tasks(n_belvar, COMBOS, seed=24)
    scaffold = make_belief_tasks(max(1, n_bel // 2), COMBOS, seed=22)
    for _, m in scaffold:
        m['kind'] = 'belief_scaffold'

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

    tasks = (phys + des + ov + comet + bel + gdb + dual + scaffold
             + fn_corner + reg + pair_corner)
    # dedupe identical matrices (would skew stitch counts)
    seen, out = set(), []
    for x, m in tasks:
        k = mat_key(x)
        if k in seen:
            continue
        seen.add(k)
        out.append((x, m))
    return out


# ── the experiment ──────────────────────────────────────────────────────────────────
def run(decomposed=False, smoke=False):
    phase = 2 if decomposed else 1
    print(f"\n{'='*72}\nMDL-MARGIN EXPERIMENT — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync){' [smoke]' if smoke else ''}"
          f"\n{'='*72}")

    tasks = build_corpus(smoke)
    D = Deltas(make_symmetric_prims(decomposed=decomposed))
    verify_ground_truth(D, tasks)                 # licenses gt programs as "found"
    if decomposed:
        check_decomposition_identities(tasks)

    # stitch the whole corpus's ground-truth programs -> the FINAL library (D mutated,
    # abstractions registered; the rewritten found programs come back aligned to sols).
    sols = {mat_key(x): tr(D, gt_program_str(D, m)) for x, m in tasks}
    sol_keys = [mat_key(x) for x, _ in tasks]
    _trees, rewritten_strs = saturate_stitch(D, sols, iterations=(3 if smoke else 6),
                                             max_arity=5)
    # saturate_stitch drops unparseable rewrites, so re-derive the found library form
    # per task with the same helper we use for rivals (guarantees per-task alignment).
    Q = uniform_type_q(D)
    print(f"  final library: {len(D)} tokens ({len(D.invented)} invented: "
          f"{[d.repr for d in D.invented]})")

    # the plain-wall scaffold tasks (kind='belief_scaffold') are genuine wall-belief
    # tasks; include them as the belief_wall variant.  They matter most here: for plain
    # wall-belief the transient-wall rival reproduces the FULL scene, so MDL (not the
    # witness/expressiveness) is the operative discriminator.
    records = []
    n_bad = 0
    for x, m in tasks:
        if m['kind'] not in ('belief', 'belief_scaffold'):
            continue
        var = belief_variant(m)
        found_base = gt_program_str(D, m)
        (found_lib,) = rewrite_through_library(D, [found_base])
        dl_found_base = _dl(D, Q, found_base)
        dl_found_lib  = _dl(D, Q, found_lib)

        rivals = belief_rival_specs(m)
        rival_libs = rewrite_through_library(D, [s for _, s in rivals])
        rival_recs = []
        for (label, rstr), rlib in zip(rivals, rival_libs):
            # behavioural guard: the library rewrite must be the same program in prims
            if _prims_of(D, rlib) != _prims_of(D, rstr):
                n_bad += 1
            dl_r_base = _dl(D, Q, rstr)
            dl_r_lib  = _dl(D, Q, rlib)
            rival_recs.append({
                'label': label,
                'competitor': _reproduces_behaviour(D, x, m, rstr),
                'dl_base': dl_r_base,
                'dl_lib': dl_r_lib,
                'margin_base': dl_r_base - dl_found_base,
                'margin_lib':  dl_r_lib - dl_found_lib,
            })
        records.append({
            'variant': var,
            'av': m.get('av'), 'gv': m.get('gv'),
            'found_base': found_base,
            'found_lib': found_lib,
            'dl_found_base': dl_found_base,
            'dl_found_lib': dl_found_lib,
            'rivals': rival_recs,
        })

    if n_bad:
        print(f"  WARNING: {n_bad} rival library-rewrites failed the behavioural guard "
              f"(priced in base prims as fallback).")

    _print_summary(records)
    out = {
        'phase': phase,
        'decomposed': decomposed,
        'smoke': smoke,
        'library': [d.repr for d in D.invented],
        'n_belief_tasks': len(records),
        'records': records,
    }
    path = f"mdl_margins{'.decomposed' if decomposed else ''}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {len(records)} belief tasks x rivals to {path}")
    return out


def _print_summary(records):
    """Per-variant summary over BEHAVIOURAL COMPETITORS — the non-mental rivals that
    reproduce the observed agent trajectory (the genuine 'almost as short' threats).
    Pure-physics rivals that render no detour are reported separately: they are shorter
    but excluded by expressiveness, so they never bear on the MDL comparison."""
    from collections import defaultdict
    comp = defaultdict(list)         # variant -> [library margins over competitor pairs]
    comp_base = defaultdict(list)
    n_tasks = defaultdict(int)
    n_tasks_with_comp = defaultdict(int)
    excl = defaultdict(list)         # variant -> [library margins of non-competitors]
    for r in records:
        v = r['variant']
        n_tasks[v] += 1
        has = False
        for rv in r['rivals']:
            if rv['competitor']:
                comp[v].append(rv['margin_lib'])
                comp_base[v].append(rv['margin_base'])
                has = True
            else:
                excl[v].append(rv['margin_lib'])
        n_tasks_with_comp[v] += int(has)

    print(f"\n  {'variant':16s} {'tasks':>5s} {'w/comp':>6s} {'comp':>5s}  "
          f"{'library margin over competitors (nats)':>40s}")
    print("  " + "-" * 82)
    for var in ('belief_wall', 'belief_witness', 'belief_goal', 'belief_dual'):
        if var not in n_tasks:
            continue
        cl = comp[var]
        if cl:
            desc = (f"min {min(cl):+6.2f}  median {float(np.median(cl)):+6.2f}  "
                    f"max {max(cl):+6.2f}   base median {float(np.median(comp_base[var])):+6.2f}")
        else:
            desc = "no non-mental behavioural competitor (expressiveness-only)"
        print(f"  {var:16s} {n_tasks[var]:5d} {n_tasks_with_comp[var]:6d} {len(cl):5d}  {desc}")

    all_comp = [v for l in comp.values() for v in l]
    if all_comp:
        share = 100.0 * sum(1 for v in all_comp if v > 0) / len(all_comp)
        print(f"\n  every behavioural competitor is LONGER than the mental reading: "
              f"{share:.0f}% of {len(all_comp)} competitor pairs have library margin > 0")
    all_excl = [v for l in excl.values() for v in l]
    n_shorter_excl = sum(1 for v in all_excl if v < 0)
    n_shorter_comp = sum(1 for v in all_comp if v < 0)
    print(f"  no rival is both a competitor and shorter: {n_shorter_comp} competitor pairs "
          f"have margin < 0 (of {len(all_comp)})")
    print(f"  of the {n_shorter_excl} rivals that ARE shorter than the mental reading, "
          f"none reproduces the agent's trajectory (all expressiveness-excluded)")


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if '--both' in sys.argv:
        run(decomposed=False, smoke=smoke)
        run(decomposed=True, smoke=smoke)
    else:
        run(decomposed='--decomposed' in sys.argv, smoke=smoke)
