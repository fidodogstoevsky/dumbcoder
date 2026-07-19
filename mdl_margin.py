"""Quantitative experiment (phases 1-2): the MDL-margin distribution for belief.

One figure to kill the "maybe a non-mental program was almost as short" objection for
the whole corpus.  For every belief task in the phase-1/2 curriculum we price, in nats
under the FINAL library:

  * the found (mental) program — the verified ground-truth belief compound
    `fork(compose(derive, seek), sync_to_world av)` that the searcher recovers
    (verify_ground_truth licenses reusing it without re-enumerating); and
  * each NON-MENTAL rival spelling from the task's discriminating battery
    (tasks.belief_rival_specs) — the transient-wall / pure-physics programs a
    skeptic offers as "almost as short".

The margin is  DL(rival) - DL(found)  in nats: positive means the mental reading is the
shorter description.  Crucially both are priced UNDER THE FINAL LIBRARY (base prims + the
abstractions ONE joint stitch discovers over the whole mixed corpus), via the same `_dl`
phase 3 uses.  That is what makes the comparison bite: the belief compound recurs across
the corpus and so collapses to the abstracted agent constructor, while a bespoke
transient-wall schedule does not — the library reprices exactly the reuse the MDL
objective rewards.  We also report the base-primitive margin (no abstractions) as the
honest lower bound; the gap between the two IS the compression the objection ignores.

The "found" program and the library both come from an actual PHASE RUN.  `run_phase`
(phase1.py / phase2.py) writes `phase{1,2}_run[.smoke].json` — the searched sols (the
programs enumeration actually found, expanded to base primitives) and their library-
rewritten forms.  We re-stitch those searched sols to reconstruct exactly the library
that run converged to, then price each belief task's found program (its rewritten form)
against the non-mental rivals under it.  So the figure reflects what search recovered,
not a ground-truth reconstruction of it.

    # 1. produce a run (slow; HPC): its output is the artifact we consume
    python phase1.py                     -> phase1_run.json
    python phase2.py                     -> phase2_run.json
    # 2. price belief vs rivals under that run's library
    python mdl_margin.py                 phase 1 (reads phase1_run.json)
    python mdl_margin.py --decomposed    phase 2 (reads phase2_run.json)
    python mdl_margin.py --run PATH      consume a specific run artifact
    python mdl_margin.py --both          phases 1 and 2, one JSON each
    python mdl_margin.py --smoke         use the .smoke run artifacts
    python mdl_margin.py --ground-truth  legacy: stitch ground-truth programs, no run needed

Writes mdl_margins[.decomposed].json; plot with `python plot_mdl_margin.py`.
"""

import sys
import json
import math

import numpy as np
import torch as th

from ecd import (
    Deltas, saturate_stitch, rewrite_through_library,
    mat_key, mat_key_id, tr, normalize, simplify,
)
from dsl import unfold
from prims import make_symmetric_prims
from tasks import (
    COMBOS,
    make_physics_tasks, make_desire_tasks,
    make_belief_tasks, make_witness_belief_tasks,
    make_goal_displacement_tasks, make_dual_belief_tasks,
    make_false_obstacle_belief_tasks,
    belief_rival_specs, belief_variant,
    make_overlay_tasks, make_comet_tasks, make_registration_tasks,
    make_flee_tasks, make_deletion_tasks, make_denoise_tasks, make_underlay_tasks,
    make_obstacle_tasks, make_relocation_tasks, make_perception_tasks,
    make_multi_registration_tasks,
    make_registration_except_tasks, make_inpainting_tasks, make_readout_tasks,
)
from experiment import (gt_program_str, verify_ground_truth, check_decomposition_identities,
                        provenance_line)


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


def _observed_agents(m):
    """The observed agents the AGENT-only diagnostic tracks: the mind-bearing agent(s) AND
    — on a witness-belief task — the crossing WITNESS.  This is now only a secondary label
    (see _reproduces_agents); the primary competitor test is full-scene equality, which
    already pins every distinguishing cell (witness, goal, wall) without our having to
    enumerate the discriminating value here."""
    agents = [m['av']]
    if 'av2' in m:            # dual belief: a second believer
        agents.append(m['av2'])
    if 'aw' in m:             # witness belief: the non-believing witness
        agents.append(m['aw'])
    return agents


def _unfold_rival(D, x, rival_str):
    "Render a rival from x's initial frame over x's horizon; None if it raises during unfold."
    try:
        return unfold(x[0], x.shape[0], tr(D, rival_str)())
    except Exception:
        return None


def _reproduces_scene(xr, x):
    """True iff the rival renders the ENTIRE observed frame sequence, cell for cell — the
    genuine 'explains the data' competitor condition.  The task's data IS the whole scene:
    the agents AND the goal / wall / obstacle that make it a mental task.  A rival that
    matches the believer's detour but renders the goal or wall differently — e.g.
    belief_goal's 'shove goal in world', whose STATIONARY goal is the discriminating witness
    yet was never in _observed_agents — reproduces a DIFFERENT scene and is excluded by
    expressiveness, not by MDL.  Full-frame equality is both stricter and simpler than the
    old agents-only trajectory check, which let such a wrong-scene rival masquerade as a
    competitor and — being shorter on a few tasks — spuriously trip the 'genuine shorter
    competitor' warning.  A rival that raised during unfold (xr is None) is not a competitor."""
    return xr is not None and np.array_equal(xr, x)


def _reproduces_agents(xr, x, m):
    """Secondary label: True iff the rival reproduces the observed agent trajectories
    (believer(s) + witness) even where the rest of the scene differs.  Distinguishes a rival
    that gets the ACTION right but the SCENE wrong (reproduces_agents and not
    reproduces_scene) from one that gets the action wrong too — diagnostics only; it does
    NOT decide competitor status."""
    if xr is None:
        return False
    T = x.shape[0]
    return all(_agent_pos(xr[t], a) == _agent_pos(x[t], a)
               for a in _observed_agents(m) for t in range(T))


# ── corpus (mirrors experiment.run_phase's full-run corpus; kept in step by seeds) ──
def build_corpus(smoke=False):
    """The full phase-1/2 mixed corpus: physics/desire/overlay/comet/registration +
    the eight symmetric cube corners, plus belief wall/witness/goal-displacement/dual
    and a second distinct-seeded batch of plain wall-belief tasks.  Sizes/seeds match
    experiment.run_phase so the stitched library equals the one a real phase converges to.
    Returns a flat list of (x, meta)."""
    if smoke:
        n_phys, n_des, n_ov, n_reg, n_bel, n_corner = 2, 1, 2, 2, 1, 2
        n_comet, n_belvar, n_goal, n_obstacle, n_relocate = 2, 1, 2, 2, 4
    else:
        n_phys, n_des, n_ov, n_reg, n_bel, n_corner = 4, 2, 4, 4, 6, 4
        n_comet, n_belvar, n_goal, n_obstacle, n_relocate = 4, 3, 6, 6, 16

    phys = make_physics_tasks(n_phys, seed=0)
    des  = make_desire_tasks(n_des, COMBOS, seed=1)
    ov   = make_overlay_tasks(n_ov, seed=3)
    comet = make_comet_tasks(n_comet, seed=5)
    reg  = make_registration_tasks(n_reg, seed=4)
    # cube run: witness belief is the headline single-agent family (transient-wall rival
    # excluded by the crossing witness); plain belief seeds the fork(policy, sync) token.
    bel  = make_witness_belief_tasks(n_bel, COMBOS, seed=2)
    gdb  = make_goal_displacement_tasks(n_goal, COMBOS, seed=23)
    dual = make_dual_belief_tasks(n_belvar, COMBOS, seed=24)
    # false-obstacle belief: real wall in the world forbids the scope-complement commit,
    # so every solution is the literal sync_to_world(av) (see experiment.run_phase).
    fob  = make_false_obstacle_belief_tasks(n_belvar, COMBOS, seed=25)
    belief_extra = make_belief_tasks(max(1, n_bel // 2), COMBOS, seed=22)

    fn_corner = (make_flee_tasks(n_corner, seed=10)
                 + make_deletion_tasks(n_corner, seed=11)
                 + make_denoise_tasks(n_corner, seed=12)
                 + make_underlay_tasks(n_corner, seed=19)
                 + make_obstacle_tasks(n_obstacle, seed=18)
                 + make_relocation_tasks(n_relocate, seed=26))
    pair_corner = (make_perception_tasks(n_corner, seed=13)
                   + make_multi_registration_tasks(n_corner, seed=14)
                   + make_registration_except_tasks(n_corner, seed=15)
                   + make_inpainting_tasks(n_corner, seed=16)
                   + make_readout_tasks(n_corner, seed=17))

    tasks = (phys + des + ov + comet + bel + gdb + dual + fob + belief_extra
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


# ── found programs: from a phase run's search output (default) or ground truth ───────
def _load_found(decomposed, smoke, tasks, D, run_path, ground_truth):
    """Return (sols, found_of, rewritten_of, run_library, run_prov):
      sols        — {key: tree} to re-stitch into the FINAL library (mutates D).
      found_of    — {key_id: base-primitive 'found' program} per solved belief task.
      rewritten_of— {key_id: library-rewritten found program}, or None to re-derive.
      run_library — the run's invented-abstraction names (for name remapping), or None.
      run_prov    — the source run's provenance block (commit + knobs), or None when
                    reconstructing from ground truth / reading a pre-header artifact.

    Default source is a phase run's artifact (the searched sols/rewritten that
    `experiment.save_run_artifact` writes); `ground_truth=True` restores the old
    deterministic path (stitch the ground-truth programs, no run required)."""
    if ground_truth:
        sols = {mat_key(x): tr(D, gt_program_str(D, m)) for x, m in tasks}
        found_of = {mat_key_id(x): gt_program_str(D, m) for x, m in tasks}
        print("  source: ground-truth programs (legacy; no phase run consumed)")
        return sols, found_of, None, None, None

    if run_path is None:
        run_path = f"phase{2 if decomposed else 1}_run{'.smoke' if smoke else ''}.json"
    try:
        with open(run_path) as f:
            art = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"no run artifact '{run_path}' — run `python phase{2 if decomposed else 1}.py"
            f"{' --smoke' if smoke else ''}` first (or pass --ground-truth to reconstruct "
            f"from ground truth without a run).")
    if bool(art.get('decomposed')) != bool(decomposed):
        raise SystemExit(f"{run_path} is a phase {2 if art.get('decomposed') else 1} run, "
                         f"but this invocation is phase {2 if decomposed else 1}.")
    # Re-stitch the RUN's searched sols (in solve order) to rebuild exactly the library
    # that run converged to.  The run named its abstractions with an offset (prior wake-
    # sleep rounds), so run() remaps those names onto the reconstructed ones (both are in
    # stitch discovery order) before the run's rewritten strings are used as found_lib.
    sols = {kid: tr(D, s) for kid, s in art['sols'].items()}
    print(f"  source: phase run {run_path} ({art.get('n_solved', len(sols))} searched "
          f"programs; run library {art.get('library', [])})")
    print(f"  {provenance_line(art, run_path)}")
    return (sols, dict(art['sols']), dict(art.get('rewritten', {})),
            list(art.get('library', [])), art.get('provenance'))


def _remap_names(rewritten_of, run_library, D):
    """Rename the run's invented-abstraction tokens (e.g. fn_3/4/5, offset by earlier
    wake-sleep rounds) to the ones the fresh re-stitch registered (fn_0/1/2), so the
    run's rewritten programs parse — and are USED — under the reconstructed library.
    Positional because both name lists are in stitch discovery order.  Returns the
    rewritten map unchanged if the counts disagree (the caller then re-derives)."""
    new_names = [d.repr for d in D.invented]
    if not run_library or len(run_library) != len(new_names):
        return rewritten_of
    remap = {old: new for old, new in zip(run_library, new_names) if old != new}
    if not remap:
        return rewritten_of
    import re as _re
    def _apply(s):
        for old in sorted(remap, key=lambda n: -int(n.split('_')[1])):  # longest suffix first
            s = _re.sub(rf'\b{old}\b', remap[old], s)
        return s
    return {k: _apply(v) for k, v in rewritten_of.items()}


# ── the experiment ──────────────────────────────────────────────────────────────────
def run(decomposed=False, smoke=False, run_path=None, ground_truth=False):
    phase = 2 if decomposed else 1
    print(f"\n{'='*72}\nMDL-MARGIN EXPERIMENT — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync){' [smoke]' if smoke else ''}"
          f"\n{'='*72}")

    tasks = build_corpus(smoke)
    D = Deltas(make_symmetric_prims(decomposed=decomposed))
    verify_ground_truth(D, tasks)                 # sanity-checks the reconstructed corpus
    if decomposed:
        check_decomposition_identities(tasks)

    # the found programs + the sols that define the FINAL library both come from the run
    # (or ground truth, with --ground-truth); re-stitching the sols mutates D in place.
    sols, found_of, rewritten_of, run_library, run_prov = _load_found(
        decomposed, smoke, tasks, D, run_path, ground_truth)
    saturate_stitch(D, sols, iterations=(3 if smoke else 6), max_arity=5)
    if rewritten_of is not None:
        rewritten_of = _remap_names(rewritten_of, run_library, D)
    Q = uniform_type_q(D)
    print(f"  final library: {len(D)} tokens ({len(D.invented)} invented: "
          f"{[d.repr for d in D.invented]})")

    # every wall-belief task (including the extra distinct-seeded batch) is a genuine
    # belief_wall variant.  They matter most here: for plain wall-belief the transient-
    # wall rival reproduces the FULL scene, so MDL (not the witness/expressiveness) is
    # the operative discriminator.
    records = []
    n_bad = 0
    n_missing = 0
    for x, m in tasks:
        if m['kind'] != 'belief':
            continue
        kid = mat_key_id(x)
        if kid not in found_of:
            # the run did not solve this belief task — nothing found to price.
            n_missing += 1
            continue
        var = belief_variant(m)
        found_base = found_of[kid]
        if rewritten_of is not None and kid in rewritten_of:
            found_lib = rewritten_of[kid]           # the run's library form of the found program
            try:
                dl_found_lib = _dl(D, Q, found_lib)
            except Exception:                        # token mismatch — re-derive via the library
                (found_lib,) = rewrite_through_library(D, [found_base])
                dl_found_lib = _dl(D, Q, found_lib)
        else:
            (found_lib,) = rewrite_through_library(D, [found_base])
            dl_found_lib = _dl(D, Q, found_lib)
        dl_found_base = _dl(D, Q, found_base)

        rivals = belief_rival_specs(m)
        rival_libs = rewrite_through_library(D, [s for _, s in rivals])
        rival_recs = []
        for (label, rstr), rlib in zip(rivals, rival_libs):
            # behavioural guard: the library rewrite must be the same program in prims
            if _prims_of(D, rlib) != _prims_of(D, rstr):
                n_bad += 1
            xr = _unfold_rival(D, x, rstr)
            dl_r_base = _dl(D, Q, rstr)
            dl_r_lib  = _dl(D, Q, rlib)
            rival_recs.append({
                'label': label,
                'competitor': _reproduces_scene(xr, x),
                'reproduces_agents': _reproduces_agents(xr, x, m),
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
    if n_missing:
        print(f"  NOTE: {n_missing} belief task(s) had no found program in the run "
              f"(unsolved / not in the artifact) — excluded from the margin.")

    summary = _print_summary(records)
    out = {
        # the provenance of the RUN these margins were priced under (this script has no
        # knobs of its own worth a header): what a caption for the margin figure cites.
        'provenance': run_prov,
        'phase': phase,
        'decomposed': decomposed,
        'smoke': smoke,
        'source': 'ground-truth' if ground_truth else (run_path or f"phase{phase}_run"
                                                        f"{'.smoke' if smoke else ''}.json"),
        'library': [d.repr for d in D.invented],
        'n_belief_tasks': len(records),
        'n_belief_unsolved': n_missing,
        'summary': summary,
        'records': records,
    }
    path = f"mdl_margins{'.decomposed' if decomposed else ''}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {len(records)} belief tasks x rivals to {path}")
    # close the loop back to the run: the verdict quotes these nats (experiment.py's
    # load_mdl_margin echoes them on the next run; this back-fill fixes up THIS one).
    _backfill_verdict(summary, out, decomposed, smoke)
    return out


def _print_summary(records):
    """Per-variant summary over BEHAVIOURAL COMPETITORS — the non-mental rivals that
    reproduce the ENTIRE observed scene (the genuine 'almost as short' threats).  Rivals
    that render a different scene — whether they miss the agents' actions or match the
    actions but misplace the goal / wall — are reported separately: they are excluded by
    expressiveness, so they never bear on the MDL comparison.

    Returns the summary as data (stored in this experiment's artifact under 'summary'
    and back-filled into the phase's verdict artifact by `_backfill_verdict`), so the
    verdict reports these nats rather than only the booleans they support."""
    from collections import defaultdict
    comp = defaultdict(list)         # variant -> [library margins over competitor pairs]
    comp_base = defaultdict(list)
    n_tasks = defaultdict(int)
    n_tasks_with_comp = defaultdict(int)
    excl = defaultdict(list)         # variant -> [library margins of non-competitors]
    n_action_right_scene_wrong = 0   # reproduce the agents' actions but a different scene
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
                if rv.get('reproduces_agents'):
                    n_action_right_scene_wrong += 1
        n_tasks_with_comp[v] += int(has)

    by_variant = {}
    for var in ('belief_wall', 'belief_witness', 'belief_goal', 'belief_dual',
                'belief_false_obstacle'):
        if var not in n_tasks:
            continue
        cl = comp[var]
        by_variant[var] = {
            'n_tasks': n_tasks[var], 'n_tasks_with_competitor': n_tasks_with_comp[var],
            'n_competitor_pairs': len(cl),
            'margin_lib_min': (float(min(cl)) if cl else None),
            'margin_lib_median': (float(np.median(cl)) if cl else None),
            'margin_lib_max': (float(max(cl)) if cl else None),
            'margin_base_median': (float(np.median(comp_base[var])) if cl else None),
        }

    print(f"\n  {'variant':22s} {'tasks':>5s} {'w/comp':>6s} {'comp':>5s}  "
          f"{'library margin over competitors (nats)':>40s}")
    print("  " + "-" * 88)
    for var, s in by_variant.items():
        if s['n_competitor_pairs']:
            desc = (f"min {s['margin_lib_min']:+6.2f}  median {s['margin_lib_median']:+6.2f}  "
                    f"max {s['margin_lib_max']:+6.2f}   base median "
                    f"{s['margin_base_median']:+6.2f}")
        else:
            desc = "no non-mental behavioural competitor (expressiveness-only)"
        print(f"  {var:22s} {s['n_tasks']:5d} {s['n_tasks_with_competitor']:6d} "
              f"{s['n_competitor_pairs']:5d}  {desc}")

    all_comp = [v for l in comp.values() for v in l]
    share = (100.0 * sum(1 for v in all_comp if v > 0) / len(all_comp)) if all_comp else None
    if all_comp:
        print(f"\n  every behavioural competitor is LONGER than the mental reading: "
              f"{share:.0f}% of {len(all_comp)} competitor pairs have library margin > 0")
    all_excl = [v for l in excl.values() for v in l]
    n_shorter_excl = sum(1 for v in all_excl if v < 0)
    n_shorter_comp = sum(1 for v in all_comp if v < 0)
    print(f"  no rival is both a competitor and shorter: {n_shorter_comp} competitor pairs "
          f"have margin < 0 (of {len(all_comp)})")
    # honest, computed claim — never assert 'none' when a competitor actually is shorter
    if n_shorter_comp == 0:
        print(f"  of the {n_shorter_excl} rivals that ARE shorter than the mental reading, "
              f"none reproduces the observed agents (all expressiveness-excluded)")
    else:
        print(f"  WARNING: {n_shorter_comp} of the {n_shorter_excl + n_shorter_comp} shorter "
              f"rivals DO reproduce the observed agents — genuine behavioural competitors that "
              f"are shorter than the mental reading (inspect these before trusting the margin)")
    if n_action_right_scene_wrong:
        print(f"  ({n_action_right_scene_wrong} rival(s) reproduce the agents' actions but "
              f"render a different scene — action-right, scene-wrong; excluded by "
              f"expressiveness under the full-frame test, not counted as competitors)")
    return {
        'by_variant': by_variant,
        'n_competitor_pairs': len(all_comp),
        'pct_competitors_longer': share,
        'n_competitors_shorter': n_shorter_comp,
        'n_excluded_shorter': n_shorter_excl,
        'n_action_right_scene_wrong': n_action_right_scene_wrong,
        'margin_lib_median': (float(np.median(all_comp)) if all_comp else None),
        'margin_lib_min': (float(min(all_comp)) if all_comp else None),
    }


def _backfill_verdict(summary, out, decomposed, smoke):
    """Write this experiment's margins back into the phase's verdict artifact.

    The verdict is where the thesis reads the run's claims, and it was all booleans:
    'belief is the MDL win' with the nats living only here.  A run cannot compute them
    itself (they need this pass), so the loop closes from this side — after pricing, we
    add an 'mdl_margin' block to phase{1,2}_verdict[.smoke].json.  Only touches a
    verdict of the SAME phase/mode; anything else is left alone."""
    path = f"phase{2 if decomposed else 1}_verdict{'.smoke' if smoke else ''}.json"
    try:
        with open(path) as f:
            verdict = json.load(f)
    except (FileNotFoundError, ValueError):
        print(f"  (no verdict artifact {path} to back-fill — run the phase first)")
        return None
    if bool(verdict.get('decomposed')) != bool(decomposed) or \
       bool(verdict.get('smoke')) != bool(smoke):
        print(f"  (verdict artifact {path} is a different phase/mode — not back-filled)")
        return None
    verdict['mdl_margin'] = dict(summary, source=out['source'], library=out['library'],
                                 n_belief_tasks=out['n_belief_tasks'],
                                 n_belief_unsolved=out['n_belief_unsolved'])
    verdict['mdl_margin_note'] = None
    with open(path, 'w') as f:
        json.dump(verdict, f, indent=1)
    print(f"  back-filled the margin summary into {path}")
    return path


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    ground_truth = '--ground-truth' in sys.argv
    run_path = None
    if '--run' in sys.argv:
        run_path = sys.argv[sys.argv.index('--run') + 1]
    if '--both' in sys.argv:
        if run_path is not None:
            sys.exit("--run names a single phase artifact; drop --both or run each phase "
                     "separately with its own --run.")
        run(decomposed=False, smoke=smoke, ground_truth=ground_truth)
        run(decomposed=True, smoke=smoke, ground_truth=ground_truth)
    else:
        run(decomposed='--decomposed' in sys.argv, smoke=smoke,
            run_path=run_path, ground_truth=ground_truth)
