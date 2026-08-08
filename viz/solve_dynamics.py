"""Solve-dynamics numbers (phases 1-2): HOW the wake-sleep loop reaches belief.

`corpus_dl.py` prices the corpus under each round's library — it shows the description
length falling.  This script reports the SEARCH behaviour that the falling DL buys,
straight from a run: which round each task was first solved in, and how long enumeration
took in the round a task missed against the round it solved.

It emits NUMBERS, NOT A FIGURE.  It used to draw two panels and both were cut; the
reasons are in the comment block above `run()`, and the short version is that one panel
duplicated a table and the other implied per-task speedups the run never measured.  The
thesis quotes this script's `headline` block in prose instead.

READ THE CENSORING NOTE BEFORE QUOTING ANY NUMBER FROM HERE.  A miss time is a TIMEOUT:
the task needed more than that, by an unknown margin.  So `median_miss_lower_bound` and
`worst_speedup_lower_bound` are bounds, not measurements, and they are named that way on
purpose.  The budget also steps (round 1 gets `--t-fn-round1`, later rounds `--t-fn`), so
a round-1 miss paired with a round-2 solve crosses a budget increase as well as a change
of library.  `n_library_attributable` and the `clean_*` quartiles are the subset where
neither problem bites — those are the per-task numbers that can be defended.

Everything is DERIVED FROM AN ACTUAL RUN, not ground truth.  `experiment.run_phase`
records, per solved task, the round it was solved (the `cumulative` block) and, per fn
enumeration attempt, the round/solved/seconds triple (the `pairs` block), into the
trajectory artifact `phase{1,2}_traj[.smoke].json` (`save_trajectory_artifact`).  Older
artifacts, or a full HPC run whose numbers only reached stdout, can instead be read
straight from the run log with `--log` (we parse the same
`OK/MISS <secs> <kind> <shape>#<crc>` lines).

    # 1. produce a run (slow; HPC) — writes the trajectory artifact we consume
    sbatch run.job phase1.py --t-fn 1200      -> phase1_traj.json  (+ phase1_run.json)
    # 2. compute the solve-dynamics numbers
    python solve_dynamics.py                  phase 1 (reads phase1_traj.json)
    python solve_dynamics.py --decomposed     phase 2 (reads phase2_traj.json)
    python solve_dynamics.py --both           phases 1 and 2, one JSON each
    python solve_dynamics.py --smoke          use the .smoke trajectory artifacts
    python solve_dynamics.py --run PATH       consume a specific trajectory artifact
    python solve_dynamics.py --log run.out    parse per-task timings out of a run log

Writes solve_dynamics[.decomposed].json.
"""

import re
import sys
import json
from collections import defaultdict

import numpy as np


# ── task family (the facet) — identical convention to corpus_dl.family ────────────────
# All belief variants (kind 'belief', including the extra plain wall-belief batch) fold
# into 'belief'; physics/desire are their own families; every non-mental maker folds into
# the single 'world' facet — the contrast the figure is about is belief vs the rest.
# ('world' is only the internal key; it is LABELLED 'non-mental', the thesis' own word for
# these families — 'world' would collide with the world model.)
def family(kind):
    if kind.startswith('belief'):
        return 'belief'
    if kind in ('physics', 'desire'):
        return kind
    return 'world'


FAM_ORDER = ['physics', 'desire', 'world', 'belief']   # belief drawn last / on top
FAM_COLOR = {
    'physics': '#eb6834',   # orange
    'desire':  '#157f5a',   # green
    'world':   '#7a5bd0',   # violet
    'belief':  '#2a78d6',   # blue — the hero
}
FAM_LABEL = {
    'physics': 'constant movement',
    'desire':  'goal-directed movement',
    'world':   'non-mental',
    'belief':  'belief',
}
MUTED, GRID, BG = '#52514e', '#e6e6e2', '#fcfcfb'


# ── data loading: trajectory artifact (default) or a raw run log (--log) ───────────────
def _load_from_artifact(decomposed, smoke, run_path):
    if run_path is None:
        run_path = f"phase{2 if decomposed else 1}_traj{'.smoke' if smoke else ''}.json"
    try:
        with open(run_path) as f:
            art = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"no trajectory artifact '{run_path}' — run "
            f"`python phase{2 if decomposed else 1}.py{' --smoke' if smoke else ''}` first "
            f"(it writes the *_traj.json), or pass --log to read a run's stdout instead.")
    if bool(art.get('decomposed')) != bool(decomposed):
        raise SystemExit(f"{run_path} is a phase {2 if art.get('decomposed') else 1} run, "
                         f"but this invocation is phase {2 if decomposed else 1}.")
    # tasks: id -> {kind, family, solve_round};  timings: [{id, kind, round, solved, elapsed}]
    tasks = {kid: {'kind': t['kind'], 'family': family(t['kind']), 'solve_round': int(t['round'])}
             for kid, t in art['tasks'].items()}
    timings = [{'id': str(r['id']), 'kind': r['kind'], 'family': family(r['kind']),
                'round': int(r['round']), 'solved': bool(r['solved']),
                'elapsed': float(r['elapsed'])}
               for r in art.get('timings', [])]
    n_rounds = int(art.get('n_rounds') or max((t['solve_round'] for t in tasks.values()), default=0))
    # provenance: the source run's commit + knobs (experiment.build_provenance).  Inlined
    # rather than importing experiment.provenance_line — this module reads artifacts with
    # numpy alone, and the header isn't worth pulling the DSL and torch in behind it.
    prov = art.get('provenance')
    print("  " + (prov['repro'] if prov else
                  f"provenance: NONE recorded in {run_path} — predates the provenance "
                  f"header; rerun the phase to make its figures citable."))
    return {'source': run_path, 'n_rounds': n_rounds, 'tasks': tasks, 'timings': timings,
            'provenance': prov}


# stdout line written by run_phase's per-task timing report:
#   "      OK    12.3  belief           (6, 5, 5)#a1b2c3"
#   "      MISS 1200.0  belief           (6, 5, 5)#a1b2c3"
# the `(shape)#crc` tag is a stable per-task id across rounds; kind is the family label.
_ROUND_RE = re.compile(r'---\s*round\s+(\d+)\s*/')
_TASK_RE  = re.compile(r'\b(OK|MISS)\s+([\d.]+)\s+(\S+)\s+(\([^)]*\)#[0-9a-f]+)')


def _load_from_log(decomposed, log_path):
    "Reconstruct tasks + timings by parsing the OK/MISS per-task lines out of a run log."
    timings, cur_round = [], 0
    with open(log_path) as f:
        for line in f:
            mr = _ROUND_RE.search(line)
            if mr:
                cur_round = int(mr.group(1))
                continue
            mt = _TASK_RE.search(line)
            if mt and cur_round:
                hit, secs, kind, tag = mt.group(1) == 'OK', float(mt.group(2)), mt.group(3), mt.group(4)
                timings.append({'id': tag, 'kind': kind, 'family': family(kind),
                                'round': cur_round, 'solved': hit, 'elapsed': secs})
    if not timings:
        raise SystemExit(f"parsed no per-task OK/MISS timing lines from '{log_path}' — is it a "
                         f"run stdout log?  (expected lines like '  OK   12.3  belief (…)#crc').")
    # a task's solve round is the first round it reported OK; only solved tasks enter `tasks`.
    solve_round = {}
    for r in sorted(timings, key=lambda r: r['round']):
        if r['solved'] and r['id'] not in solve_round:
            solve_round[r['id']] = (r['round'], r['kind'])
    tasks = {tag: {'kind': kd, 'family': family(kd), 'solve_round': rnd}
             for tag, (rnd, kd) in solve_round.items()}
    n_rounds = max(r['round'] for r in timings)
    return {'source': log_path, 'n_rounds': n_rounds, 'tasks': tasks, 'timings': timings}


# ── derived series ────────────────────────────────────────────────────────────────────
def compute(data):
    """From loaded {tasks, timings}, build the cumulative-solve series (Plot 1) and the
    per-task solve-time collapse pairs (Plot 2)."""
    tasks, timings, n_rounds = data['tasks'], data['timings'], data['n_rounds']

    # Plot 1: cumulative solves per family, rounds 0..n_rounds (round 0 = nothing solved).
    totals = {f: sum(1 for t in tasks.values() if t['family'] == f) for f in FAM_ORDER}
    cumulative = {}
    for f in FAM_ORDER:
        rs = sorted(t['solve_round'] for t in tasks.values() if t['family'] == f)
        cumulative[f] = [sum(1 for sr in rs if sr <= r) for r in range(0, n_rounds + 1)]

    # Plot 2: for each task, its per-round attempt timeline; the collapse pair is the
    # (miss round r-1, solve round r) time pair — the round before it solves vs the round
    # it solves.  A task solved in round 1 has no prior attempt: it never paid the toll.
    timeline = defaultdict(dict)        # id -> {round: (solved, elapsed, kind)}
    for r in timings:
        timeline[r['id']][r['round']] = (r['solved'], r['elapsed'], r['kind'])
    pairs, immediate = [], []
    for tid, tl in timeline.items():
        solved_rounds = sorted(rd for rd, (hit, _e, _k) in tl.items() if hit)
        if not solved_rounds:
            continue
        rs = solved_rounds[0]
        kind = tl[rs][2]
        solve_t = tl[rs][1]
        if (rs - 1) in tl:              # the miss the round before it landed
            miss_t = tl[rs - 1][1]
            pairs.append({'id': tid, 'kind': kind, 'family': family(kind),
                          'solve_round': rs, 'miss_time': miss_t, 'solve_time': solve_t,
                          'speedup': (miss_t / solve_t) if solve_t > 0 else float('inf')})
        else:
            immediate.append({'id': tid, 'kind': kind, 'family': family(kind),
                              'solve_round': rs, 'solve_time': solve_t})

    # Headline collapse.  Two things make the naive miss→solve comparison unusable, and
    # both are handled here rather than left to the reader:
    #
    #   CENSORING.  A miss time is a TIMEOUT — the task needed *more* than that, by an
    #     unknown margin.  So `median_miss` is a lower bound, not a measurement, and no
    #     per-task speedup is recoverable.  `worst_speedup` is likewise a lower bound; it
    #     is kept because a lower bound on a 160x drop is still worth quoting, but it must
    #     never be reported as *the* speedup.
    #   BUDGET STEP.  Round 1 allows `t_fn_round1` per task and later rounds `t_fn` — 1200 s
    #     against 3600 s in the thesis runs — so a round-1 miss paired with a round-2 solve
    #     has crossed a budget increase as well as a change of library.
    #
    # The subset that survives both is `n_library_attributable`: solves that fit inside the
    # budget the MISS round already had.  There the extra clock cannot be the explanation,
    # so the library is, and the quartiles of that subset are the honest per-task numbers.
    # The complement (`n_budget_confounded`) is evidence for neither side.
    bel_pairs = [p for p in pairs if p['family'] == 'belief']
    headline = None
    if bel_pairs:
        worst = max(bel_pairs, key=lambda p: p['miss_time'] - p['solve_time'])
        clean = [p for p in bel_pairs if p['solve_time'] < p['miss_time']]
        cq = (np.percentile([p['solve_time'] for p in clean], [25, 50, 75]).tolist()
              if clean else None)
        headline = {
            'worst_miss': worst['miss_time'], 'worst_solve': worst['solve_time'],
            'worst_speedup_lower_bound': worst['speedup'],
            'median_miss_lower_bound': float(np.median([p['miss_time'] for p in bel_pairs])),
            'median_solve_all': float(np.median([p['solve_time'] for p in bel_pairs])),
            'slowest_solve': max(p['solve_time'] for p in bel_pairs),
            'n_belief_collapsed': len(bel_pairs),
            'n_library_attributable': len(clean),
            'n_budget_confounded': len(bel_pairs) - len(clean),
            'clean_q25': cq[0] if cq else None,
            'clean_median': cq[1] if cq else None,
            'clean_q75': cq[2] if cq else None,
            'clean_fastest': min((p['solve_time'] for p in clean), default=None),
        }

    return {
        # None on the --log path: a stdout log carries no provenance block
        'provenance': data.get('provenance'),
        'phase': 2 if data.get('decomposed') else 1,
        'source': data['source'],
        'n_rounds': n_rounds,
        'totals': totals,
        'cumulative': cumulative,
        'pairs': pairs,
        'immediate': immediate,
        'has_timings': bool(timings),
        'headline': headline,
    }


def run(decomposed=False, smoke=False, run_path=None, log_path=None):
    phase = 2 if decomposed else 1
    print(f"\n{'='*72}\nSOLVE-DYNAMICS — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync){' [smoke]' if smoke else ''}"
          f"\n{'='*72}")
    if log_path is not None:
        data = _load_from_log(decomposed, log_path)
    else:
        data = _load_from_artifact(decomposed, smoke, run_path)
    data['decomposed'] = decomposed
    out = compute(data)

    # report
    for f in FAM_ORDER:
        print(f"  {FAM_LABEL[f]:20s} cumulative solves: {out['cumulative'][f]}  "
              f"(of {out['totals'][f]})")
    if out['has_timings']:
        h = out['headline']
        if h:
            print(f"  belief tasks missed then solved: {h['n_belief_collapsed']}; "
                  f"slowest solve {h['slowest_solve']:.0f}s")
            print(f"  library-attributable (solve fits the miss round's own budget): "
                  f"{h['n_library_attributable']}/{h['n_belief_collapsed']}; "
                  f"quartiles {h['clean_q25']:.0f} / {h['clean_median']:.0f} / "
                  f"{h['clean_q75']:.0f}s; fastest {h['clean_fastest']:.1f}s")
            print(f"  budget-confounded (solve exceeds it): {h['n_budget_confounded']} "
                  f"— evidence for neither side")
            print(f"  NB miss times are TIMEOUTS (censored): median miss "
                  f">={h['median_miss_lower_bound']:.0f}s and the {h['worst_speedup_lower_bound']:.0f}× "
                  f"steepest drop are LOWER BOUNDS, not measured speedups")
        else:
            print("  no belief task had a prior-round miss to pair (all solved on first "
                  "attempt this run — nothing to collapse).")
    else:
        print("  NOTE: this artifact carries no per-task timings. "
              "Re-run the phase (timings are recorded now) or pass --log <run stdout>.")

    path = f"solve_dynamics{'.decomposed' if decomposed else ''}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {path}")
    return out


# ── no figure ─────────────────────────────────────────────────────────────────────────
# This script used to emit a two-panel figure and now emits none; both panels were cut,
# for different reasons, and the reasons are worth keeping so neither gets rebuilt.
#
#   Panel 1, cumulative solves per round, was REDUNDANT.  Its whole content is the belief
#     row of the per-round library table (0 / 75 / 107 / 132 against a flat non-mental
#     144), the prose quotes those four numbers anyway, and on the page it read as a
#     near-twin of the corpus-DL figure — same x-axis, same four series, same colours — so
#     the same round-2 step got told three times.
#   Panel 2, the miss→solve slopegraph, was MISLEADING.  Every left endpoint was a timeout,
#     so it took exactly two values (1200 s and 3600 s) and no line's slope carried
#     anything the right endpoint did not.  Worse, a timeout is CENSORED — the task needed
#     more than that by an unknown margin — so drawing a line from it to a measured solve
#     time implies a magnitude the run never observed.  Reading that picture is what
#     produced the "large mass pressed against the ceiling" and "most tasks spend most of
#     the window" claims the thesis used to make; both are false against the 3600 s budget
#     (median 32%, three tasks of 132 within 90% of it).
#
# The numbers that replaced them are in `headline`, which is censoring- and budget-aware.
# The `cumulative` and `pairs` blocks are still computed into the JSON — they are the
# run's record, and only their plots are gone.


# ── CLI ───────────────────────────────────────────────────────────────────────────────
def main(argv):
    smoke = '--smoke' in argv
    both = '--both' in argv
    run_path = argv[argv.index('--run') + 1] if '--run' in argv else None
    log_path = argv[argv.index('--log') + 1] if '--log' in argv else None

    if both:
        if log_path is not None:
            sys.exit("--log names a single phase's stdout; drop --both or run each phase "
                     "separately with its own --log.")
        run(decomposed=False, smoke=smoke)
        run(decomposed=True, smoke=smoke)
    else:
        decomposed = '--decomposed' in argv
        run(decomposed=decomposed, smoke=smoke, run_path=run_path, log_path=log_path)


if __name__ == '__main__':
    main(sys.argv)
