"""Solve-dynamics figures (phases 1-2): HOW the wake-sleep loop reaches belief.

`corpus_dl.py` prices the corpus under each round's library — it shows the description
length falling.  This script instead charts the SEARCH behaviour that the falling DL
buys, straight from a run, in two panels:

  Plot 1 — CUMULATIVE SOLVES vs round, one line per family.  The non-mental families
    (physics / desire / world) are shallow from the base primitives and solve in the
    first round, then sit flat at their ceiling.  Belief is deep — it only becomes
    reachable once the joint stitch has abstracted the parts it reuses — so its curve is
    an S: nothing, then a step up at the round its agency signature lands in the library.

  Plot 2 — PER-TASK SOLVE-TIME COLLAPSE, the round before a task solves vs the round it
    solves (a paired slope-graph, log time axis).  A belief task burns the whole `--t-fn`
    budget MISSING while its abstraction is still absent, then, the very next round, solves
    in seconds once stitch has added it — the ~1100 s → ~10 s collapse that is otherwise
    buried in the per-round `OK/MISS` log lines.  Non-mental tasks that solve immediately
    never pay that toll; they are the fast cluster with no prior miss.

Both plots are DERIVED FROM AN ACTUAL RUN, not ground truth.  `experiment.run_phase`
records, per solved task, the round it was solved (for Plot 1) and, per fn enumeration
attempt, the round/solved/seconds triple (for Plot 2), into the trajectory artifact
`phase{1,2}_traj[.smoke].json` (`save_trajectory_artifact`).  Older artifacts, or a full
HPC run whose dramatic numbers only reached stdout, can instead be read straight from the
run log with `--log` (we parse the same `OK/MISS <secs> <kind> <shape>#<crc>` lines).

    # 1. produce a run (slow; HPC) — writes the trajectory artifact we consume
    sbatch run.job phase1.py --t-fn 1200      -> phase1_traj.json  (+ phase1_run.json)
    # 2. build the solve-dynamics figure
    python solve_dynamics.py                  phase 1 (reads phase1_traj.json)
    python solve_dynamics.py --decomposed     phase 2 (reads phase2_traj.json)
    python solve_dynamics.py --both           phases 1 and 2, one JSON each, one PNG
    python solve_dynamics.py --smoke          use the .smoke trajectory artifacts
    python solve_dynamics.py --run PATH        consume a specific trajectory artifact
    python solve_dynamics.py --log run.out     parse per-task timings out of a run log
    python solve_dynamics.py --no-plot         recompute the JSON only
    python solve_dynamics.py out.png           choose the figure path

Writes solve_dynamics[.decomposed].json and solve_dynamics.png/.pdf.
"""

import re
import sys
import json
from collections import defaultdict

import numpy as np


# ── task family (the facet) — identical convention to corpus_dl.family ────────────────
# Belief variants and the plain-wall scaffold (kind 'belief'/'belief_scaffold') fold into
# 'belief'; physics/desire are their own families; every tasks_world maker folds into the
# single non-mental 'world' facet — the contrast the figure is about is belief vs the rest.
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
    'physics': 'physics',
    'desire':  'desire',
    'world':   'world (non-mental)',
    'belief':  'belief',
}
INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e6e6e2', '#fcfcfb'


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
    return {'source': run_path, 'n_rounds': n_rounds, 'tasks': tasks, 'timings': timings}


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

    # headline collapse: the biggest belief miss→solve drop, plus the belief medians.
    bel_pairs = [p for p in pairs if p['family'] == 'belief']
    headline = None
    if bel_pairs:
        worst = max(bel_pairs, key=lambda p: p['miss_time'] - p['solve_time'])
        headline = {
            'worst_miss': worst['miss_time'], 'worst_solve': worst['solve_time'],
            'worst_speedup': worst['speedup'],
            'median_miss': float(np.median([p['miss_time'] for p in bel_pairs])),
            'median_solve': float(np.median([p['solve_time'] for p in bel_pairs])),
            'n_belief_collapsed': len(bel_pairs),
        }

    return {
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
            print(f"  belief solve-time collapse: {h['n_belief_collapsed']} tasks; "
                  f"steepest {h['worst_miss']:.1f}s → {h['worst_solve']:.1f}s "
                  f"({h['worst_speedup']:.0f}×); median {h['median_miss']:.1f}s → "
                  f"{h['median_solve']:.1f}s")
        else:
            print("  no belief task had a prior-round miss to pair (all solved on first "
                  "attempt this run — nothing to collapse).")
    else:
        print("  NOTE: this artifact carries no per-task timings — Plot 2 will be a stub. "
              "Re-run the phase (timings are recorded now) or pass --log <run stdout>.")

    path = f"solve_dynamics{'.decomposed' if decomposed else ''}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {path}")
    return out


# ── plotting ──────────────────────────────────────────────────────────────────────────
def _panel_cumulative(ax, d):
    "Plot 1: cumulative solves vs round — belief's S-curve against the flat non-mental lines."
    xs = list(range(0, d['n_rounds'] + 1))
    for f in FAM_ORDER:
        ys = d['cumulative'][f]
        hero = f == 'belief'
        ax.plot(xs, ys, color=FAM_COLOR[f], lw=2.8 if hero else 1.9,
                alpha=1.0 if hero else 0.85, zorder=6 if hero else 3,
                marker='o', ms=6.5 if hero else 4.5, solid_capstyle='round',
                drawstyle='steps-post' if not hero else 'default')
        ax.text(xs[-1] + 0.08, ys[-1], f"{FAM_LABEL[f]}  ({ys[-1]}/{d['totals'][f]})",
                color=FAM_COLOR[f], fontsize=8.8 if hero else 8.2, va='center', ha='left',
                fontweight='bold' if hero else 'normal', zorder=7)

    # the round belief's cumulative rises the most: where its agency signature lands.
    bel = d['cumulative']['belief']
    steps = [(bel[i] - bel[i - 1], i) for i in range(1, len(bel))]
    if steps and max(steps)[0] > 0:
        sr = max(steps)[1]
        ax.axvline(sr, color=FAM_COLOR['belief'], lw=1.4, ls=(0, (4, 3)), zorder=2, alpha=0.8)
        ax.annotate('', xy=(sr, bel[sr]), xytext=(sr, bel[sr - 1]),
                    arrowprops=dict(arrowstyle='-|>', color=FAM_COLOR['belief'], lw=2))
        ax.text(sr - 0.08, (bel[sr] + bel[sr - 1]) / 2,
                f"agency signature\nenters library\n(+{bel[sr] - bel[sr - 1]} belief solves)",
                color=FAM_COLOR['belief'], fontsize=8.4, ha='right', va='center',
                zorder=7, fontweight='bold')

    _style_axes(ax, xs, xlabel='ECD round  (enumerate ↦ joint stitch)')
    ax.set_ylabel('cumulative tasks solved', fontsize=9.5, color=MUTED)
    ymax = max(max(d['cumulative'][f]) for f in FAM_ORDER)
    ax.set_ylim(-ymax * 0.03, ymax * 1.08)


def _panel_collapse(ax, d):
    "Plot 2: per-task solve-time collapse — round before solve vs solve round (log time)."
    pairs = d['pairs']
    if not d['has_timings'] or not pairs:
        ax.text(0.5, 0.5, "no per-task timings in this run\n"
                          "(re-run the phase, or pass --log <run stdout>)",
                transform=ax.transAxes, ha='center', va='center', color=MUTED, fontsize=10)
        ax.set_axis_off()
        return

    x0, x1 = 0.0, 1.0
    # order so belief draws on top of the non-mental lines
    for p in sorted(pairs, key=lambda p: p['family'] == 'belief'):
        hero = p['family'] == 'belief'
        c = FAM_COLOR[p['family']]
        ax.plot([x0, x1], [p['miss_time'], p['solve_time']], color=c,
                lw=2.0 if hero else 1.0, alpha=0.9 if hero else 0.5,
                zorder=6 if hero else 3, solid_capstyle='round')
        ax.scatter([x0, x1], [p['miss_time'], p['solve_time']], s=26 if hero else 12,
                   color=c, alpha=0.95 if hero else 0.6, zorder=7 if hero else 4)

    # the timeout ceiling: misses pile up at ~t_fn (the largest miss time observed).
    t_fn = max(p['miss_time'] for p in pairs)
    ax.axhline(t_fn, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1, alpha=0.7)
    ax.text(1.28, t_fn, f"~timeout {t_fn:.0f}s", color=MUTED, fontsize=8,
            ha='right', va='bottom')

    h = d['headline']
    if h:
        ax.annotate(f"belief\n{h['worst_miss']:.0f}s → {h['worst_solve']:.0f}s"
                    f"  ({h['worst_speedup']:.0f}× faster)",
                    xy=(x1, h['worst_solve']), xytext=(x1 - 0.32, h['worst_solve']),
                    color=FAM_COLOR['belief'], fontsize=8.6, fontweight='bold',
                    va='center', ha='right', zorder=8,
                    arrowprops=dict(arrowstyle='-|>', color=FAM_COLOR['belief'], lw=1.6))

    ax.set_yscale('log')
    ax.set_xlim(-0.42, 1.30)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(['round before\n(miss)', 'solve round\n(hit)'])
    ax.grid(True, axis='y', color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_ylabel('per-task enumeration time  (s, log)', fontsize=9.5, color=MUTED)

    # direct family labels at the right (solve) column
    fam_seen = {}
    for p in pairs:
        fam_seen.setdefault(p['family'], p['solve_time'])
    n_imm = len(d['immediate'])
    ax.text(0.015, 0.985, f"{n_imm} tasks solved on the first\nattempt (no prior miss)",
            transform=ax.transAxes, fontsize=8, color=MUTED, ha='left', va='top')


def _style_axes(ax, xs, xlabel):
    ax.set_xlabel(xlabel, fontsize=9.5, color=MUTED)
    ax.set_xlim(min(xs) - 0.15, max(xs) + 1.35)
    ax.set_xticks(xs)
    ax.set_xticklabels([('0\n(base)' if x == 0 else str(x)) for x in xs])
    ax.grid(True, axis='y', color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def plot(datasets, out_path='solve_dynamics.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(datasets)
    fig, axes = plt.subplots(n, 2, figsize=(13.4, 4.7 * n + 0.8), squeeze=False)
    fig.patch.set_facecolor(BG)
    for row, d in enumerate(datasets):
        kind = 'atomic fork/sync' if d['phase'] == 1 else 'decomposed plumbing'
        axA, axB = axes[row, 0], axes[row, 1]
        for ax in (axA, axB):
            ax.set_facecolor(BG)
        axA.set_title(f"Phase {d['phase']}  —  {kind}   (cumulative solves)",
                      fontsize=11, color=INK, loc='left', pad=8)
        axB.set_title("per-task solve-time collapse", fontsize=11, color=INK, loc='left', pad=8)
        _panel_cumulative(axA, d)
        _panel_collapse(axB, d)

    top = 1 - 1.0 / fig.get_figheight()
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.text(0.02, 1 - 0.32 / fig.get_figheight(),
             'Belief is reached late and cheaply — an S-curve of solves, and a solve-time '
             'collapse the round its abstraction lands',
             fontsize=13.5, fontweight='bold', color=INK, ha='left', va='top')
    fig.text(0.02, 1 - 0.62 / fig.get_figheight(),
             'Non-mental families solve in round 1 and sit flat at their ceiling. Belief '
             'becomes reachable only once the joint stitch abstracts the parts it reuses —\n'
             'its enumeration time then drops from the full timeout to seconds in a single '
             'round, the round its S-curve steps up.',
             fontsize=9, color=MUTED, ha='left', va='top', linespacing=1.5)

    for ext in ({out_path, out_path.rsplit('.', 1)[0] + '.pdf'}):
        fig.savefig(ext, dpi=200, facecolor=fig.get_facecolor())
        print(f"wrote {ext}")


# ── CLI ───────────────────────────────────────────────────────────────────────────────
def main(argv):
    smoke = '--smoke' in argv
    both = '--both' in argv
    no_plot = '--no-plot' in argv
    run_path = argv[argv.index('--run') + 1] if '--run' in argv else None
    log_path = argv[argv.index('--log') + 1] if '--log' in argv else None
    out_path = next((a for a in argv[1:] if a.endswith('.png')), 'solve_dynamics.png')

    if both:
        if log_path is not None:
            sys.exit("--log names a single phase's stdout; drop --both or run each phase "
                     "separately with its own --log.")
        datasets = [run(decomposed=False, smoke=smoke),
                    run(decomposed=True, smoke=smoke)]
    else:
        decomposed = '--decomposed' in argv
        datasets = [run(decomposed=decomposed, smoke=smoke, run_path=run_path, log_path=log_path)]

    if not no_plot:
        plot(datasets, out_path)


if __name__ == '__main__':
    main(sys.argv)
