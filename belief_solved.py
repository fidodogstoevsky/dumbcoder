"""Belief solve-rate figure (phases 1-2): solved / total per belief variant.

`solve_dynamics.py` charts WHEN belief becomes reachable (the S-curve) and `corpus_dl.py`
charts the compression it buys.  This script asks the flatter, prior question: of the four
belief families the curriculum poses — wall / witness / goal-displacement / dual — HOW MANY
did the search actually solve?  It is the honest denominator behind every later belief
claim: a bar per variant, its solved height against the full-corpus total.

  wall    (belief_wall)    — plain transient-wall belief; the transient-wall rival
                             reproduces the whole scene, so MDL is the discriminator.
  witness (belief_witness) — a crossing witness rules the transient-wall rival out.
  goal    (belief_goal)    — the agent chases a privately-displaced goal.
  dual    (belief_dual)    — two mind-bearing agents, two private worlds.

The bar heights are DERIVED FROM AN ACTUAL RUN, not ground truth.  `experiment.run_phase`
(phase1.py / phase2.py) writes `phase{1,2}_run[.smoke].json` — the programs enumeration
actually FOUND, keyed by task matrix.  We rebuild the exact phase corpus with
`mdl_margin.build_corpus` (same sizes/seeds as the run), label each belief task by
`tasks.belief_variant`, and count a variant's task as SOLVED iff its matrix key is in
the run's `sols`.  So a bar that falls short of its total is a task the search missed, not
an artefact of reconstruction.

    # 1. produce a run (slow; HPC): its output is the artifact we consume
    python phase1.py                     -> phase1_run.json
    python phase2.py                     -> phase2_run.json
    # 2. build the solve-rate figure
    python belief_solved.py              phase 1 (reads phase1_run.json)
    python belief_solved.py --decomposed phase 2 (reads phase2_run.json)
    python belief_solved.py --both       phases 1 and 2, one JSON each, one PNG
    python belief_solved.py --smoke      use the .smoke run artifacts
    python belief_solved.py --run PATH   consume a specific run artifact
    python belief_solved.py --no-plot    recompute the JSON only
    python belief_solved.py out.png      choose the figure path

Writes belief_solved[.decomposed].json and belief_solved.png/.pdf.
"""

import sys
import json

from ecd import mat_key_id
from tasks import belief_variant
from mdl_margin import build_corpus


# ── belief variants (the bars) — same fine labels as tasks.belief_variant ───────
VAR_ORDER = ['belief_wall', 'belief_witness', 'belief_goal', 'belief_dual',
             'belief_false_obstacle']
VAR_LABEL = {
    'belief_wall':    'wall',
    'belief_witness': 'witness',
    'belief_goal':    'goal',
    'belief_dual':    'dual',
    'belief_false_obstacle': 'false-obst.',
}
VAR_SUB = {
    'belief_wall':    'transient-wall belief',
    'belief_witness': 'crossing witness',
    'belief_goal':    'displaced private goal',
    'belief_dual':    'two private worlds',
    'belief_false_obstacle': 'forced literal commit',
}
# tints of the belief blue: all belief, subtypes (deep → light).
VAR_COLOR = {
    'belief_wall':    '#0b4f9e',
    'belief_witness': '#2a78d6',
    'belief_goal':    '#5a9de8',
    'belief_dual':    '#8fbdf0',
    'belief_false_obstacle': '#b9d5f5',
}
INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e6e6e2', '#fcfcfb'
MISS = '#e6e6e2'        # the unsolved remainder (light neutral, drawn behind the solved bar)


# ── data loading: a phase run artifact (the found programs, keyed by task matrix) ─────
def _load_run(decomposed, smoke, run_path):
    if run_path is None:
        run_path = f"phase{2 if decomposed else 1}_run{'.smoke' if smoke else ''}.json"
    try:
        with open(run_path) as f:
            art = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"no run artifact '{run_path}' — run "
            f"`python phase{2 if decomposed else 1}.py{' --smoke' if smoke else ''}` first "
            f"(it writes the *_run.json the found programs live in).")
    if bool(art.get('decomposed')) != bool(decomposed):
        raise SystemExit(f"{run_path} is a phase {2 if art.get('decomposed') else 1} run, "
                         f"but this invocation is phase {2 if decomposed else 1}.")
    return art, run_path


# ── the count ─────────────────────────────────────────────────────────────────────────
def compute(decomposed, smoke, run_path):
    """Rebuild the phase corpus, then per belief variant count solved / total against a
    run's found programs.  A task is solved iff its matrix key is in the run's `sols`."""
    art, run_path = _load_run(decomposed, smoke, run_path)
    solved_keys = set(art['sols'])
    tasks = build_corpus(smoke)

    per = {v: {'total': 0, 'solved': 0} for v in VAR_ORDER}
    for x, m in tasks:
        if m['kind'] != 'belief':
            continue
        v = belief_variant(m)
        per[v]['total'] += 1
        if mat_key_id(x) in solved_keys:
            per[v]['solved'] += 1

    for v in VAR_ORDER:
        t, s = per[v]['total'], per[v]['solved']
        per[v]['rate'] = (s / t) if t else 0.0

    tot_total = sum(per[v]['total'] for v in VAR_ORDER)
    tot_solved = sum(per[v]['solved'] for v in VAR_ORDER)
    return {
        'phase':      2 if decomposed else 1,
        'decomposed': decomposed,
        'smoke':      bool(smoke),
        'source':     run_path,
        'per_variant': per,
        'total':      tot_total,
        'solved':     tot_solved,
        'rate':       (tot_solved / tot_total) if tot_total else 0.0,
    }


def run(decomposed=False, smoke=False, run_path=None):
    phase = 2 if decomposed else 1
    print(f"\n{'='*72}\nBELIEF SOLVE-RATE — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync){' [smoke]' if smoke else ''}"
          f"\n{'='*72}")
    out = compute(decomposed, smoke, run_path)
    for v in VAR_ORDER:
        p = out['per_variant'][v]
        bar = '#' * round(20 * p['rate'])
        print(f"  {VAR_LABEL[v].replace(chr(10), ' '):20s} "
              f"{p['solved']:>3}/{p['total']:<3} ({p['rate']*100:5.1f}%)  {bar}")
    print(f"  {'ALL belief':20s} {out['solved']:>3}/{out['total']:<3} "
          f"({out['rate']*100:5.1f}%)")
    path = f"belief_solved{'.decomposed' if decomposed else ''}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {path}")
    return out


# ── plotting ──────────────────────────────────────────────────────────────────────────
def _panel(ax, d):
    "One bar per variant: full-height total (faint) with the solved count overlaid."
    xs = list(range(len(VAR_ORDER)))
    totals = [d['per_variant'][v]['total'] for v in VAR_ORDER]
    solved = [d['per_variant'][v]['solved'] for v in VAR_ORDER]
    ymax = max(totals + [1])

    for i, v in enumerate(VAR_ORDER):
        t, s = totals[i], solved[i]
        # faint full-height total, then the solved bar on top of it
        ax.bar(i, t, width=0.62, color=MISS, edgecolor=GRID, lw=1, zorder=2)
        ax.bar(i, s, width=0.62, color=VAR_COLOR[v], edgecolor='none', zorder=3)
        # count label above the bar, sub-label of the variant meaning below the axis
        ax.text(i, t + ymax * 0.02, f"{s}/{t}", ha='center', va='bottom',
                color=VAR_COLOR[v], fontsize=11, fontweight='bold', zorder=5)
        pct = d['per_variant'][v]['rate'] * 100
        ax.text(i, max(s * 0.5, ymax * 0.06), f"{pct:.0f}%", ha='center', va='center',
                color='white' if s > ymax * 0.12 else MUTED, fontsize=9,
                fontweight='bold', zorder=5)
        ax.text(i, -ymax * 0.105, VAR_SUB[v], ha='center', va='top', color=MUTED,
                fontsize=7.6, zorder=5)

    ax.set_xticks(xs)
    ax.set_xticklabels([VAR_LABEL[v] for v in VAR_ORDER], fontsize=10, color=INK)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.62, len(VAR_ORDER) - 0.38)
    ax.set_ylim(0, ymax * 1.14)
    ax.set_ylabel('belief tasks', fontsize=9.5, color=MUTED)
    ax.grid(True, axis='y', color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)

    # overall solve-rate chip, top-right
    ax.text(0.98, 0.96, f"{d['solved']}/{d['total']} belief tasks solved\n"
                        f"({d['rate']*100:.0f}% of the curriculum)",
            transform=ax.transAxes, fontsize=9, color=INK, ha='right', va='top', zorder=6,
            bbox=dict(boxstyle='round,pad=0.35', fc='#eef3fb', ec='#cfe0f4', lw=1))


def plot(datasets, out_path='belief_solved.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n + 2.6, 5.2), squeeze=False)
    fig.patch.set_facecolor(BG)
    for col, d in enumerate(datasets):
        kind = 'atomic fork/sync' if not d['decomposed'] else 'decomposed plumbing'
        ax = axes[0, col]
        ax.set_facecolor(BG)
        ax.set_title(f"Phase {d['phase']}  —  {kind}", fontsize=11, color=INK,
                     loc='left', pad=8)
        _panel(ax, d)

    top = 1 - 1.0 / fig.get_figheight()
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.text(0.02, 1 - 0.30 / fig.get_figheight(),
             'The search solves every belief family the curriculum poses',
             fontsize=13.5, fontweight='bold', color=INK, ha='left', va='top')
    fig.text(0.02, 1 - 0.58 / fig.get_figheight(),
             'Solved / total per belief variant, counted from the programs the run actually '
             'found.\nThe faint bar is the full-corpus total; the coloured bar is what '
             'search recovered.',
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
    out_path = next((a for a in argv[1:] if a.endswith('.png')), 'belief_solved.png')

    if both:
        if run_path is not None:
            sys.exit("--run names a single phase artifact; drop --both or run each phase "
                     "separately with its own --run.")
        datasets = [run(decomposed=False, smoke=smoke),
                    run(decomposed=True, smoke=smoke)]
    else:
        decomposed = '--decomposed' in argv
        datasets = [run(decomposed=decomposed, smoke=smoke, run_path=run_path)]

    if not no_plot:
        plot(datasets, out_path)


if __name__ == '__main__':
    main(sys.argv)
