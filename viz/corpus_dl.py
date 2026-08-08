"""Corpus description-length trajectory (DreamCoder-style figure) for phases 1-2.

The wake-sleep loop in `experiment.run_phase` alternates ENUMERATE ↦ joint STITCH: each
round solves more tasks and re-compresses the whole pool into a shared library.  The MDL
objective it is implicitly minimising is the total description length of the corpus:

        total DL(round r)  =  Σ_task  DL(task's program | library_r)   +   DL(library_r)
                              └──────── corpus term ────────┘    └─ structure term ─┘

This script reconstructs that curve from a run's TRAJECTORY ARTIFACT
(`phase{1,2}_traj[.smoke].json`, written by `save_trajectory_artifact`).  For each round r
it re-stitches the sols solved THROUGH round r to rebuild that round's library — the same
reproducibility `mdl_margin.py` relies on for the final round, because `saturate_stitch`
re-discovers the library from the fully-expanded sols each call — then prices the FIXED
full corpus (every task's base-primitive solution) under it, in nats, via the same `_dl`
convention phase 3 / mdl_margin use.

Pricing a FIXED corpus under an IMPROVING library is what makes the total fall: as reuse
is captured, each recurring compound collapses to a cheap abstraction token.  The story is
in the per-family split — belief's contribution drops STEEPLY at the round the agency
signature (fork ∧ a sync-family commit) first recurs enough for stitch to abstract it,
while a bespoke non-mental schedule never does.  Round 0 is the base-primitive corpus
(no library), the honest starting point every later round is measured against.

    # 1. produce a run (slow; HPC) — its trajectory artifact is what we consume
    sbatch run.job phase1.py --t-fn 1200      -> phase1_traj.json  (+ phase1_run.json)
    sbatch run.job phase2.py --t-fn 1200      -> phase2_traj.json
    # 2. build the DL curve + figure
    python corpus_dl.py                       phase 1 (reads phase1_traj.json)
    python corpus_dl.py --decomposed          phase 2 (reads phase2_traj.json)
    python corpus_dl.py --both                phases 1 and 2, one JSON each, one PNG
    python corpus_dl.py --smoke               use the .smoke trajectory artifacts
    python corpus_dl.py --run PATH            consume a specific trajectory artifact
    python corpus_dl.py --no-plot             recompute the JSON only
    python corpus_dl.py --replot              redraw from an existing corpus_dl*.json
    python corpus_dl.py out.png               choose the figure path

Writes corpus_dl[.decomposed].json and corpus_dl.png/.pdf.
"""

import os
import sys
# the analysis modules live in the repo root, one level up from viz/; the sibling
# figure scripts live here, so put both on the path however this file is launched
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import json
import math
from collections import defaultdict

import torch as th

from ecd import (
    Deltas, saturate_stitch, rewrite_through_library, tr,
)
from prims import make_symmetric_prims
from experiment import provenance_line


# ── DL pricing (identical convention to mdl_margin._dl / phase3_arity._dl) ────────────
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


def _body_dl(D, Q, tree):
    """Structure cost (nats) of one abstraction body: the same per-node -Q[index] sum as
    `_dl`, but bound-variable holes ($i; ishole/isarg) are FREE — they name arguments, they
    do not select a library symbol.  This is the library's own contribution to total DL."""
    if getattr(tree, 'ishole', False) or getattr(tree, 'isarg', False):
        return 0.0
    idx = D.index(tree)
    cost = 0.0 if idx is None else float(-Q[idx])
    if tree.tails:
        for t in tree.tails:
            cost += _body_dl(D, Q, t)
    return cost


# ── task family (the facet) ───────────────────────────────────────────────────────────
# Belief variants all carry kind 'belief' (including the extra plain wall-belief batch);
# they fold into the single 'belief' facet.  physics / desire are their own families;
# every non-mental maker folds into the single 'world' facet — the contrast the figure
# is about is belief vs the rest.  ('world' is only the internal key; it is LABELLED
# 'non-mental', the thesis' own word for these families — 'world' would collide with the
# world model.)
def family(kind):
    if kind.startswith('belief'):
        return 'belief'
    if kind in ('physics', 'desire'):
        return kind
    return 'world'


FAM_ORDER = ['physics', 'desire', 'world', 'belief']   # priced order; belief last
# ...but only two of them are DRAWN.  physics (4 tasks) and desire (16) sit flat against
# the axis at this scale and carry no contrast the pooled non-mental curve does not
# already carry; they stay in the JSON, out of the figure.
PLOT_FAMS = ['world', 'belief']
FAM_COLOR = {
    'physics': '#eb6834',   # orange
    'desire':  '#157f5a',   # green (darkened for label contrast on the light surface)
    'world':   '#7a5bd0',   # violet
    'belief':  '#2a78d6',   # blue — the hero
}
FAM_LABEL = {
    'physics': 'constant movement',
    'desire':  'goal-directed movement',
    'world':   'non-mental',
    'belief':  'belief',
}
MUTED, GRID = '#52514e', '#e6e6e2'


# ── the trajectory computation ────────────────────────────────────────────────────────
def compute_trajectory(art):
    """From a trajectory artifact, price the fixed full corpus under each round's library.

    Returns a dict with the per-round series and the detected agency-signature round."""
    decomposed = bool(art['decomposed'])
    stitch_iters = int(art['stitch_iters'])
    n_rounds = int(art['n_rounds'])
    tasks = art['tasks']                       # kid -> {sol, kind, round}

    # the FIXED corpus: every solved task's base-primitive program, tagged with the round
    # it entered the solved pool (so we can rebuild each round's library from the subset).
    corpus = [(kid, t['sol'], t['kind'], int(t['round'])) for kid, t in tasks.items()]
    fam_of = {kid: family(t['kind']) for kid, t in tasks.items()}
    base_strs = [s for _, s, _, _ in corpus]

    rounds = []
    for r in range(0, n_rounds + 1):
        # a fresh base DSL each round keeps the reconstruction isolated; stitch mutates it.
        D = Deltas(make_symmetric_prims(decomposed=decomposed))
        solved_upto = {kid: tr(D, s) for kid, s, _, rd in corpus if rd <= r}
        if r >= 1 and solved_upto:
            # re-discover round r's library from the fully-expanded solved-through-r sols.
            saturate_stitch(D, solved_upto, iterations=stitch_iters, max_arity=5)
        Q = uniform_type_q(D)

        # price the WHOLE corpus under this round's library (rewrite base → library form).
        libs = rewrite_through_library(D, base_strs) if D.invented else base_strs
        by_family = defaultdict(float)
        by_count = defaultdict(int)
        corpus_dl = 0.0
        for (kid, _s, _k, _rd), lib in zip(corpus, libs):
            dl = _dl(D, Q, lib)
            by_family[fam_of[kid]] += dl
            by_count[fam_of[kid]] += 1
            corpus_dl += dl
        lib_cost = float(sum(_body_dl(D, Q, d.hiddentail) for d in D.invented))

        rounds.append({
            'round':       r,
            'n_invented':  len(D.invented),
            'invented':    [d.repr for d in D.invented],
            'n_solved':    len(solved_upto),
            'corpus_dl':   corpus_dl,
            'library_cost': lib_cost,
            'total_dl':    corpus_dl + lib_cost,
            'by_family':   {f: by_family.get(f, 0.0) for f in FAM_ORDER},
        })
        print(f"  round {r:>2}: |D|+{len(D.invented):<2} invented, "
              f"{len(solved_upto):>3}/{len(corpus)} solved | corpus {corpus_dl:8.1f} + "
              f"lib {lib_cost:6.1f} = {corpus_dl + lib_cost:8.1f} nats  "
              f"[belief {by_family.get('belief', 0.0):7.1f}]", flush=True)

    # the agency-signature round: where belief's family DL drops the most (round r vs r-1).
    bel = [rd['by_family']['belief'] for rd in rounds]
    drops = [(bel[i - 1] - bel[i], i) for i in range(1, len(bel))]
    sig_round = max(drops)[1] if drops else 0
    sig_drop = max(drops)[0] if drops else 0.0

    return {
        'decomposed':   decomposed,
        'phase':        2 if decomposed else 1,
        'smoke':        bool(art.get('smoke')),
        'n_rounds':     n_rounds,
        'n_tasks':      len(corpus),
        'family_counts': {f: sum(1 for v in fam_of.values() if v == f) for f in FAM_ORDER},
        'signature_round': sig_round,
        'signature_drop':  sig_drop,
        'final_library':   rounds[-1]['invented'] if rounds else [],
        'rounds':       rounds,
    }


def _load_traj(decomposed, smoke, run_path):
    if run_path is None:
        run_path = f"phase{2 if decomposed else 1}_traj{'.smoke' if smoke else ''}.json"
    try:
        with open(run_path) as f:
            art = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"no trajectory artifact '{run_path}' — run "
            f"`python phase{2 if decomposed else 1}.py{' --smoke' if smoke else ''}` first "
            f"(it writes the *_traj.json alongside the *_run.json).")
    if bool(art.get('decomposed')) != bool(decomposed):
        raise SystemExit(f"{run_path} is a phase {2 if art.get('decomposed') else 1} run, "
                         f"but this invocation is phase {2 if decomposed else 1}.")
    return art


def run(decomposed=False, smoke=False, run_path=None):
    phase = 2 if decomposed else 1
    print(f"\n{'='*72}\nCORPUS-DL TRAJECTORY — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync){' [smoke]' if smoke else ''}"
          f"\n{'='*72}")
    art = _load_traj(decomposed, smoke, run_path)
    print(f"  {provenance_line(art, run_path)}")
    data = compute_trajectory(art)
    # carry the source run's commit + knobs into this figure's own artifact, so the
    # caption can cite them without re-opening the trajectory file
    data['provenance'] = art.get('provenance')
    out = f"corpus_dl{'.decomposed' if decomposed else ''}.json"
    with open(out, 'w') as f:
        json.dump(data, f, indent=1)
    r0, rN = data['rounds'][0], data['rounds'][-1]
    print(f"  total DL {r0['total_dl']:.1f} → {rN['total_dl']:.1f} nats "
          f"({r0['total_dl'] - rN['total_dl']:+.1f} compressed over {data['n_rounds']} rounds); "
          f"belief's steepest drop {data['signature_drop']:.1f} nats at round "
          f"{data['signature_round']}.  wrote {out}")
    return data


# ── plotting ──────────────────────────────────────────────────────────────────────────
# Figures carry no titles, captions or annotations: the thesis body explains them, and
# the only text on the surface is the axes plus the direct series labels that stand in
# for a legend.  The figure is ONE panel — the per-family lines.  It used to be paired
# with a family-stacked area of the same four numbers, whose only extra content was the
# total and the library's own structure cost; the latter is under 1% of the axis (47 of
# 3315 nats), so it was never legible, and the totals are given in the prose instead.


def _panel_families(ax, data):
    "Per-family corpus DL vs round — belief's steep drop against the flat rest."
    rounds = data['rounds']
    xs = [rd['round'] for rd in rounds]
    for f in PLOT_FAMS:
        ys = [rd['by_family'][f] for rd in rounds]
        hero = f == 'belief'
        ax.plot(xs, ys, color=FAM_COLOR[f], lw=2.6 if hero else 1.8,
                alpha=1.0 if hero else 0.85, zorder=5 if hero else 3,
                marker='o', ms=6 if hero else 4, solid_capstyle='round')

    _style_axes(ax, xs)
    ax.set_ylabel('description length  (nats)', fontsize=11, color=MUTED)
    ymax = max(rd['by_family'][f] for rd in rounds for f in PLOT_FAMS)
    ax.set_ylim(0, ymax * 1.10)

    # label each line at its final point, pushed apart if the two families end up close
    # together.
    ends = sorted(((rounds[-1]['by_family'][f], f) for f in PLOT_FAMS))
    gap = ymax * 0.055
    ys_lab = []
    for y, _f in ends:
        ys_lab.append(y if not ys_lab else max(y, ys_lab[-1] + gap))
    for (y, f), yl in zip(ends, ys_lab):
        hero = f == 'belief'
        ax.text(xs[-1] + 0.08, yl, FAM_LABEL[f], color=FAM_COLOR[f],
                fontsize=13 if hero else 12, va='center', ha='left',
                fontweight='bold' if hero else 'normal', zorder=6)


def _style_axes(ax, xs):
    ax.set_xlabel('round', fontsize=11, color=MUTED)
    ax.set_xlim(min(xs) - 0.15, max(xs) + 1.15)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.grid(True, axis='y', color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)


def plot(datasets, out_path='corpus_dl.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 4.6 * n), squeeze=False)
    fig.patch.set_facecolor('#fcfcfb')
    for row, d in enumerate(datasets):
        ax = axes[row, 0]
        ax.set_facecolor('#fcfcfb')
        _panel_families(ax, d)

    fig.tight_layout()

    for ext in ({out_path, out_path.rsplit('.', 1)[0] + '.pdf'}):
        fig.savefig(ext, dpi=200, facecolor=fig.get_facecolor())
        print(f"wrote {ext}")


# ── CLI ───────────────────────────────────────────────────────────────────────────────
def replot(decomposed=False):
    """Redraw from this script's OWN artifact, skipping the re-stitch.

    The per-round per-family series is already in corpus_dl[.decomposed].json, so a purely
    presentational change to the figure need not re-price the corpus — and must not, when
    the working tree has moved past the commit the run was priced at."""
    out = f"corpus_dl{'.decomposed' if decomposed else ''}.json"
    try:
        with open(out) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"no {out} — run without --replot to compute it first.")
    prov = (data.get('provenance') or {}).get('repro', 'unknown provenance')
    print(f"replot phase {data['phase']} from {out}  [{prov}]")
    return data


def main(argv):
    smoke = '--smoke' in argv
    both = '--both' in argv
    no_plot = '--no-plot' in argv
    run_path = None
    if '--run' in argv:
        run_path = argv[argv.index('--run') + 1]
    out_path = next((a for a in argv[1:] if a.endswith('.png')), 'corpus_dl.png')

    if '--replot' in argv:
        datasets = ([replot(False), replot(True)] if both
                    else [replot('--decomposed' in argv)])
        plot(datasets, out_path)
        return

    if both:
        datasets = [run(decomposed=False, smoke=smoke),
                    run(decomposed=True, smoke=smoke)]
    else:
        decomposed = '--decomposed' in argv
        datasets = [run(decomposed=decomposed, smoke=smoke, run_path=run_path)]

    if not no_plot:
        plot(datasets, out_path)


if __name__ == '__main__':
    main(sys.argv)
