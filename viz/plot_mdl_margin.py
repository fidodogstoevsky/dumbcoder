"""Plot the MDL-margin distribution written by mdl_margin.py.

One figure per phase (rows if both present).  Per belief variant, the empirical CDF of
the description-length margin  DL(rival) - DL(found)  in nats, over the NON-MENTAL rivals
that reproduce the observed agent trajectory (the genuine 'almost as short' competitors).

Two line styles carry the crux:
  solid  = priced under the FINAL library (base prims + the abstractions one joint stitch
           discovers over the whole corpus) — the belief compound collapses to the
           agent constructor, a bespoke transient-wall schedule does not;
  dashed = priced in base primitives (no abstractions) — the honest lower bound.

Everything to the right of the 0 line means the mental reading is the shorter description.
Variants with no non-mental behavioural competitor (goal-displacement, dual) are named in
the annotation: there, expressiveness alone excludes every non-mental program.

    python plot_mdl_margin.py                 # reads mdl_margins[.decomposed].json
    python plot_mdl_margin.py out.png         # choose output path
"""

import sys
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# categorical hues in fixed order (validated: blue↔orange CVD ΔE 96.7, both >3:1 on light)
VAR_COLOR = {
    'belief_wall':    '#2a78d6',   # slot 1 — blue
    'belief_witness': '#eb6834',   # slot 8 — orange
    'belief_goal':    '#1baf7a',   # aqua (used only if it ever gains competitors)
    'belief_dual':    '#4a3aa7',   # violet
    'belief_false_obstacle': '#b8860b',  # dark goldenrod (forced-literal family)
}
VAR_LABEL = {
    'belief_wall':    'wall (phantom obstacle)',
    'belief_witness': 'witness (false belief + bystander)',
    'belief_goal':    'goal-displacement',
    'belief_dual':    'dual (contradictory beliefs)',
    'belief_false_obstacle': 'false-obstacle (forced literal commit)',
}
VAR_ORDER = ['belief_wall', 'belief_witness', 'belief_goal', 'belief_dual',
             'belief_false_obstacle']

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e6e2'


def _ecdf(vals):
    "step ECDF: returns (x, y) with a leading/trailing flat so the step reads cleanly."
    v = np.sort(np.asarray(vals, float))
    n = len(v)
    y = np.arange(1, n + 1) / n
    return v, y


def _competitor_margins(rec_list):
    "per variant: lists of (library, base) margins over behavioural-competitor rivals."
    lib, base = {}, {}
    n_tasks, n_comp_tasks = {}, {}
    for r in rec_list:
        v = r['variant']
        n_tasks[v] = n_tasks.get(v, 0) + 1
        got = False
        for rv in r['rivals']:
            if rv.get('competitor'):
                lib.setdefault(v, []).append(rv['margin_lib'])
                base.setdefault(v, []).append(rv['margin_base'])
                got = True
        if got:
            n_comp_tasks[v] = n_comp_tasks.get(v, 0) + 1
    return lib, base, n_tasks, n_comp_tasks


def _step(ax, margins, color, lw, alpha, zorder, ls='-'):
    x, y = _ecdf(margins)
    xx = np.concatenate([[x[0]], x]); yy = np.concatenate([[0.0], y])
    ax.step(xx, yy, where='post', color=color, lw=lw, alpha=alpha, ls=ls,
            solid_capstyle='round', solid_joinstyle='round',
            dash_capstyle='round', zorder=zorder)


def _panel(ax, data):
    lib, base, n_tasks, n_comp_tasks = _competitor_margins(data['records'])
    present = [v for v in VAR_ORDER if lib.get(v)]

    all_base = [m for v in base.values() for m in v]
    lo = min(all_base + [0.0])
    hi = max([m for v in lib.values() for m in v] + [1.0])
    ax.set_xlim(min(lo - 2, -2), hi + 2)

    # objection region: where a non-mental competitor would be SHORTER than belief
    ax.axvspan(ax.get_xlim()[0], 0, color='#f4d4cd', alpha=0.30, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=1)

    for v in present:
        c = VAR_COLOR[v]
        _step(ax, base[v], c, 1.7, 0.7, 2, ls='--')  # base prims — honest lower bound (dashed)
        _step(ax, lib[v],  c, 2.4, 1.0, 4)           # final library — the headline (solid)

    # One legend row per line ACTUALLY drawn, in that line's own colour and style, so the
    # key maps one-to-one onto the axes: solid-bold = priced under the final library,
    # dashed-faint = priced in base primitives (the honest lower bound).  A neutral style
    # key was tried but read as a phantom grey series — with only belief_wall carrying
    # non-mental competitors (the others are expressiveness-excluded, see the note), a
    # per-line coloured legend is both shorter and unambiguous.
    handles = []
    for v in present:
        c = VAR_COLOR[v]
        handles.append(Line2D([0], [0], color=c, lw=2.4, ls='-',
                              label="final library"))
        #handles.append(Line2D([0], [0], color=c, lw=2.4, ls='-',
        #                      label=f"{VAR_LABEL[v]} — final library"))
        handles.append(Line2D([0], [0], color=c, lw=1.7, ls='--', alpha=0.75,
                              label="base primitives"))
        #handles.append(Line2D([0], [0], color=c, lw=1.7, ls='--', alpha=0.75,
        #                      label=f"{VAR_LABEL[v]} — base primitives"))
    leg = ax.legend(handles=handles, loc='center left', fontsize=8.6, frameon=True,
                    framealpha=0.96, edgecolor=GRID, borderpad=0.7, labelspacing=0.5,
                    bbox_to_anchor=(0.02, 0.62))
    leg.set_zorder(6)

    ax.set_xlabel('description-length margin   DL(rival) − DL(found)   (nats)',
                  fontsize=9.5, color=MUTED)
    ax.set_ylabel('cumulative fraction of\ncompetitor rivals', fontsize=9.5, color=MUTED)
    ax.set_ylim(0, 1.03)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main(out_path='mdl_margin.png'):
    files = [f for f in ('mdl_margins.json', 'mdl_margins.decomposed.json')
             if os.path.exists(f)]
    if not files:
        sys.exit("no mdl_margins*.json found — run `python mdl_margin.py [--both]` first.")
    datasets = [json.load(open(f)) for f in files]

    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(8.8, 4.8 * n + 0.7), squeeze=False)
    fig.patch.set_facecolor('#fcfcfb')
    for ax, d in zip(axes[:, 0], datasets):
        ax.set_facecolor('#fcfcfb')
        kind = 'atomic fork/sync' if not d['decomposed'] else 'decomposed plumbing'
        ax.set_title(f"Phase {d['phase']}  —  {kind}   "
                     f"(final library: {len(d['library'])} invented abstractions)",
                     fontsize=11, color=INK, loc='left', pad=8)
        _panel(ax, d)

    top = 1 - 0.95 / fig.get_figheight()         # reserve a constant ~0.95in header
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.text(0.02, 1 - 0.30 / fig.get_figheight(),
             'No non-mental program is both behaviourally adequate and as short as belief',
             fontsize=13.5, fontweight='bold', color=INK, ha='left', va='top')
    fig.text(0.02, 1 - 0.62 / fig.get_figheight(),
             'ECDF of the MDL margin over rivals that reproduce the observed agent '
             'trajectory; the shaded band is where the objection would live.',
             fontsize=9, color=MUTED, ha='left', va='top')

    for ext_path in ({out_path, out_path.rsplit('.', 1)[0] + '.pdf'}):
        fig.savefig(ext_path, dpi=200, facecolor=fig.get_facecolor())
        print(f"wrote {ext_path}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'mdl_margin.png')
