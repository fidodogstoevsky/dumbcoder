"""The Sally-Anne illustration figure: what the belief compound computes, drawn.

*** THIS IS NOT A TEST, AND THE THESIS NO LONGER PRESENTS IT AS ONE. ***

Read `compute()` before citing anything this prints.  NOTHING HERE IS ENUMERATED.  The
"mental" program is `experiment.gt_program_str(D, m)` — the family's ground-truth
s-expression, built out of the scene's OWN metadata (`m['dirs']`, `m['displaced_to']`) —
and the "rival" is the shortest member of `tasks.belief_rival_specs(m)`, likewise authored.
The search is never run on these scenes.  So the headline count (24/24 to the believed
cell) is ANALYTIC, not a measurement: a program whose body plans against a copy in which
the goal has been shoved one cell up lands on the shoved cell by construction, and no
scene could have made it do otherwise.  There is no outcome here that could have come
out the other way, which is exactly why it is not a criterion.

What it IS good for is the picture, and the thesis uses it that way — chapter 4 §4.5
(`fig-sally-anne`), as a second worked example of the derive-run-commit frame beside the
false-wall walkthrough.  Keep the frame, change the derive, and the agent is wrong about
where the GOAL is rather than about a wall:

  * the belief compound derives a private copy with the goal shoved one cell up, plans
    the agent's move against THAT, and commits the agent's position and nothing else —
    so the agent settles on the BELIEVED cell and the true goal never moves.
  * plain desire `(optimize (neg_dist gv) av)`, the shortest program that attributes
    nothing, walks the agent onto the TRUE goal.

The two ending in different cells is the whole point of the figure: unlike the wall and
witness families (whose transient-wall rival reproduces the trajectory exactly, so only
description length separates the readings — that is the `mdl_margin` story), a displaced
goal makes the two readings behaviourally distinguishable.  Note the DL numbers run the
other way here — plain desire is SHORTER (7.8 vs 17.8) — which costs nothing, since
length only ranks programs that already reproduce every scene, and desire does not.

The genuine behavioural experiment, which nobody has run, is in the thesis's limitations
section: enumerate under the converged library over held-out scenes, at the same per-task
budget and success criterion, and report what the SEARCH returns.  That would need a real
enumeration pass here, not `gt_program_str`.

The scenes are drawn from a seed that never entered the run (`--seed`, default 101) and
we assert their matrices are absent from the phase corpus — true, and worth keeping, but
it buys much less than "held out" normally buys, since no learning happens on them.  The
library is reconstructed from an actual PHASE RUN (`run_phase` writes
`phase{1,2}_run[.smoke].json`; we re-stitch its searched sols exactly as `mdl_margin.py`
does), which is what makes the DL numbers the converged library's numbers.

The behavioural divergence is DSL-independent — phase 1 (atomic fork/sync) and phase 2
(decomposed plumbing) are the same machine, so the agent lands on the same cells either
way.  The phase only changes how the mental program is DISPLAYED: through the atomic
`fork(derive, commit)` abstraction (phase 1, the more legible spelling) or its decomposed
product-category plumbing (phase 2).  Default is phase 1, matching `mdl_margin.py` /
`belief_solved.py`.

    # 1. produce a run (slow; HPC): its output is the library we consume
    python phase1.py                        -> phase1_run.json
    python phase2.py                        -> phase2_run.json
    # 2. run the probe on held-out Sally-Anne scenes
    python behavioral_probe.py              phase 1 (reads phase1_run.json)
    python behavioral_probe.py --decomposed phase 2 (reads phase2_run.json)
    python behavioral_probe.py --both       phases 1 and 2, one JSON + PNG each
    python behavioral_probe.py --smoke      use the .smoke run artifact
    python behavioral_probe.py --run PATH   consume a specific run artifact
    python behavioral_probe.py --seed 101   held-out scene seed (default 101)
    python behavioral_probe.py --index 0    which held-out scene to render (default 0)
    python behavioral_probe.py --no-plot    recompute the JSON only
    python behavioral_probe.py out.png      choose the figure path (single phase)

Writes behavioral_probe[.decomposed].json and behavioral_probe[.decomposed].png/.pdf.
"""

import os
import sys
# the analysis modules live in the repo root, one level up from viz/; the sibling
# figure scripts live here, so put both on the path however this file is launched
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import json
from ecd import (
    Deltas, saturate_stitch, rewrite_through_library, mat_key, tr,
)
from dsl import unfold
from prims import make_symmetric_prims
from scenes import as_scenes
from tasks import (
    COMBOS, make_goal_displacement_tasks, belief_rival_specs, _agent_pos,
)
from experiment import verify_ground_truth, gt_program_str, check_decomposition_identities
from mdl_margin import build_corpus, uniform_type_q, _load_found, _remap_names


# ── notation (mirrors illc-mol-thesis/viz.typ) ───────────────────────────────────────
# The probe draws the same object on the same grid as the task figures, so it draws it
# the same way: cells are coloured by the OBJECT ID they carry (the agent and the goal
# keep the colours they have in every other grid in the thesis), a trajectory collapses
# onto ONE grid with an arrow per transition, a cell an object only reaches at the end is
# ghosted in, and a cell the PROGRAM posits rather than the scene supplying it — here the
# believed goal cell — is hollow, dashed and hatched.
PALETTE = {0: '#f4f4f6', 1: '#4e79a7', 2: '#59a14f', 3: '#54585c', 4: '#e1812c',
           5: '#b07aa1', 6: '#e15759', 7: '#76b7b2', 8: '#b8a02e', 9: '#9c755f'}
CELL_EDGE = '#c2c2c8'     # viz.typ's 0.3pt cell stroke
LABEL     = '#6e6e6e'     # viz.typ's luma(110) panel label
BG        = 'white'
GUTTER    = 0.05          # gap between cells, as a fraction of the cell pitch
ROUNDING  = 0.06          # cell corner radius, likewise (viz.typ: 1.2pt on a 20pt cell)


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _lighten(h, p):        # Typst's colour.lighten: mix towards white
    return tuple(c + (1 - c) * p for c in _rgb(h))


def _darken(h, p):         # Typst's colour.darken: mix towards black
    return tuple(c * (1 - p) for c in _rgb(h))


def _col(v):
    return PALETTE.get(int(v), '#cccccc')


# ── DL pricing (identical convention to mdl_margin._dl / phase3_arity._dl) ────────────
def _dl(D, Q, prog):
    t = tr(D, prog) if isinstance(prog, str) else prog
    return float(-D.logp(Q, t))


# ── library reconstruction (mirrors mdl_margin.run: rebuild the run's final library) ──
def _rebuild_library(decomposed, smoke, run_path):
    """Rebuild exactly the library a run converged to, returning (D, Q, tasks,
    corpus_keys, run_prov).  Same reconstruction mdl_margin performs: verify the corpus, re-stitch
    the run's searched sols into the base DSL (atomic fork/sync in phase 1, decomposed
    plumbing in phase 2), remap the run's fn names.  The behavioural result is the same
    either way — the two DSLs are the same machine — but the mental program is displayed
    through the atomic `fork(derive, commit)` abstraction (phase 1) or its decomposed
    plumbing (phase 2)."""
    tasks = build_corpus(smoke, decomposed=decomposed)
    D = Deltas(make_symmetric_prims(decomposed=decomposed))
    verify_ground_truth(D, tasks)
    if decomposed:
        check_decomposition_identities(tasks)
    sols, _found, _rw, run_library, run_prov, _kinds = _load_found(decomposed, smoke,
                                                                   tasks, D, run_path, False)
    saturate_stitch(D, sols, iterations=(3 if smoke else 6), max_arity=5)
    _remap_names(_rw, run_library, D)          # register the run's names on D (side-effect free here)
    Q = uniform_type_q(D)
    corpus_keys = {mat_key(x) for x, _ in tasks}
    print(f"  final library: {len(D)} tokens ({len(D.invented)} invented: "
          f"{[d.repr for d in D.invented]})")
    return D, Q, tasks, corpus_keys, run_prov


# ── the probe over a battery of held-out Sally-Anne scenes ────────────────────────────
def _probe_scene(D, Q, task, m):
    """On one held-out goal-displacement scene, evaluate the mental reading and the
    shortest non-mental rival under the library.  Returns a record with each program's
    library form, DL, and the cell its agent settles on in the final frame.

    A held-out TASK is k scenes sharing one latent program; the probe is a
    single-scene picture (the Sally-Anne figure shows one grid), so it reads the
    task's representative scene.  `m` is that scene's own meta, so `displaced_to`
    below is the believed cell for exactly the scene being drawn."""
    x = as_scenes(task).rep
    av, gv = m['av'], m['gv']
    believed = tuple(m['displaced_to'])
    true_goal = _agent_pos(x[-1], gv)            # the true goal never moves
    observed = _agent_pos(x[-1], av)             # what the agent actually did (= believed)

    # MENTAL: the verified belief compound, rewritten through the run's library.
    mental_str = gt_program_str(D, m)
    (mental_lib,) = rewrite_through_library(D, [mental_str])
    mental_frames = unfold(x[0], x.shape[0], tr(D, mental_str)())
    mental_final = _agent_pos(mental_frames[-1], av)

    # NON-MENTAL: the shortest rival from the discriminating battery, under the library.
    rivals = belief_rival_specs(m)
    rival_libs = rewrite_through_library(D, [s for _, s in rivals])
    scored = sorted(zip(rivals, rival_libs), key=lambda pr: _dl(D, Q, pr[1]))
    (rival_label, rival_str), rival_lib = scored[0]
    rival_frames = unfold(x[0], x.shape[0], tr(D, rival_str)())
    rival_final = _agent_pos(rival_frames[-1], av)

    return {
        'av': av, 'gv': gv, 'dir': '+'.join(m['dirs']), 'T': int(x.shape[0]),
        'believed_cell': list(believed),
        'true_goal': list(true_goal),
        'observed_final': list(observed),
        'mental': {
            'program': mental_str, 'program_lib': mental_lib,
            'dl': _dl(D, Q, mental_lib),
            'final': list(mental_final),
            'searches_believed': mental_final == believed,
        },
        'rival': {
            'label': rival_label, 'program': rival_str, 'program_lib': rival_lib,
            'dl': _dl(D, Q, rival_lib),
            'final': list(rival_final),
            'searches_true': rival_final == true_goal,
        },
        # carried for plotting (not serialised to keep the JSON small)
        '_x': x, '_mental_frames': mental_frames, '_rival_frames': rival_frames,
    }


def compute(decomposed, smoke, run_path, seed, index):
    D, Q, _tasks, corpus_keys, run_prov = _rebuild_library(decomposed, smoke, run_path)

    heldout = make_goal_displacement_tasks(3, COMBOS, seed=seed)
    heldout = [(x, m) for x, m in heldout if mat_key(x) not in corpus_keys]
    if not heldout:
        raise SystemExit(f"no held-out goal-displacement scene at seed {seed} (all in corpus?)")

    recs = [_probe_scene(D, Q, x, m) for x, m in heldout]
    n = len(recs)
    n_mental_pass = sum(r['mental']['searches_believed'] for r in recs)
    n_rival_true  = sum(r['rival']['searches_true'] for r in recs)

    if index >= n:
        raise SystemExit(f"--index {index} out of range ({n} held-out scenes)")
    return {
        # the source run's commit + knobs; the probe's own knobs (held-out seed, index)
        # are the two below
        'provenance': run_prov,
        'phase': 2 if decomposed else 1, 'decomposed': decomposed, 'smoke': bool(smoke),
        'source': run_path or f"phase{2 if decomposed else 1}_run{'.smoke' if smoke else ''}.json",
        'seed': seed, 'index': index,
        'library': [d.repr for d in D.invented],
        'n_heldout': n,
        'n_mental_searches_believed': n_mental_pass,
        'n_rival_searches_true': n_rival_true,
        'records': recs,
    }


def _json_path(decomposed):
    return f"behavioral_probe{'.decomposed' if decomposed else ''}.json"


def run(decomposed=False, smoke=False, run_path=None, seed=101, index=0):
    phase = 2 if decomposed else 1
    print(f"\n{'='*72}\nSALLY-ANNE ILLUSTRATION — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync)"
          f"{' [smoke]' if smoke else ''}\n"
          f"  authored programs rendered on fresh scenes — NOT a test, see the module "
          f"docstring\n{'='*72}")
    out = compute(decomposed, smoke, run_path, seed, index)
    n = out['n_heldout']
    print(f"  {n} goal-displacement scenes (seed {seed}, none in the run's corpus)")
    print(f"  belief compound settles on the BELIEVED cell:  "
          f"{out['n_mental_searches_believed']}/{n}  (analytic — its derive shoves the goal)")
    print(f"  shortest non-mental rival settles on the TRUE cell: "
          f"{out['n_rival_searches_true']}/{n}")
    r = out['records'][index]
    print(f"\n  rendered scene #{index}  (av={r['av']} gv={r['gv']} goal displaced "
          f"{r['dir']}, T={r['T']}):")
    print(f"    true goal   {tuple(r['true_goal'])}   believed cell {tuple(r['believed_cell'])}")
    print(f"    MENTAL  {r['mental']['program_lib']}")
    print(f"      -> agent settles on {tuple(r['mental']['final'])}  "
          f"({'BELIEVED' if r['mental']['searches_believed'] else 'MISS — a real bug'}); "
          f"DL {r['mental']['dl']:.2f} nats")
    print(f"    RIVAL   {r['rival']['program_lib']}   [{r['rival']['label']}]")
    print(f"      -> agent settles on {tuple(r['rival']['final'])}  "
          f"({'TRUE' if r['rival']['searches_true'] else 'other'}); "
          f"DL {r['rival']['dl']:.2f} nats")

    # serialise (drop the numpy frames carried for plotting)
    slim = json.loads(json.dumps(out, default=lambda o: None))
    for rec in slim['records']:
        for k in ('_x', '_mental_frames', '_rival_frames'):
            rec.pop(k, None)
    path = _json_path(decomposed)
    with open(path, 'w') as f:
        json.dump(slim, f, indent=1)
    print(f"  wrote {path}")
    return out


# ── plotting: the held-out scene + the two predicted trajectories, viz.typ's path view ─
# Object identity is the cell value.  Between consecutive frames each cell a value GAINS
# is matched to the nearest cell it LOST (greedy, Manhattan) — that pairing is the move;
# a gain with nothing lost is a growth step, drawn from an adjacent cell the value already
# occupied, and a loss with nothing gained is a deletion.  This is viz.typ's
# `_transitions`, ported so the two renderers say the same thing about the same frames.
def _cells_of(g, v):
    return [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r][c] == v]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _values_in(frames):
    return sorted({int(v) for f in frames for v in f.flatten().tolist() if v != 0})


def _transitions(frames):
    moves, gone = [], []
    for t in range(len(frames) - 1):
        a, b = frames[t], frames[t + 1]
        for v in _values_in((a, b)):
            pa, pb = _cells_of(a, v), _cells_of(b, v)
            added = [p for p in pb if p not in pa]
            lost = [p for p in pa if p not in pb]
            for p in added:
                # prefer a cell the value just vacated; else an adjacent cell it kept
                src = lost if lost else [q for q in pa if _manhattan(q, p) <= 1]
                frm = min(src, key=lambda q: _manhattan(q, p)) if src else None
                if frm is not None:
                    moves.append({'from': frm, 'to': p, 'v': v, 't': t})
                    lost = [q for q in lost if q != frm]
            gone += [{'cell': q, 'v': v, 't': t} for q in lost]
    return moves, gone


def _box(r, c, *, scale=1.0, **kw):
    """One cell-shaped rounded square centred on (r, c), in grid coordinates."""
    import matplotlib.patches as mp
    s = (1 - GUTTER) * scale
    return mp.FancyBboxPatch((c - s / 2, r - s / 2), s, s,
                             boxstyle=f'round,pad=0,rounding_size={ROUNDING * scale}', **kw)


def _draw_cell(ax, r, c, v, *, fs, scale=1.0, ghost=False, z=2):
    """A grid cell carrying value `v`.  `ghost` is viz.typ's washed-out outline: where an
    object ENDS UP, without pretending it is there at t0."""
    col = _col(v)
    if ghost:
        ax.add_patch(_box(r, c, scale=scale, facecolor=_lighten(col, 0.78),
                          edgecolor=_lighten(col, 0.30), lw=0.7, ls=(0, (1, 1.6)),
                          zorder=z))
    else:
        ax.add_patch(_box(r, c, scale=scale, facecolor=col, edgecolor=CELL_EDGE,
                          lw=0.5, zorder=z))
    if v != 0:
        ax.text(c, r, str(int(v)), ha='center', va='center', zorder=z + 1,
                fontsize=fs * scale, fontweight='bold',
                color=_darken(col, 0.15) if ghost else 'white')


def _draw_posited(ax, r, c, ink, *, z=2):
    """viz.typ's POSITED cell: hollow, dashed and hatched, carrying no digit.  The
    believed goal cell is not scenery the scene arrives with — it is a cell the program
    the learner found posits — so it gets the notation the obstacle family's `wall_at`
    cell gets, in the colour of the object posited there."""
    ax.add_patch(_box(r, c, facecolor='none', edgecolor=_lighten(ink, 0.35),
                      lw=0.0, hatch='///', zorder=z))
    ax.add_patch(_box(r, c, facecolor='none', edgecolor=ink, lw=1.0,
                      ls=(0, (2.4, 1.6)), zorder=z + 1))


def _draw_arrow(ax, m, *, lane, cell_pt, z=6):
    """One transition: a shaft with a flat triangular head, in the mover's own colour,
    offset onto its own lane so crossing paths stay readable."""
    import matplotlib.patches as mp
    from math import hypot
    (y1, x1), (y2, x2) = m['from'], m['to']
    dy, dx = y2 - y1, x2 - x1
    length = hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy * lane, ux * lane                  # perpendicular lane offset
    head = min(0.30 * (1 - GUTTER), 6.0 / cell_pt)  # viz.typ: min(size * 0.30, 6pt)
    tail = 0.28 * (1 - GUTTER)                      # keep the shaft clear of the digits
    tip = (x2 - ux * tail + px, y2 - uy * tail + py)
    foot = (x1 + ux * tail + px, y1 + uy * tail + py)
    end = (tip[0] - ux * head, tip[1] - uy * head)
    ink = _darken(_col(m['v']), 0.22)
    ax.plot([foot[0], end[0]], [foot[1], end[1]], color=ink, lw=1.0,
            solid_capstyle='round', zorder=z)
    ax.add_patch(mp.Polygon([tip, (end[0] - uy * head * 0.42, end[1] + ux * head * 0.42),
                             (end[0] + uy * head * 0.42, end[1] - ux * head * 0.42)],
                            closed=True, facecolor=ink, edgecolor='none', zorder=z))


def _draw_strike(ax, g, *, z=6):
    """A value that was there and is not any more (and was not merely moved onto)."""
    y, x = g['cell']
    d, ink = 0.24, _darken(_col(g['v']), 0.22)
    for sx in (1, -1):
        ax.plot([x - d * sx, x + d * sx], [y - d, y + d], color=ink, lw=0.9,
                solid_capstyle='round', zorder=z)


def _path_grid(ax, frames, *, believed, believed_ink, cell_pt, size=5):
    """ONE grid carrying a whole trajectory: the t0 state, an arrow per transition, cells
    reached only later ghosted in.  A single frame just draws the state."""
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)               # row 0 at top
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)

    base, final = frames[0], frames[-1]
    moves, gone = _transitions(frames)
    # a value that disappears because something MOVED ONTO it (the agent reaching the
    # goal) was occluded, not deleted — the incoming arrowhead already says so
    dead = [g for g in gone
            if not any(m['t'] == g['t'] and m['to'] == g['cell'] for m in moves)]
    fs = 0.55 * cell_pt

    for r in range(size):
        for c in range(size):
            v, f = int(base[r][c]), int(final[r][c])
            on_believed = believed is not None and (r, c) == tuple(believed)
            if on_believed:
                _draw_posited(ax, r, c, believed_ink)
            else:
                _draw_cell(ax, r, c, 0, fs=fs)   # the empty cell under everything
            # an occupant of the believed cell sits INSIDE its hatched square, so the
            # marker survives the agent arriving on it — which is the whole point
            scale = 0.72 if on_believed else 1.0
            if v != 0:
                _draw_cell(ax, r, c, v, fs=fs, scale=scale, z=4,
                           ghost=any(g['cell'] == (r, c) for g in dead))
            elif f != 0:
                _draw_cell(ax, r, c, f, fs=fs, scale=scale, z=4, ghost=True)

    movers = [v for v in _values_in(frames) if any(m['v'] == v for m in moves)]
    for m in moves:
        i = movers.index(m['v'])
        lane = 0.0 if len(movers) < 2 else (i - (len(movers) - 1) / 2) * 1.7 / cell_pt
        _draw_arrow(ax, m, lane=lane, cell_pt=cell_pt)
    for g in dead:
        _draw_strike(ax, g)


# The figure carries no headline, no descriptive subtitles, no verdict lines and no
# program strings: the thesis body explains all of it, quotes both programs and gives
# their nats.  What stays is one short label per panel — with three grids side by side
# nothing else says which reading produced which — and the legend, which is what names
# the two object ids and the hatched cell.
def _panel_label(ax, label, fs):
    ax.text(0.0, 1.03, label, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=fs, color=LABEL)


def _legend(ax, entries, *, posited_ink, cell_pt, width):
    """viz.typ's legend row: the swatch as it is drawn in the grids, then its name."""
    ax.set_xlim(0, width); ax.set_ylim(0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    fs = 0.36 * cell_pt
    pad, per_char = 0.34, 0.30 * fs / cell_pt       # crude text metrics, for centring
    span = sum(1 + pad + per_char * len(t) + 0.9 for _, t in entries) - 0.9
    x = max(0.0, (width - span) / 2)
    for swatch, text in entries:
        if swatch is None:
            _draw_posited(ax, 0, x, posited_ink)
        else:
            _draw_cell(ax, 0, x, swatch, fs=0.55 * cell_pt)
        ax.text(x + 0.5 + pad, 0, text, ha='left', va='center', fontsize=fs, color=LABEL)
        x += 1 + pad + per_char * len(text) + 0.9


def plot(out, out_path='behavioral_probe.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matplotlib.rcParams['hatch.linewidth'] = 0.6
    # the thesis sets New Computer Modern; matplotlib ships Computer Modern (cmr10),
    # the same letterform.  Take the first of these that is actually installed, so a
    # machine without NCM renders in CM rather than printing a findfont warning per label.
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    matplotlib.rcParams['font.family'] = [f for f in ('New Computer Modern', 'cmr10',
                                                      'STIX Two Text', 'DejaVu Serif')
                                          if f in have] or ['serif']

    # The held-out scene panel shows the OBSERVED trajectory — the frames a successful
    # program must reproduce — drawn exactly as the task figures draw a scene.  The
    # believed-cell hatch appears only on the belief-compound panel: it is that program's
    # posit, not scenery, so neither the scene nor the beliefless program carries it.
    r = out['records'][out['index']]
    av, gv = r['av'], r['gv']
    believed = tuple(r['believed_cell'])
    panels = [('held-out scene', list(r['_x']), None),
              ('belief compound', list(r['_mental_frames']), believed),
              ('shortest non-mental program', list(r['_rival_frames']), None)]

    # deterministic geometry: fixing the cell size in inches fixes it in POINTS, so every
    # stroke, digit and label can keep viz.typ's ratio to the cell
    size = int(r['_x'].shape[1])
    cell_in, gap_in, margin_in = 0.44, 0.40, 0.16
    label_in, legend_in = 0.30, 0.62
    grid_in = size * cell_in
    fig_w = 3 * grid_in + 2 * gap_in + 2 * margin_in
    fig_h = margin_in + label_in + grid_in + legend_in + margin_in
    cell_pt = cell_in * 72

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    y0 = (margin_in + legend_in) / fig_h
    for i, (label, frames, bel) in enumerate(panels):
        x0 = (margin_in + i * (grid_in + gap_in)) / fig_w
        ax = fig.add_axes([x0, y0, grid_in / fig_w, grid_in / fig_h])
        _path_grid(ax, frames, believed=bel, believed_ink=_col(gv),
                   cell_pt=cell_pt, size=size)
        _panel_label(ax, label, 0.40 * cell_pt)

    lax = fig.add_axes([margin_in / fig_w, margin_in / fig_h,
                        (fig_w - 2 * margin_in) / fig_w, cell_in / fig_h])
    _legend(lax, [(av, 'agent'), (gv, 'goal'), (None, 'believed goal cell')],
            posited_ink=_col(gv), cell_pt=cell_pt,
            width=(fig_w - 2 * margin_in) / cell_in)

    for ext in ({out_path, out_path.rsplit('.', 1)[0] + '.pdf'}):
        fig.savefig(ext, dpi=200, facecolor=fig.get_facecolor())
        print(f"wrote {ext}")


# ── CLI ───────────────────────────────────────────────────────────────────────────────
def _default_png(decomposed):
    return f"behavioral_probe{'.decomposed' if decomposed else ''}.png"


def main(argv):
    smoke = '--smoke' in argv
    both = '--both' in argv
    no_plot = '--no-plot' in argv
    run_path = argv[argv.index('--run') + 1] if '--run' in argv else None
    seed = int(argv[argv.index('--seed') + 1]) if '--seed' in argv else 101
    index = int(argv[argv.index('--index') + 1]) if '--index' in argv else 0
    out_png = next((a for a in argv[1:] if a.endswith('.png')), None)

    if both:
        if run_path is not None:
            sys.exit("--run names a single phase artifact; drop --both or run each phase "
                     "separately with its own --run.")
        for decomposed in (False, True):
            out = run(decomposed=decomposed, smoke=smoke, seed=seed, index=index)
            if not no_plot:
                plot(out, _default_png(decomposed))
    else:
        decomposed = '--decomposed' in argv
        out = run(decomposed=decomposed, smoke=smoke, run_path=run_path, seed=seed, index=index)
        if not no_plot:
            plot(out, out_png or _default_png(decomposed))


if __name__ == '__main__':
    main(sys.argv)
