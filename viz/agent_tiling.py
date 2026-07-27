"""Agent-constructor tiling figure (phases 1-2): (gv,av) × abstraction-used.

`mdl_margin.py` prices belief against its non-mental rivals; this script shows the flip
side of the same coin — that the win is not a per-task fluke but ONE reused abstraction.
The stitch discovers a single agent constructor and every belief task, whatever its goal
value `gv` and agent value `av`, spells its found program through that one token:

        found belief program  =  (fn_k  av  gv  policy)          # fn_k = the agent constructor

The figure is a heatmap.  Each ROW is a (gv,av) combination the curriculum poses; each
COLUMN is an invented library abstraction (fn_3, fn_4, …).  A cell counts the solved
belief tasks at that (gv,av) whose library-rewritten program uses that abstraction.  The
agent-constructor column lights up for EVERY (gv,av) row — the abstraction tiles the whole
combination space — while the other invented tokens (used by the non-mental corpus) stay
dark for belief.  That single filled column is the picture of a discovered agent concept.

Everything is DERIVED FROM AN ACTUAL RUN, not ground truth.  `experiment.run_phase` writes
`phase{1,2}_run[.smoke].json` with, per solved task, both the found program and its
library-rewritten form (`rewritten`) — the exact spelling the run's stitch produced,
naming its own invented tokens (`library`).  We rebuild the phase corpus with
`mdl_margin.build_corpus` to recover each belief task's (gv,av) and variant, match it to
the run by matrix key, and read which library tokens its rewritten program actually
contains.  So the tiling is what the run compressed to, not a reconstruction of it.

    # 1. produce a run (slow; HPC): its output is the artifact we consume
    python phase1.py                     -> phase1_run.json
    python phase2.py                     -> phase2_run.json
    # 2. build the tiling figure
    python agent_tiling.py               phase 1 (reads phase1_run.json)
    python agent_tiling.py --decomposed  phase 2 (reads phase2_run.json)
    python agent_tiling.py --both        phases 1 and 2, one JSON each, one PNG
    python agent_tiling.py --smoke       use the .smoke run artifacts
    python agent_tiling.py --run PATH    consume a specific run artifact
    python agent_tiling.py --no-plot     recompute the JSON only
    python agent_tiling.py out.png       choose the figure path

Writes agent_tiling[.decomposed].json and agent_tiling.png/.pdf.
"""

import re
import sys
import json
from collections import Counter, defaultdict

from ecd import mat_key_id
from tasks import belief_variant
from mdl_margin import build_corpus


INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e6e6e2', '#fcfcfb'
HERO = '#2a78d6'        # the belief blue — the agent-constructor column
_FN_RE = re.compile(r'\bfn_\d+\b')

# variant marker glyphs, so a cell can show WHICH belief families sit at each (gv,av).
VAR_MARK = {'belief_wall': '■', 'belief_witness': '◆', 'belief_goal': '▲',
            'belief_observers': '●', 'belief_false_obstacle': '★'}
VAR_LABEL = {'belief_wall': 'wall', 'belief_witness': 'witness',
             'belief_goal': 'goal', 'belief_observers': 'two observers',
             'belief_false_obstacle': 'false-obstacle'}


# ── data loading: a phase run artifact (found + rewritten programs, keyed by matrix) ──
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
            f"(it writes the *_run.json with the found + rewritten programs).")
    if bool(art.get('decomposed')) != bool(decomposed):
        raise SystemExit(f"{run_path} is a phase {2 if art.get('decomposed') else 1} run, "
                         f"but this invocation is phase {2 if decomposed else 1}.")
    return art, run_path


# ── the tiling ──────────────────────────────────────────────────────────────────────
def compute(decomposed, smoke, run_path):
    """For every solved belief task, read which invented library tokens its rewritten
    program uses, and tabulate them against the task's (gv,av).  Identify the agent
    constructor as the invented token most belief programs share."""
    art, run_path = _load_run(decomposed, smoke, run_path)
    sols = art['sols']
    rewritten = art.get('rewritten', {})
    library = list(art.get('library', []))
    lib_set = set(library)
    tasks = build_corpus(smoke)

    # combo -> abstraction -> count;  combo -> variant markers;  per-program token record.
    cell = defaultdict(Counter)                 # (gv,av) -> Counter(abstraction -> n tasks)
    combo_variants = defaultdict(set)           # (gv,av) -> {variant}
    combo_solved = Counter()                    # (gv,av) -> n solved belief tasks
    token_uses = Counter()                      # abstraction -> n belief programs using it
    n_solved = 0
    for x, m in tasks:
        if m['kind'] != 'belief':
            continue
        kid = mat_key_id(x)
        if kid not in sols:
            continue
        n_solved += 1
        combo = (m['gv'], m['av'])
        combo_solved[combo] += 1
        combo_variants[combo].add(belief_variant(m))
        toks = set(_FN_RE.findall(rewritten.get(kid, ''))) & lib_set
        for t in toks:
            cell[combo][t] += 1
            token_uses[t] += 1

    # the agent constructor: the invented token the most belief programs spell through.
    agent_ctor = token_uses.most_common(1)[0][0] if token_uses else None
    combos = sorted(cell)
    # column order: agent constructor first, then the remaining library tokens (stable).
    cols = ([agent_ctor] if agent_ctor else []) + [t for t in library if t != agent_ctor]

    # does the agent-constructor tile every solved combo?  (used per combo == solved there)
    tiled = bool(agent_ctor) and all(
        cell[c].get(agent_ctor, 0) == combo_solved[c] for c in combos) and bool(combos)

    return {
        'phase':      2 if decomposed else 1,
        'decomposed': decomposed,
        'smoke':      bool(smoke),
        'source':     run_path,
        'library':    library,
        'agent_ctor': agent_ctor,
        'agent_ctor_uses': token_uses.get(agent_ctor, 0),
        'n_solved_belief': n_solved,
        'tiled':      tiled,
        'columns':    cols,
        'combos':     [{'gv': g, 'av': a,
                        'solved': combo_solved[(g, a)],
                        'variants': sorted(combo_variants[(g, a)]),
                        'counts': {t: cell[(g, a)].get(t, 0) for t in cols}}
                       for (g, a) in combos],
    }


def run(decomposed=False, smoke=False, run_path=None):
    phase = 2 if decomposed else 1
    print(f"\n{'='*72}\nAGENT-CONSTRUCTOR TILING — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync){' [smoke]' if smoke else ''}"
          f"\n{'='*72}")
    out = compute(decomposed, smoke, run_path)
    if out['agent_ctor'] is None:
        print("  no belief program used an invented abstraction in this run "
              "(nothing to tile — is this a solved, stitched run?).")
    else:
        print(f"  agent constructor: {out['agent_ctor']}  "
              f"(used by {out['agent_ctor_uses']}/{out['n_solved_belief']} solved belief "
              f"programs, over {len(out['combos'])} (gv,av) combos)")
        for c in out['combos']:
            marks = ' '.join(f"{VAR_MARK[v]}{VAR_LABEL[v]}" for v in c['variants'])
            row = '  '.join(f"{t}:{c['counts'][t]}" for t in out['columns'])
            print(f"    gv{c['gv']}·av{c['av']}  solved {c['solved']:>2}  [{row}]   {marks}")
        print(f"  agent constructor tiles every solved (gv,av) cell: {out['tiled']}")
    path = f"agent_tiling{'.decomposed' if decomposed else ''}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {path}")
    return out


# ── plotting ──────────────────────────────────────────────────────────────────────────
def _blue(frac):
    "white → belief-blue sequential ramp for a fill fraction in [0,1]."
    frac = max(0.0, min(1.0, frac))
    r0, g0, b0 = 0xfc, 0xfc, 0xfb      # near-white
    r1, g1, b1 = 0x2a, 0x78, 0xd6      # HERO
    return (
        (r0 + (r1 - r0) * frac) / 255,
        (g0 + (g1 - g0) * frac) / 255,
        (b0 + (b1 - b0) * frac) / 255,
    )


def _panel(ax, d):
    "Heatmap: (gv,av) rows × abstraction columns; the agent constructor is a full column."
    combos, cols = d['combos'], d['columns']
    if not combos or not cols:
        ax.text(0.5, 0.5, "no stitched belief programs in this run",
                transform=ax.transAxes, ha='center', va='center', color=MUTED, fontsize=10)
        ax.set_axis_off()
        return

    nrow, ncol = len(combos), len(cols)
    vmax = max((c['solved'] for c in combos), default=1) or 1

    for i, c in enumerate(combos):           # rows top→bottom
        y = nrow - 1 - i
        for j, t in enumerate(cols):
            n = c['counts'][t]
            ax.add_patch(_rect(j, y, _blue(n / vmax)))
            if n:
                ax.text(j, y, str(n), ha='center', va='center',
                        color='white' if n / vmax > 0.5 else INK, fontsize=9,
                        fontweight='bold', zorder=4)

    # row labels (gv·av on the left) + variant glyphs (to the RIGHT of the grid)
    for i, c in enumerate(combos):
        y = nrow - 1 - i
        marks = ' '.join(VAR_MARK[v] for v in c['variants'])
        ax.text(-0.65, y, f"gv{c['gv']}·av{c['av']}", ha='right', va='center',
                fontsize=8.6, color=INK)
        ax.text(ncol - 0.35, y, marks, ha='left', va='center', fontsize=8.5, color=MUTED)
    for j, t in enumerate(cols):
        hero = (t == d['agent_ctor'])
        ax.text(j, nrow - 0.35, t, ha='center', va='bottom', rotation=0,
                fontsize=9 if hero else 8.2, color=HERO if hero else MUTED,
                fontweight='bold' if hero else 'normal')

    # frame the agent-constructor column to name the tiling
    if d['agent_ctor'] in cols:
        jc = cols.index(d['agent_ctor'])
        ax.add_patch(_frame(jc, -0.5, nrow))
        ax.annotate('agent constructor\ntiles every (gv,av)',
                    xy=(jc, -0.5), xytext=(jc, -1.35),
                    ha='center', va='top', fontsize=8.6, color=HERO, fontweight='bold',
                    arrowprops=dict(arrowstyle='-|>', color=HERO, lw=1.6))

    # variant-glyph legend, down the right-hand gutter beneath the marks column
    present = [v for v in VAR_MARK if any(v in c['variants'] for c in combos)]
    ax.text(ncol - 0.35, nrow - 0.35, 'variants', fontsize=8, color=MUTED,
            ha='left', va='bottom')
    for k, v in enumerate(present):
        ax.text(ncol - 0.35, -0.65 - 0.42 * k, f"{VAR_MARK[v]} {VAR_LABEL[v]}",
                fontsize=8, color=MUTED, ha='left', va='top')

    ax.set_xlim(-2.7, ncol + 1.5)
    ax.set_ylim(-1.9, nrow + 0.4)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.text(-2.65, nrow - 0.5, '(gv, av)', fontsize=8.5, color=MUTED, ha='left', va='bottom')
    ax.text(-2.65, -0.6, 'invented library abstraction →', fontsize=8.5, color=MUTED,
            ha='left', va='top')


def _rect(cx, cy, color):
    import matplotlib.patches as mp
    return mp.Rectangle((cx - 0.46, cy - 0.46), 0.92, 0.92, facecolor=color,
                        edgecolor=GRID, lw=0.8, zorder=2)


def _frame(cx, y0, nrow):
    import matplotlib.patches as mp
    return mp.Rectangle((cx - 0.5, y0), 1.0, nrow, facecolor='none',
                        edgecolor=HERO, lw=2.2, zorder=5)


def plot(datasets, out_path='agent_tiling.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n + 2.4, 6.4), squeeze=False)
    fig.patch.set_facecolor(BG)
    for col, d in enumerate(datasets):
        kind = 'atomic fork/sync' if not d['decomposed'] else 'decomposed plumbing'
        ax = axes[0, col]
        ax.set_facecolor(BG)
        ctor = d['agent_ctor'] or '—'
        ax.set_title(f"Phase {d['phase']}  —  {kind}   (agent constructor: {ctor})",
                     fontsize=11, color=INK, loc='left', pad=8)
        _panel(ax, d)

    top = 1 - 1.0 / fig.get_figheight()
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.text(0.02, 1 - 0.30 / fig.get_figheight(),
             'One discovered abstraction is the agent — it tiles every (gv, av) belief task',
             fontsize=13.5, fontweight='bold', color=INK, ha='left', va='top')
    fig.text(0.02, 1 - 0.56 / fig.get_figheight(),
             'Rows are the (goal-value, agent-value) combinations the curriculum poses; '
             'columns are\nthe invented library abstractions. A cell counts solved belief '
             'programs whose rewritten\nform uses that abstraction — the agent-constructor '
             'column is filled for every combination.',
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
    out_path = next((a for a in argv[1:] if a.endswith('.png')), 'agent_tiling.png')

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
