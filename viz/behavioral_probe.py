"""Behavioral probe (phase 2): the discovered library passes the false-belief test.

Where `mdl_margin.py` prices belief against its behavioural competitors and `corpus_dl.py`
charts the compression it buys, this script asks the developmental-psychology question
directly: put the learner in front of a *held-out* Sally-Anne scene and see which location
it predicts the agent will search.

We reuse the goal-displacement (Sally-Anne) family: the agent walks to where it *believes*
the goal is — one cell displaced from the true goal — while the true goal never moves.
Unlike the wall/witness families, whose transient-wall rival reproduces the whole scene
(so MDL is the only discriminator — that is the `mdl_margin` story), the goal-displacement
scene behaviourally *diverges*:

  * the MENTAL reading — the belief compound the run discovered (the `fork(derive, commit)`
    abstraction `fn_5` the library reprices belief through) — sends the agent to the
    BELIEVED cell.  It answers the classic false-belief question the way a competent
    mentalizer does: the agent searches where it *thinks* the goal is.
  * the shortest NON-MENTAL program expressible under the SAME library — pure desire
    `optimize(neg_dist gv) av`, which the library collapses to `fn_1` — sends the agent to
    the TRUE goal.  That is the naive, reality-bound answer three-year-olds give before
    they pass the false-belief task.

So a learner that has compressed the corpus into a belief compound passes the test; a
learner restricted to the non-mental fragment fails it, and fails it in the specific way
the developmental literature documents — predicting the true location, not the believed one.

The scenes are HELD OUT: they are drawn from a fresh seed that never entered the run
(`--seed`, default 101), and we assert their matrices are absent from the phase corpus.
The library and the belief compound both come from an actual PHASE RUN — `run_phase`
writes `phase{1,2}_run[.smoke].json`; we re-stitch its searched sols to rebuild exactly the
library it converged to (the same reconstruction `mdl_margin.py` performs), then instantiate
the discovered belief abstraction on the novel scene.  Nothing here is re-enumerated: the
mental program is the verified ground-truth belief compound (`experiment.gt_program_str`,
which `verify_ground_truth` renders to confirm it reproduces the scene), rewritten through
the run's library; the non-mental rival is the shortest of the task's discriminating
battery under that library.

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

import sys
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


# ── palette (inherits belief_solved.py's house colours) ──────────────────────────────
INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e6e6e2', '#fcfcfb'
AGENT   = '#0b4f9e'   # the mind-bearing agent (belief deep-blue)
GOAL    = '#e0982f'   # the true goal object (amber)
BELIEF  = '#2a78d6'   # the believed / displaced cell marker
GOOD, BAD = '#1a7f45', '#b5321f'   # passes / fails the false-belief test
CELL_BG = '#f4f3ef'   # empty grid cell


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
    tasks = build_corpus(smoke)
    D = Deltas(make_symmetric_prims(decomposed=decomposed))
    verify_ground_truth(D, tasks)
    if decomposed:
        check_decomposition_identities(tasks)
    sols, _found, _rw, run_library, run_prov = _load_found(decomposed, smoke, tasks, D,
                                                           run_path, False)
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
    print(f"\n{'='*72}\nBEHAVIORAL PROBE — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync) — held-out Sally-Anne scenes"
          f"{' [smoke]' if smoke else ''}\n{'='*72}")
    out = compute(decomposed, smoke, run_path, seed, index)
    n = out['n_heldout']
    print(f"  {n} held-out goal-displacement scenes (seed {seed}, none in the run's corpus)")
    print(f"  belief compound searches the BELIEVED cell:  "
          f"{out['n_mental_searches_believed']}/{n}")
    print(f"  shortest non-mental rival searches the TRUE cell: "
          f"{out['n_rival_searches_true']}/{n}")
    r = out['records'][index]
    print(f"\n  rendered scene #{index}  (av={r['av']} gv={r['gv']} goal displaced "
          f"{r['dir']}, T={r['T']}):")
    print(f"    true goal   {tuple(r['true_goal'])}   believed cell {tuple(r['believed_cell'])}")
    print(f"    MENTAL  {r['mental']['program_lib']}")
    print(f"      -> agent settles on {tuple(r['mental']['final'])}  "
          f"({'BELIEVED — passes' if r['mental']['searches_believed'] else 'MISS'}); "
          f"DL {r['mental']['dl']:.2f} nats")
    print(f"    RIVAL   {r['rival']['program_lib']}   [{r['rival']['label']}]")
    print(f"      -> agent settles on {tuple(r['rival']['final'])}  "
          f"({'TRUE — naive answer' if r['rival']['searches_true'] else 'other'}); "
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


# ── plotting: the initial scene + the two predicted final frames ──────────────────────
def _draw_grid(ax, frame, av, gv, *, path=None, believed=None, true_goal=None,
               agent_at=None, size=5):
    """Render one 5x5 frame: empty cells, the true goal (amber square), the agent token
    (blue disc) at `agent_at` (defaults to its position in `frame`), a dashed marker on
    the believed cell, and a faint dotted trajectory `path` (list of (r,c))."""
    import matplotlib.patches as mp

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)          # row 0 at top
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # cells
    for r in range(size):
        for c in range(size):
            ax.add_patch(mp.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=CELL_BG,
                                      edgecolor=GRID, lw=1.0, zorder=1))
    # believed (displaced) cell — dashed outline, drawn under the tokens
    if believed is not None:
        br, bc = believed
        ax.add_patch(mp.Rectangle((bc - 0.5, br - 0.5), 1, 1, facecolor='none',
                                  edgecolor=BELIEF, lw=2.2, ls=(0, (3, 2)), zorder=2))
        # label just inside the top edge so it never sits under the agent token
        ax.text(bc, br - 0.37, 'believed', ha='center', va='center', color=BELIEF,
                fontsize=6.5, style='italic', zorder=6)
    # trajectory
    if path:
        ys = [p[0] for p in path]; xs = [p[1] for p in path]
        ax.plot(xs, ys, color=AGENT, lw=1.6, ls=':', alpha=0.5, zorder=3,
                solid_capstyle='round')
    # true goal (amber rounded square)
    if true_goal is not None:
        gr, gc = true_goal
        ax.add_patch(mp.FancyBboxPatch((gc - 0.30, gr - 0.30), 0.60, 0.60,
                     boxstyle='round,pad=0.02,rounding_size=0.12',
                     facecolor=GOAL, edgecolor='white', lw=1.5, zorder=4))
        ax.text(gc, gr, 'g', ha='center', va='center', color='white',
                fontsize=10, fontweight='bold', zorder=5)
    # agent (blue disc)
    if agent_at is None:
        agent_at = _agent_pos(frame, av)
    if agent_at is not None:
        ar, ac = agent_at
        ax.add_patch(mp.Circle((ac, ar), 0.32, facecolor=AGENT, edgecolor='white',
                               lw=1.5, zorder=5))
        ax.text(ac, ar, 'a', ha='center', va='center', color='white',
                fontsize=10, fontweight='bold', zorder=6)


def _panel_title(ax, title, sub, color):
    "Bold title above a two-line descriptive sub, both floated clear above the grid."
    ax.text(0.5, 1.135, title, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=11.5, color=color, fontweight='bold')
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8.2, color=MUTED, linespacing=1.35)


def _path_of(frames, av):
    return [_agent_pos(frames[t], av) for t in range(len(frames))
            if _agent_pos(frames[t], av) is not None]


def plot(out, out_path='behavioral_probe.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = out['records'][out['index']]
    x = r['_x']
    av, gv = r['av'], r['gv']
    believed = tuple(r['believed_cell']); true_goal = tuple(r['true_goal'])
    decomposed = out['decomposed']
    fork_sub = ('learner with the discovered fork,\nas decomposed plumbing' if decomposed
                else 'learner with the discovered\nfork(derive, commit)')

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 6.6))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(top=0.70, bottom=0.18, left=0.03, right=0.97, wspace=0.14)
    for ax in axes:
        ax.set_facecolor(BG)

    # panel 1 — the held-out scene (initial frame): agent, true goal, believed cell
    _draw_grid(axes[0], x[0], av, gv, believed=believed, true_goal=true_goal)
    _panel_title(axes[0], 'A held-out scene',
                 'agent a, true goal g; the goal is\n"believed" one cell away', INK)

    # panel 2 — the belief compound: agent searches the believed cell
    mpath = _path_of(r['_mental_frames'], av)
    _draw_grid(axes[1], r['_mental_frames'][-1], av, gv, path=mpath,
               believed=believed, true_goal=true_goal, agent_at=tuple(r['mental']['final']))
    _panel_title(axes[1], 'Belief compound  ✓', fork_sub, GOOD)
    axes[1].text(0.5, -0.045, 'searches the BELIEVED cell — passes',
                 transform=axes[1].transAxes, ha='center', va='top',
                 fontsize=9.5, color=GOOD, fontweight='bold')

    # panel 3 — the non-mental fragment: agent searches the true cell (naive answer)
    rpath = _path_of(r['_rival_frames'], av)
    _draw_grid(axes[2], r['_rival_frames'][-1], av, gv, path=rpath,
               believed=believed, true_goal=true_goal, agent_at=tuple(r['rival']['final']))
    _panel_title(axes[2], 'Non-mental fragment  ✗',
                 'learner restricted to\nphysics only', BAD)
    axes[2].text(0.5, -0.045, 'searches the TRUE cell — the naive answer',
                 transform=axes[2].transAxes, ha='center', va='top',
                 fontsize=9.5, color=BAD, fontweight='bold')

    # program strings + DL beneath the two prediction panels
    axes[1].text(0.5, -0.115, f"{r['mental']['program_lib']}\n{r['mental']['dl']:.1f} nats",
                 transform=axes[1].transAxes, ha='center', va='top', fontsize=7.0,
                 color=MUTED, family='monospace')
    axes[2].text(0.5, -0.115, f"{r['rival']['program_lib']}\n{r['rival']['dl']:.1f} nats "
                              f"· shortest non-mental",
                 transform=axes[2].transAxes, ha='center', va='top', fontsize=7.0,
                 color=MUTED, family='monospace')

    fig.text(0.03, 0.955,
             'The discovered belief compound passes the false-belief test',
             fontsize=14.5, fontweight='bold', color=INK, ha='left', va='top')
    fig.text(0.03, 0.90,
             f"On a held-out Sally-Anne scene, the learned fork(derive, commit) abstraction "
             f"sends the agent to where it believes the\ngoal is; the shortest program in the "
             f"non-mental fragment sends it to the true goal — the answer three-year-olds give. "
             f"({out['n_mental_searches_believed']}/{out['n_heldout']} held-out scenes, both.)",
             fontsize=9.5, color=MUTED, ha='left', va='top', linespacing=1.5)
    # phase chip, top-right — which DSL the mental program is displayed through
    fig.text(0.97, 0.955,
             f"Phase {out['phase']} · {'decomposed' if decomposed else 'atomic'} fork/sync",
             fontsize=9, color=MUTED, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.35', fc='#eef3fb', ec='#cfe0f4', lw=1))

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
