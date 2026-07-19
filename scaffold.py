"""Results-side SCAFFOLD identifier — which tasks/abstractions built up to belief.

The curriculum no longer *labels* any task a "scaffold": `tasks.py` is one
undifferentiated bag, and which rungs actually seeded belief is meant to be read off a
run's results rather than stipulated in code (see the one-bag collapse — nothing tags
`obstacle` or `desire` as belief's donor).  This script recovers that structure from a
phase run's search output alone.

The claim it makes concrete is that belief's solved program is *assembled* from the
solutions of shallower tasks.  Two mechanisms, both discovered from data, both priced on
the run's own programs (`phase{1,2}_run[.smoke].json`):

  (1) STRUCTURAL containment.  A lower-level task is a scaffold for a belief task when the
      whole capability it certifies — its solved program, with concrete cell values /
      coordinates abstracted to typed holes — occurs as a SUBTERM of the belief program.
      This is theory-independent: it holds of the base-primitive solutions, before any
      library exists.  Examples the run surfaces on its own:
        * desire   `(optimize (neg_dist #v) #v)`                    → belief's action policy
        * obstacle `(compose (wall_at #c #c) (optimize (neg_dist #v) #v))`
                                                                    → belief's fork DERIVE
        * registration `(sync_to_world #v)`                         → belief's agency COMMIT
        * plain wall-belief `(fork (compose … ) (sync_to_world #v))`→ a PROPER subterm of
                                                                      witness-belief
      A donor can therefore be a non-mental task OR a shallower belief variant; the
      belief→belief edges are exactly the "plain belief seeds witness/goal belief"
      curriculum story, recovered rather than asserted.

  (2) ABSTRACTION sharing.  The joint stitch invents abstractions and rewrites every
      solution through them.  When a belief program and a NON-belief family's program are
      rewritten through the SAME invented token, that token is shared reuse the MDL
      objective actually paid for — the strongest form of "this rung fed belief".  We
      report each abstraction's role, its belief users, and the non-belief families that
      co-induced it (its donors).

The headline output is an ASSEMBLY view: one belief solution per variant, printed as its
program tree with each maximal donor-covered fragment annotated by the family (and role)
that certifies it — belief read back as a composition of the rungs below it.

    # 1. produce a run (its artifact is what we consume)
    python phase1.py                      -> phase1_run.json
    python phase2.py                      -> phase2_run.json
    # 2. identify the scaffold from that run
    python scaffold.py                    phase 1 (reads phase1_run.json)
    python scaffold.py --decomposed       phase 2 (reads phase2_run.json)
    python scaffold.py --both             phases 1 and 2
    python scaffold.py --smoke            use the .smoke run artifacts
    python scaffold.py --run PATH         consume a specific run artifact

Writes scaffold[.decomposed][.smoke].json.
"""

import sys
import json
import re as _re
from copy import deepcopy
from collections import Counter, defaultdict

from ecd import (
    Deltas, saturate_stitch, tr, normalize, simplify, mat_key_id,
)
from dsl import coord, cellvalue, Delta
from prims import make_symmetric_prims
from tasks import belief_variant
from experiment import provenance_line, _abstraction_role
from mdl_margin import build_corpus, _remap_names


# ── canonical program SHAPES (literal-abstracted subterm identity) ──────────────────
# A shape is the s-expression with every *content* leaf replaced by a per-type
# placeholder: cell values (agent/goal ids, step magnitudes) -> #v, grid coordinates
# (the invisible wall latent) -> #c.  Two programs share a shape iff they are the same
# skeleton up to which concrete values/positions they name — exactly the granularity the
# scaffold claim lives at ("desire's policy", not "desire's policy for gv=5,av=4").  A
# bound-variable hole ($i) in an abstraction body carries its own type, so an abstraction
# and a concrete subterm canonicalise to the SAME shape when they are the same skeleton.
def _canon(node):
    if not isinstance(node, Delta):
        return str(node)
    if not node.tails:
        if node.type == coord:
            return '#c'
        if node.type == cellvalue:
            return '#v'
        return node.repr                      # directions, nullary commits (overlay, …)
    return '(' + node.repr + ' ' + ' '.join(_canon(t) for t in node.tails) + ')'


def _subtrees(node, depth=0):
    "every Delta subtree with its depth (0 = the root)."
    if not isinstance(node, Delta):
        return
    yield node, depth
    for t in (node.tails or []):
        yield from _subtrees(t, depth + 1)


def _frag_role(shape):
    """Semantic role of a matched fragment, keyed off its shape string.  Mirrors the
    labels experiment._abstraction_role gives invented abstractions, so a donated
    fragment and the abstraction it becomes read as the same thing in the report."""
    has_fork = 'fork' in shape or ('pipe_gpg' in shape and 'compose_gp' in shape)
    has_sync = 'sync_to_world' in shape or 'register' in shape
    if has_fork and has_sync and 'wall_at' in shape:
        return 'agent_constructor (belief)'
    if has_fork and has_sync:
        return 'agent block (fork ∧ commit)'
    if 'clear_at' in shape and 'wall_at' in shape:
        return 'relocate prefix (clear ▸ stamp)'
    if 'wall_at' in shape and ('neg_dist' in shape or 'optimize' in shape):
        return "belief's derive (wall policy)"
    if has_sync:
        return 'agency commit'
    if 'distance' in shape and 'optimize' in shape:
        return 'flee policy'
    if 'neg_dist' in shape or 'optimize' in shape:
        return 'action policy (seek)'
    if 'step' in shape:
        return 'physics step'
    if 'overlay' in shape or 'underlay' in shape:
        return 'motion-trail commit'
    if 'wall_at' in shape:
        return 'wall stamp'
    if 'clear_at' in shape or 'erase' in shape:
        return 'grid edit'
    return 'fragment'


# ── loading: a phase run's searched programs + the library it converged to ──────────
def _load(decomposed, smoke, run_path):
    """Return (art, D, sols_str, rewritten, meta_of).

      art        — the raw run artifact (for provenance / kinds).
      D          — base DSL with the run's library re-stitched back in (bodies + names).
      sols_str   — {kid: base-primitive solution string}.
      rewritten  — {kid: library-rewritten string}, remapped onto D's reconstructed names.
      meta_of    — {kid: task metadata} from the regenerated corpus (for belief variants).

    Re-stitching the run's own searched sols reproduces exactly the library it converged
    to (saturate_stitch re-discovers abstractions deterministically from the expanded
    sols — the reproducibility mdl_margin / corpus_dl already rely on).  The run named its
    tokens with an offset from earlier wake-sleep rounds, so we remap the rewritten strings
    onto the reconstructed names, positionally, exactly as mdl_margin does."""
    if run_path is None:
        run_path = f"phase{2 if decomposed else 1}_run{'.smoke' if smoke else ''}.json"
    try:
        with open(run_path) as f:
            art = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"no run artifact '{run_path}' — run `python phase{2 if decomposed else 1}.py"
            f"{' --smoke' if smoke else ''}` first.")
    if bool(art.get('decomposed')) != bool(decomposed):
        raise SystemExit(f"{run_path} is a phase {2 if art.get('decomposed') else 1} run, "
                         f"but this invocation is phase {2 if decomposed else 1}.")

    D = Deltas(make_symmetric_prims(decomposed=decomposed))
    sols = {kid: tr(D, s) for kid, s in art['sols'].items()}
    stitch_iters = int(((art.get('provenance') or {}).get('knobs') or {})
                       .get('stitch_iters', 3 if smoke else 6))
    saturate_stitch(D, sols, iterations=stitch_iters, max_arity=5)
    rewritten = _remap_names(dict(art.get('rewritten', {})), list(art.get('library', [])), D)

    # regenerate the corpus to recover per-task metadata (belief variant labels), keyed by
    # the same stable mat_key_id the artifact uses.  Sizes/seeds match run_phase, so the
    # keys line up (the coupling mdl_margin.build_corpus already documents).
    meta_of = {mat_key_id(x): m for x, m in build_corpus(smoke)}
    return art, D, dict(art['sols']), rewritten, meta_of


def _label_of(kid, kinds, meta_of):
    "family label: the fine belief variant when known, else the coarse kind."
    m = meta_of.get(kid)
    if m is not None and m.get('kind') == 'belief':
        return belief_variant(m)
    return (m or {}).get('kind') or kinds.get(kid, '?')


# ── the analysis ────────────────────────────────────────────────────────────────────
def analyse(art, D, sols_str, rewritten, meta_of):
    kinds = art.get('kinds', {})
    label_of = {kid: _label_of(kid, kinds, meta_of) for kid in sols_str}
    is_belief = {kid: (meta_of.get(kid, {}).get('kind') == 'belief'
                       or kinds.get(kid) == 'belief') for kid in sols_str}

    # parse every solved program to a canonical base-primitive tree.
    trees = {kid: simplify(normalize(tr(D, s))) for kid, s in sols_str.items()}

    # ── belief targets: their proper-subterm shapes (depth ≥ 1) + all shapes ──────────
    targets = {}          # kid -> {'var', 'proper': {shape: min_depth}, 'all': set(shape)}
    for kid, t in trees.items():
        if not is_belief[kid]:
            continue
        proper, allsh = {}, set()
        for node, depth in _subtrees(t):
            sh = _canon(node)
            allsh.add(sh)
            if depth >= 1:
                proper[sh] = min(depth, proper.get(sh, depth))
        targets[kid] = {'var': label_of[kid], 'proper': proper, 'all': allsh}

    # ── donors: each solved COMPOUND program's whole shape ────────────────────────────
    # A donor is any solved task whose program is compound (has structure to reuse); its
    # whole-program shape is the capability it certifies.  We match that whole shape
    # against belief's proper subterms — "this task's entire solution sits inside belief".
    donor_shape = {}          # kid -> shape (compound donors only)
    shape_labels = defaultdict(set)   # shape -> set(family label) offering it as a whole sol
    shape_kids = defaultdict(set)     # shape -> set(kid)
    for kid, t in trees.items():
        if not (isinstance(t, Delta) and t.tails):
            continue                  # bare nullary sols (snd_gg / sync_all / underlay)
        sh = _canon(t)
        donor_shape[kid] = sh
        shape_labels[sh].add(label_of[kid])
        shape_kids[sh].add(kid)

    # ── structural edges: donor whole-shape ⊆ belief proper subterms ──────────────────
    edges = []
    by_family = {}            # donor label -> aggregate
    for dk, sh in donor_shape.items():
        for tk, tinfo in targets.items():
            if dk == tk or sh not in tinfo['proper']:
                continue
            depth = tinfo['proper'][sh]
            edges.append({'donor_label': label_of[dk], 'donor_kid': dk,
                          'target_kid': tk, 'target_variant': tinfo['var'],
                          'shape': sh, 'role': _frag_role(sh), 'depth': depth})
            rec = by_family.setdefault(label_of[dk], {
                'targets': set(), 'variants': Counter(), 'fragments': Counter(),
                'roles': set(), 'min_depth': depth})
            rec['targets'].add(tk)
            rec['variants'][tinfo['var']] += 1
            rec['fragments'][sh] += 1
            rec['roles'].add(_frag_role(sh))
            rec['min_depth'] = min(rec['min_depth'], depth)

    n_belief = len(targets)

    # ── abstraction sharing: which invented tokens belief and non-belief co-use ───────
    abst_names = [d.repr for d in D.invented]

    def _absts_in(s):
        return {a for a in abst_names if s and _re.search(rf'\b{a}\b', s)}

    belief_all_shapes = set().union(*(t['all'] for t in targets.values())) if targets else set()
    abst_body = {d.repr: str(simplify(normalize(deepcopy(d)))) for d in D.invented}
    abst_shape = {d.repr: _canon(simplify(normalize(deepcopy(d)))) for d in D.invented}
    abst_role = {d.repr: _abstraction_role(abst_body[d.repr]) for d in D.invented}

    belief_users = defaultdict(set)     # abst -> set(belief variant)
    donor_users = defaultdict(set)      # abst -> set(non-belief family)
    for kid, s in rewritten.items():
        if kid not in label_of:
            continue
        for a in _absts_in(s):
            (belief_users if is_belief.get(kid) else donor_users)[a].add(label_of[kid])

    abstractions = []
    for name in abst_names:
        beliefs = sorted(belief_users.get(name, set()))
        donors = sorted(donor_users.get(name, set()))
        sub = abst_shape[name] in belief_all_shapes
        abstractions.append({
            'repr': name, 'role': abst_role[name], 'body': abst_body[name],
            'body_shape': abst_shape[name],
            'is_belief_subshape': sub,
            'belief_users': beliefs, 'donor_families': donors,
            # a scaffold abstraction is one belief actually uses that also has a
            # non-belief origin (co-inducer) or whose body sits inside belief structurally
            'scaffold': bool(beliefs) and (bool(donors) or sub),
        })

    # ── assembly: one belief solution per variant, decomposed into donor fragments ────
    # pick, per variant, the target whose tree is MOST covered by donor fragments (the
    # clearest illustration); ties broken by the shorter program.
    donor_all_shapes = set(shape_labels)
    reps = {}
    for tk, tinfo in targets.items():
        covered = sum(1 for sh in tinfo['proper'] if sh in donor_all_shapes)
        cur = reps.get(tinfo['var'])
        cand = (covered, -len(sols_str[tk]), tk)
        if cur is None or cand > cur[0]:
            reps[tinfo['var']] = (cand, tk)

    assembly = {}
    for var, (_score, tk) in reps.items():
        t = trees[tk]
        # the constructor abstraction (if any) belief's own rewrite folded the root into
        root_abst = sorted(_absts_in(rewritten.get(tk, '')) &
                           {a['repr'] for a in abstractions
                            if a['role'].startswith('agent_constructor')})
        assembly[var] = {
            'target_kid': tk,
            'sol': str(t),
            'root_abstraction': root_abst,
            'lines': _render(t, donor_all_shapes, shape_labels),
        }

    return {
        'n_belief_solved': n_belief,
        'belief_variants_solved': dict(Counter(t['var'] for t in targets.values())),
        'structural': {
            'by_donor_family': {
                lbl: {
                    'n_belief_targets': len(rec['targets']),
                    'target_variants': dict(rec['variants']),
                    'roles': sorted(rec['roles']),
                    'min_depth': rec['min_depth'],
                    'fragments': [{'shape': sh, 'role': _frag_role(sh), 'count': n}
                                  for sh, n in rec['fragments'].most_common()],
                }
                for lbl, rec in by_family.items()
            },
            'edges': edges,
        },
        'abstraction': {'library': abst_names, 'items': abstractions},
        'assembly': assembly,
    }


def _render(node, donor_shapes, shape_labels, depth=0):
    """Pretty-print a belief tree, stopping at each MAXIMAL donor-covered subterm and
    annotating it with the families that certify it.  The root (depth 0) is always
    expanded — a belief program is only a scaffold for something deeper, never for
    itself."""
    sh = _canon(node)
    if depth > 0 and node.tails and sh in donor_shapes:
        fams = ', '.join(sorted(shape_labels[sh]))
        return [f"{'  ' * depth}{node}   ⟵ {fams}  [{_frag_role(sh)}]"]
    if not node.tails:
        return [f"{'  ' * depth}{node.repr}"]
    lines = [f"{'  ' * depth}({node.repr}"]
    for t in node.tails:
        lines += _render(t, donor_shapes, shape_labels, depth + 1)
    lines[-1] += ')'
    return lines


# ── reporting ─────────────────────────────────────────────────────────────────────
def _print_report(data, phase):
    print(f"\n  belief tasks solved: {data['n_belief_solved']}"
          + (f"  ({', '.join(f'{v}×{n}' for v, n in data['belief_variants_solved'].items())})"
             if data['belief_variants_solved'] else ""))

    fam = data['structural']['by_donor_family']
    print("\n  " + "-" * 76)
    print("  (1) STRUCTURAL SCAFFOLD — lower-level solutions that sit inside belief")
    print("  " + "-" * 76)
    if not fam:
        print("      (no solved task's whole program occurs as a belief subterm this run)")
    else:
        print(f"      {'donor family':22s} {'feeds':>5s}  {'depth':>5s}  reused fragment (role)")
        for lbl, rec in sorted(fam.items(), key=lambda kv: (-kv[1]['n_belief_targets'],
                                                            kv[1]['min_depth'])):
            frag = rec['fragments'][0]
            print(f"      {lbl:22s} {rec['n_belief_targets']:5d}  {rec['min_depth']:5d}  "
                  f"{frag['shape']}")
            print(f"      {'':22s} {'':5s}  {'':5s}  → {frag['role']}"
                  + (f"; also {', '.join(f['shape'] for f in rec['fragments'][1:])}"
                     if len(rec['fragments']) > 1 else ""))
        # punchline: belief's parts, named by the rungs that certify them
        pieces = sorted(fam.items(), key=lambda kv: kv[1]['min_depth'])
        print("\n      => belief is assembled from: "
              + "; ".join(f"{lbl} ({next(iter(sorted(rec['roles'])), 'fragment')})"
                          for lbl, rec in pieces))

    print("\n  " + "-" * 76)
    print("  (2) ABSTRACTION SCAFFOLD — invented tokens belief shares with other families")
    print("  " + "-" * 76)
    items = data['abstraction']['items']
    scaffolds = [a for a in items if a['scaffold']]
    if not scaffolds:
        print("      (no invented abstraction is shared between belief and another family)")
    for a in items:
        tag = '  *** shared with belief' if a['scaffold'] else ''
        print(f"      {a['repr']:6s} «{a['role']}»{tag}")
        print(f"          body: {a['body']}")
        if a['belief_users']:
            print(f"          belief users : {', '.join(a['belief_users'])}")
        if a['donor_families']:
            print(f"          donor families: {', '.join(a['donor_families'])}")
        elif a['belief_users'] and a['is_belief_subshape']:
            print(f"          (belief-specific, but its body is a belief subterm)")

    print("\n  " + "-" * 76)
    print("  (3) ASSEMBLY — a belief solution per variant, decomposed into its scaffold")
    print("  " + "-" * 76)
    for var, asm in data['assembly'].items():
        print(f"\n      [{var}]")
        if asm['root_abstraction']:
            print(f"      root folded by stitch into {', '.join(asm['root_abstraction'])} "
                  f"(the agent constructor)")
        for line in asm['lines']:
            print("      " + line)


# ── driver ──────────────────────────────────────────────────────────────────────────
def run(decomposed=False, smoke=False, run_path=None):
    phase = 2 if decomposed else 1
    print(f"\n{'='*78}\nSCAFFOLD IDENTIFIER — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync){' [smoke]' if smoke else ''}"
          f"\n{'='*78}")
    art, D, sols_str, rewritten, meta_of = _load(decomposed, smoke, run_path)
    src = run_path or f"phase{phase}_run{'.smoke' if smoke else ''}.json"
    print(f"  source: {src} ({art.get('n_solved', len(sols_str))} solved programs; "
          f"library {[d.repr for d in D.invented]})")
    print(f"  {provenance_line(art, src)}")

    data = analyse(art, D, sols_str, rewritten, meta_of)
    _print_report(data, phase)

    out = {
        'provenance': art.get('provenance'),
        'phase': phase, 'decomposed': decomposed, 'smoke': smoke, 'source': src,
        'library': [d.repr for d in D.invented],
        **data,
    }
    path = f"scaffold{'.decomposed' if decomposed else ''}{'.smoke' if smoke else ''}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\n  wrote scaffold analysis to {path}")
    return out


def main(argv):
    smoke = '--smoke' in argv
    run_path = argv[argv.index('--run') + 1] if '--run' in argv else None
    if '--both' in argv:
        if run_path is not None:
            sys.exit("--run names a single phase artifact; drop --both or run each phase "
                     "separately with its own --run.")
        run(decomposed=False, smoke=smoke)
        run(decomposed=True, smoke=smoke)
    else:
        run(decomposed='--decomposed' in argv, smoke=smoke, run_path=run_path)


if __name__ == '__main__':
    main(sys.argv)
