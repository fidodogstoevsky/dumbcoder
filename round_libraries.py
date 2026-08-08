"""Results-side ROUND-BY-ROUND library reconstruction — what each round's stitch invented.

The run artifacts keep only the FINAL library (`save_run_artifact` overwrites the same
path at the end of every round), so the thesis could report only where the loop landed,
never how it got there.  But §5.1's whole claim is about the *order*: the parts belief
needs — hold a second grid, move one value between two grids — are invented in round 1,
out of solutions to non-mental tasks, BEFORE a single belief task has been solved.  That
is a claim about round 1's library, and it is unverifiable from a table of round 4's.

This script rebuilds every round's library from the TRAJECTORY artifact
(`phase{1,2}_traj[.smoke].json`), which does record, per task, the round it was first
solved.  For round r we re-stitch the solutions solved THROUGH round r — the exact input
that round's compression step saw — and read the invented abstractions back out.  That is
the same reconstruction `corpus_dl.py` prices each round's corpus under, and it is sound
for the same reason: `saturate_stitch` calls `D.reset()` and re-discovers the whole
library from the fully-expanded solutions on every call, so round r's library is a
function of round r's solved pool and nothing else.

For each abstraction we report:

  * the run's own name.  Live, stitch's names are offset by the number of abstractions
    already registered, so round 1 emits fn_0…fn_5 and every later round fn_6…fn_11
    (a fresh Deltas per round here, so the offset is re-applied for printing).  The
    names are therefore NOT stable across rounds: round 2's fn_7 and round 3's fn_7 are
    different bodies.  `stable_id` groups the ones that are genuinely the same term.
  * the body as stitch emitted it (which may reference the round's own earlier
    abstractions) and the body expanded to base primitives (`normalize`), so a reader can
    count symbols or match a round-2 body against its round-1 parts.
  * `built_from`: which of the SAME round's abstractions the body invokes, and which
    round-(r−1) terms it contains as a subterm once both are expanded — the bootstrapping
    edge, read structurally rather than asserted.
  * WHO PAID: the solved-through-r programs rewritten through the abstraction, split
    belief vs non-mental and broken down by family.  In round 1 that split is the whole
    point: every user of the derive-and-commit frame and the single-value commit is a
    non-mental task, because no belief task is solved yet.

    # 1. produce a run (its trajectory artifact is what we consume)
    python phase2.py                       -> phase2_traj.json
    # 2. reconstruct the per-round libraries
    python round_libraries.py              phase 1 (reads phase1_traj.json)
    python round_libraries.py --decomposed phase 2 (reads phase2_traj.json)
    python round_libraries.py --both       phases 1 and 2
    python round_libraries.py --smoke      use the .smoke trajectory artifacts
    python round_libraries.py --run PATH   consume a specific trajectory artifact

Writes round_libraries[.decomposed][.smoke].json.
"""

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, 'viz'), _HERE]
import json
import re as _re
from copy import deepcopy
from collections import defaultdict

from ecd import (
    Deltas, saturate_stitch, rewrite_through_library, tr, normalize, simplify,
)
from prims import make_symmetric_prims
from experiment import provenance_line, _abstraction_role
from corpus_dl import family, _load_traj


# ── helpers ───────────────────────────────────────────────────────────────────────────
def _expanded_body(d):
    "The abstraction's body with every nested invented token inlined to base primitives."
    return str(simplify(normalize(deepcopy(d.hiddentail))))


def _canonical(body):
    """Body with argument indices erased, so the same term found in two rounds under
    different argument ORDERS ($0/$2 swapped by the searcher) still compares equal."""
    return _re.sub(r'\$\d+', '$', body)


def _tokens(body):
    "the invented tokens a body invokes (its intra-round dependencies)"
    return sorted(set(_re.findall(r'\bfn_\d+\b', body)))


def _uses(lib_form, name):
    "does this rewritten program invoke abstraction `name`?"
    return bool(_re.search(rf'\b{name}\b', lib_form))


# ── the reconstruction ────────────────────────────────────────────────────────────────
def compute_round_libraries(art):
    decomposed = bool(art['decomposed'])
    stitch_iters = int(art['stitch_iters'])
    n_rounds = int(art['n_rounds'])
    tasks = art['tasks']                       # kid -> {sol, kind, round}

    corpus = [(kid, t['sol'], t['kind'], int(t['round'])) for kid, t in tasks.items()]
    fam_of = {kid: family(t['kind']) for kid, t in tasks.items()}
    kind_of = {kid: t['kind'] for kid, t in tasks.items()}

    stable = {}          # canonical expanded body -> stable id ('t1', 't2', …)
    prev_expanded = {}   # stable id -> expanded body, from the previous round
    rounds = []

    for r in range(1, n_rounds + 1):
        # a fresh base DSL per round: stitch mutates D, and each round's library is a
        # function of that round's solved pool alone (saturate_stitch resets D.invented).
        D = Deltas(make_symmetric_prims(decomposed=decomposed))
        solved = [(kid, s) for kid, s, _k, rd in corpus if rd <= r]
        solved_trees = {kid: tr(D, s) for kid, s in solved}
        if not solved_trees:
            continue
        saturate_stitch(D, solved_trees, iterations=stitch_iters, max_arity=5)

        # the run's own naming: stitch offsets by the count already registered, which is
        # 0 in round 1 and (the previous round's) 6 in every later round.
        offset = 0 if r == 1 else len(rounds[-1]['abstractions'])
        names = {d.repr: f"fn_{int(d.repr.split('_')[1]) + offset}" for d in D.invented}

        # rewrite the pool this round's stitch actually saw, to attribute each token
        lib_forms = (rewrite_through_library(D, [s for _kid, s in solved])
                     if D.invented else [s for _kid, s in solved])
        per_family = defaultdict(lambda: defaultdict(int))   # d.repr -> family -> n
        per_kind = defaultdict(lambda: defaultdict(int))     # d.repr -> fine kind -> n
        for (kid, _s), lib in zip(solved, lib_forms):
            for d in D.invented:
                if _uses(lib, d.repr):
                    per_family[d.repr][fam_of[kid]] += 1
                    per_kind[d.repr][kind_of[kid]] += 1

        solved_by_family = defaultdict(int)
        for kid, _s in solved:
            solved_by_family[fam_of[kid]] += 1

        abstractions = []
        for d in D.invented:
            raw = str(d.hiddentail)
            exp = _expanded_body(d)
            canon = _canonical(exp)
            sid = stable.setdefault(canon, f"t{len(stable) + 1}")
            fams = dict(per_family[d.repr])
            n_bel = fams.get('belief', 0)
            abstractions.append({
                'name':        names[d.repr],
                'local_name':  d.repr,
                'stable_id':   sid,
                'first_seen':  r if canon not in prev_expanded.values() else None,
                'role':        _abstraction_role(exp),
                'type':        str(d.type),
                'argtypes':    [str(t) for t in (d.tailtypes or [])],
                'arity':       len(d.tailtypes or []),
                'body':        raw,
                'body_expanded': exp,
                'calls':       [names.get(t, t) for t in _tokens(raw)],
                'users':       sum(fams.values()),
                'users_belief': n_bel,
                'users_nonmental': sum(fams.values()) - n_bel,
                'users_by_family': fams,
                'users_by_kind': dict(sorted(per_kind[d.repr].items(),
                                             key=lambda kv: -kv[1])),
            })

        # bootstrapping edges: which of the PREVIOUS round's terms survive inside this
        # round's bodies, once both sides are expanded to base primitives.
        for a in abstractions:
            carried = [sid for sid, body in prev_expanded.items()
                       if sid != a['stable_id']
                       and _canonical(body).replace('$', '') in
                           _canonical(a['body_expanded']).replace('$', '')]
            a['contains_prev_round'] = carried
            a['persisted'] = a['stable_id'] in prev_expanded

        prev_expanded = {a['stable_id']: a['body_expanded'] for a in abstractions}
        rounds.append({
            'round': r,
            'n_solved_through': len(solved),
            'solved_by_family': dict(solved_by_family),
            'n_abstractions': len(abstractions),
            'abstractions': abstractions,
        })

    return {
        'decomposed': decomposed,
        'phase': 2 if decomposed else 1,
        'smoke': bool(art.get('smoke')),
        'n_rounds': n_rounds,
        'rounds': rounds,
    }


# ── reporting ─────────────────────────────────────────────────────────────────────────
def report(data):
    for rd in data['rounds']:
        fams = rd['solved_by_family']
        bel = fams.get('belief', 0)
        print(f"\n{'-'*88}\nROUND {rd['round']} — stitched over {rd['n_solved_through']} "
              f"solutions ({bel} belief, {rd['n_solved_through'] - bel} non-mental)"
              f"\n{'-'*88}")
        for a in rd['abstractions']:
            tag = ('persists' if a['persisted'] else 'NEW')
            print(f"  {a['name']:<6} «{a['role']}»  {tag}   [{', '.join(a['argtypes'])}] "
                  f"-> {a['type']}   ({a['stable_id']})")
            print(f"    body      : {a['body']}")
            if a['body'] != a['body_expanded']:
                print(f"    expanded  : {a['body_expanded']}")
            if a['calls']:
                print(f"    calls     : {', '.join(a['calls'])}")
            if a['contains_prev_round']:
                print(f"    contains  : {', '.join(a['contains_prev_round'])} "
                      f"(round {rd['round'] - 1} terms)")
            top = ', '.join(f"{k} x{n}" for k, n in list(a['users_by_kind'].items())[:6])
            print(f"    used by   : {a['users']} solves "
                  f"({a['users_belief']} belief / {a['users_nonmental']} non-mental)"
                  + (f"  — {top}" if top else ""))


def run(decomposed=False, smoke=False, run_path=None):
    phase = 2 if decomposed else 1
    print(f"\n{'='*88}\nPER-ROUND LIBRARIES — phase {phase} "
          f"({'decomposed' if decomposed else 'atomic'} fork/sync)"
          f"{' [smoke]' if smoke else ''}\n{'='*88}")
    art = _load_traj(decomposed, smoke, run_path)
    print(f"  {provenance_line(art, run_path)}")
    data = compute_round_libraries(art)
    data['provenance'] = art.get('provenance')
    report(data)
    out = f"round_libraries{'.decomposed' if decomposed else ''}{'.smoke' if smoke else ''}.json"
    with open(out, 'w') as f:
        json.dump(data, f, indent=1)
    print(f"\n  wrote {out}")
    return data


if __name__ == '__main__':
    args = sys.argv[1:]
    smoke = '--smoke' in args
    run_path = None
    if '--run' in args:
        run_path = args[args.index('--run') + 1]
    if '--both' in args:
        run(decomposed=False, smoke=smoke)
        run(decomposed=True, smoke=smoke)
    else:
        run(decomposed='--decomposed' in args, smoke=smoke, run_path=run_path)
