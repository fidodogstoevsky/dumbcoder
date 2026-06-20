"""File 15: decompose `fork` into the product-category combinators.

file13 made belief a discoverable compound — `fork(policy, sync_to_world av)` —
rather than a `believe` primitive.  But `fork` itself still hid structure inside
one closure: the private copy (`w.copy()`) and the application of the policy to
it.  Spell `fork` out and it falls into three textbook combinators of the
product (×) category:

    fork(derive, commit)(w) = commit((w, derive(w)))
                            ≡ (commit ∘ mapsnd(derive) ∘ dup)(w)

    dup      :: grid -> pair     w |-> (w, w)          -- diagonal Δ
    mapsnd f :: pair -> pair     (a,b) |-> (a, f(b))   -- bifunctor 'second'
    commit   :: pair -> grid                           -- product eliminator

Every part already existed in dsl.py: `dup` is `dup_g`, `mapsnd` is `on_model`
(the file11 model-channel map), `fst_gg`/`snd_gg` are the projections.  So
"decompose fork" means: bring those pieces into single-grid program space.  The
private copy stops being implicit in `w.copy()` and becomes a discoverable `dup`
node; "run the policy on the copy" becomes a `mapsnd` node.  `fork` is no longer
a primitive — it is the compound `(pipe_gpg (compose_gp dup (mapsnd P)) C)`,
which stitch can re-extract as a `fork` abstraction.

Why decompose *fork* but not *sync*.  sync_to_world decomposes symmetrically —
`sync_to_world(v) ≡ place(v) ∘ ⟨ fst , locate(v) ∘ snd ⟩` (read v's coordinate
through one channel, impose it on the other; see `locate`/`place` in dsl.py).
But there is a real caution: push the decomposition too far and the agent
signature dissolves.  Agency is discoverable here because the SAME value `av` is
threaded through the actor `(optimize … av)` and the committer `(sync … av)`.  If
the committer were itself a 4-node `place(av) ∘ ⟨fst, locate(av) ∘ snd⟩` subtree,
that av-coincidence would be buried and stitch would be less likely to surface it
as the signature.  The sweet spot: decompose fork (cheap, and `dup` is a great
independent primitive), but keep `sync_to_world` one node deep so the shared-`av`
signature stays visible.

Expected solutions (same trajectories as file13; only the encoding of fork changes):

  physics  (3 nodes):  (step v d)
  desire   (4 nodes):  (optimize (neg_dist gv) av)
  belief  (14 nodes):  (pipe_gpg
                          (compose_gp dup
                            (mapsnd (compose (wall_at r c)
                                             (optimize (neg_dist gv) av))))
                          (sync_to_world av))

av appears twice — in optimize (who acts on the model) and in sync_to_world
(whose move is committed to the world).  The hoped-for stitch discovery is the
agent type constructor with the fork-shape folded back into one node:

  fn_agent($r, $c, $gv, $av) =
    (pipe_gpg (compose_gp dup
                (mapsnd (compose (wall_at $r $c) (optimize (neg_dist $gv) $av))))
              (sync_to_world $av))

Run:
    python file15.py            # full run
    python file15.py --smoke    # tiny corpus, short timeouts
"""

import sys
import re as _re
from collections import Counter
from copy import deepcopy

import numpy as np

from ecd import Deltas, Delta, ECD, normalize, mat_key
from dsl import (
    fn, util, direction, fn_g_p, fn_p_p, fn_p_g,
    RIGHT, LEFT, UP, DOWN,
    dup, mapsnd, compose_gp, pipe_gpg, sync_to_world,
    fork, fork_decomposed, sync_decomposed,
    compose, step, optimize, neg_distance, wall_at,
    unfold, tr, simplify,
)
from file13 import (
    COMBOS, SIZE, DIRS,
    make_physics_tasks, make_desire_tasks, make_belief_tasks,
)


# ── Decomposition self-check ─────────────────────────────────────────────────────
# Before encoding anything as a Delta tree, prove the python identities hold on the
# actual task trajectories: the decomposed combinators must be *numerically* the
# same machine as file13's fork/sync, otherwise the "decomposition" would be a
# different program that merely happens to share a name.

def check_decomposition_identities(tasks):
    n = 0
    for x, m in tasks:
        if m['kind'] != 'belief':
            continue
        av, gv, (pr, pc) = m['av'], m['gv'], m['pw']
        derive = compose(wall_at(pr, pc), optimize(neg_distance(gv), av))
        orig = unfold(x[0], x.shape[0], fork(derive, sync_to_world(av)))
        deco = unfold(x[0], x.shape[0], fork_decomposed(derive, sync_decomposed(av)))
        assert np.array_equal(orig, x),  f"orig fork != task for {m}"
        assert np.array_equal(deco, x),  f"decomposed != task for {m}"
        n += 1
    print(f"decomposition identity: fork ≡ commit∘mapsnd(derive)∘dup and "
          f"sync ≡ place∘⟨fst,locate∘snd⟩ verified on {n} belief tasks")


# ── DSL ──────────────────────────────────────────────────────────────────────────
# file13's DSL with `fork` removed and replaced by the three product-category
# combinators (`dup`, `mapsnd`, plus the two typed composers that wire the
# grid/pair arrows).  `sync_to_world` is left atomic on purpose (see docstring).

def make_core_prims():
    return [
        # Decomposed fork: dup (Δ) ▸ mapsnd (bifunctor second) ▸ commit, wired by
        # the two typed composers.  No `fork` primitive — it is now a compound.
        Delta(pipe_gpg,      fn,     [fn_g_p, fn_p_g],   repr='pipe_gpg'),
        Delta(compose_gp,    fn_g_p, [fn_g_p, fn_p_p],   repr='compose_gp'),
        Delta(dup,           fn_g_p,                     repr='dup'),
        Delta(mapsnd,        fn_p_p, [fn],               repr='mapsnd'),

        # Commit (the product eliminator) — kept one node deep to preserve the
        # shared-av agent signature.
        Delta(sync_to_world, fn_p_g, [int],              repr='sync_to_world'),

        # Grid-state primitives (fn = grid -> grid)
        Delta(compose,      fn,   [fn, fn],          repr='compose'),
        Delta(step,         fn,   [int, direction],  repr='step'),
        Delta(optimize,     fn,   [util, int],       repr='optimize'),
        Delta(wall_at,      fn,   [int, int],        repr='wall_at'),

        # Utility
        Delta(neg_distance, util, [int],             repr='neg_dist'),

        # Direction terminals
        Delta(RIGHT, direction, repr='right'),
        Delta(LEFT,  direction, repr='left'),
        Delta(UP,    direction, repr='up'),
        Delta(DOWN,  direction, repr='down'),

        # Int terminals
        *[Delta(i, int, repr=str(i)) for i in range(6)],
    ]


def _belief_sexp(m):
    "the decomposed belief ground-truth s-expression for a task's metadata"
    pr, pc = m['pw']
    return (f"(pipe_gpg (compose_gp dup "
            f"(mapsnd (compose (wall_at {pr} {pc}) "
            f"(optimize (neg_dist {m['gv']}) {m['av']})))) "
            f"(sync_to_world {m['av']}))")


def verify_ground_truth(D, tasks_meta):
    """Re-express each kind's ground truth as a Delta tree (the searcher's own
    encoding) and check it reproduces the task — catches eager/typing mistakes in
    the Delta encoding rather than in the raw python closures."""
    for x, m in tasks_meta:
        if m['kind'] == 'physics':
            prog = f"(step {m['val']} {m['dir']})"
        elif m['kind'] == 'desire':
            prog = f"(optimize (neg_dist {m['gv']}) {m['av']})"
        else:
            prog = _belief_sexp(m)
        tree = tr(D, prog)
        out  = unfold(x[0], x.shape[0], tree())
        assert np.array_equal(out, x), f"ground truth failed for {m}: {prog}"
    print(f"ground-truth check: {len(tasks_meta)} tasks verified via Delta trees")


# ── Reporting helpers ────────────────────────────────────────────────────────────

def _is_agent_body(body_str):
    "an invented body realises the (decomposed) fork-shape committed via sync"
    return ('pipe_gpg' in body_str and 'compose_gp' in body_str
            and 'dup' in body_str and 'sync_to_world' in body_str)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(smoke=False):
    n_phys = 2 if smoke else 6
    n_des  = 1 if smoke else 2   # per combo
    n_bel  = 1 if smoke else 6   # per combo
    per_task_timeout = 10 if smoke else 300
    max_iterations   = 2  if smoke else 10

    print("Generating tasks…")
    phys = make_physics_tasks(n_phys, seed=0)
    des  = make_desire_tasks(n_des, COMBOS, seed=1)
    bel  = make_belief_tasks(n_bel, COMBOS, seed=2)

    # dedupe across the whole corpus (identical mats would skew counts)
    seen, all_tasks = set(), []
    for x, m in phys + des + bel:
        k = mat_key(x)
        if k in seen:
            continue
        seen.add(k)
        all_tasks.append((x, m))

    Xs   = [x for x, _ in all_tasks]
    meta = [m for _, m in all_tasks]
    by_kind = Counter(m['kind'] for m in meta)
    print(f"  {by_kind['physics']} physics, {by_kind['desire']} desire, "
          f"{by_kind['belief']} belief — {len(Xs)} total\n")

    check_decomposition_identities(all_tasks)

    D = Deltas(make_core_prims())
    print(f"\nDSL: {len(D)} primitives (fork decomposed → dup / mapsnd / "
          f"compose_gp / pipe_gpg; sync_to_world kept atomic)")
    print("  expected physics solution (3 nodes):  (step v d)")
    print("  expected desire solution  (4 nodes):  (optimize (neg_dist gv) av)")
    print("  expected belief solution (14 nodes):")
    print("    (pipe_gpg (compose_gp dup")
    print("      (mapsnd (compose (wall_at r c) (optimize (neg_dist gv) av))))")
    print("      (sync_to_world av))")
    print("  expected stitch discovery — the agent type constructor:")
    print("    fn_agent($r,$c,$gv,$av) with $av shared ×2 (optimize + sync_to_world)\n")

    verify_ground_truth(D, all_tasks)

    print("\nRunning ECD…\n")
    Z, rewritten = ECD(
        Xs, D,
        per_task_timeout=per_task_timeout,
        max_iterations=max_iterations,
        max_arity=5,
        stitch_iterations=4,
        root_type=fn,
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n=== Results ===")
    for kind in ('physics', 'desire', 'belief'):
        ks = [x for x, m in all_tasks if m['kind'] == kind]
        n  = sum(1 for x in ks if mat_key(x) in Z and Z[mat_key(x)] is not None)
        print(f"  {kind:8s}: {n}/{len(ks)}")

    print("\n=== Invented primitives ===")
    if not D.invented:
        print("  (none)")
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(simplify(normalize(deepcopy(d))))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        print(f"  {d.repr}  [{argtypes}]  -> {d.type}")
        print(f"    body: {body_str}")
        if _is_agent_body(body_str):
            print(f"    *** AGENT TYPE CONSTRUCTOR (belief) ***")
            print(f"        fork re-extracted from dup/mapsnd/commit; policy runs on a")
            print(f"        private model (dup), move committed via sync_to_world")
            if shared:
                shared_str = ', '.join(f'{v} (×{c})' for v, c in shared.items())
                print(f"        shared: {shared_str}  — actor AND committer")
        elif 'pipe_gpg' in body_str or 'compose_gp' in body_str or 'dup' in body_str:
            print(f"    *** fork fragment (pair plumbing: Δ / mapsnd / commit) ***")
        elif 'optimize' in body_str or 'neg_dist' in body_str:
            print(f"    *** desire fragment (goal-directed motion) ***")
        elif 'step' in body_str:
            print(f"    *** physics fragment ***")

    # Structural census of the belief solutions: programs with ints stripped.
    print("\n=== Belief solution shapes ===")
    skeletons = Counter()
    for x, m in all_tasks:
        if m['kind'] != 'belief':
            continue
        k = mat_key(x)
        if k in Z and Z[k] is not None:
            sol = simplify(normalize(deepcopy(Z[k])))
            skeletons[_re.sub(r'\b\d+\b', '_', str(sol))] += 1
    for shape, cnt in skeletons.most_common():
        print(f"  ×{cnt}  {shape}")

    print("\n=== Sample solutions ===")
    kind_seen = Counter()
    for x, m in all_tasks:
        # print all belief solutions (to see shape variants); 2 samples otherwise
        if m['kind'] != 'belief' and kind_seen[m['kind']] >= 2:
            continue
        kind_seen[m['kind']] += 1
        k = mat_key(x)
        tag = {k2: v for k2, v in m.items() if k2 != 'kind'}
        if k in Z and Z[k] is not None:
            sol = simplify(normalize(deepcopy(Z[k])))
            rw  = rewritten.get(k, '')
            print(f"  [{m['kind']}] {tag}")
            print(f"    found:     {sol}")
            if rw:
                print(f"    rewritten: {rw}")
        else:
            print(f"  [{m['kind']}] {tag}  → unsolved")


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)
