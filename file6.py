"""File 6: multi-agent false belief — lambda abstraction.

Programs are mat (root_type=mat).  The belief function emerges from lambda abstraction
over the agent value — ECD must discover the pairing structure itself.

DSL:
  sim(g, wm)               :: grid, fn_belief -> mat
  lam(body)                :: fn -> fn_belief      (body contains var)
  var                      :: int                  (lambda-bound agent value)
  if_int_eq(a,b,ft,ff)     :: int,int,fn,fn -> fn
  set_at(r,c,v)            :: int,int,int -> fn
  id_fn                    :: fn                   (identity transform)

─────────────────────────────────────────────────────────────────────────────
Tier 1 (single-agent false belief):

  Solution (11 nodes):
    sim(ig_i, lam(if_int_eq(var, $av, set_at($r, $c, 3), id_fn)))

  Stitch discovers:
    fn_cond($av, $r, $c, $fallback) = if_int_eq(var, $av, set_at($r, $c, 3), $fallback)

─────────────────────────────────────────────────────────────────────────────
Tier 2 (two-agent false belief):

  Solution (18 nodes):
    sim(ig_i, lam(if_int_eq(var, av1, set_at(r1,c1,3),
                  if_int_eq(var, av2, set_at(r2,c2,3), id_fn))))

  After fn_cond available (8 nodes):
    sim(ig_i, lam(fn_cond(av1, r1, c1, fn_cond(av2, r2, c2, id_fn))))

Run:
  python file6.py
"""

from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    normalize, mat_key, task_terminals,
)
from dsl import (
    mat, fn, fn_belief, grid,
    sim, _lam_impl, _if_int_eq, _var_sentinel,
    id_fn, set_at,
)
from tasks import make_multi_agent_false_belief_tasks, make_false_belief_tasks

# ── Tasks ──────────────────────────────────────────────────────────────────
AGENTS = [(1, 2), (4, 5)]   # (agent_val, goal_val) pairs

# Tier 1a: single-agent false belief, agent 1 (val=1, goal=2)
Xs_a1 = make_false_belief_tasks(n=10, size=5, seed=10)

# Tier 1b: single-agent false belief, agent 4 (val=4, goal=5)
def remap_agents(Xs, av_from, av_to, gv_from, gv_to):
    out = []
    for x in Xs:
        x2 = x.copy()
        x2[x == av_from] = av_to
        x2[x == gv_from] = gv_to
        out.append(x2)
    return out

raw_a4 = make_false_belief_tasks(n=10, size=5, seed=20)
Xs_a4  = remap_agents(raw_a4, av_from=1, av_to=4, gv_from=2, gv_to=5)

# Tier 2: two agents, two distinct phantom walls
raw_multi = make_multi_agent_false_belief_tasks(n=20, size=5, seed=0)
Xs_multi  = [x for x, _ in raw_multi]
meta      = [m for _, m in raw_multi]

Xs = Xs_a1 + Xs_a4 + Xs_multi
print(f"\n{len(Xs_a1)} single-agent false-belief tasks (agent 1, 5×5)")
print(f"{len(Xs_a4)} single-agent false-belief tasks (agent 4, 5×5)")
print(f"{len(Xs_multi)} two-agent false-belief tasks (5×5, agents {AGENTS})")
print(f"{len(Xs)} total tasks\n")

# ── Task terminals ──────────────────────────────────────────────────────────
ig_terminals = task_terminals(Xs, mode='full')

# ── DSL ────────────────────────────────────────────────────────────────────
core_prims = [
    Delta(sim,           mat,       [grid, fn_belief],  repr='sim'),
    Delta(_lam_impl,     fn_belief, [fn],               repr='lam'),
    Delta(_if_int_eq,    fn,        [int, int, fn, fn], repr='if_int_eq'),
    Delta(set_at,        fn,        [int, int, int],    repr='set_at'),
    Delta(_var_sentinel, int,                           repr='var'),
    Delta(id_fn,         fn,                            repr='id_fn'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
] + ig_terminals

D = Deltas(core_prims)
print(f"DSL: {len(core_prims)} primitives ({len(ig_terminals)} ig terminals)")
print("  tier-1 target (11 nodes):")
print("    sim(ig_i, lam(if_int_eq(var, $av, set_at($r, $c, 3), id_fn)))")
print("  tier-2 target (18 nodes, 8 after fn_cond):")
print("    sim(ig_i, lam(if_int_eq(var, av1, set_at(r1,c1,3),")
print("                  if_int_eq(var, av2, set_at(r2,c2,3), id_fn))))\n")

# ── ECD ────────────────────────────────────────────────────────────────────
print("Running ECD…\n")
Z, rewritten = ECD(
    Xs, D,
    per_task_timeout=60,
    max_iterations=8,
    max_arity=4,
    root_type=mat,
    agents=AGENTS,
)

# ── Report ─────────────────────────────────────────────────────────────────
n_a1    = sum(1 for x in Xs_a1    if mat_key(x) in Z and Z[mat_key(x)] is not None)
n_a4    = sum(1 for x in Xs_a4    if mat_key(x) in Z and Z[mat_key(x)] is not None)
n_multi = sum(1 for x in Xs_multi if mat_key(x) in Z and Z[mat_key(x)] is not None)
print(f"\n=== Results ===")
print(f"  single-agent (a1): {n_a1}/{len(Xs_a1)}")
print(f"  single-agent (a4): {n_a4}/{len(Xs_a4)}")
print(f"  two-agent:         {n_multi}/{len(Xs_multi)}")

print("\n=== Invented primitives ===")
if not D.invented:
    print("  (none)")
else:
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        print(f"  {d.repr}  [{argtypes}]  -> {d.type}")
        print(f"    body: {body_str}")
        if 'if_int_eq' in body_str and 'var' in body_str and 'set_at' in body_str:
            print(f"    *** BELIEF STRUCTURE — if_int_eq(var, agent, set_at(...)) ***")

print("\n=== Two-agent solutions (first 4) ===")
for x, m in list(zip(Xs_multi, meta))[:4]:
    k   = mat_key(x)
    tag = (f"a1={m['agent1']} pw1={m['pw1']}  "
           f"a2={m['agent2']} pw2={m['pw2']}  T={x.shape[0]}")
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  {tag}")
        print(f"    found:     {sol}")
        if rw:
            print(f"    rewritten: {rw}")
    else:
        print(f"  {tag}  → unsolved")
