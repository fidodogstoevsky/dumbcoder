"""File 6: multi-agent false belief — fn-body enumeration.

Programs are fn-typed lam bodies (root_type='fn_lam_body').  Each program is
evaluated by wrapping it in lam and simulating via unfold_multiagent_fn_belief_steps.
This lets stitch find reusable fn-level abstractions that nest naturally:

DSL:
  if_int_eq(a,b,ft,ff)  :: int,int,fn,fn -> fn
  set_at(r,c,v)         :: int,int,int -> fn
  var                   :: int   (lambda-bound agent value, substituted at sim time)
  id_fn                 :: fn    (identity transform)

─────────────────────────────────────────────────────────────────────────────
Tier 1 (single-agent false belief):

  Body (7 nodes):
    if_int_eq(var, $av, set_at($r, $c, 3), id_fn)

  Stitch discovers (fn type, 4 holes — fallback is free):
    fn_cond($av, $r, $c, $fallback) = if_int_eq(var, $av, set_at($r, $c, 3), $fallback)

─────────────────────────────────────────────────────────────────────────────
Tier 2 (two-agent false belief):

  Body (13 nodes):
    if_int_eq(var, av1, set_at(r1,c1,3), if_int_eq(var, av2, set_at(r2,c2,3), id_fn))

  After fn_cond available (8 nodes):
    fn_cond(av1, r1, c1, fn_cond(av2, r2, c2, id_fn))

Run:
  python file6.py
"""

from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    normalize, mat_key,
)
from dsl import (
    fn,
    _if_int_eq, _var_sentinel,
    id_fn, set_at,
)
from tasks import (
    make_multi_agent_false_belief_tasks,
    make_two_agent_one_false_belief_tasks,
)

# ── Tasks ──────────────────────────────────────────────────────────────────
AGENTS = [(1, 2), (4, 5)]   # (agent_val, goal_val) pairs

# Tier 1a: agent 1 has false belief, agent 4 navigates directly
raw_a1 = make_two_agent_one_false_belief_tasks(
    n=3, false_agent_val=1, false_agent_goal_val=2,
    direct_agent_val=4, direct_agent_goal_val=5, size=5, seed=10)
Xs_a1 = [x for x, _ in raw_a1]

# Tier 1b: agent 4 has false belief, agent 1 navigates directly
raw_a4 = make_two_agent_one_false_belief_tasks(
    n=20, false_agent_val=4, false_agent_goal_val=5,
    direct_agent_val=1, direct_agent_goal_val=2, size=5, seed=20)
Xs_a4 = [x for x, _ in raw_a4]

# Tier 2: two agents, two distinct phantom walls
raw_multi = make_multi_agent_false_belief_tasks(n=20, size=5, seed=0)
Xs_multi  = [x for x, _ in raw_multi]
meta      = [m for _, m in raw_multi]

Xs = Xs_a1 + Xs_a4 + Xs_multi
for idx, task in enumerate(Xs):
    print("----------")
    print(f"task no. {idx}")
    print(task)
print(f"\n{len(Xs_a1)} single-agent false-belief tasks (agent 1, 5×5)")
print(f"{len(Xs_a4)} single-agent false-belief tasks (agent 4, 5×5)")
print(f"{len(Xs_multi)} two-agent false-belief tasks (5×5, agents {AGENTS})")
print(f"{len(Xs)} total tasks\n")

# ── DSL ────────────────────────────────────────────────────────────────────
core_prims = [
    Delta(_if_int_eq,    fn,  [int, int, fn, fn], repr='if_int_eq'),
    Delta(set_at,        fn,  [int, int, int],    repr='set_at'),
    Delta(_var_sentinel, int,                     repr='var'),
    Delta(id_fn,         fn,                      repr='id_fn'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
]

D = Deltas(core_prims)
print(f"DSL: {len(core_prims)} primitives (fn-body enumeration, no sim/lam/ig_i)")
print("  tier-1 body (7 nodes):  if_int_eq(var, $av, set_at($r, $c, 3), id_fn)")
print("  tier-2 body (13 nodes): if_int_eq(var, av1, set_at(r1,c1,3),")
print("                            if_int_eq(var, av2, set_at(r2,c2,3), id_fn))")
print("  after fn_cond (8 nodes): fn_cond(av1, r1, c1, fn_cond(av2, r2, c2, id_fn))\n")

# ── ECD ────────────────────────────────────────────────────────────────────
print("Running ECD…\n")
Z, rewritten = ECD(
    Xs, D,
    per_task_timeout=20,
    max_iterations=8,
    max_arity=4,
    root_type='fn_lam_body',
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
            parts_after_set_at = body_str.split('set_at', 1)
            fallback_is_hole = len(parts_after_set_at) > 1 and '$' in parts_after_set_at[1]
            tag = '*** fn_cond WITH FREE FALLBACK ***' if fallback_is_hole else '*** belief structure ***'
            #print(f"    {tag}")

print("\n=== Two-agent solutions (first 4) ===")
for x, m in list(zip(Xs_multi, meta))[:4]:
    k   = mat_key(x)
    tag = (f"a1={m['agent1']} pw1={m['pw1']}  "
           f"a2={m['agent2']} pw2={m['pw2']}  T={x.shape[0]}")
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  {tag}")
        print(f"    body:      {sol}")
        if rw:
            print(f"    rewritten: {rw}")
    else:
        print(f"  {tag}  → unsolved")
