"""File 6: multi-agent false belief — synthesize a belief function (fn_belief).

Two agents navigate simultaneously. Each acts optimally on a different world
model (a different phantom wall). The observed trajectory shows both agents
taking suboptimal paths that can only be jointly explained by attributing
distinct world models to each agent.

The synthesis target is a belief function of type fn_belief (int -> fn):
given an agent value, return a Grid->Grid transformation that produces the
world that agent navigates on.

DSL:
  assign_belief(agent_val, transform_fn, fallback) :: int, fn, fn_belief -> fn_belief
  no_belief_fn                                      :: fn_belief
  set_at(r, c, v)                                   :: int, int, int -> fn

─────────────────────────────────────────────────────────────────────────────
Bootstrap (tier 1) — single-agent false belief:

  One agent (val=1 or val=4) takes a detour caused by one phantom wall.
  No second agent is present, so assign_belief is dispatched once.

  Solution (7 nodes):
    assign_belief($av, set_at($r, $c, 3), no_belief_fn)

  Stitch discovers:
    fn_one_belief($av, $r, $c) =
      assign_belief($av, set_at($r, $c, 3), no_belief_fn)

─────────────────────────────────────────────────────────────────────────────
Two-agent (tier 2) — two distinct phantom walls:

  Solution (11 nodes):
    assign_belief(1, set_at(r1, c1, 3),
      assign_belief(4, set_at(r2, c2, 3),
        no_belief_fn))

  After fn_one_belief is available (7 nodes compressed to 1):
    assign_belief(1, set_at(r1, c1, 3),
      fn_one_belief(4, r2, c2))    ← 7 nodes

  Note: actual grid does not appear in programs — it is passed at evaluation
  time, so no task-specific ig_i terminals are needed.

  Stitch discovers:
    fn_two_beliefs($r1, $c1, $r2, $c2) =
      assign_belief(1, set_at($r1, $c1, 3),
        fn_one_belief(4, $r2, $c2))

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
    fn, fn_belief,
    set_at, assign_belief, no_belief_fn,
)
from tasks import make_multi_agent_false_belief_tasks, make_false_belief_tasks

# ── Tasks ──────────────────────────────────────────────────────────────────
AGENTS = [(1, 2), (4, 5)]   # (agent_val, goal_val) pairs — held constant in evaluator

# Tier 1a: single-agent false belief, agent 1 (val=1, goal=2)
Xs_a1 = make_false_belief_tasks(n=10, size=5, seed=10)

# Tier 1b: single-agent false belief, agent 4 (val=4, goal=5) — remap 1→4, 2→5
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

# ── DSL ────────────────────────────────────────────────────────────────────
# Enumerate programs of type fn_belief (int -> fn).
# No task-specific ig_i terminals needed: actual grid is passed at eval time.
core_prims = [
    Delta(assign_belief, fn_belief, [int, fn, fn_belief], repr='assign_belief'),
    Delta(no_belief_fn,  fn_belief,                       repr='no_belief_fn'),
    Delta(set_at,        fn,        [int, int, int],      repr='set_at'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
]

D = Deltas(core_prims)
print(f"DSL: {len(core_prims)} primitives (no task terminals)")
print("  tier-1 solution (7 nodes):  assign_belief($av, set_at($r, $c, 3), no_belief_fn)")
print("  tier-2 solution (11 nodes): assign_belief(1, set_at(r1, c1, 3),")
print("                                assign_belief(4, set_at(r2, c2, 3), no_belief_fn))\n")

# ── ECD ────────────────────────────────────────────────────────────────────
print("Running ECD…\n")
Z, rewritten = ECD(
    Xs, D,
    per_task_timeout=60,
    max_iterations=8,
    max_arity=6,
    root_type=fn_belief,
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
        if 'assign_belief' in body_str and 'set_at' in body_str:
            if shared:
                label = ', '.join(f'{v} (×{c})' for v, c in shared.items())
                print(f"    *** BELIEF FN — {label} ***")
            else:
                print(f"    *** BELIEF FN — assign_belief(agent, set_at(...), ...) ***")

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
