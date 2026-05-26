"""File 7: multi-agent false belief — per-agent grid synthesis (non-mental).

Same two-agent false-belief scenario as file6, but the DSL has no world-map
type (int -> grid), no case/default_world dispatch primitives.

Instead: each agent's world model is synthesized as a plain `grid` — the
grid state on which that agent's observed path is BFS-optimal.  Two
separate ECD runs, one per agent.  The joint explanation (both agents share
the same `place_wall($ig, $r, $c)` pattern) emerges from stitch compression
across both solution sets, not from the DSL structure.

DSL (same for both agents):
  place_wall(g, r, c)  :: grid, int, int -> grid

Root type: grid
Evaluator: unfold_belief_steps(actual_g, believed_g, T, step_fn)
           where step_fn = approach(agent_val, goal_val)

Agent 1 solution: place_wall(ig1_i, r1, c1)
Agent 4 solution: place_wall(ig4_i, r2, c2)

Stitch discovers across both corpora:
  fn_world_mod($ig, $r, $c) = place_wall($ig, $r, $c)

Run:
  python file7.py
"""

import numpy as np
from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    task_terminals,
    normalize, mat_key,
)
from dsl import (
    grid,
    place_wall,
    approach,
)
from tasks import make_multi_agent_false_belief_tasks

# ── Tasks ──────────────────────────────────────────────────────────────────
AGENTS = [(1, 2), (4, 5)]   # (agent_val, goal_val) pairs

raw = make_multi_agent_false_belief_tasks(n=20, size=5, seed=0)
Xs_joint = [x for x, _ in raw]
meta      = [m for _, m in raw]

def extract_agent_tasks(Xs, agent_val, goal_val):
    """Isolate one agent's trajectory from multi-agent observation matrices.

    Returns a list of (T, H, W) arrays containing only agent_val and goal_val
    cells — all other values zeroed out.  The initial frame becomes that
    agent's initial grid (their actual starting world, without phantom walls).
    """
    out = []
    for x in Xs:
        x1 = np.zeros_like(x)
        x1[x == agent_val] = agent_val
        x1[x == goal_val]  = goal_val
        out.append(x1)
    return out

Xs_1 = extract_agent_tasks(Xs_joint, agent_val=1, goal_val=2)
Xs_4 = extract_agent_tasks(Xs_joint, agent_val=4, goal_val=5)

print(f"{len(Xs_joint)} multi-agent false-belief tasks (5×5, agents {AGENTS})")
print(f"Split into {len(Xs_1)} agent-1 tasks and {len(Xs_4)} agent-4 tasks\n")

# ── DSL ────────────────────────────────────────────────────────────────────
# Identical DSL for both agents: just place_wall + coordinates.
# No world-map type, no case/default_world.
def make_prims():
    return [
        Delta(place_wall, grid, [grid, int, int], repr='place_wall'),
        Delta(0, int, repr='0'), Delta(1, int, repr='1'),
        Delta(2, int, repr='2'), Delta(3, int, repr='3'),
        Delta(4, int, repr='4'), Delta(5, int, repr='5'),
    ]

ig_1 = task_terminals(Xs_1, mode='full')
ig_4 = task_terminals(Xs_4, mode='full')

D_1 = Deltas(make_prims() + ig_1)
D_4 = Deltas(make_prims() + ig_4)

print(f"Agent-1 DSL: {len(make_prims())} core prims + {len(ig_1)} task terminals = {len(D_1)} total")
print(f"Agent-4 DSL: {len(make_prims())} core prims + {len(ig_4)} task terminals = {len(D_4)} total")
print("  agent-1 solution: place_wall(ig1_i, r1, c1)")
print("  agent-4 solution: place_wall(ig4_i, r2, c2)\n")

# ── ECD — Agent 1 ──────────────────────────────────────────────────────────
# root_type=grid: program produces the believed initial grid.
# Evaluator: unfold_belief_steps(actual_g, believed_g, T, approach(1, 2))
print("Running ECD for agent 1…\n")
Z_1, rewritten_1 = ECD(
    Xs_1, D_1,
    per_task_timeout=90,
    max_iterations=8,
    max_arity=4,
    root_type=grid,
    step_fn=approach(1, 2),
)

# ── ECD — Agent 4 ──────────────────────────────────────────────────────────
print("Running ECD for agent 4…\n")
Z_4, rewritten_4 = ECD(
    Xs_4, D_4,
    per_task_timeout=90,
    max_iterations=8,
    max_arity=4,
    root_type=grid,
    step_fn=approach(4, 5),
)

# ── Report ─────────────────────────────────────────────────────────────────
n1 = sum(1 for x in Xs_1 if mat_key(x) in Z_1 and Z_1[mat_key(x)] is not None)
n4 = sum(1 for x in Xs_4 if mat_key(x) in Z_4 and Z_4[mat_key(x)] is not None)
print(f"\n=== Results ===")
print(f"  agent 1: solved {n1}/{len(Xs_1)}")
print(f"  agent 4: solved {n4}/{len(Xs_4)}")

def report_invented(label, D):
    print(f"\n=== Invented primitives ({label}) ===")
    if not D.invented:
        print("  (none)")
        return
    for d in D.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        print(f"  {d.repr}  [{argtypes}]  -> {d.type}")
        print(f"    body: {body_str}")
        if 'place_wall' in body_str:
            print(f"    *** WORLD MODEL — grid modification rationalising agent path ***")

report_invented("agent 1", D_1)
report_invented("agent 4", D_4)

print("\n=== Solutions (first 4 tasks) ===")
for i, (x1, x4, m) in enumerate(list(zip(Xs_1, Xs_4, meta))[:4]):
    k1, k4 = mat_key(x1), mat_key(x4)
    tag = (f"a1={m['agent1']} pw1={m['pw1']}  "
           f"a2={m['agent2']} pw2={m['pw2']}")
    print(f"  task {i}: {tag}")
    if k1 in Z_1 and Z_1[k1] is not None:
        sol1 = normalize(deepcopy(Z_1[k1]))
        rw1  = rewritten_1.get(k1, '')
        print(f"    agent 1 found:     {sol1}")
        if rw1: print(f"    agent 1 rewritten: {rw1}")
    else:
        print(f"    agent 1 → unsolved")
    if k4 in Z_4 and Z_4[k4] is not None:
        sol4 = normalize(deepcopy(Z_4[k4]))
        rw4  = rewritten_4.get(k4, '')
        print(f"    agent 4 found:     {sol4}")
        if rw4: print(f"    agent 4 rewritten: {rw4}")
    else:
        print(f"    agent 4 → unsolved")
