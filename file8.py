"""File 8: multi-agent desire assignment synthesis.

Goal: synthesize the desire assignment function for heterogeneous multi-agent
navigation. Agent 4 seeks goal 2; agent 1 seeks goal 5 (crossed assignment).

DSL:
  assign_step(av, step_fn, fallback) :: int, fn, agent_step -> agent_step
  no_step                             :: agent_step (identity for any agent)
  seek(av, gv)                        :: int, int -> fn   (= optimize(neg_dist(gv), av))

──────────────────────────────────────────────────────────────────────────────
Stage 1 (bootstrap, single-agent tasks):

  Bootstrap solution (5 nodes):
    assign_step(av, seek(av, gv), no_step)    ← av appears TWICE

  Stitch discovers:
    fn_desire_base($gv, $av) = assign_step($av, seek($av, $gv), no_step)
    with $av (×2) as the shared variable — same agent is both the subject
    of the assignment AND the executor inside seek.

──────────────────────────────────────────────────────────────────────────────
Stage 2 (multi-agent, generalize fn_desire_base → fn_desire with free fallback):

  fn_desire($gv, $av, $fallback) = assign_step($av, seek($av, $gv), $fallback)
  (fn_desire_base is fn_desire with $fallback = no_step)

  Two-agent solution (7 nodes):
    fn_desire(2, 4, fn_desire(5, 1, no_step))

  seek(av, gv) = optimize(neg_dist(gv), av), so:
    fn_desire(2, 4, …) → optimize(neg_dist(2), 4)   agent 4 seeks goal 2
    fn_desire(5, 1, …) → optimize(neg_dist(5), 1)   agent 1 seeks goal 5

Run:
  python file8.py
"""

import numpy as np
from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    solve_enumeration, mat_key, normalize,
)
from dsl import (
    fn, agent_step,
    assign_step, no_step,
    seek, freeze,
)
import torch as th

from tasks import (
    make_desire_task,
    make_multi_agent_desire_tasks,
)

# ── Configuration ───────────────────────────────────────────────────────────
AGENT_GOAL_PAIRS = [(4, 2), (1, 5)]   # agent 4 seeks goal 2, agent 1 seeks goal 5

# ── Tasks ───────────────────────────────────────────────────────────────────
def make_bootstrap_tasks(n_per_pair=10, size=4, seed=0):
    """Single-agent tasks for all four (av, gv) pairings."""
    rng = np.random.default_rng(seed)
    tasks = []
    for av in (4, 1):
        for gv in (2, 5):
            count = 0
            while count < n_per_pair:
                result = make_desire_task(gv, size=size,
                                          seed=int(rng.integers(1 << 31)))
                if result is None:
                    continue
                x, meta = result
                if av != 1:
                    x2 = x.copy(); x2[x == 1] = av; x = x2
                tasks.append((x, {**meta, 'av': av, 'gv': gv}))
                count += 1
            print(f"  bootstrap av={av}→gv={gv}: {count} tasks")
    return tasks

print("Generating tasks…")
bootstrap = make_bootstrap_tasks(n_per_pair=10, size=4, seed=0)
Xs_boot   = [x for x, _ in bootstrap]

multi_raw  = make_multi_agent_desire_tasks(n=20, agent_goal_pairs=AGENT_GOAL_PAIRS,
                                           size=5, seed=0)
Xs_multi   = [x for x, _ in multi_raw]
meta_multi = [m for _, m in multi_raw]
print(f"  {len(Xs_boot)} bootstrap + {len(Xs_multi)} multi-agent tasks\n")

# ── Stage 1 DSL ─────────────────────────────────────────────────────────────
core_prims = [
    Delta(assign_step, agent_step, [int, fn, agent_step], repr='assign_step'),
    Delta(no_step,     agent_step,                        repr='no_step'),
    Delta(seek,        fn,         [int, int],            repr='seek'),
    Delta(1, int, repr='1'),
    Delta(2, int, repr='2'),
    Delta(4, int, repr='4'),
    Delta(5, int, repr='5'),
]
D1 = Deltas(core_prims)
print(f"Stage 1 DSL: {len(D1)} primitives")
print("  target: assign_step($av, seek($av, $gv), no_step)  ← $av appears twice\n")

# ── Stage 1 ECD: bootstrap ───────────────────────────────────────────────────
print("=== Stage 1: bootstrap ECD ===\n")
Z1, rw1 = ECD(
    Xs_boot, D1,
    per_task_timeout=30,
    max_iterations=4,
    max_arity=3,
    stitch_iterations=4,
    root_type=agent_step,
    agents=AGENT_GOAL_PAIRS,
)

n1 = sum(1 for x in Xs_boot if mat_key(x) in Z1 and Z1[mat_key(x)] is not None)
print(f"\n--- Stage 1 results: {n1}/{len(Xs_boot)} bootstrap solved")

print("\n--- Stage 1 invented primitives:")
fn_desire_base = None
for d in D1.invented:
    body_str = str(normalize(deepcopy(d)))
    shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
    print(f"  {d.repr}  [{', '.join(str(t) for t in (d.tailtypes or []))}]  -> {d.type}")
    print(f"    body: {body_str}")
    if 'assign_step' in body_str and 'seek' in body_str and shared:
        fn_desire_base = d
        label = ', '.join(f'{v} (×{c})' for v, c in shared.items())
        print(f"    *** SHARED VARIABLE: {label} — same agent carries and executes desire ***")
        print(f"    *** seek(av,gv) = optimize(neg_dist(gv), av) ***")

print("\n--- Stage 1 bootstrap solutions (first 2 per pairing):")
pair_seen = {}
for x, m in bootstrap:
    av, gv = m['av'], m['gv']
    key = (av, gv)
    if pair_seen.get(key, 0) >= 2:
        continue
    pair_seen[key] = pair_seen.get(key, 0) + 1
    k = mat_key(x)
    if k in Z1 and Z1[k] is not None:
        print(f"  av={av}→gv={gv}: {normalize(deepcopy(Z1[k]))}")
    else:
        print(f"  av={av}→gv={gv}: unsolved")

# ── Stage 2: generalize fn_desire_base → fn_desire (free fallback) ──────────
print("\n\n=== Stage 2: generalize to fn_desire with free fallback ===")
print("fn_desire_base($gv, $av) = assign_step($av, seek($av, $gv), no_step)")
print("fn_desire($gv, $av, $fallback) = assign_step($av, seek($av, $gv), $fallback)")
print("  → fn_desire_base is fn_desire with $fallback = no_step (the base case)")
print("  → seek(av,gv) = optimize(neg_dist(gv), av)")
print()

# Construct fn_desire as a Delta with hiddentail:
#   body = assign_step($1, seek($1, $0), $2)   — $0=gv, $1=av (×2), $2=fallback
_body = Delta(assign_step, agent_step, [int, fn, agent_step], repr='assign_step')
_av_a = Delta('$1', ishole=True, type=int)     # $av for assign_step first arg
_av_b = Delta('$1', ishole=True, type=int)     # $av for seek first arg (shared!)
_gv   = Delta('$0', ishole=True, type=int)     # $gv for seek second arg
_seek = Delta(seek, fn, [int, int], repr='seek')
_seek.tails = [_av_b, _gv]
_fb   = Delta('$2', ishole=True, type=agent_step)   # $fallback
_body.tails = [_av_a, _seek, _fb]

fn_desire = Delta(
    'fn_desire', type=agent_step,
    tailtypes=[int, int, agent_step],
    hiddentail=_body, repr='fn_desire'
)
freeze(fn_desire)

# Build Stage 2 DSL: core prims + fn_desire
D2 = Deltas(core_prims + [fn_desire])
print(f"Stage 2 DSL: {len(D2)} primitives (added fn_desire)")
print("  target: fn_desire(2, 4, fn_desire(5, 1, no_step))\n")

# ── Stage 2 ECD: multi-agent tasks ──────────────────────────────────────────
print("=== Stage 2: multi-agent ECD ===\n")
Z2, rw2 = ECD(
    Xs_multi, D2,
    per_task_timeout=60,
    max_iterations=4,
    max_arity=3,
    stitch_iterations=4,
    root_type=agent_step,
    agents=AGENT_GOAL_PAIRS,
)

n2 = sum(1 for x in Xs_multi if mat_key(x) in Z2 and Z2[mat_key(x)] is not None)
print(f"\n--- Stage 2 results: {n2}/{len(Xs_multi)} multi-agent solved")

print("\n--- Multi-agent solutions (first 4):")
for x, m in list(zip(Xs_multi, meta_multi))[:4]:
    k = mat_key(x)
    pairs_str = ', '.join(f'av={av}→gv={gv}' for av, gv in m['agent_goal_pairs'])
    if k in Z2 and Z2[k] is not None:
        compact  = str(Z2[k])                     # compact: shows fn_desire(…)
        expanded = str(normalize(deepcopy(Z2[k])))  # expanded: shows assign_step/seek
        print(f"  [{pairs_str}]")
        print(f"    compact:  {compact}")
        print(f"    expanded: {expanded}")
    else:
        print(f"  [{pairs_str}]  → unsolved")

print("\n--- Stage 2 invented primitives:")
if not D2.invented:
    print("  (none beyond fn_desire)")
else:
    for d in D2.invented:
        body_str = str(normalize(deepcopy(d)))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        print(f"  {d.repr}  [{', '.join(str(t) for t in (d.tailtypes or []))}]  -> {d.type}")
        print(f"    body: {body_str}")
