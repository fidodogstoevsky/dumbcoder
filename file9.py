"""File 9: joint belief+desire synthesis via scene_model product type.

Introduces scene_model = (fn_belief, agent_step) to jointly synthesize
what each agent believes AND desires.  Previously belief and desire were
either split across separate ECD runs (file7) or one was hardcoded (files
6/8).  Here both are free parameters of the synthesized program.

DSL combinator:
  mk_agent_scene(av, belief_fn, step_fn, rest) :: int, fn, fn, scene_model -> scene_model
  empty_scene                                   :: scene_model (terminal)

Single-agent solution (8 nodes):
  mk_agent_scene($av, set_at($r,$c,3), seek($av,$gv), empty_scene)

  $av appears TWICE: as the first arg to mk_agent_scene (belief subject)
  AND inside seek (desire executor). This shared variable is the structural
  signature of agency — the entity is both the one whose belief is modelled
  AND the one who executes the goal-seeking behaviour.

Expected stitch discovery — the agent type constructor:
  fn_agent($av, $r, $c, $gv, $rest) =
    mk_agent_scene($av, set_at($r,$c,$3), seek($av,$gv), $rest)

Two-agent solution (using fn_agent, 9 nodes):
  fn_agent(1, r1, c1, gv1, fn_agent(4, r2, c2, gv2, empty_scene))

Tasks vary both the phantom wall position (belief content) and the goal
value (desire content) across agents and task instances, forcing stitch
to discover both as free holes in fn_agent.

Run:
  python file9.py
"""

from copy import deepcopy
import re as _re
from collections import Counter

from ecd import (
    Deltas, Delta, ECD,
    normalize, mat_key,
)
from dsl import (
    fn, scene_model,
    mk_agent_scene, empty_scene,
    set_at, id_fn, seek,
)
from tasks import (
    make_false_belief_desire_tasks,
    make_joint_false_belief_desire_tasks,
)

# ── Configuration ──────────────────────────────────────────────────────────────
# agent_vals for simulation ordering; goal values are encoded in the program
AGENTS = [(1, None), (4, None)]

# Single-agent combos: (agent_val, goal_val)
# Vary both av and gv so stitch discovers both as free holes in fn_agent
SINGLE_COMBOS = [(1, 2), (1, 5), (4, 2), (4, 5)]

# Two-agent combos: [(av1,gv1),(av2,gv2)]
# Include crossed goal assignment so gv is genuinely variable per agent
TWO_AGENT_COMBOS = [
    [(1, 2), (4, 5)],   # standard
    [(1, 5), (4, 2)],   # crossed
]

# ── Tasks ──────────────────────────────────────────────────────────────────────
print("Generating tasks…")
raw_single = make_false_belief_desire_tasks(
    n_per_combo=8, agent_goal_combos=SINGLE_COMBOS, size=5, seed=0)
Xs_single   = [x for x, _ in raw_single]
meta_single = [m for _, m in raw_single]

raw_multi = []
for pairs in TWO_AGENT_COMBOS:
    raw_multi += make_joint_false_belief_desire_tasks(
        n=8, agent_goal_pairs=pairs, size=5, seed=len(raw_multi))
Xs_multi   = [x for x, _ in raw_multi]
meta_multi = [m for _, m in raw_multi]

Xs = Xs_single + Xs_multi
print(f"\n{len(Xs_single)} single-agent tasks (combos {SINGLE_COMBOS})")
print(f"{len(Xs_multi)} two-agent tasks (combos {TWO_AGENT_COMBOS})")
print(f"{len(Xs)} total tasks\n")

# ── DSL ────────────────────────────────────────────────────────────────────────
empty_scene_delta = Delta(empty_scene, scene_model, repr='empty_scene')

core_prims = [
    Delta(mk_agent_scene, scene_model, [int, fn, fn, scene_model], repr='mk_agent_scene'),
    empty_scene_delta,
    Delta(set_at, fn,  [int, int, int], repr='set_at'),
    Delta(id_fn,  fn,                   repr='id_fn'),
    Delta(seek,   fn,  [int, int],      repr='seek'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
]

D = Deltas(core_prims)
print(f"DSL: {len(core_prims)} primitives")
print("  single-agent solution (8 nodes):")
print("    mk_agent_scene($av, set_at($r,$c,3), seek($av,$gv), empty_scene)")
print("  $av shared ×2: belief subject AND desire executor")
print("  expected stitch discovery:")
print("    fn_agent($av,$r,$c,$gv,$rest) =")
print("      mk_agent_scene($av, set_at($r,$c,3), seek($av,$gv), $rest)\n")

# ── ECD ────────────────────────────────────────────────────────────────────────
print("Running ECD…\n")
Z, rewritten = ECD(
    Xs, D,
    per_task_timeout=60,
    max_iterations=8,
    max_arity=5,
    root_type=scene_model,
    agents=AGENTS,
)

# ── Report ─────────────────────────────────────────────────────────────────────
n_single = sum(1 for x in Xs_single if mat_key(x) in Z and Z[mat_key(x)] is not None)
n_multi  = sum(1 for x in Xs_multi  if mat_key(x) in Z and Z[mat_key(x)] is not None)
print(f"\n=== Results ===")
print(f"  single-agent: {n_single}/{len(Xs_single)}")
print(f"  two-agent:    {n_multi}/{len(Xs_multi)}")

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
        if 'mk_agent_scene' in body_str and 'seek' in body_str and shared:
            shared_str = ', '.join(f'{v} (×{c})' for v, c in shared.items())
            print(f"    *** AGENT TYPE CONSTRUCTOR")
            print(f"        shared: {shared_str}")
            print(f"        $av in mk_agent_scene (belief subject) AND seek (desire executor) ***")
        elif 'mk_agent_scene' in body_str and shared:
            shared_str = ', '.join(f'{v} (×{c})' for v, c in shared.items())
            print(f"    *** scene fragment — shared: {shared_str} ***")

print("\n=== Sample solutions ===")
print("--- Single-agent (1 per combo) ---")
combo_seen = {}
for x, m in zip(Xs_single, meta_single):
    key = (m['agent_val'], m['goal_val'])
    if combo_seen.get(key, 0) >= 1:
        continue
    combo_seen[key] = combo_seen.get(key, 0) + 1
    k = mat_key(x)
    tag = f"av={m['agent_val']}→gv={m['goal_val']}  pw={m['phantom_wall']}"
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  {tag}")
        print(f"    found:     {sol}")
        if rw:
            print(f"    rewritten: {rw}")
    else:
        print(f"  {tag}  → unsolved")

print("\n--- Two-agent (first 4) ---")
for x, m in list(zip(Xs_multi, meta_multi))[:4]:
    k = mat_key(x)
    pairs_str = ', '.join(f'av={av}→gv={gv}' for av, gv in m['agent_goal_pairs'])
    if k in Z and Z[k] is not None:
        sol = normalize(deepcopy(Z[k]))
        rw  = rewritten.get(k, '')
        print(f"  [{pairs_str}]  pws={m['phantom_walls']}")
        print(f"    found:     {sol}")
        if rw:
            print(f"    rewritten: {rw}")
    else:
        print(f"  [{pairs_str}]  → unsolved")

# ── Stage 2: generalize fn_0 → fn_agent with free $rest ───────────────────────
# ECD is stuck: no two-agent solutions → stitch sees only single-agent programs
# → empty_scene is constant → gets baked in → never becomes a hole.
# Fix: manually construct the composable form by replacing empty_scene with $rest.
#
# fn_agent($gv, $av, $c, $r, $rest) =
#   mk_agent_scene($av, set_at($r, $c, 3), seek($av, $gv), $rest)
# $av shared ×2 (belief subject + desire executor), v=3 hardcoded.
#
# Two-agent solution (11 nodes):
#   fn_agent(gv1, av1, c1, r1, fn_agent(gv2, av2, c2, r2, empty_scene))

print("\n\n=== Stage 2: fn_agent (composable) → two-agent tasks ===")
print("Generalising fn_0: replacing baked-in empty_scene with free $rest hole.")

from dsl import freeze

_ht   = Delta(mk_agent_scene, scene_model, [int, fn, fn, scene_model], repr='mk_agent_scene')
_av_a = Delta('$1', ishole=True, type=int)
_av_b = Delta('$1', ishole=True, type=int)   # shared with _av_a — same $1 hole
_gv   = Delta('$0', ishole=True, type=int)
_c    = Delta('$2', ishole=True, type=int)
_r    = Delta('$3', ishole=True, type=int)
_rest = Delta('$4', ishole=True, type=scene_model)
_sat  = Delta(set_at, fn, [int, int, int], repr='set_at')
_sat.tails  = [_r, _c, Delta(3, int, repr='3')]   # v=3 hardcoded
_sk   = Delta(seek, fn, [int, int], repr='seek')
_sk.tails   = [_av_b, _gv]
_ht.tails   = [_av_a, _sat, _sk, _rest]

fn_agent = Delta('fn_agent', type=scene_model,
                 tailtypes=[int, int, int, int, scene_model],
                 hiddentail=_ht, repr='fn_agent')
freeze(fn_agent)

import numpy as _np

D2 = Deltas(core_prims + [fn_agent])
print(f"  fn_agent DSL: {len(D2)} primitives\n")

# ── Per-agent decomposition ─────────────────────────────────────────────────────
# Two-agent enumeration has 8 free int params (av,gv,r,c per agent) — ~10,000×
# harder than single-agent. Fix: solve each agent independently (4 params each,
# same tractability as Stage 1), assemble, validate, then seed stitch so it sees
# varied $rest values and discovers the composable fn_agent form automatically.

def extract_agent_view(x, m, av):
    gv = next(gv_ for av_, gv_ in m['agent_goal_pairs'] if av_ == av)
    xv = _np.zeros_like(x)
    xv[x == av] = av
    xv[x == gv] = gv
    return xv

Xs_v1 = [extract_agent_view(x, m, 1) for x, m in zip(Xs_multi, meta_multi)]
Xs_v4 = [extract_agent_view(x, m, 4) for x, m in zip(Xs_multi, meta_multi)]

print("Stage 2a: ECD on agent-1 views (4 free int params)…")
Z_v1, _ = ECD(Xs_v1, D2, per_task_timeout=90, max_iterations=4, max_arity=5,
               root_type=scene_model, agents=[(1, None)])
n_v1 = sum(1 for x in Xs_v1 if mat_key(x) in Z_v1 and Z_v1[mat_key(x)])
print(f"  {n_v1}/{len(Xs_v1)} agent-1 views solved")

print("\nStage 2b: ECD on agent-4 views (4 free int params)…")
Z_v4, _ = ECD(Xs_v4, D2, per_task_timeout=90, max_iterations=4, max_arity=5,
               root_type=scene_model, agents=[(4, None)])
n_v4 = sum(1 for x in Xs_v4 if mat_key(x) in Z_v4 and Z_v4[mat_key(x)])
print(f"  {n_v4}/{len(Xs_v4)} agent-4 views solved")

# Assemble: replace empty_scene in agent-1 solution with agent-4 solution.
# fn_agent(gv1,1,c1,r1, empty_scene) + fn_agent(gv2,4,c2,r2, empty_scene)
# → fn_agent(gv1,1,c1,r1, fn_agent(gv2,4,c2,r2, empty_scene))
# Valid because agents navigate independently (BFS avoids only value-3 walls,
# not other agent values), so per-agent trajectories compose correctly.
from dsl import unfold_scene as _unfold_scene

combined_seeds = {}
for x, xv1, xv4 in zip(Xs_multi, Xs_v1, Xs_v4):
    k1, k4, kj = mat_key(xv1), mat_key(xv4), mat_key(x)
    if not (k1 in Z_v1 and Z_v1[k1] and k4 in Z_v4 and Z_v4[k4]):
        continue
    sol = deepcopy(Z_v1[k1])
    sol.tails[-1] = deepcopy(Z_v4[k4])   # graft agent-4 program in place of empty_scene
    try:
        out = _unfold_scene(x[0].copy(), x.shape[0], sol(), [1, 4])
        if _np.array_equal(out, x):
            combined_seeds[kj] = sol
    except Exception:
        pass

n_comb = len(combined_seeds)
print(f"\nAssembled and validated {n_comb}/{len(Xs_multi)} two-agent programs")

# ── Stage 2c: stitch on full seeded corpus ─────────────────────────────────────
# Seeds bypass enumeration. Stitch now sees:
#   single-agent: fn_agent(gv, av, c, r, empty_scene)   ← $rest = empty_scene
#   two-agent:    fn_agent(gv1,1,c1,r1, fn_agent(gv2,4,c2,r2, empty_scene))  ← $rest varies
# With $rest varying across programs, stitch makes it a free hole and discovers
# the composable agent type constructor (not baked to empty_scene).

all_seeds = {mat_key(x): Z.get(mat_key(x)) for x in Xs_single if Z.get(mat_key(x))}
all_seeds.update(combined_seeds)

D3 = Deltas(core_prims + [fn_agent])
print("\nStage 2c: ECD with full seeded corpus…\n")
Z3, rw3 = ECD(
    Xs, D3,
    per_task_timeout=30,
    max_iterations=3,
    max_arity=6,
    root_type=scene_model,
    agents=AGENTS,
    seeds=all_seeds,
)

n3_s = sum(1 for x in Xs_single if mat_key(x) in Z3 and Z3[mat_key(x)])
n3_m = sum(1 for x in Xs_multi  if mat_key(x) in Z3 and Z3[mat_key(x)])
print(f"\n=== Stage 2c results ===")
print(f"  single-agent: {n3_s}/{len(Xs_single)}")
print(f"  two-agent:    {n3_m}/{len(Xs_multi)}")

print("\n=== Stage 2c invented primitives ===")
if not D3.invented:
    print("  (none)")
else:
    for d in D3.invented:
        argtypes = ', '.join(str(t) for t in (d.tailtypes or []))
        body_str = str(normalize(deepcopy(d)))
        shared   = {v: c for v, c in Counter(_re.findall(r'\$\d+', body_str)).items() if c > 1}
        print(f"  {d.repr}  [{argtypes}]  -> {d.type}")
        print(f"    body: {body_str}")
        if 'mk_agent_scene' in body_str and shared:
            shared_str = ', '.join(f'{v} (×{c})' for v, c in shared.items())
            composable = 'empty_scene' not in body_str
            tag = 'COMPOSABLE agent constructor' if composable else 'agent constructor (empty_scene baked in)'
            print(f"    *** {tag}")
            print(f"        shared: {shared_str} ***")

print("\n=== Stage 2c two-agent solutions (first 4) ===")
for x, m in list(zip(Xs_multi, meta_multi))[:4]:
    k = mat_key(x)
    pairs_str = ', '.join(f'av={av}→gv={gv}' for av, gv in m['agent_goal_pairs'])
    if k in Z3 and Z3[k]:
        compact  = str(Z3[k])
        expanded = str(normalize(deepcopy(Z3[k])))
        print(f"  [{pairs_str}]  pws={m['phantom_walls']}")
        print(f"    compact:  {compact}")
        print(f"    expanded: {expanded}")
    else:
        print(f"  [{pairs_str}]  → unsolved")
