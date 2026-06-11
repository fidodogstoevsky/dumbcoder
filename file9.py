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
    unfold_scene,
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
    per_task_timeout=120,
    max_iterations=12,
    max_arity=5,
    root_type=scene_model,
    agents=AGENTS,
    content_aware_q=True,
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
