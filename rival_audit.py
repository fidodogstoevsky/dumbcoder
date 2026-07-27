"""Audit: which belief/scaffold tasks do the Jul-12 round-1 base-prim rivals solve?

Rebuilds the fn corpus with run_phase's exact seeds/counts (full, non-smoke) and
replays the suspicious caught programs from the Jul-12 logs against every task.

Since tasks became k-scene sets (scenes.py), a rival only counts as a leak if it
reproduces EVERY scene of a task — that is the condition under which the searcher
would actually return it.  The audit reports both numbers, because the gap between
them is the measurement that justifies deleting the generators' rival batteries:
"fits some scene" is what the old per-scene filters had to reject one coincidence at
a time, and "fits all scenes" is what survives the format change.
"""
import sys
sys.path.insert(0, '/Users/gidonkaminer/Documents/mol/s26/dumbcoder')
import numpy as np

from dsl import (RIGHT, DOWN, compose, step, optimize, neg_distance,
                 distance, wall_at, clear_at, erase, unfold, fork, sync_to_world,
                 snd_gg)
from scenes import solves, solves_any
from tasks import (make_belief_tasks,
                   make_witness_belief_tasks, make_goal_displacement_tasks,
                   make_two_observer_tasks, make_false_obstacle_belief_tasks,
                   COMBOS, DIRS, belief_variant, belief_rival_specs)

# full-run corpus (run_phase non-smoke): n_bel=6, n_goal=6, n_belvar=3, belief_extra=3
print("generating corpora (same seeds as run_phase)…", flush=True)
bel  = make_witness_belief_tasks(6, COMBOS, seed=2)
gdb  = make_goal_displacement_tasks(6, COMBOS, seed=23)
obs  = make_two_observer_tasks(3, COMBOS, seed=27)
fob  = make_false_obstacle_belief_tasks(3, COMBOS, seed=25)
wall = make_belief_tasks(3, COMBOS, seed=22)   # plain wall-belief (kind='belief')
tasks = ([(x, m, 'witness') for x, m in bel]
         + [(x, m, 'goal') for x, m in gdb]
         + [(x, m, 'observers') for x, m in obs]
         + [(x, m, 'fob') for x, m in fob]
         + [(x, m, 'wall') for x, m in wall])
print(f"  {len(bel)} witness, {len(gdb)} goal, {len(obs)} observers, {len(fob)} fob, "
      f"{len(wall)} wall = {len(tasks)}", flush=True)

RIVALS = {
    # the 189730 (phase1) / 25110 (phase2) pure-compose program
    'R1 step-right,flee-self,flee-goal':
        compose(compose(step(9, RIGHT), optimize(distance(9), 9)),
                optimize(distance(4), 9)),
    # phase2-long line 74
    'R2 flee(9)8,step down,step right':
        compose(compose(optimize(distance(9), 8), step(8, DOWN)), step(8, RIGHT)),
    # phase1-long round1: mental-form but flee-derive (distance) belief solve
    'R3 fork(wall(1,0),flee 9->8, sync 8)':
        fork(compose(wall_at(1, 0), optimize(distance(9), 8)), sync_to_world(8)),
    # erase-3 transient-wall rivals (phase2-long round 1)
    'E1 wall(2,3),seek4/9,erase3':
        compose(compose(wall_at(2, 3), optimize(neg_distance(4), 9)), erase(3)),
    'E2 wall(1,2),seek1/7,erase3':
        compose(compose(wall_at(1, 2), optimize(neg_distance(1), 7)), erase(3)),
    'E3 wall(2,1),seek1/7,erase3':
        compose(compose(wall_at(2, 1), optimize(neg_distance(1), 7)), erase(3)),
    'E4 wall(3,4),seek4/9,erase3':
        compose(compose(wall_at(3, 4), optimize(neg_distance(4), 9)), erase(3)),
    # clear_at transient-wall rivals (both phase2 runs + phase1? round 1)
    'C1 wall(3,1),seek2/1,clear(3,1)':
        compose(compose(wall_at(3, 1), optimize(neg_distance(2), 1)), clear_at(3, 1)),
    'C2 wall(2,2),seek6/2,clear(2,2)':
        compose(compose(wall_at(2, 2), optimize(neg_distance(6), 2)), clear_at(2, 2)),
    'C3 wall(0,1),seek4/9,clear(0,1)':
        compose(compose(wall_at(0, 1), optimize(neg_distance(4), 9)), clear_at(0, 1)),
}

hits, partial = [], []
for name, prog in RIVALS.items():
    for x, m, fam in tasks:
        if solves(prog, x):
            hits.append((name, fam, m))
        elif solves_any(prog, x):
            partial.append((name, fam, m))

print("\n=== rival hits (reproduce EVERY scene — a real leak) ===")
for name, fam, m in hits:
    keys = {k: v for k, v in m.items() if k not in ('kind', 'per_scene')}
    print(f"  {name:38s} -> {fam:9s} {belief_variant(m) if m['kind']=='belief' else m['kind']} {keys}")
if not hits:
    print("  (none — no rival reproduces every scene of any regenerated task)")

print("\n=== near misses (reproduce SOME scene, die on another) ===")
print(f"  {len(partial)} task/rival pairs a single-scene corpus would have leaked")
for name, fam, m in partial[:12]:
    print(f"    {name:38s} -> {fam:9s} "
          f"{belief_variant(m) if m['kind']=='belief' else m['kind']}")
if len(partial) > 12:
    print(f"    … and {len(partial) - 12} more")

# Regression gate.  Scaffold is EXPECTED to remain reproducible by the transient-wall
# rivals: on a single-agent scene the transient real wall is extensionally identical to
# the private-copy belief in every frame (the reason witness tasks exist).  That leak is
# unfixable in DATA and is instead handled by run_phase's round-1 t_fn cap, which keeps
# the deep transient-wall program out of the round where belief must win.  Every OTHER
# family (goal / witness / observers / false-obstacle) must have zero hits after (a)+(c).
other_hits = [(name, fam, m) for name, fam, m in hits if fam != 'scaffold']
print(f"\n=== regression gate ===")
print(f"  scaffold hits (by design, handled by round-1 t_fn cap): "
      f"{sum(1 for _, fam, _ in hits if fam == 'scaffold')}")
if other_hits:
    print(f"  FAIL: {len(other_hits)} non-scaffold rival hit(s) — a real leak:")
    for name, fam, m in other_hits:
        print(f"    {name} -> {fam} {belief_variant(m) if m['kind']=='belief' else m['kind']}")
else:
    print("  PASS: 0 non-scaffold rival hits (goal/witness/observers/fob all clean).")

# generic snd_gg-readout audit: is any belief scene reproducible by fork(derive, snd_gg)
# where derive = [optional wall stamp anywhere] ∘ seek(gv',av') over scene values?
print("\n=== snd_gg-readout audit (fork(derive, snd_gg) sweep) ===", flush=True)
n_vuln, n_vuln_1 = 0, 0
for x, m, fam in tasks:
    vals = sorted(x.values)
    found = found_1 = None
    for av in vals:
        for gv in vals + [3]:
            if av == gv:
                continue
            seeks = [optimize(neg_distance(gv), av), optimize(distance(gv), av)]
            derives = []
            for s in seeks:
                derives.append(s)
                for r in range(5):
                    for c in range(5):
                        derives.append(compose(wall_at(r, c), s))
                        derives.append(compose(s, wall_at(r, c)))
            for d in derives:
                p = fork(d, snd_gg)
                if found_1 is None and solves_any(p, x):
                    found_1 = (av, gv)
                if solves(p, x):
                    found = (av, gv)
                    break
            if found:
                break
        if found:
            break
    if found_1:
        n_vuln_1 += 1
    if found:
        n_vuln += 1
        print(f"  VULNERABLE [{fam}] {belief_variant(m) if m['kind']=='belief' else m['kind']} "
              f"av={m.get('av')} gv={m.get('gv')} via seek({found[1]})/{found[0]}")
print(f"  {n_vuln}/{len(tasks)} tasks reproducible by a snd_gg readout on EVERY scene")
print(f"  {n_vuln_1}/{len(tasks)} would have been on a single scene "
      f"(the leak the k-scene format closes)")

# ── transient-shove audit (the displaced-goal families' tightest single-grid rival) ──
# The wall families' hardest rival stamps a wall, acts, and erases it, so the wall never
# shows up in a rendered frame.  The displaced-goal families (Sally-Anne, two-observer)
# have the exact analogue with the goal in place of the wall: shove the goal, let the
# agent seek it there, shove it back — every rendered frame then shows the goal where it
# really is, and no private copy is used.  It is the rival those families must beat, and
# it is not in RIVALS above because it is task-parameterised (it names the task's own
# goal value and shove direction).  Expectation: zero hits, because the agent settles ON
# the believed cell, which is where the transiently shoved goal is standing — so the
# restore has nothing to move and the goal vanishes from the final frame.
OPP = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


def _transient_shove(m):
    "shove goal ▸ (observer seek, if any) ▸ believer seek ▸ shove goal back"
    av, gv = m['av'], m['gv']
    fs = [step(gv, DIRS[d]) for d in m['dirs']]
    back = [step(gv, DIRS[OPP[d]]) for d in reversed(m['dirs'])]
    pre = [optimize(neg_distance(gv), m['observer'])] if 'observer' in m else []
    prog = (pre + fs + [optimize(neg_distance(gv), av)] + back)[0]
    for f in (pre + fs + [optimize(neg_distance(gv), av)] + back)[1:]:
        prog = compose(prog, f)
    return prog


print("\n=== transient-shove audit (displaced-goal families) ===", flush=True)
shove_tasks = [(x, m, fam) for x, m, fam in tasks
               if m['kind'] == 'belief' and 'displaced_to' in m and 'real_wall' not in m]
n_all = sum(solves(_transient_shove(m), x) for x, m, _ in shove_tasks)
n_any = sum(solves_any(_transient_shove(m), x) for x, m, _ in shove_tasks)
print(f"  {n_all}/{len(shove_tasks)} tasks reproduced on EVERY scene "
      f"({n_any} on at least one)")
if n_all:
    print("  FAIL: the transient shove explains a displaced-goal task without a private copy.")
else:
    print("  PASS: the arrival cell kills the restore — no transient-shove leak.")
