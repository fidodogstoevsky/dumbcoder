"""Audit: which belief/scaffold tasks do the Jul-12 round-1 base-prim rivals solve?

Rebuilds the fn corpus with run_phase's exact seeds/counts (full, non-smoke) and
replays the suspicious caught programs from the Jul-12 logs against every task.
"""
import sys
sys.path.insert(0, '/Users/gidonkaminer/Documents/mol/s26/dumbcoder')
import numpy as np

from dsl import (RIGHT, LEFT, UP, DOWN, compose, step, optimize, neg_distance,
                 distance, wall_at, clear_at, erase, unfold, fork, sync_to_world,
                 sync_all, sync_except, snd_gg)
from tasks_minds import (make_physics_tasks, make_desire_tasks, make_belief_tasks,
                         make_witness_belief_tasks, make_goal_displacement_tasks,
                         make_dual_belief_tasks, make_false_obstacle_belief_tasks,
                         COMBOS, belief_variant)

# full-run corpus (run_phase non-smoke): n_bel=6, n_goal=6, n_belvar=3, scaffold=3
print("generating corpora (same seeds as run_phase)…", flush=True)
bel  = make_witness_belief_tasks(6, COMBOS, seed=2)
gdb  = make_goal_displacement_tasks(6, COMBOS, seed=23)
dual = make_dual_belief_tasks(3, COMBOS, seed=24)
fob  = make_false_obstacle_belief_tasks(3, COMBOS, seed=25)
scaf = make_belief_tasks(3, COMBOS, seed=22)
for _, m in scaf:
    m['kind'] = 'belief_scaffold'
tasks = ([(x, m, 'witness') for x, m in bel]
         + [(x, m, 'goal') for x, m in gdb]
         + [(x, m, 'dual') for x, m in dual]
         + [(x, m, 'fob') for x, m in fob]
         + [(x, m, 'scaffold') for x, m in scaf])
print(f"  {len(bel)} witness, {len(gdb)} goal, {len(dual)} dual, {len(fob)} fob, "
      f"{len(scaf)} scaffold = {len(tasks)}", flush=True)

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

hits = []
for name, prog in RIVALS.items():
    for x, m, fam in tasks:
        try:
            if np.array_equal(unfold(x[0], x.shape[0], prog), x):
                hits.append((name, fam, m))
        except Exception:
            pass

print("\n=== rival hits ===")
for name, fam, m in hits:
    keys = {k: v for k, v in m.items() if k != 'kind'}
    print(f"  {name:38s} -> {fam:9s} {belief_variant(m) if m['kind']=='belief' else m['kind']} {keys}")
if not hits:
    print("  (none — rivals do not reproduce any regenerated scene)")

# generic snd_gg-readout audit: is any belief scene reproducible by fork(derive, snd_gg)
# where derive = [optional wall stamp anywhere] ∘ seek(gv',av') over scene values?
print("\n=== snd_gg-readout audit (fork(derive, snd_gg) sweep) ===", flush=True)
n_vuln = 0
for x, m, fam in tasks:
    g = x[0]
    vals = [int(v) for v in np.unique(g) if v not in (0, 3)]
    found = None
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
                try:
                    if np.array_equal(unfold(g, x.shape[0], fork(d, snd_gg)), x):
                        found = (av, gv)
                        raise StopIteration
                except StopIteration:
                    raise
                except Exception:
                    pass
    if found:
        n_vuln += 1
        print(f"  VULNERABLE [{fam}] {belief_variant(m) if m['kind']=='belief' else m['kind']} "
              f"av={m.get('av')} gv={m.get('gv')} via seek({found[1]})/{found[0]}")
print(f"  {n_vuln}/{len(tasks)} scenes reproducible by a snd_gg readout (this sweep)")
