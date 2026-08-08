"""Frame-by-frame render of a REFUTED rival program, for chapter 4's exposition.

Every other script in this directory visualises something the search FOUND.  This one
visualises something that cannot be found, because showing why is half of what makes the
goal-displacement family an argument rather than a stipulation.

The rival is the world shove: _"at each time step, move the goal `dirs`, then move the
agent to the adjacent cell closest to the goal"_ — the obvious reading of a Sally-Anne
scene, and the one `tasks.belief_rival_specs` prices as 'shove goal in world'.  It is
perfectly expressible; it is refuted by the scenes.  Run it forward from the SAME t0 the
reader is looking at in @fig-belief-goal and the refutation is visual:

    t0  agent and goal as the scene begins
    t1  the goal has jumped two cells up — already wrong, since every frame of the
        real scene shows it sitting still
    t2  the goal has been shoved off the top edge and no longer exists (`step` drops
        whatever leaves the grid)
    t3+ the agent is frozen: with no goal on the grid `neg_distance` is -inf at its own
        cell and at all four neighbours alike, `optimize` finds no strict improvement,
        and it never moves again

which is the point the prose needs — not merely that the rival is wrong, but that it
destroys the very thing the planner was aiming at.

The t0 grid is READ FROM `task_samples.json` rather than regenerated, so the rival's
opening frame is by construction the same one @fig-belief-goal shows; a regenerated task
would be a different instance and the comparison would be a cheat.  av/gv/dirs come off
that sample's tag for the same reason.

    python viz/rival_frames.py                    # scene 1 of belief_goal, 4 frames
    python viz/rival_frames.py --scene 2 --frames 5
    python viz/rival_frames.py --print            # ASCII frames on stdout, write nothing

Output is `illc-mol-thesis/rival_samples.json`, schema-identical to task_samples.json
(experiment.export_task_samples), so viz.typ renders it with no changes:

    #figure(
      task-figure("belief_goal_rival", mode: "frames",
                  data: json("rival_samples.json")),
      kind: image,
      caption: [...],
    ) <fig-belief-goal-rival>

It is a SEPARATE file on purpose: task_samples.json is overwritten wholesale by a phase
run's `--samples` path, which would drop anything added to it.
"""

import argparse
import ast
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl import compose, neg_distance, optimize, step, unfold   # noqa: E402
from tasks import DIRS                                          # noqa: E402

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS = os.path.join(ROOT, 'illc-mol-thesis')


def _tag_fields(tag):
    """(av, gv, dirs) off a belief_goal sample tag.

    The tag is the instance's provenance line, e.g.
    "av=1, gv=2, displaced_to=(1, 1), dirs=('up', 'up'), distractors={}".
    """
    def _int(name):
        m = re.search(rf'\b{name}=(\d+)', tag)
        if not m:
            raise SystemExit(f"tag has no {name}=: {tag!r}")
        return int(m.group(1))

    m = re.search(r"\bdirs=(\([^)]*\))", tag)
    if not m:
        raise SystemExit(f"tag has no dirs=: {tag!r}")
    dirs = ast.literal_eval(m.group(1))
    return _int('av'), _int('gv'), tuple(dirs)


def shove_rival(av, gv, dirs):
    """The 'shove goal in world' program as an executable fn: grid -> grid.

    Exactly `tasks.belief_rival_specs`' spelling for this family, left-folded the same
    way `tasks._seq` folds it — every shove, then the planner:

        (compose (compose (step gv d1) (step gv d2)) (optimize (neg_dist gv) av))

    The ONLY difference from the family's true program is where the shoves land: here
    on the world, there on a private copy that the commit never publishes.  That one
    difference is what the figure is for.
    """
    fs = [step(gv, DIRS[d]) for d in dirs] + [optimize(neg_distance(gv), av)]
    prog = fs[0]
    for f in fs[1:]:
        prog = compose(prog, f)
    return prog


def build(samples_path, kind, scene, frames):
    with open(samples_path) as f:
        data = json.load(f)
    sample = next((s for s in data['samples'] if s['kind'] == kind), None)
    if sample is None:
        raise SystemExit(f"no {kind!r} sample in {samples_path}")
    if not 1 <= scene <= len(sample['scenes']):
        raise SystemExit(f"--scene {scene} out of range (task has "
                         f"{len(sample['scenes'])} scenes)")

    av, gv, dirs = _tag_fields(sample['tag'])
    g0 = np.array(sample['scenes'][scene - 1]['panels'][0]['grid'], dtype=int)
    T = frames if frames else sample['scenes'][scene - 1]['T']
    traj = unfold(g0, T, shove_rival(av, gv, dirs))

    shove = ' then '.join(dirs)
    out = {
        'size': data['size'],
        'samples': [{
            'kind': f'{kind}_rival',
            'tag': (f'REFUTED rival: shove {gv} {shove} in the world, then seek it — '
                    f'av={av}, gv={gv}, dirs={dirs}, scene {scene} of the '
                    f'{kind} sample'),
            'k': 1,
            'scenes': [{'T': T, 'panels': [{'label': f't{t}',
                                            'grid': traj[t].astype(int).tolist()}
                                           for t in range(T)]}],
            'T': T,
            'panels': [{'label': f't{t}', 'grid': traj[t].astype(int).tolist()}
                       for t in range(T)],
        }],
    }
    return out, traj, (av, gv, dirs)


def _describe(traj, av, gv):
    "One line per frame: where the agent and the goal are, or that the goal is gone."
    def _pos(g, v):
        w = np.argwhere(g == v)
        return tuple(int(i) for i in w[0]) if len(w) else None

    for t, g in enumerate(traj):
        a, go = _pos(g, av), _pos(g, gv)
        print(f"  t{t}: agent {av} at {a}   goal {gv} "
              f"{'at ' + str(go) if go else 'OFF THE GRID — nothing to seek'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--kind', default='belief_goal',
                    help="family whose sample supplies t0 (default: belief_goal)")
    ap.add_argument('--scene', type=int, default=1,
                    help="which scene of that task, 1-based (default: 1)")
    ap.add_argument('--frames', type=int, default=4,
                    help="how many frames to render (default: 4, the scene's own length)")
    ap.add_argument('--samples', default=os.path.join(THESIS, 'task_samples.json'),
                    help="the task_samples.json to read t0 from")
    ap.add_argument('--out', default=os.path.join(THESIS, 'rival_samples.json'))
    ap.add_argument('--print', dest='show', action='store_true',
                    help="print the frames and write nothing")
    args = ap.parse_args()

    out, traj, (av, gv, dirs) = build(args.samples, args.kind, args.scene, args.frames)
    print(f"rival: shove {gv} {' then '.join(dirs)} in the world, then seek it "
          f"(agent {av}), from scene {args.scene} of the {args.kind} sample")
    _describe(traj, av, gv)

    if args.show:
        for t, g in enumerate(traj):
            print(f"\nt{t}")
            print(g)
        return

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
