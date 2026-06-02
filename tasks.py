import numpy as np
from dsl import UP, DOWN, LEFT, RIGHT


def make_task(path, size=4):
    "create a (T, size, size) matrix with a single 1-cell moving along path"
    x = np.zeros((len(path), size, size), dtype=int)
    for t, (r, c) in enumerate(path):
        x[t, r, c] = 1
    return x

def simple_walk_tasks(size):
    """create size*2*2 tasks: walk each direction of each row/col"""
    tasks = []
    for row in range(size):
        x = np.zeros((size, size, size), dtype=int)
        for t in range(size):
            x[t,row,t] = 1
        tasks.append(x)
    for row in range(size):
        x = np.zeros((size, size, size), dtype=int)
        for t in range(size):
            x[t,row,size-(t+1)] = 1
        tasks.append(x)
    for col in range(size):
        x = np.zeros((size, size, size), dtype=int)
        for t in range(size):
            x[t,t,col] = 1
        tasks.append(x)
    for col in range(size):
        x = np.zeros((size, size, size), dtype=int)
        for t in range(size):
            x[t,size-(t+1),col] = 1
        tasks.append(x)
    return tasks

def make_nav_task(size=6, n_walls=6, agent=None, goal=None, min_dist=None, seed=None):
    """Generate a navigation task on a size×size grid.

    Values: agent=1, goal=2, wall=3.
    Agent follows the BFS shortest path to the goal, one step per frame.
    Walls and goal are visible on every frame; goal disappears when reached.

    min_dist: minimum Manhattan distance between agent and goal.
              Defaults to size // 2.

    Returns (T, size, size) int array, or None if no valid placement exists.
    """
    from collections import deque

    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)

    # place walls first
    it = iter(cells)
    walls = set()
    while len(walls) < n_walls:
        walls.add(next(it))

    # pick agent and goal from remaining cells, enforcing min_dist
    free = [c for c in cells if c not in walls]
    if agent is None and goal is None:
        placed = False
        for i, a in enumerate(free):
            for g in free[i+1:]:
                if abs(a[0]-g[0]) + abs(a[1]-g[1]) >= min_dist:
                    agent, goal = a, g
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    elif agent is None:
        candidates = [c for c in free if c != goal and abs(c[0]-goal[0]) + abs(c[1]-goal[1]) >= min_dist]
        if not candidates:
            return None
        agent = candidates[0]
    elif goal is None:
        candidates = [c for c in free if c != agent and abs(c[0]-agent[0]) + abs(c[1]-agent[1]) >= min_dist]
        if not candidates:
            return None
        goal = candidates[0]

    # BFS shortest path (4-directional)
    queue = deque([(agent, [agent])])
    visited = {agent}
    path = None
    while queue:
        pos, cur = queue.popleft()
        if pos == goal:
            path = cur
            break
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nb = (pos[0]+dr, pos[1]+dc)
            if 0 <= nb[0] < size and 0 <= nb[1] < size and nb not in walls and nb not in visited:
                visited.add(nb)
                queue.append((nb, cur + [nb]))

    if path is None:
        return None

    T = len(path)
    x = np.zeros((T, size, size), dtype=int)
    for t, (ar, ac) in enumerate(path):
        for wr, wc in walls:
            x[t, wr, wc] = 3
        gr, gc = goal
        x[t, gr, gc] = 2 if (ar, ac) != goal else 1  # goal replaced by agent on arrival
        x[t, ar, ac] = 1
    return x

def make_nav_tasks(n=12, size=6, n_walls=6, seed=0):
    "generate n random navigation tasks, retrying on unsolvable layouts"
    rng = np.random.default_rng(seed)
    tasks = []
    attempt = 0
    while len(tasks) < n:
        t = make_nav_task(size=size, n_walls=n_walls, seed=int(rng.integers(1<<31)))
        attempt += 1
        if t is not None:
            tasks.append(t)
    print(f"generated {n} nav tasks ({attempt} attempts, {size}x{size}, {n_walls} walls)")
    return tasks


def make_fixed_wall_task(walls, size=4, seed=None, min_dist=None):
    """Generate a nav task with fixed wall positions and variable agent/goal.

    walls: sequence of (r, c) tuples — identical across every task in the corpus.
    Agent and goal are placed randomly on free cells.
    Because wall positions are hard-coded constants shared across all programs,
    stitch can factor them into a reusable abstraction (unlike pure nav tasks
    where only leaf integers vary).

    Returns (T, size, size) int array, or None if no valid placement or path.
    """
    from collections import deque
    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)
    wall_set = set(map(tuple, walls))
    free = [(r, c) for r in range(size) for c in range(size) if (r, c) not in wall_set]
    rng.shuffle(free)

    agent = goal = None
    for i, a in enumerate(free):
        for g in free[i+1:]:
            if abs(a[0]-g[0]) + abs(a[1]-g[1]) >= min_dist:
                agent, goal = a, g
                break
        if agent:
            break
    if agent is None:
        return None

    queue = deque([(agent, [agent])])
    visited = {agent}
    path = None
    while queue:
        pos, cur = queue.popleft()
        if pos == goal:
            path = cur
            break
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nb = (pos[0]+dr, pos[1]+dc)
            if (0 <= nb[0] < size and 0 <= nb[1] < size
                    and nb not in wall_set and nb not in visited):
                visited.add(nb)
                queue.append((nb, cur + [nb]))

    if path is None:
        return None

    T = len(path)
    x = np.zeros((T, size, size), dtype=int)
    for t, (ar, ac) in enumerate(path):
        for wr, wc in wall_set:
            x[t, wr, wc] = 3
        gr, gc = goal
        x[t, gr, gc] = 2 if (ar, ac) != goal else 1
        x[t, ar, ac] = 1
    return x


def make_fixed_wall_tasks(n, walls, size=4, seed=0):
    """Generate n nav tasks with fixed wall positions, variable agent/goal.

    walls: sequence of (r, c) tuples fixed across all tasks.
    Every solution program shares the same place_wall(…, r, c) subtrees,
    giving stitch a concrete repeated structure to compress into an abstraction.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    attempts = 0
    while len(tasks) < n:
        t = make_fixed_wall_task(walls, size=size, seed=int(rng.integers(1<<31)))
        attempts += 1
        if t is not None:
            tasks.append(t)
    wall_str = ' '.join(f'({r},{c})' for r, c in walls)
    print(f"generated {n} fixed-wall tasks ({attempts} attempts, {size}x{size}, walls=[{wall_str}])")
    return tasks


def make_false_belief_task(size=6, n_phantoms=1, seed=None, min_dist=None, return_meta=False):
    """Generate a false belief navigation task.

    The agent navigates optimally on their BELIEVED grid (which has phantom
    walls not present in the actual world).  The observed trajectory X shows
    only agent (1) and goal (2) — phantom walls are hidden — so the path
    appears suboptimal on the blank actual grid.

    Solution program: mask(unfold(place_wall(place_ag(blank,...),pwr,pwc), T, navigate), 3)

    Bootstrap fails: reading frame 0 finds no walls, so the bootstrapped
    0-wall solution produces the optimal path, which doesn't match X.
    ECD must search over phantom wall positions to explain the detour.

    Returns (T, size, size) int array, or None if no valid layout found.
    If return_meta=True, returns (x, {'agent', 'goal', 'phantom_walls'}) or None.
    """
    from collections import deque

    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)

    def bfs(start, end, walls):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == end:
                return path
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nb = (pos[0]+dr, pos[1]+dc)
                if 0<=nb[0]<size and 0<=nb[1]<size and nb not in walls and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path+[nb]))
        return None

    for _ in range(200):
        cells = [(r,c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)

        # pick agent and goal satisfying min_dist
        agent = goal = None
        for i, a in enumerate(cells):
            for g in cells[i+1:]:
                if abs(a[0]-g[0]) + abs(a[1]-g[1]) >= min_dist:
                    agent, goal = a, g
                    break
            if agent:
                break
        if agent is None:
            continue

        # optimal path on blank grid
        optimal = bfs(agent, goal, set())
        if not optimal:
            continue

        # phantom wall candidates: interior cells of the optimal path
        # that force a genuine detour when blocked
        interior = [p for p in optimal[1:-1]
                    if p != agent and p != goal]
        if len(interior) < n_phantoms:
            continue

        rng.shuffle(interior)
        phantom_walls = tuple(interior[:n_phantoms])
        phantom_set = set(phantom_walls)

        # believed path: BFS avoiding phantom walls
        believed = bfs(agent, goal, phantom_set)
        if believed is None or believed == optimal:
            continue  # no actual detour

        T = len(believed)

        # build X: believed trajectory with walls hidden
        x = np.zeros((T, size, size), dtype=int)
        for t, (ar, ac) in enumerate(believed):
            x[t, goal[0], goal[1]] = 2 if (ar, ac) != goal else 1
            x[t, ar, ac] = 1

        if return_meta:
            return x, {'agent': agent, 'goal': goal, 'phantom_walls': list(phantom_walls)}
        return x

    return None


def make_false_belief_tasks(n=6, size=6, n_phantoms=1, seed=0, return_meta=False):
    "generate n false belief tasks, retrying on failures"
    rng = np.random.default_rng(seed)
    tasks = []
    attempt = 0
    while len(tasks) < n:
        t = make_false_belief_task(size=size, n_phantoms=n_phantoms,
                                   seed=int(rng.integers(1<<31)),
                                   return_meta=return_meta)
        attempt += 1
        if t is not None:
            tasks.append(t)
    print(f"generated {n} false-belief tasks ({attempt} attempts, {size}x{size}, "
          f"{n_phantoms} phantom wall(s))")
    return tasks


def make_desire_task(goal_val, size=4, seed=None, min_dist=None):
    """Generate a desire task: agent(1) navigates toward goal(goal_val), no walls.

    goal_val: the value the agent is 'desiring' (e.g. 2, 4, 5 — not 3, which is walls).
    The solution program uses gset to place agent and goal, and approach(1, goal_val)
    as the step function — so goal_val appears twice, once in the world representation
    and once in the agent's driving function.  That shared variable is the desire.

    Returns (x, meta) where x is (T, size, size) and
    meta = {'agent', 'goal', 'goal_val', 'T'}, or None on failure.
    """
    from collections import deque
    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)

    agent = goal_pos = None
    for i, a in enumerate(cells):
        for g in cells[i+1:]:
            if abs(a[0]-g[0]) + abs(a[1]-g[1]) >= min_dist:
                agent, goal_pos = a, g
                break
        if agent:
            break
    if agent is None:
        return None

    queue = deque([(agent, [agent])])
    visited = {agent}
    path = None
    while queue:
        pos, cur = queue.popleft()
        if pos == goal_pos:
            path = cur
            break
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nb = (pos[0]+dr, pos[1]+dc)
            if 0 <= nb[0] < size and 0 <= nb[1] < size and nb not in visited:
                visited.add(nb)
                queue.append((nb, cur + [nb]))

    if path is None:
        return None

    T = len(path)
    x = np.zeros((T, size, size), dtype=int)
    gr, gc = goal_pos
    for t, (ar, ac) in enumerate(path):
        x[t, gr, gc] = goal_val if (ar, ac) != goal_pos else 1
        x[t, ar, ac] = 1

    return x, {'agent': agent, 'goal': goal_pos, 'goal_val': goal_val, 'T': T}


def make_desire_tasks(n_per_goal, goal_vals=(2, 4, 5), size=4, seed=0):
    """Generate desire tasks for each goal_val in goal_vals.

    n_per_goal tasks are generated per goal value, giving stitch enough variation
    to discover the general desire abstraction: the shared goal_val variable
    appearing in both gset(…, goal_val) and approach(1, goal_val).

    goal_vals: avoid 3 (reserved for walls in approach's BFS).
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for gv in goal_vals:
        count = 0
        while count < n_per_goal:
            result = make_desire_task(gv, size=size, seed=int(rng.integers(1<<31)))
            if result is not None:
                tasks.append(result)
                count += 1
    print(f"generated {len(tasks)} desire tasks "
          f"({n_per_goal}/goal_val, goal_vals={list(goal_vals)}, {size}x{size})")
    return tasks


def make_physics_task(direction, size=4, seed=None):
    """Generate a physics task: agent (1) drifts in direction toward goal (2).

    Agent starts at a random position on the grid (not on the goal edge).
    Goal (2) is placed at the far edge in the same row/col so the trajectory
    is a straight line.  This makes the task explainable by both:
      step(1, dir)    — physics: move in fixed direction
      approach(1, 2)  — intention: navigate toward goal
    Both produce the same frames; step is the simpler hypothesis.

    Returns (x, meta) where x is (T, size, size) and
    meta = {'direction': direction, 'agent': (ar,ac), 'goal': (gr,gc)},
    or None on failure.
    """
    rng = np.random.default_rng(seed)
    dr, dc = direction

    if dr == 0:  # horizontal
        row = int(rng.integers(size))
        if dc > 0:  # RIGHT: start anywhere left of the goal edge
            col = int(rng.integers(size - 1))   # 0 .. size-2
            ar, ac, gr, gc = row, col, row, size - 1
        else:       # LEFT: start anywhere right of the goal edge
            col = int(rng.integers(1, size))     # 1 .. size-1
            ar, ac, gr, gc = row, col, row, 0
    else:          # vertical
        col = int(rng.integers(size))
        if dr > 0:  # DOWN: start anywhere above goal edge
            row = int(rng.integers(size - 1))
            ar, ac, gr, gc = row, col, size - 1, col
        else:       # UP: start anywhere below goal edge
            row = int(rng.integers(1, size))
            ar, ac, gr, gc = row, col, 0, col

    g = np.zeros((size, size), dtype=int)
    g[ar, ac] = 1
    g[gr, gc] = 2

    frames = [g.copy()]
    curr = g.copy()
    for _ in range(size * 2):
        agents = [(r, c) for r in range(size) for c in range(size) if curr[r, c] == 1]
        if not agents:
            break
        r, c = agents[0]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < size and 0 <= nc < size):
            break
        nxt = curr.copy()
        nxt[r, c] = 0
        nxt[nr, nc] = 1   # overwrites goal (2) on arrival
        curr = nxt
        frames.append(curr.copy())
        if (nr, nc) == (gr, gc):
            break  # agent reached goal — stop before step would overshoot

    if len(frames) < 2:
        return None
    return np.stack(frames), {'direction': direction, 'agent': (ar, ac), 'goal': (gr, gc)}


def make_sequential_desire_task(gv1, gv2, size=4, seed=None, min_dist=None):
    """Sequential desire task: agent(1) approaches goal(gv1) then goal(gv2).

    Both goals are placed in the initial frame. The step function
      if_fn(exists(gv1), approach_1(gv1), approach_1(gv2))
    pursues gv1 while it exists; once consumed, switches to gv2.

    gv1 appears 3× in the solution (gset + exists + approach_1):
      world, trigger, and behaviour share the same value — sequential desire.
    gv2 appears 2× (gset + approach_1).

    Requires gv1 ≠ gv2 and neither equal to 1 (agent) or 3 (wall).

    Returns (x, meta) or None on failure.
    """
    from dsl import _approach_grid
    assert gv1 != gv2, "gv1 must differ from gv2"
    assert gv1 not in (1, 3) and gv2 not in (1, 3), \
        "goal vals cannot be 1 (agent) or 3 (wall)"
    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(size) for c in range(size)]

    for _ in range(500):
        rng.shuffle(cells)
        a_pos, g1_pos, g2_pos = cells[0], cells[1], cells[2]

        if abs(a_pos[0] - g1_pos[0]) + abs(a_pos[1] - g1_pos[1]) < min_dist:
            continue

        # Build initial grid: agent at a_pos, gv1 at g1_pos, gv2 at g2_pos
        g = np.zeros((size, size), dtype=int)
        g[a_pos[0], a_pos[1]] = 1
        g[g1_pos[0], g1_pos[1]] = gv1
        g[g2_pos[0], g2_pos[1]] = gv2

        # Simulate: if gv1 exists → approach gv1, else → approach gv2
        frames = [g.copy()]
        curr = g.copy()

        for _ in range(size * 6):
            if bool(np.any(curr == gv1)):
                nxt = _approach_grid(curr, 1, gv1)
            else:
                nxt = _approach_grid(curr, 1, gv2)

            if np.array_equal(nxt, curr):
                break  # stuck — no reachable goal

            curr = nxt
            frames.append(curr.copy())

            if not bool(np.any(curr == gv2)):
                break  # agent consumed gv2 — done

        # Valid only if agent consumed both goals in order
        if bool(np.any(frames[-1] == gv1)) or bool(np.any(frames[-1] == gv2)):
            continue
        if len(frames) < 3:
            continue

        x = np.stack(frames)
        meta = {
            'agent': a_pos,
            'goal1': g1_pos, 'gv1': gv1,
            'goal2': g2_pos, 'gv2': gv2,
            'T': len(frames),
        }
        return x, meta

    return None


def make_sequential_desire_tasks(n_per_combo, goal_combos=((2, 4), (4, 5), (2, 5)),
                                   size=4, seed=0):
    """Generate sequential desire tasks for each (gv1, gv2) goal combo.

    n_per_combo tasks per combo. goal_combos: pairs (gv1, gv2) with gv1≠gv2,
    neither 1 (agent) nor 3 (wall). Stitch discovers fn_cond_desire from the
    variation in gv1/gv2 across combos.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for gv1, gv2 in goal_combos:
        count = 0
        while count < n_per_combo:
            result = make_sequential_desire_task(
                gv1, gv2, size=size, seed=int(rng.integers(1 << 31))
            )
            if result is not None:
                tasks.append(result)
                count += 1
    combos_str = ', '.join(f'({gv1},{gv2})' for gv1, gv2 in goal_combos)
    print(f"generated {len(tasks)} sequential desire tasks "
          f"({n_per_combo}/combo, combos=[{combos_str}], {size}x{size})")
    return tasks


def make_two_agent_one_false_belief_task(false_agent_val, false_agent_goal_val,
                                          direct_agent_val, direct_agent_goal_val,
                                          size=5, seed=None, min_dist=None):
    """Two agents present; only one has a phantom wall (false belief).

    false_agent navigates on its believed grid (actual + phantom wall).
    direct_agent navigates optimally on the actual grid (no phantom wall).
    Both agents are visible in every frame, so the enumerator must assign
    beliefs correctly per-agent: wm(false_agent_val) = set_at(pw_r, pw_c, 3)
    and wm(direct_agent_val) = id_fn.

    Returns (x, meta) or None on failure.
    """
    from collections import deque
    from dsl import _step_belief, approach, _approach_grid

    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)

    def bfs(start, end, wall_set, sz):
        q = deque([(start, [start])])
        vis = {start}
        while q:
            pos, path = q.popleft()
            if pos == end:
                return path
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nb = (pos[0]+dr, pos[1]+dc)
                if 0<=nb[0]<sz and 0<=nb[1]<sz and nb not in wall_set and nb not in vis:
                    vis.add(nb)
                    q.append((nb, path+[nb]))
        return None

    for _ in range(2000):
        cells = [(r,c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)

        # 4 distinct positions: false_agent, direct_agent, false_goal, direct_goal
        fa_pos, da_pos, fg_pos, dg_pos = cells[0], cells[1], cells[2], cells[3]
        if len({fa_pos, da_pos, fg_pos, dg_pos}) < 4:
            continue

        if (abs(fa_pos[0]-fg_pos[0]) + abs(fa_pos[1]-fg_pos[1]) < min_dist or
                abs(da_pos[0]-dg_pos[0]) + abs(da_pos[1]-dg_pos[1]) < min_dist):
            continue

        opt_fa = bfs(fa_pos, fg_pos, set(), size)
        opt_da = bfs(da_pos, dg_pos, set(), size)
        if not opt_fa or not opt_da:
            continue

        # Phantom wall candidates: interior of false_agent's optimal path,
        # not occupying direct_agent or direct_goal positions
        interior = [p for p in opt_fa[1:-1] if p not in (da_pos, dg_pos)]
        if not interior:
            continue

        rng.shuffle(interior)
        pw = bel_path = None
        for p in interior:
            b = bfs(fa_pos, fg_pos, {p}, size)
            if b and b != opt_fa:
                pw, bel_path = p, b
                break
        if not pw:
            continue

        # Reject if paths collide at any timestep
        max_t = max(len(bel_path), len(opt_da))
        ext_fa = bel_path + [bel_path[-1]] * (max_t - len(bel_path))
        ext_da = opt_da  + [opt_da[-1]]  * (max_t - len(opt_da))
        if any(p1 == p2 for p1, p2 in zip(ext_fa, ext_da)):
            continue

        ig = np.zeros((size, size), dtype=int)
        ig[fa_pos[0], fa_pos[1]] = false_agent_val
        ig[da_pos[0], da_pos[1]] = direct_agent_val
        ig[fg_pos[0], fg_pos[1]] = false_agent_goal_val
        ig[dg_pos[0], dg_pos[1]] = direct_agent_goal_val

        # false agent's believed grid: actual + phantom wall
        bg_fa = ig.copy(); bg_fa[pw[0], pw[1]] = 3

        step_fa = approach(false_agent_val, false_agent_goal_val)
        act = ig.copy()
        bfa = bg_fa.copy()

        frames = [act.copy()]
        for _ in range(size * 4):
            act, bfa = _step_belief(act, bfa, step_fa)
            act = _approach_grid(act, direct_agent_val, direct_agent_goal_val)
            frames.append(act.copy())
            false_done  = not np.any(act == false_agent_goal_val)
            direct_done = not np.any(act == direct_agent_goal_val)
            if false_done and direct_done:
                break

        if len(frames) < 3:
            continue

        x = np.stack(frames)
        meta = {
            'false_agent': fa_pos, 'false_goal': fg_pos, 'phantom_wall': pw,
            'direct_agent': da_pos, 'direct_goal': dg_pos,
            'false_agent_val': false_agent_val,
            'direct_agent_val': direct_agent_val,
        }
        return x, meta

    return None


def make_two_agent_one_false_belief_tasks(n, false_agent_val, false_agent_goal_val,
                                           direct_agent_val, direct_agent_goal_val,
                                           size=5, seed=0):
    """Generate n tasks: false_agent has a phantom wall, direct_agent navigates directly."""
    rng = np.random.default_rng(seed)
    tasks = []
    attempts = 0
    while len(tasks) < n:
        t = make_two_agent_one_false_belief_task(
            false_agent_val, false_agent_goal_val,
            direct_agent_val, direct_agent_goal_val,
            size=size, seed=int(rng.integers(1<<31))
        )
        attempts += 1
        if t is not None:
            tasks.append(t)
    print(f"generated {n} two-agent-one-false-belief tasks ({attempts} attempts, {size}x{size}, "
          f"false={false_agent_val}→{false_agent_goal_val}, "
          f"direct={direct_agent_val}→{direct_agent_goal_val})")
    return tasks


def make_multi_agent_false_belief_task(size=5, seed=None, min_dist=None):
    """Two agents with different phantom wall beliefs navigate to their goals.

    Agent 1 (val=1) → goal (val=2), phantom wall at PW1.
    Agent 2 (val=4) → goal (val=5), phantom wall at PW2.
    PW1 ≠ PW2 so each agent's detour can only be explained by their own belief.

    Agents move simultaneously each frame; observed trajectory shows the actual
    world (no phantom walls). Both agents' suboptimal paths appear in the output.

    Returns (x, meta) or None on failure.
    """
    from collections import deque
    from dsl import _step_belief, approach

    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)

    def bfs(start, end, wall_set, sz):
        q = deque([(start, [start])])
        vis = {start}
        while q:
            pos, path = q.popleft()
            if pos == end:
                return path
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nb = (pos[0]+dr, pos[1]+dc)
                if 0<=nb[0]<sz and 0<=nb[1]<sz and nb not in wall_set and nb not in vis:
                    vis.add(nb)
                    q.append((nb, path+[nb]))
        return None

    for _ in range(2000):
        cells = [(r,c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)

        # Need 4 distinct positions
        a1, a2, g1, g2 = cells[0], cells[1], cells[2], cells[3]

        if (abs(a1[0]-g1[0]) + abs(a1[1]-g1[1]) < min_dist or
                abs(a2[0]-g2[0]) + abs(a2[1]-g2[1]) < min_dist):
            continue

        # Optimal paths on blank grid
        opt1 = bfs(a1, g1, set(), size)
        opt2 = bfs(a2, g2, set(), size)
        if not opt1 or not opt2:
            continue

        # Phantom wall candidates: interior cells, not occupied by the other agent/goal
        interior1 = [p for p in opt1[1:-1] if p not in (a2, g2)]
        interior2 = [p for p in opt2[1:-1] if p not in (a1, g1)]
        if not interior1 or not interior2:
            continue

        rng.shuffle(interior1)
        rng.shuffle(interior2)

        pw1 = pw2 = bel1 = bel2 = None
        for p1 in interior1:
            b1 = bfs(a1, g1, {p1}, size)
            if not b1 or b1 == opt1:
                continue
            for p2 in interior2:
                if p2 == p1:
                    continue
                b2 = bfs(a2, g2, {p2}, size)
                if b2 and b2 != opt2:
                    pw1, pw2, bel1, bel2 = p1, p2, b1, b2
                    break
            if pw1:
                break
        if not pw1:
            continue

        # Ensure believed paths don't collide at any timestep
        max_t = max(len(bel1), len(bel2))
        ext1 = bel1 + [bel1[-1]] * (max_t - len(bel1))
        ext2 = bel2 + [bel2[-1]] * (max_t - len(bel2))
        if any(p1 == p2 for p1, p2 in zip(ext1, ext2)):
            continue

        # Build initial grid (no phantom walls visible)
        ig = np.zeros((size, size), dtype=int)
        ig[a1[0], a1[1]] = 1
        ig[a2[0], a2[1]] = 4
        ig[g1[0], g1[1]] = 2
        ig[g2[0], g2[1]] = 5

        # Believed grids: actual grid + each agent's phantom wall
        bg1 = ig.copy(); bg1[pw1[0], pw1[1]] = 3
        bg2 = ig.copy(); bg2[pw2[0], pw2[1]] = 3

        step1, step2 = approach(1, 2), approach(4, 5)
        act = ig.copy()
        b1, b2 = bg1.copy(), bg2.copy()

        frames = [act.copy()]
        for _ in range(size * 4):
            act, b1 = _step_belief(act, b1, step1)
            act, b2 = _step_belief(act, b2, step2)
            frames.append(act.copy())
            if not np.any(act == 2) and not np.any(act == 5):
                break

        if len(frames) < 3:
            continue

        x = np.stack(frames)
        meta = {
            'agent1': a1, 'goal1': g1, 'pw1': pw1,
            'agent2': a2, 'goal2': g2, 'pw2': pw2,
        }
        return x, meta

    return None


def make_multi_agent_false_belief_tasks(n=20, size=5, seed=0):
    "Generate n multi-agent false-belief tasks (two agents, two distinct phantom walls)."
    rng = np.random.default_rng(seed)
    tasks = []
    attempts = 0
    while len(tasks) < n:
        t = make_multi_agent_false_belief_task(size=size, seed=int(rng.integers(1<<31)))
        attempts += 1
        if t is not None:
            tasks.append(t)
    print(f"generated {n} multi-agent false-belief tasks ({attempts} attempts, {size}x{size})")
    return tasks


def make_multi_agent_desire_task(agent_goal_pairs, size=5, seed=None, min_dist=None):
    """Two agents navigate toward assigned goals on a blank grid (no walls).

    agent_goal_pairs: [(av1, gv1), (av2, gv2)] — agent av_i seeks goal gv_i.
    Agents move sequentially each frame in the order given.
    Returns (x, meta) or None.
    """
    from dsl import _approach_grid
    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)

    for _ in range(500):
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)

        n = len(agent_goal_pairs)
        if len(cells) < 2 * n:
            return None

        agent_pos = cells[:n]
        goal_pos  = cells[n:2*n]

        if len(set(agent_pos + goal_pos)) < 2 * n:
            continue

        if not all(abs(ap[0]-gp[0]) + abs(ap[1]-gp[1]) >= min_dist
                   for ap, gp in zip(agent_pos, goal_pos)):
            continue

        ig = np.zeros((size, size), dtype=int)
        for (av, gv), ap, gp in zip(agent_goal_pairs, agent_pos, goal_pos):
            ig[ap[0], ap[1]] = av
            ig[gp[0], gp[1]] = gv

        frames = [ig.copy()]
        curr = ig.copy()
        for _ in range(size * 4):
            prev = curr.copy()
            for av, gv in agent_goal_pairs:
                curr = _approach_grid(curr, av, gv)
            frames.append(curr.copy())
            if np.array_equal(curr, prev):
                break

        if len(frames) < 3:
            continue

        return np.stack(frames), {
            'agent_goal_pairs': agent_goal_pairs,
            'agent_pos': agent_pos,
            'goal_pos': goal_pos,
        }

    return None


def make_multi_agent_desire_tasks(n, agent_goal_pairs, size=5, seed=0):
    """Generate n multi-agent desire tasks for given agent_goal_pairs."""
    rng = np.random.default_rng(seed)
    tasks = []
    attempts = 0
    while len(tasks) < n:
        t = make_multi_agent_desire_task(
            agent_goal_pairs, size=size, seed=int(rng.integers(1 << 31))
        )
        attempts += 1
        if t is not None:
            tasks.append(t)
    pairs_str = ', '.join(f'{av}→{gv}' for av, gv in agent_goal_pairs)
    print(f"generated {n} multi-agent desire tasks ({attempts} attempts, "
          f"pairs=[{pairs_str}], {size}x{size})")
    return tasks


def make_physics_tasks(n_per_dir, directions=None, size=4, seed=0):
    """Generate physics tasks: agent drifts linearly to goal, one per direction bucket.

    With diverse starting positions and all four directions, stitch discovers:
      fn_physics($ig, $dir) = (unfold $ig (step 1 $dir))
    Direction is an explicit free parameter — the physical degree of freedom
    that intentional vocabulary (approach) would otherwise hide.
    """
    if directions is None:
        directions = [UP, DOWN, LEFT, RIGHT]

    rng = np.random.default_rng(seed)
    tasks = []
    for d in directions:
        count = 0
        while count < n_per_dir:
            result = make_physics_task(d, size=size, seed=int(rng.integers(1<<31)))
            if result is not None:
                tasks.append(result)
                count += 1

    dir_names = {(-1,0):'UP', (1,0):'DOWN', (0,-1):'LEFT', (0,1):'RIGHT'}
    names = [dir_names.get(tuple(d), str(d)) for d in directions]
    print(f"generated {len(tasks)} physics tasks "
          f"({n_per_dir}/dir, dirs={names}, {size}x{size})")
    return tasks


def make_false_belief_desire_task(agent_val, goal_val, size=5, seed=None, min_dist=None):
    """Single-agent false-belief + variable desire.

    Agent agent_val navigates on a believed grid (actual + phantom wall) toward goal_val.
    The observed trajectory shows the actual world — phantom wall is hidden.
    agent_val and goal_val must differ and neither may be 3 (reserved for walls).

    Returns (x, meta) or None.
    """
    from collections import deque

    assert agent_val != goal_val, "agent_val and goal_val must differ"
    assert 3 not in (agent_val, goal_val), "3 is reserved for walls"

    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)

    def bfs(start, end, walls):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == end:
                return path
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (pos[0] + dr, pos[1] + dc)
                if (0 <= nb[0] < size and 0 <= nb[1] < size
                        and nb not in walls and nb not in visited):
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    for _ in range(200):
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)

        agent = goal = None
        for i, a in enumerate(cells):
            for g in cells[i + 1:]:
                if abs(a[0] - g[0]) + abs(a[1] - g[1]) >= min_dist:
                    agent, goal = a, g
                    break
            if agent:
                break
        if agent is None:
            continue

        optimal = bfs(agent, goal, set())
        if not optimal:
            continue

        interior = [p for p in optimal[1:-1] if p != agent and p != goal]
        if not interior:
            continue

        rng.shuffle(interior)
        phantom_wall = believed_path = None
        for pw in interior:
            b = bfs(agent, goal, {pw})
            if b and b != optimal:
                phantom_wall, believed_path = pw, b
                break

        if phantom_wall is None:
            continue

        T = len(believed_path)
        x = np.zeros((T, size, size), dtype=int)
        for t, (ar, ac) in enumerate(believed_path):
            x[t, goal[0], goal[1]] = goal_val if (ar, ac) != goal else agent_val
            x[t, ar, ac] = agent_val

        meta = {
            'agent': agent, 'goal': goal,
            'agent_val': agent_val, 'goal_val': goal_val,
            'phantom_wall': phantom_wall, 'T': T,
        }
        return x, meta

    return None


def make_false_belief_desire_tasks(n_per_combo, agent_goal_combos, size=5, seed=0):
    """Generate false-belief+desire tasks for each (agent_val, goal_val) combo."""
    rng = np.random.default_rng(seed)
    tasks = []
    for av, gv in agent_goal_combos:
        count = 0
        while count < n_per_combo:
            result = make_false_belief_desire_task(
                av, gv, size=size, seed=int(rng.integers(1 << 31))
            )
            if result is not None:
                tasks.append(result)
                count += 1
    combos_str = ', '.join(f'av={av}→gv={gv}' for av, gv in agent_goal_combos)
    print(f"generated {len(tasks)} false-belief+desire tasks "
          f"({n_per_combo}/combo, combos=[{combos_str}], {size}x{size})")
    return tasks


def make_joint_false_belief_desire_task(agent_goal_pairs, size=5, seed=None, min_dist=None):
    """Two agents each with a phantom wall and a variable desired goal.

    agent_goal_pairs: [(av1, gv1), (av2, gv2)] — agent avi seeks goal gvi.
    Each agent navigates on its believed grid (actual + its own phantom wall).
    Returns (x, meta) or None.
    """
    from collections import deque
    from dsl import seek as _seek, _step_belief

    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)
    (av1, gv1), (av2, gv2) = agent_goal_pairs

    def bfs(start, end, wall_set):
        q = deque([(start, [start])])
        vis = {start}
        while q:
            pos, path = q.popleft()
            if pos == end:
                return path
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (pos[0] + dr, pos[1] + dc)
                if (0 <= nb[0] < size and 0 <= nb[1] < size
                        and nb not in wall_set and nb not in vis):
                    vis.add(nb)
                    q.append((nb, path + [nb]))
        return None

    for _ in range(2000):
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)

        a1, a2, g1, g2 = cells[0], cells[1], cells[2], cells[3]
        if len({a1, a2, g1, g2}) < 4:
            continue
        if (abs(a1[0] - g1[0]) + abs(a1[1] - g1[1]) < min_dist or
                abs(a2[0] - g2[0]) + abs(a2[1] - g2[1]) < min_dist):
            continue

        opt1 = bfs(a1, g1, set())
        opt2 = bfs(a2, g2, set())
        if not opt1 or not opt2:
            continue

        interior1 = [p for p in opt1[1:-1] if p not in (a2, g2)]
        interior2 = [p for p in opt2[1:-1] if p not in (a1, g1)]
        if not interior1 or not interior2:
            continue

        rng.shuffle(interior1)
        rng.shuffle(interior2)
        pw1 = pw2 = bel1 = bel2 = None
        for p1 in interior1:
            b1 = bfs(a1, g1, {p1})
            if not b1 or b1 == opt1:
                continue
            for p2 in interior2:
                if p2 == p1:
                    continue
                b2 = bfs(a2, g2, {p2})
                if b2 and b2 != opt2:
                    pw1, pw2, bel1, bel2 = p1, p2, b1, b2
                    break
            if pw1:
                break
        if not pw1:
            continue

        max_t = max(len(bel1), len(bel2))
        ext1 = bel1 + [bel1[-1]] * (max_t - len(bel1))
        ext2 = bel2 + [bel2[-1]] * (max_t - len(bel2))
        if any(p1 == p2 for p1, p2 in zip(ext1, ext2)):
            continue

        ig = np.zeros((size, size), dtype=int)
        ig[a1[0], a1[1]] = av1
        ig[a2[0], a2[1]] = av2
        ig[g1[0], g1[1]] = gv1
        ig[g2[0], g2[1]] = gv2

        bg1 = ig.copy(); bg1[pw1[0], pw1[1]] = 3
        bg2 = ig.copy(); bg2[pw2[0], pw2[1]] = 3

        step1, step2 = _seek(av1, gv1), _seek(av2, gv2)
        act = ig.copy()
        b1, b2 = bg1.copy(), bg2.copy()

        frames = [act.copy()]
        for _ in range(size * 4):
            act, b1 = _step_belief(act, b1, step1)
            act, b2 = _step_belief(act, b2, step2)
            frames.append(act.copy())
            if not np.any(act == gv1) and not np.any(act == gv2):
                break

        if len(frames) < 3:
            continue

        x = np.stack(frames)
        meta = {
            'agent_goal_pairs': agent_goal_pairs,
            'agent_pos': [a1, a2], 'goal_pos': [g1, g2],
            'phantom_walls': [pw1, pw2],
        }
        return x, meta

    return None


def make_joint_false_belief_desire_tasks(n, agent_goal_pairs, size=5, seed=0):
    """Generate n two-agent false-belief+desire tasks."""
    rng = np.random.default_rng(seed)
    tasks = []
    attempts = 0
    while len(tasks) < n:
        t = make_joint_false_belief_desire_task(
            agent_goal_pairs, size=size, seed=int(rng.integers(1 << 31))
        )
        attempts += 1
        if t is not None:
            tasks.append(t)
    pairs_str = ', '.join(f'av={av}→gv={gv}' for av, gv in agent_goal_pairs)
    print(f"generated {n} joint false-belief+desire tasks ({attempts} attempts, "
          f"pairs=[{pairs_str}], {size}x{size})")
    return tasks
