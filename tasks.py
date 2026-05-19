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
