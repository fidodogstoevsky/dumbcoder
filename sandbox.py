import numpy as np


def make_static_nav(size=4, n_walls=3, seed=None, min_dist=None):
    """Generate a static navigation scene as a single-frame (1, size, size) matrix.

    Values: agent=1, goal=2, wall=3.
    Returns (1, size, size) int array, or None if no valid placement exists.

    min_dist: minimum Manhattan distance between agent and goal.
              Defaults to size // 2.
    """
    if min_dist is None:
        min_dist = size // 2

    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)

    # place walls
    it = iter(cells)
    walls = set()
    while len(walls) < n_walls:
        walls.add(next(it))

    # pick agent and goal from remaining cells, enforcing min_dist
    free = [c for c in cells if c not in walls]
    placed = False
    agent, goal = None, None
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

    x = np.zeros((1, size, size), dtype=int)
    for wr, wc in walls:
        x[0, wr, wc] = 3
    x[0, goal[0], goal[1]] = 2
    x[0, agent[0], agent[1]] = 1
    return x


def make_static_tasks(n=12, size=4, n_walls=3, seed=0):
    "generate n random static navigation scenes"
    rng = np.random.default_rng(seed)
    tasks = []
    attempts = 0
    while len(tasks) < n:
        t = make_static_nav(size=size, n_walls=n_walls, seed=int(rng.integers(1<<31)))
        attempts += 1
        if t is not None:
            tasks.append(t)
    print(f"generated {n} static scenes ({attempts} attempts, {size}x{size}, {n_walls} walls)")
    return tasks


if __name__ == '__main__':
    Xs = make_static_tasks(n=4)
    for i, x in enumerate(Xs):
        print(f'task {i}:')
        print(x[0])
