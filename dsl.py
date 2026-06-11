from operator import add, mul
import numpy as np
from copy import deepcopy

class _VarSentinel:
    "Marker for lambda-bound variable nodes; deepcopy returns the same object."
    def __deepcopy__(self, memo):
        return self
    def __repr__(self):
        return 'var'

_var_sentinel = _VarSentinel()

# types
mat          = 'mat'          # 3d numpy array (T, H, W)
grid         = 'grid'         # 2d numpy array (H, W)
fn           = 'fn'           # grid -> grid
fn2          = 'fn2'          # grid -> grid -> grid
fn_pred      = 'fn_pred'      # grid -> bool
util         = 'util'         # (grid, int, int) -> float  — positional utility
belief       = 'belief'       # agent's subjective world model (structurally a grid)

# int is used as a type string too
# direction is used as a type string
direction = 'dir'

# direction terminals
RIGHT   = ( 0,  1)
LEFT    = ( 0, -1)
UP      = (-1,  0)
DOWN    = ( 1,  0)

# ── grid primitives ────────────────────────────────────────────────────────────

def zeros(h, w):
    "int, int -> grid: blank h×w grid"
    if h <= 0 or w <= 0:
        raise ValueError(f"zeros: need positive dims, got h={h} w={w}")
    return np.zeros((h, w), dtype=int)

# Common blank grids as terminals (saves 2 int-holes compared to zeros(h,w))
blank33 = np.zeros((3, 3), dtype=int)
blank44 = np.zeros((4, 4), dtype=int)
blank55 = np.zeros((5, 5), dtype=int)
blank66 = np.zeros((6, 6), dtype=int)

def gset(g, r, c, v):
    "grid, int, int, int -> grid: set cell (r,c) to v"
    if r < 0 or r >= g.shape[0] or c < 0 or c >= g.shape[1]:
        raise ValueError(f"gset out of bounds: ({r},{c}) in {g.shape}")
    out = g.copy()
    out[r, c] = v
    return out

# ── int arithmetic ─────────────────────────────────────────────────────────────
# add and mul are imported from operator: add(a,b) = a+b, mul(a,b) = a*b

# ── fn constructors (grid -> grid) ────────────────────────────────────────────

def _step_grid(g, v, d):
    "move all cells with value v one step in direction d, clearing vacated cells"
    dr, dc = d
    h, w = g.shape
    old = [(r, c) for r in range(h) for c in range(w) if g[r, c] == v]
    out = g.copy()
    for r, c in old:
        out[r, c] = 0
    for r, c in old:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr, nc] = v
    return out

def compose(f, g):
    "fn, fn -> fn: returns h where h(x) = g(f(x))"
    def _composed(x):
        return g(f(x))
    return _composed

def _id_fn_impl(g):
    return g.copy()

id_fn = _id_fn_impl

def approach_from(agent_val):
    "int -> fn→fn: partially apply approach, fixing the agent value"
    def _approach_from(goal_val):
        return approach(agent_val, goal_val)
    return _approach_from

def step(v, d):
    "int, dir -> fn: move all cells with value v one step in direction d"
    def _step(g):
        return _step_grid(g, v, d)
    return _step

# ── intentional motion ─────────────────────────────────────────────────────────

def _approach_grid(g, agent_val, goal_val):
    "move agent_val one BFS-optimal step toward goal_val, treating value 3 as walls"
    from collections import deque
    h, w = g.shape
    agents = [(r, c) for r in range(h) for c in range(w) if g[r, c] == agent_val]
    goals  = [(r, c) for r in range(h) for c in range(w) if g[r, c] == goal_val]
    if not agents or not goals:
        return g.copy()
    agent = agents[0]
    goal  = goals[0]
    if agent == goal:
        return g.copy()
    queue   = deque([(agent, None)])   # (pos, first_step)
    visited = {agent}
    first_step = None
    while queue:
        pos, step = queue.popleft()
        if pos == goal:
            first_step = step
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (pos[0] + dr, pos[1] + dc)
            if (0 <= nb[0] < h and 0 <= nb[1] < w and nb not in visited
                    and (g[nb[0], nb[1]] != 3 or nb == goal)):
                visited.add(nb)
                queue.append((nb, (dr, dc) if step is None else step))
    if first_step is None:
        return g.copy()
    dr, dc = first_step
    nr, nc = agent[0] + dr, agent[1] + dc
    out = g.copy()
    out[agent[0], agent[1]] = 0
    out[nr, nc] = agent_val
    return out

def place_wall(g, r, c):
    "grid, int, int -> grid: place a wall (value 3) at (r, c)"
    return gset(g, r, c, 3)

def approach(agent_val, goal_val):
    "int, int -> fn: move agent_val one BFS step toward goal_val each frame"
    def _approach(g):
        return _approach_grid(g, agent_val, goal_val)
    return _approach

# ── utility-based motion ────────────────────────────────────────────────────────
# approach(av, gv) = optimize(neg_distance(gv), av)
# Greedy on negative BFS distance is equivalent to BFS-optimal first step:
# optimal neighbours have distance d-1, all others d+1 or more, so the
# greedy choice always picks an optimal first move.

def _bfs_distance(g, r, c, target_val):
    "BFS distance from (r,c) to nearest target_val cell; inf if unreachable."
    from collections import deque
    h, w = g.shape
    if g[r, c] == target_val:
        return 0
    queue = deque([(r, c, 0)])
    visited = {(r, c)}
    while queue:
        cr, cc, d = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if g[nr, nc] == target_val:
                    return d + 1
                if g[nr, nc] != 3:
                    visited.add((nr, nc))
                    queue.append((nr, nc, d + 1))
    return float('inf')

def neg_distance(target_val):
    "int -> util: u(g,r,c) = -(BFS distance from (r,c) to nearest target_val cell)"
    def _u(g, r, c):
        return -_bfs_distance(g, r, c, target_val)
    return _u

def distance(target_val):
    "int -> util: u(g,r,c) = BFS distance from (r,c) to nearest target_val cell"
    def _u(g, r, c):
        return _bfs_distance(g, r, c, target_val)
    return _u

def neg_util(u):
    "util -> util: negate a utility function"
    def _u(g, r, c):
        return -u(g, r, c)
    return _u

def add_util(u1, u2):
    "util, util -> util: additive combination of two utility functions"
    def _u(g, r, c):
        return u1(g, r, c) + u2(g, r, c)
    return _u

def optimize(u, agent_val):
    "util, int -> fn: move agent_val one greedy step maximising u at the landing cell"
    def _step(g):
        h, w = g.shape
        agents = [(r, c) for r in range(h) for c in range(w) if g[r, c] == agent_val]
        if not agents:
            return g.copy()
        ar, ac = agents[0]
        best_r, best_c, best_u = ar, ac, u(g, ar, ac)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 3:
                uu = u(g, nr, nc)
                if uu > best_u:
                    best_u, best_r, best_c = uu, nr, nc
        if best_r == ar and best_c == ac:
            return g.copy()
        out = g.copy()
        out[ar, ac] = 0
        out[best_r, best_c] = agent_val
        return out
    return _step

# ── belief-based motion ────────────────────────────────────────────────────────
# unfold_belief separates the agent's subjective model (belief) from the actual
# world.  The step function runs on the believed grid; the resulting move is
# extracted and applied to the actual grid.  The output trajectory shows the
# actual world — no mask needed, because the phantom wall never existed there.

def add_phantom_wall(g, r, c):
    "grid, int, int -> belief: agent's false model — actual grid plus a wall at (r,c)"
    b = g.copy()
    if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
        b[r, c] = 3
    return b

def _step_belief(actual_g, believed_g, f):
    "apply f on believed_g, extract the agent's move, replay it on actual_g"
    new_believed = f(believed_g)
    h, w = believed_g.shape
    # find the cell that was vacated (non-zero → 0): that's the agent's old position
    old_pos = agent_val = None
    for r in range(h):
        for c in range(w):
            if believed_g[r, c] != 0 and new_believed[r, c] == 0:
                old_pos, agent_val = (r, c), int(believed_g[r, c])
                break
        if old_pos is not None:
            break
    if old_pos is None:
        return actual_g.copy(), new_believed
    # find where agent_val newly appeared (agent's new position)
    new_pos = None
    for r in range(h):
        for c in range(w):
            if new_believed[r, c] == agent_val and believed_g[r, c] != agent_val:
                new_pos = (r, c)
                break
        if new_pos is not None:
            break
    if new_pos is None:
        return actual_g.copy(), new_believed
    out = actual_g.copy()
    out[old_pos] = 0
    out[new_pos] = agent_val
    return out, new_believed

def unfold_belief(actual_g, believed_g, f):
    "grid, belief, fn -> mat: navigate using believed world, record actual world"
    if _unfold_steps is None:
        raise ValueError("unfold_belief: _unfold_steps not set")
    frames = [actual_g.copy()]
    for _ in range(_unfold_steps - 1):
        actual_g, believed_g = _step_belief(actual_g, believed_g, f)
        frames.append(actual_g.copy())
    return np.stack(frames)

def unfold_belief_steps(actual_g, believed_g, T, f):
    "grid, grid, int, fn -> mat: unfold under belief for T frames (no global needed)"
    frames = [actual_g.copy()]
    for _ in range(T - 1):
        actual_g, believed_g = _step_belief(actual_g, believed_g, f)
        frames.append(actual_g.copy())
    return np.stack(frames)

# ── agent_step primitives (int -> fn) ─────────────────────────────────────────
# Desire assignment: maps each agent to the step function it should execute.
# assign_step(av, f, rest)(av) = f; assign_step(av, f, rest)(x) = rest(x) for x≠av.
# no_step: identity step for any agent (agent doesn't move).

agent_step = 'agent_step'  # int -> fn
fn_belief  = 'fn_belief'   # int -> fn  (agent -> world-transformation function)

def set_at(r, c, v):
    "int, int, int -> fn: set cell (r,c) to value v"
    def _f(g): return gset(g, r, c, v)
    return _f

def assign_belief(agent_val, transform_fn, fallback_fn):
    "int, fn, fn_belief -> fn_belief: assign a grid transformation to one agent"
    def _ab(a):
        return transform_fn if a == agent_val else fallback_fn(a)
    return _ab

def _no_belief_fn(a):
    return lambda g: g.copy()

no_belief_fn = _no_belief_fn

def unfold_multiagent_fn_belief_steps(actual_g, T, fn_belief_fn, agents):
    """grid, int, agent_step, [(int,int)] -> mat

    fn_belief_fn :: int -> fn: maps agent_val to a Grid->Grid world transformation.
    Each agent's believed initial grid = fn_belief_fn(av)(actual_g).
    """
    believed_gs = [fn_belief_fn(av)(actual_g) for av, _ in agents]

    if len(believed_gs) > 1 and all(np.array_equal(believed_gs[0], bg) for bg in believed_gs[1:]):
        raise ValueError("all agents hold identical beliefs — program rejected")

    step_fns = [approach(av, gv) for av, gv in agents]

    frames = [actual_g.copy()]
    for _ in range(T - 1):
        for i, step_fn in enumerate(step_fns):
            actual_g, new_bg = _step_belief(actual_g, believed_gs[i], step_fn)
            believed_gs[i] = new_bg
        frames.append(actual_g.copy())
    return np.stack(frames)

def assign_step(agent_val, step_fn, fallback_fn):
    "int, fn, agent_step -> agent_step: assign a step function to one agent"
    def _as(a):
        if a == agent_val:
            return step_fn
        return fallback_fn(a)
    return _as

def _no_step_fn(a):
    def _id(g): return g.copy()
    return _id

no_step = _no_step_fn  # agent_step terminal

# ── scene_model product type ───────────────────────────────────────────────────
# scene_model = (fn_belief, agent_step) — packages both per-agent belief transforms
# and per-agent step functions into a single jointly-synthesized object.

scene_model = 'scene_model'

def mk_agent_scene(av, belief_fn, step_fn, rest):
    """int, fn, fn, scene_model -> scene_model

    Add one agent to a scene model.
    belief_fn: grid -> grid  (transforms actual_g to the agent's believed grid)
    step_fn:   grid -> grid  (navigation function run on the believed grid)
    rest: scene_model for the remaining agents
    """
    wm_rest, desire_rest = rest
    return (assign_belief(av, belief_fn, wm_rest),
            assign_step(av, step_fn, desire_rest))

empty_scene = (no_belief_fn, no_step)

def unfold_scene(actual_g, T, scene, agent_vals):
    """grid, int, scene_model, [int] -> mat

    Simulate T frames under a joint (belief, desire) scene model.
    Only agents present in actual_g are simulated; agent_vals fixes ordering.
    """
    wm_fn, desire_fn = scene

    present = [av for av in agent_vals if np.any(actual_g == av)]
    if not present:
        raise ValueError("unfold_scene: no agents in initial grid")

    believed_gs = {av: wm_fn(av)(actual_g) for av in present}

    if len(present) > 1:
        bg_list = [believed_gs[av] for av in present]
        if all(np.array_equal(bg_list[0], bg) for bg in bg_list[1:]):
            raise ValueError("unfold_scene: all agents hold identical beliefs — rejected")

    step_fns = {av: desire_fn(av) for av in present}

    frames = [actual_g.copy()]
    for _ in range(T - 1):
        for av in present:
            actual_g, new_bg = _step_belief(actual_g, believed_gs[av], step_fns[av])
            believed_gs[av] = new_bg
        frames.append(actual_g.copy())
    return np.stack(frames)

def seek(agent_val, goal_val):
    "int, int -> fn: BFS-optimal step for agent_val toward goal_val (= optimize(neg_dist(gv), av))"
    return optimize(neg_distance(goal_val), agent_val)

def unfold_multiagent_desire_steps(actual_g, T, desire_fn, agent_vals):
    """grid, int, agent_step, [int] -> mat

    Each agent's step function is provided by desire_fn(av).
    Only agents actually present in actual_g are simulated — this ensures that
    bootstrap tasks (single-agent) reject programs that assign the step function
    to the wrong agent slot.
    Agents apply steps sequentially each frame, preserving the order of agent_vals.
    """
    present = [av for av in agent_vals if np.any(actual_g == av)]
    step_fns = [desire_fn(av) for av in present]
    frames = [actual_g.copy()]
    for _ in range(T - 1):
        for step_fn in step_fns:
            actual_g = step_fn(actual_g)
        frames.append(actual_g.copy())
    return np.stack(frames)

# ── state-threading combinator calculus (file11) ──────────────────────────────
# The non-mental substrate for synthesizing the agent structure itself.
# A scene unfolds as the iteration of a synthesized transition function over
# state = (world, model), a pair of grids.  The simulator is purely mechanical:
# it initializes the model as a copy of the world, applies the sfn each frame,
# and renders the world channel.  Belief semantics lives nowhere in here — a
# program that modifies the model channel and acts through sync_w implements
# it; a program that ignores the model channel is ordinary physics.

sfn = 'sfn'  # state -> state, where state = (world grid, model grid)

def on_world(f):
    "fn -> sfn: apply a grid transformation to the world channel only"
    def _s(s):
        w, m = s
        return f(w), m
    return _s

def on_model(f):
    "fn -> sfn: apply a grid transformation to the model channel only"
    def _s(s):
        w, m = s
        return w, f(m)
    return _s

def sync_w(v):
    "int -> sfn: move value v in the world to its position in the model"
    def _s(s):
        w, m = s
        mpos = np.argwhere(m == v)
        wpos = np.argwhere(w == v)
        if len(mpos) == 0 or len(wpos) == 0:
            return w, m
        mr, mc = int(mpos[0][0]), int(mpos[0][1])
        wr, wc = int(wpos[0][0]), int(wpos[0][1])
        if (mr, mc) == (wr, wc):
            return w, m
        out = w.copy()
        out[wr, wc] = 0
        out[mr, mc] = v
        return out, m
    return _s

def compose_s(a, b):
    "sfn, sfn -> sfn: apply a, then b"
    def _s(s):
        return b(a(s))
    return _s

def wall_at(r, c):
    "int, int -> fn: a wall (value 3) appears at (r, c)"
    def _f(g):
        return gset(g, r, c, 3)
    return _f

def unfold_state(g, T, sf):
    """grid, int, sfn -> mat: iterate sf from (g, copy(g)); render the world.

    The model channel starts as a copy of the world (memory initialized from
    the senses) and is never rendered.
    """
    s = (g.copy(), g.copy())
    frames = [s[0].copy()]
    for _ in range(T - 1):
        s = sf(s)
        frames.append(s[0].copy())
    return np.stack(frames)

# ── conditionals ───────────────────────────────────────────────────────────────

def exists(v):
    "int -> fn_pred: True if any cell in the grid equals v"
    def _exists(g):
        return bool(np.any(g == v))
    return _exists

def if_fn(pred, f1, f2):
    "fn_pred, fn, fn -> fn: apply f1 if pred(grid) else f2"
    def _if(g):
        return f1(g) if pred(g) else f2(g)
    return _if

def _if_int_eq(a, b, f_true, f_false):
    "int, int, fn, fn -> fn: return f_true if a==b else f_false"
    return f_true if a == b else f_false

# ── fn2 terminals (grid -> grid -> grid) ──────────────────────────────────────

def _overlay(g1, g2):
    "element-wise maximum of two grids (union / overlay)"
    if g1.shape != g2.shape:
        raise ValueError(f"overlay shape mismatch {g1.shape} {g2.shape}")
    return np.maximum(g1, g2)

overlay = _overlay   # fn2 terminal

# ── fn_pred terminals (grid -> bool) ──────────────────────────────────────────

def _nonempty(g):
    "True if grid has any nonzero cell"
    return bool(np.any(g != 0))

nonempty = _nonempty  # fn_pred terminal

# ── mat construction ───────────────────────────────────────────────────────────

def place_agent_goal(g, ar, ac, gr, gc):
    "grid, int,int,int,int -> grid: place agent(1) at (ar,ac) and goal(2) at (gr,gc) on g"
    return gset(gset(g, ar, ac, 1), gr, gc, 2)

def unfold(g, n, f):
    "grid, int, fn -> mat: produce n frames [g, f(g), f²(g), …, f^(n-1)(g)]"
    if n <= 0:
        raise ValueError(f"unfold: need n>0, got {n}")
    frames = [g.copy()]
    for _ in range(n - 1):
        g = f(g)
        frames.append(g.copy())
    return np.stack(frames)

# Set by solve_enumeration to the current task's T before each enumeration.
_unfold_steps = None

def unfold_auto(g, f):
    "grid, fn -> mat: unfold for _unfold_steps frames (T injected by solve_enumeration)"
    if _unfold_steps is None:
        raise ValueError("unfold_auto: _unfold_steps not set")
    return unfold(g, _unfold_steps, f)

def predict_auto(g, f):
    "grid, fn -> grid: apply f (_unfold_steps - 1) times to predict the last frame"
    if _unfold_steps is None:
        raise ValueError("predict_auto: _unfold_steps not set")
    for _ in range(_unfold_steps - 1):
        g = f(g)
    return g

# ── mat transformations ────────────────────────────────────────────────────────

def mask(m, v):
    "mat, int -> mat: zero out all cells with value v across all frames"
    out = m.copy()
    out[out == v] = 0
    return out

def replace_val(src, dst):
    "int, int -> fn: replace all cells of value src with dst in a grid"
    def _replace(g):
        out = g.copy()
        out[out == src] = dst
        return out
    return _replace

def map_mat(f, m):
    "fn, mat -> mat: apply grid→grid function f to every frame of mat m"
    return np.stack([f(m[t]) for t in range(m.shape[0])])

def filter_mat(pred, m):
    "fn_pred, mat -> mat: keep only frames where pred(frame) is True"
    frames = [m[t] for t in range(m.shape[0]) if pred(m[t])]
    if not frames:
        raise ValueError("filter_mat: no frames passed predicate")
    return np.stack(frames)

# ── Delta expression tree ──────────────────────────────────────────────────────

class Delta:
    # a single node in an expression tree
    def __init__(self, head, type=None, tailtypes=None, tails=None, repr=None, hiddentail=None, arrow=None, ishole=False, isarg=False):
        self.head = head
            # the function/value
        self.tails = tails
            # the arguments the function takes
        self.tailtypes = tailtypes
            # the expected argument types
        self.type = type
            # the return type
        self.ishole = ishole
        self.isarg = isarg

        if arrow:
            self.arrow = arrow
            self.type = arrow
        else:
            if tailtypes:
                self.arrow = (tuple(tailtypes), type)
            else:
                self.arrow = type

        self.hiddentail = hiddentail

        if repr is None:
            repr = str(head)

        self.repr = repr
        self.idx = 0

    def __call__(self):
        # calling delta() evaluates the expression
        if self.tails is None:
            return self.head

        if self.hiddentail:
            body = deepcopy(self.hiddentail)

            for tidx, tail in enumerate(self.tails):
                # arg in hiddentail should only match itself for replacement
                body = replace_hidden(body, Delta(f'${tidx}', isarg=True, type=tail.type), tail)

            return body()

        tails = []
        for a in self.tails:
            if isinstance(a, Delta):
                if self.head is _lam_impl:
                    tails.append(a)  # lazy: pass unevaluated Delta tree to _lam_impl
                else:
                    tails.append(a())
            else:
                tails.append(a)

        return self.head(*tails)

    def balance(self):
        if not self.tails:
            return self

        if not any(map(isterminal, self.tails)):
            self.tails = sorted(self.tails, key=str)

        if self.hiddentail:
            self.hiddentail.balance()

        for tail in self.tails:
            tail.balance()

        return self

    def __eq__(self, other):
        if other is None:
            return False

        if not isinstance(other, Delta):
            return False

        return isequal(self, other)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        if self.tails is None or len(self.tails) == 0:
            return f'{self.repr}'
        else:
            tails = self.tails

        return f'({self.repr} {" ".join(map(str, tails))})'

def _sub_var(node, value):
    "Substitute all var-sentinel leaves in tree with value, in-place."
    if node.head is _var_sentinel:
        node.head = value
        return
    if node.hiddentail:
        _sub_var(node.hiddentail, value)
    if node.tails:
        for tail in node.tails:
            if isinstance(tail, Delta):
                _sub_var(tail, value)

def _lam_impl(body_delta):
    "fn -> fn_belief: lambda over int, binding var in body"
    def _lam(a):
        body = deepcopy(body_delta)
        _sub_var(body, a)
        return body()
    return _lam

_sim_agents = None

def sim(actual_g, wm):
    "grid, fn_belief -> mat: simulate agents under their believed worlds"
    if _sim_agents is None:
        raise ValueError("sim: _sim_agents not set")
    return unfold_multiagent_fn_belief_steps(actual_g, _unfold_steps, wm, _sim_agents)

def isterminal(d: Delta) -> bool:
    if d.tailtypes == None:
        return True

    if d.tails is None or len(d.tails) == 0:
        return False

    for tail in d.tails:
        if not isterminal(tail):
            return False

    return True


def length(tree: Delta) -> int:
    if not tree:
        return 0

    if not tree.tails:
        return 1

    return 1 + sum(map(length, tree.tails))

def countholes(tree: Delta) -> int:
    if not tree:
        return 0

    if tree.ishole:
        return 1

    if not tree.tails:
        return 0

    return sum(map(countholes, tree.tails))


def getast(expr):
    ast = []
    idx = 0

    while idx < len(expr):
        if expr[idx] == '(':
            nopen = 1
            sidx = idx

            while nopen != 0:
                idx += 1
                if expr[idx] == '(':
                    nopen += 1
                if expr[idx] == ')':
                    nopen -= 1

            ast.append(getast(expr[sidx+1:idx]))

        elif not expr[idx] in "() ":
            se_idx = idx
            idx += 1

            while idx < len(expr) and not expr[idx] in "() ":
                idx += 1

            ast.append(expr[se_idx:idx])

        idx += 1

    # Unwrap only when the entire expression was a single nested s-expression,
    # e.g. getast('(fn_1 3 1)') called on the inner content of outer parens.
    # Do NOT unwrap when there are trailing tokens, e.g. '(fn_1 3 1) 1 2 0'
    # — those trailing tokens are additional arguments that must be preserved.
    if len(ast) == 1 and isinstance(ast[0], list):
        return ast[0]

    return ast

def todelta(D, ast):
    if not isinstance(ast, list):
        if ast.startswith('$'):
            return Delta(ast)

        if (idx := D.index(ast)) is None:
            raise ValueError(f"what's a {ast}?")

        return D[idx]

    newast = []
    idx = 0
    while idx < len(ast):
        d = todelta(D, ast[idx])

        args = []

        idx += 1
        while idx < len(ast):
            args.append(todelta(D, ast[idx]))
            idx += 1

        if len(args) > 0:
            # Append rather than overwrite — handles stitch's partial-application
            # encoding where ((fn_1 3 1) 1 2 0) means fn_1 called with [3,1,1,2,0].
            if d.tails:
                d.tails = list(d.tails) + args
            else:
                d.tails = args

        newast.append(d)

        idx += 1

    return newast[0]

def tr(D, expr):
    return todelta(D, getast(expr))

def isequal(n1, n2):
    if n1.ishole or n2.ishole:
        return n1.type == n2.type

    if n1.isarg and n2.isarg:
        return n1.head == n2.head and n1.type == n2.type

    # An arg node can never equal a non-arg node; short-circuit before the
    # head comparison which may involve numpy arrays and raise ValueError.
    if n1.isarg or n2.isarg:
        return False

    if n1.head == n2.head:
        # no kids
        if not n1.tails and not n2.tails:
            return True

        if not n1.tails or not n2.tails:
            return False

        if len(n1.tails) != len(n2.tails):
            return False

        for t1, t2 in zip(n1.tails, n2.tails):
            if not isequal(t1, t2):
                return False

        return True

    return False

def extract_matches(tree, treeholed):
    """
    given a healthy tree, find in it part covering holes in a given treeholed
    return pairs of holes and covered parts
    """
    if not tree or not treeholed:
        return []

    if treeholed.ishole or treeholed.isarg:
        return [(treeholed.head, deepcopy(tree))]

    out = []
    if not tree.tails:
        return []

    for tail, holedtail in zip(tree.tails, treeholed.tails):
        out += extract_matches(tail, holedtail)

    return out


def replace_hidden(tree, arg, tail):
    if isequal(tree, arg):
        return deepcopy(tail)

    if not tree.tails:
        return tree

    qq = [tree]
    while len(qq) > 0:
        n = qq.pop(0)
        if not n.tails: continue

        for idx, nt in enumerate(n.tails):
            if isequal(nt, arg):
                n.tails[idx] = deepcopy(tail)
            else:
                qq.append(nt)

    return tree

def replace(tree, matchbranch, newbranch):
    if isequal(tree, matchbranch):
        branch = deepcopy(newbranch)

        if not tree.tails:
            return branch

        args = {arg: tail for arg, tail in extract_matches(tree, matchbranch)}
        if len(args) > 0:
            branch.tails = list(args.values())

        return branch

    qq = [tree]
    while len(qq) > 0:
        n = qq.pop(0)
        if not n.tails: continue

        for i in range(len(n.tails)):
            if isequal(n.tails[i], matchbranch):
                branch = deepcopy(newbranch)
                args = {arg: tail for arg, tail in extract_matches(n.tails[i], matchbranch)}
                branch.tails = list(args.values())
                n.tails[i] = branch
            else:
                qq.append(n.tails[i])

    return tree

# d.type $ has property of wildcard matching
# making it impossible to modify hiddentails
def freeze(tree: Delta):
    if tree.ishole:
        tree.ishole = False
        tree.isarg = True

    if tree.hiddentail:
        freeze(tree.hiddentail)

    if tree.tails:
        for tail in tree.tails:
            freeze(tail)

def normalize(tree):
    if tree.hiddentail:
        ht = normalize(deepcopy(tree.hiddentail))

        if tree.tails:
            for tidx, tail in enumerate(tree.tails):
                replace_hidden(ht, Delta(f'${tidx}', isarg=True, type=tail.type), normalize(tail))

        return ht

    qq = [tree]
    while len(qq) > 0:
        n = qq.pop(0)

        if not n.tails:
            continue

        for idx in range(len(n.tails)):
            if n.tails[idx].hiddentail:
                tails = n.tails[idx].tails
                n.tails[idx] = normalize(deepcopy(n.tails[idx].hiddentail))

                if tails:
                    for tidx, tail in enumerate(tails):
                        n.tails[idx] = replace_hidden(n.tails[idx], Delta(f'${tidx}', isarg=True, type=tail.type), normalize(tail))
            else:
                qq.append(normalize(n.tails[idx]))

    return tree


# not reentrant
def typize(tree: Delta):
    """Collect types of $i holes keyed by stitch's own index; do NOT rename.

    Stitch uses #i as explicit argument positions (not BFS order).  After the
    #{i} → ${i} substitution, holes already carry the correct name.  Renaming
    them to BFS order breaks the tidx→$i mapping in __call__.

    Duplicate occurrences of the same $i (stitch shared variables) are handled
    by replace_hidden replacing all matches in one pass.
    """
    seen = {}   # '$i' → type
    qq = [tree]
    while qq:
        n = qq.pop(0)
        if not n.tails:
            continue
        for idx in range(len(n.tails)):
            child = n.tails[idx]
            if child.ishole:
                name = child.head   # e.g. '$4'
                if name not in seen:
                    seen[name] = child.type
                # Leave hole name unchanged — preserve stitch's $i index
            else:
                qq.append(child)

    if not seen:
        return []
    max_i = max(int(k[1:]) for k in seen)
    return [seen.get(f'${i}') for i in range(max_i + 1)]


def alld(tree):
    "enumerate all heads in tree"
    if not tree.tails:
        return [tree]

    heads = [tree]

    for t in tree.tails:
        heads.extend(alld(t))

    return heads
