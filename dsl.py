from operator import add, mul
import numpy as np
from copy import deepcopy

# types
mat      = 'mat'      # 3d numpy array (T, H, W)
grid     = 'grid'     # 2d numpy array (H, W)
fn       = 'fn'       # grid -> grid
fn2      = 'fn2'      # grid -> grid -> grid
fn_pred  = 'fn_pred'  # grid -> bool
goal     = 'goal'     # goal specification (interpreted by optimize)

# int is used as a type string too
# direction is used as a type string
direction = 'dir'

# direction terminals
RIGHT   = ( 0,  1)
LEFT    = ( 0, -1)
UP      = (-1,  0)
DOWN    = ( 1,  0)
# DIAG_DR = ( 1,  1)
# DIAG_DL = ( 1, -1)
# DIAG_UR = (-1,  1)
# DIAG_UL = (-1, -1)

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

def step_fn(v, d):
    "int, direction -> fn: returns grid->grid that moves value v in direction d"
    def _step(g):
        return _step_grid(g, v, d)
    return _step

def compose(f, g):
    "fn, fn -> fn: returns h where h(x) = g(f(x))"
    def _composed(x):
        return g(f(x))
    return _composed

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

# ── goal algebra ───────────────────────────────────────────────────────────────

def exists(v):
    "int -> fn_pred: True if any cell in the grid equals v"
    def _exists(g):
        return bool(np.any(g == v))
    return _exists

def be_at(target_val):
    "int -> fn: move agent (value 1) one BFS step toward target_val"
    return approach(1, target_val)

def if_goal(pred, goal1, goal2):
    "fn_pred, fn, fn -> fn: each step apply goal1 if pred(grid) else goal2"
    def _if(g):
        return goal1(g) if pred(g) else goal2(g)
    return _if

# ── goal type and optimize ─────────────────────────────────────────────────────
# goal is a declarative specification of what the agent wants to achieve.
# optimize(g) → fn converts a goal spec into an executable step function.
# This separates *what* (goal) from *how* (optimize), allowing ECD to compose
# goals independently and letting optimize be the single locus of planning logic.

def at(target_val):
    "int -> goal: agent (value 1) should reach cell with value target_val"
    return ('at', target_val)

def if_else(pred, goal_then, goal_else):
    "fn_pred, goal, goal -> goal: conditional goal — use goal_then if pred else goal_else"
    return ('if', pred, goal_then, goal_else)

def optimize(goal_spec):
    "goal -> fn: BFS-optimal step function that pursues goal_spec each frame"
    def _step(g):
        kind = goal_spec[0]
        if kind == 'at':
            _, target = goal_spec
            return _approach_grid(g, 1, target)
        elif kind == 'if':
            _, pred, g_then, g_else = goal_spec
            active = g_then if pred(g) else g_else
            return optimize(active)(g)
        else:
            raise ValueError(f"optimize: unknown goal kind '{kind}'")
    return _step

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

# def singleton(g):
#     "grid -> mat: wrap a single grid as a 1-frame matrix"
#     return g[np.newaxis].copy()

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

def nav_unfold(g, n):
    "grid, int -> mat: unfold g for n steps using navigate (agent 1 approaches goal 2)"
    return unfold(g, n, approach(1, 2))

# ── mat transformations ────────────────────────────────────────────────────────

def hide_walls(m):
    "mat -> mat: remove wall cells (value 3) from all frames, leaving only agent and goal"
    out = m.copy()
    out[out == 3] = 0
    return out

def believed_nav_unfold(g, n):
    "grid, int -> mat: nav_unfold from agent's believed grid, hiding phantom walls from output"
    return hide_walls(nav_unfold(g, n))

def map_mat(f, m):
    "fn, mat -> mat: apply f to each frame"
    return np.stack([f(m[t]) for t in range(m.shape[0])])

def zip_mat(f, m1, m2):
    "fn2, mat, mat -> mat: combine frames pairwise with f"
    if m1.shape != m2.shape:
        raise ValueError(f"zip_mat shape mismatch {m1.shape} {m2.shape}")
    return np.stack([f(m1[t], m2[t]) for t in range(m1.shape[0])])

def fold_mat(f, acc, m):
    "fn2, grid, mat -> grid: fold mat frames into single grid"
    for t in range(m.shape[0]):
        acc = f(acc, m[t])
    return acc

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

    if isinstance(ast[0], list):
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
        return n1.type == n2.type

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
                n.tails[idx] = tail
                break
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
    "replace each hole with $arg, returning all $arg's types"
    qq = [tree]
    tailtypes = []
    z = 0

    while len(qq) > 0:
        n = qq.pop(0)

        if not n.tails:
            continue

        for idx in range(len(n.tails)):
            # is this a hole?
            if n.tails[idx].ishole:
                type = n.tails[idx].type
                tailtypes.append(type)

                # need to hole it for next tree replacement
                n.tails[idx] = Delta(f'${z}', ishole=True, type=type)
                z += 1
            else:
                qq.append(n.tails[idx])

    return tailtypes

def showoff_types(tree: Delta):
    qq = [tree]
    types = set()

    while len(qq) > 0:
        n = qq.pop(0)

        types.add(n.type)

        if not n.tails:
            continue

        for t in n.tails:
            qq.append(t)

    return types

def alld(tree):
    "enumerate all heads in tree"
    if not tree.tails:
        return [tree]

    heads = [tree]

    for t in tree.tails:
        heads.extend(alld(t))

    return heads
