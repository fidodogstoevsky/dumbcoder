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
fn           = 'fn'           # grid -> grid
util         = 'util'         # (grid, int, int) -> float  — positional utility

# Integer literals are split into two distinct value types so that dataset
# diversity and the latent (wall) search are decoupled:
#   coord     — a grid POSITION (wall_at / clear_at args).  Terminals range over
#               0..SIZE-1, fixed by grid geometry; never grows with the corpus, so
#               the invisible wall coordinate is a bounded latent.
#   cellvalue — a cell VALUE (agent/goal id, step magnitude, …).  Visible in the
#               grid, so content_q prices it ~0; diversifying these costs nothing at
#               the coord slot.  (Was a single overloaded `int` type, whose shared
#               terminal pool made value-diversity inflate the coord branching.)
coord        = 'coord'
cellvalue    = 'cellvalue'

# direction is used as a type string
direction = 'dir'

# direction terminals
RIGHT   = ( 0,  1)
LEFT    = ( 0, -1)
UP      = (-1,  0)
DOWN    = ( 1,  0)

# ── grid primitives ────────────────────────────────────────────────────────────

def gset(g, r, c, v):
    "grid, int, int, int -> grid: set cell (r,c) to v"
    if r < 0 or r >= g.shape[0] or c < 0 or c >= g.shape[1]:
        raise ValueError(f"gset out of bounds: ({r},{c}) in {g.shape}")
    out = g.copy()
    out[r, c] = v
    return out

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

def step(v, d):
    "int, dir -> fn: move all cells with value v one step in direction d"
    def _step(g):
        return _step_grid(g, v, d)
    return _step

# ── utility-based motion ────────────────────────────────────────────────────────
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

# ── single-grid calculus (file13) ───────────────────────────────────────────────
# file11/12 hardwired a second grid into the interpreter's state (the pair) so
# that intensionality could be *expressed as a compound* and then discovered by
# compression.  Here the interpreter shrinks to threading a single grid; programs
# are plain fn = grid -> grid.  The second grid (the agent's private model) is no
# longer interpreter state — it is introduced *locally in program space* by a
# general, non-mental combinator `fork`, computed over, and collapsed back to one
# grid in the same step.  No primitive on its own denotes a mental state: `fork`
# is the S/fork combinator (apply a derived transform to a copy, then reconcile
# against the original) and `sync_to_world` is a grid-diff that transfers one
# value's position.  Belief is the *composition* fork(policy-on-modified-copy,
# sync_to_world av) — which is exactly what stitch is meant to extract as fn_agent.

def fork(derive, commit):
    """fn, fn_p_g -> fn: w |-> commit((w, derive(w))).

    `derive` builds a private grid from a copy of the world (e.g. stamp a phantom
    wall and run the policy on it); `commit` reconciles the (world, derived) pair
    back to a single grid (e.g. move the agent to the position it reached in the
    derived grid).  The second grid lives only for the duration of this call.
    """
    def _f(w):
        return commit((w.copy(), derive(w.copy())))
    return _f

def sync_to_world(v):
    "int -> fn_p_g: move value v in world (first) to its position in derived (second)"
    def _c(p):
        w, m = p
        wpos = np.argwhere(w == v)
        mpos = np.argwhere(m == v)
        if len(wpos) == 0 or len(mpos) == 0:
            return w.copy()
        wr, wc = int(wpos[0][0]), int(wpos[0][1])
        mr, mc = int(mpos[0][0]), int(mpos[0][1])
        if (wr, wc) == (mr, mc):
            return w.copy()
        out = w.copy()
        out[wr, wc] = 0
        out[mr, mc] = v
        return out
    return _c

# ── non-mental inhabitants of the pair interface (file14) ───────────────────────
# fork produces pairs; fn_p_g consumes them.  If fork only ever fed sync_to_world
# (and sync only ever ate fork), the fork/sync split would be a disguised belief
# primitive.  These give the interface independent extension: `overlay` is a second
# commit (graphics, not mind), and `unfold_with_template` is a second producer of
# pairs (two given grids, not a derived private model).  Belief is then just one
# path through a general pair-calculus, not a reverse-engineered decomposition.

def overlay(p):
    "pair_gg -> grid (fn_p_g): union the channels — model's nonzero cells win ties"
    w, m = p
    out = w.copy()
    out[m != 0] = m[m != 0]
    return out

def then_sync(c, v):
    "fn_p_g, int -> fn_p_g: run commit c on the working channel, then sync v"
    def _c(p):
        w, m = p
        return sync_to_world(v)((c((w, m)), m))
    return _c

def unfold_with_template(g, template, T, c):
    """grid, grid, int, fn_p_g -> mat: thread g; each frame pair it with a
    *constant external* template and apply commit c.  Unlike fork, the second
    channel is a given input, not a privately derived model — so a program that
    uses sync_to_world here is doing registration, not holding a belief.
    """
    frames = [g.copy()]
    for _ in range(T - 1):
        g = c((g, template))
        frames.append(g.copy())
    return np.stack(frames)

def unfold(g, T, f):
    "grid, int, fn -> mat: iterate f:grid->grid from g, rendering each frame."
    frames = [g.copy()]
    for _ in range(T - 1):
        g = f(g)
        frames.append(g.copy())
    return np.stack(frames)

# ── monomorphic pair-calculus (file12) ─────────────────────────────────────────
# file11 buried "state = (world, model), init by copy, render world" inside
# unfold_state.  Here that triple is exposed as program data, with two
# monomorphic state shapes: grid (no private channel) and pair_gg (one
# private channel).  Whether a synthesized program *uses* a private channel
# becomes a structural feature of the program — namely, which mk_machine_*
# constructor it picks — instead of a type-system commitment in the interpreter.

pair_gg = 'pair_gg'   # (grid, grid)
fn_g_p  = 'fn_g_p'    # grid -> pair_gg   (init for pair machines)
fn_p_g  = 'fn_p_g'    # pair_gg -> grid   (render for pair machines)
machine = 'machine'   # bundled (kind, init, step, render)

def _dup_g(g):
    "grid -> pair_gg: model channel starts as a copy of the world"
    return (g.copy(), g.copy())

dup_g = _dup_g

def _fst_gg(p):
    "pair_gg -> grid: render the world (first) channel"
    return p[0].copy()

fst_gg = _fst_gg

def _snd_gg(p):
    "pair_gg -> grid: render the model (second) channel"
    return p[1].copy()

snd_gg = _snd_gg

# ── decomposed fork: product-category combinators (file15) ───────────────────────
# file13's `fork(derive, commit)` hides two operations inside one closure: the
# private copy (`w.copy()`) and the application of `derive` to it.  Spelled out,
#
#     fork(derive, commit)(w) = commit((w, derive(w)))
#                             ≡ (commit ∘ mapsnd(derive) ∘ dup)(w)
#
# which is three textbook combinators of the product (×) category:
#
#     dup      :: grid -> pair     w |-> (w, w)         -- diagonal Δ      (= dup_g)
#     mapsnd f :: pair -> pair     (a,b) |-> (a, f(b))  -- bifunctor second (= on_model)
#     commit   :: pair -> grid                          -- product eliminator (sync/…)
#
# Exposing these as DSL primitives turns the implicit copy into a discoverable
# `dup` node and "run the policy on the copy" into a `mapsnd` node.  `fork` is then
# no longer a primitive but a compound the searcher builds and stitch re-extracts.
# Wiring needs two typed composers (the monomorphic type system has no generic
# `compose` across grid/pair arrows): `compose_gp` chains a pair-producer with a
# pair-endomorphism, and `pipe_gpg` runs a pair-producer into a pair-consumer.

fn_p_p = 'fn_p_p'   # pair_gg -> pair_gg

dup = dup_g         # grid -> pair_gg: the diagonal Δ; the private copy made explicit

mapsnd = on_model   # fn -> fn_p_p: (a,b) |-> (a, f(b)) — bifunctor 'second'.
                    # Literally the file11 model-channel map, reused verbatim: the
                    # second grid is computed over while the first is carried along.

def compose_gp(produce, endo):
    "fn_g_p, fn_p_p -> fn_g_p: (grid -> pair) then (pair -> pair)"
    def _f(g):
        return endo(produce(g))
    return _f

def pipe_gpg(produce, commit):
    "fn_g_p, fn_p_g -> fn: w |-> commit(produce(w)) — pair-producer then -consumer"
    def _f(g):
        return commit(produce(g))
    return _f

def fork_decomposed(derive, commit):
    "the decomposition identity (for ground-truth checks): == fork(derive, commit)"
    return pipe_gpg(compose_gp(dup, mapsnd(derive)), commit)

# ── decomposed sync: locate / place (file15 — defined, but kept atomic) ──────────
# sync_to_world(v) decomposes the same way on the key v: read v's coordinate
# through one channel, impose it on the other —
#
#     sync_to_world(v) ≡ place(v) ∘ ⟨ fst , locate(v) ∘ snd ⟩
#
#     locate(v) :: grid -> coord     argwhere(g == v)
#     place(v)  :: grid, coord -> grid    move v to that cell, clearing its old one
#
# These are honest, non-mental parts (a find and a move).  We DEFINE them and
# prove the identity, but DELIBERATELY keep `sync_to_world` a single DSL node (see
# file15's docstring): folding the committer into a 4-node subtree would bury the
# shared-`av` coincidence — av in the actor `(optimize … av)` AND the committer
# `(sync_to_world av)` — which is the structural signature of agency that stitch
# is meant to surface.  The sweet spot is to decompose fork (cheap; `dup` is a
# great independent primitive) while leaving the agent signature one node deep.
#
# file15 (search-path variant) wires the decomposition in anyway, to *measure*
# whether the prediction holds: `register(loc, plc)` is the av-free plumbing of
# the commit (the analog of pipe_gpg/compose_gp for fork), so the committer
# becomes `(register (locate av) (place av))` and av now appears 3× — in optimize,
# locate, and place.  The question is whether stitch still re-extracts a single
# agent constructor binding that shared av, or fragments it into a generic
# `register`/locate/place idiom that buries the signature.

coord   = 'coord'    # (row, col) — a position read off a grid
fn_g_c  = 'fn_g_c'   # grid -> coord          (locate v)
fn_gc_g = 'fn_gc_g'  # (grid, coord) -> grid  (place v)

def locate(v):
    "int -> (grid -> coord): position of value v, or None if absent"
    def _l(g):
        pos = np.argwhere(g == v)
        return (int(pos[0][0]), int(pos[0][1])) if len(pos) else None
    return _l

def place(v):
    "int -> (grid, coord -> grid): move value v to coord, clearing its old cell"
    def _p(g, coord):
        if coord is None:
            return g.copy()
        out = g.copy()
        wpos = np.argwhere(g == v)
        if len(wpos):
            out[int(wpos[0][0]), int(wpos[0][1])] = 0
        out[coord[0], coord[1]] = v
        return out
    return _p

def register(loc, plc):
    """fn_g_c, fn_gc_g -> fn_p_g: (w, m) |-> plc(w, loc(m)).

    The av-free plumbing of the commit: read a coordinate off the model channel
    with `loc`, impose it on the world with `plc`.  This is exactly registration
    (alignment) — a generic, non-mental pair-consumer, the commit-side analog of
    pipe_gpg/compose_gp.  `sync_to_world(v) ≡ register(locate(v), place(v))` on any
    trajectory where v is present in the world (always true for the agent av).
    """
    def _c(p):
        w, m = p
        return plc(w, loc(m))
    return _c

def sync_decomposed(v):
    "the decomposition identity (for checks): == sync_to_world(v)"
    def _c(p):
        w, m = p
        if not len(np.argwhere(w == v)):
            return w.copy()
        return place(v)(w, locate(v)(m))
    return _c

# ── symmetric complements (file16 — the "cube") ─────────────────────────────────
# Every commit/combinator above bakes in a *choice* that belief happens to want:
# sync_to_world reads the coordinate off the MODEL, writes it into the WORLD,
# returns the WORLD, and moves exactly ONE value.  Those are independent axes
#
#     direction   : read model→write world (sync_to_world)  vs  read world→write model (sync_to_model)
#     scope       : one value (sync_to_world)  vs  all (sync_all)  vs  all-but-one (sync_except)
#     z-order     : model wins ties (overlay)   vs  world wins ties (underlay)
#     projection  : keep world (fst_gg)         vs  keep model (snd_gg)
#     bifunctor   : map second (mapsnd/on_model) vs map first (mapfst/on_world) vs both (bimap)
#     pairing     : diagonal (dup)              vs  fresh scratch (pair_blank)
#     utility     : attract (neg_distance)      vs  repel (distance)
#     grid edit   : add a wall (wall_at)        vs  remove a cell (clear_at / erase)
#
# Populating the DSL with the *other* corners gives the searcher the full
# symmetric field.  None of these help a theory-of-mind task; each is the natural
# tool for some non-mental one (perception/recording, multi-object registration,
# inpainting, motion away from a hazard, denoising).  If joint MDL still selects
# exactly the (read-model, write-world, single-av) corner for belief while these
# corners attach to their non-mental tasks, the agency signature is discovered,
# not gerrymandered into the primitive set.

def sync_to_model(v):
    """int -> fn_p_g: the DIRECTION complement of sync_to_world.

    Read v's coordinate off the WORLD (first), impose it on the MODEL (second),
    and return the MODEL.  Where sync_to_world commits a belief to the world,
    this records a sensation into the private channel — perception, not action.
    """
    def _c(p):
        w, m = p
        wpos = np.argwhere(w == v)
        mpos = np.argwhere(m == v)
        if len(wpos) == 0 or len(mpos) == 0:
            return m.copy()
        wr, wc = int(wpos[0][0]), int(wpos[0][1])
        mr, mc = int(mpos[0][0]), int(mpos[0][1])
        if (wr, wc) == (mr, mc):
            return m.copy()
        out = m.copy()
        out[mr, mc] = 0
        out[wr, wc] = v
        return out
    return _c

def sync_all(p):
    """pair_gg -> grid (fn_p_g): the SCOPE complement of sync_to_world.

    Move EVERY value shared by both channels to its model-position (wholesale
    state adoption / multi-object registration).  Clear-all-then-place so swaps
    don't clobber.  No single value is privileged — the agency signature, which
    needs av to be the *one* committed value, cannot hide in here.
    """
    w, m = p
    out = w.copy()
    moves = []
    for v in (int(x) for x in np.unique(w) if x != 0):
        wpos = np.argwhere(w == v)
        mpos = np.argwhere(m == v)
        if len(wpos) and len(mpos):
            moves.append((v, (int(wpos[0][0]), int(wpos[0][1])),
                             (int(mpos[0][0]), int(mpos[0][1]))))
    for _v, (wr, wc), _t in moves:
        out[wr, wc] = 0
    for v, _s, (mr, mc) in moves:
        out[mr, mc] = v
    return out

def sync_except(v):
    "int -> fn_p_g: SCOPE complement — sync every shared value EXCEPT v (the set complement of sync_to_world)"
    def _c(p):
        w, m = p
        out = w.copy()
        moves = []
        for u in (int(x) for x in np.unique(w) if x != 0 and x != v):
            wpos = np.argwhere(w == u)
            mpos = np.argwhere(m == u)
            if len(wpos) and len(mpos):
                moves.append((u, (int(wpos[0][0]), int(wpos[0][1])),
                                 (int(mpos[0][0]), int(mpos[0][1]))))
        for _u, (wr, wc), _t in moves:
            out[wr, wc] = 0
        for u, _s, (mr, mc) in moves:
            out[mr, mc] = u
        return out
    return _c

def underlay(p):
    "pair_gg -> grid (fn_p_g): the Z-ORDER complement of overlay — WORLD's nonzero cells win ties (model only fills holes / inpainting)"
    w, m = p
    out = m.copy()
    out[w != 0] = w[w != 0]
    return out

def mapfst(f):
    "fn -> fn_p_p: the BIFUNCTOR complement of mapsnd — (a,b) |-> (f(a), b); transform world, carry model as pristine reference.  (== on_world)"
    return on_world(f)

def bimap(f, g):
    "fn, fn -> fn_p_p: the full product bifunctor map — (a,b) |-> (f(a), g(b))"
    def _s(p):
        a, b = p
        return f(a), g(b)
    return _s

def swap(p):
    "pair_gg -> pair_gg (fn_p_p): the symmetry witness — exchange the two channels so neither is privileged as 'the world'"
    a, b = p
    return (b.copy(), a.copy())

def via_swap(c):
    """fn_p_g -> fn_p_g: run commit c on the swapped pair.

    Makes channel direction a search *choice* rather than a hardwired privilege:
    via_swap(sync_to_world(v)) == sync_to_model(v).  Lets the searcher express
    belief through the 'wrong' channel, so picking the direct wiring is informative.
    """
    def _c(p):
        return c((p[1], p[0]))
    return _c

def pair_blank(g):
    "grid -> pair_gg (fn_g_p): the PAIRING complement of dup — pair g with a fresh empty scratch channel instead of a copy of itself"
    return (g.copy(), np.zeros_like(g))

def distance(target_val):
    "int -> util: the UTILITY complement of neg_distance — +BFS distance, so maximising it flees the nearest target_val (predator/prey, hazard avoidance)"
    def _u(g, r, c):
        return _bfs_distance(g, r, c, target_val)
    return _u

def clear_at(r, c):
    "int, int -> fn: the EDIT complement of wall_at — clear cell (r,c) (write 0) instead of stamping a blocker"
    def _f(g):
        return gset(g, r, c, 0)
    return _f

def erase(v):
    "int -> fn: EDIT complement — remove every cell of value v (denoising / deletion)"
    def _f(g):
        out = g.copy()
        out[out == v] = 0
        return out
    return _f

# ── arity generalization: the cons/nil grid-stack (file17) ───────────────────────
# The cube above introduces *role* symmetry (every choice baked into the two
# channels gets its complementary corner), but every cube combinator is hardwired
# to ARITY 2: pair_gg = (grid, grid), `swap` exchanges "the two" channels, `dup`
# makes a binary product.  The number of private channels is held fixed, so "why
# one world + one model, not three?" is answered by an interpreter commitment (the
# pair_gg type), not by search.
#
# Replacing the fixed pair with a single recursive grid-stack makes the *number of
# private channels* a discoverable structural feature.  Belief's private channel is
# within-step (atomic `fork` is fn = grid->grid; the model is rederived each frame
# by dup INSIDE the step and collapsed back by the commit), so "arity" here is the
# arity of fork's product.  One type, one combinator set, any depth:
#
#     gstack ::= ()  |  (grid, *gstack)        -- a stack of grids; index 0 = top
#
# Each combinator below is the n-ary, depth-polymorphic lift of a cube op (dup,
# pair_blank, on_model/mapsnd, swap, overlay, sync_to_world, fst_gg).  depth-1
# reproduces fork+sync exactly (see fork_stack_decomposed); depth-0 is a bare fn
# (no stack); depth-2 is the natural home of a two-buffer non-mental task.

gstack = 'gstack'   # a tuple of grids; index 0 is the top (most-recent private channel)
fn_g_s = 'fn_g_s'   # grid   -> gstack   (lift / base)
fn_s_s = 'fn_s_s'   # gstack -> gstack   (stack endomorphism)
fn_s_g = 'fn_s_g'   # gstack -> grid     (render / peek)

def base(g):
    "grid -> gstack (fn_g_s): the world as a 1-element stack (0 private channels)"
    return (g.copy(),)

def dup_top(s):
    "gstack -> gstack (fn_s_s): the diagonal Δ at the top — n-ary `dup` (push a copy of the head)"
    return (s[0].copy(),) + s

def blank_top(s):
    "gstack -> gstack (fn_s_s): the PAIRING complement of dup_top — push a fresh empty scratch channel (n-ary `pair_blank`)"
    return (np.zeros_like(s[0]),) + s

def swap_top(s):
    "gstack -> gstack (fn_s_s): exchange the top two channels — n-ary `swap`; the symmetry witness and the only way to reach a non-top channel"
    if len(s) < 2:
        return s
    return (s[1], s[0]) + s[2:]

def map_top(f):
    "fn -> fn_s_s: run a grid policy on the head only — n-ary `mapsnd`/`on_model` (the world below is carried untouched)"
    def _s(s):
        return (f(s[0]),) + s[1:]
    return _s

def zip_top(c):
    """fn_p_g -> fn_s_s: binary combine of the top two channels, popping one.

    `c` is any pair-consumer (overlay is the canonical one); the top two grids are
    fed in as (head, next).  This is what makes depth>1 genuinely necessary: two
    derived views must both be live before they can be combined.
    """
    def _s(s):
        return (c((s[0], s[1])),) + s[2:]
    return _s

def commit_top(v):
    """int -> fn_s_s: the n-ary `sync_to_world`.

    Read v's coordinate off the TOP private channel, impose it on the channel
    BELOW (the world), pop the top.  Collapses depth by one.  At depth-1 this is
    exactly fork's commit: sync_to_world(v) applied to (world-below, model-top).
    """
    def _s(s):
        return (sync_to_world(v)((s[1], s[0])),) + s[2:]
    return _s

def peek(s):
    "gstack -> grid (fn_s_g): render the top of the stack — n-ary `fst_gg`"
    return s[0].copy()

def compose_gs(produce, endo):
    "fn_g_s, fn_s_s -> fn_g_s: (grid -> stack) then (stack -> stack); n-ary `compose_gp`"
    def _f(g):
        return endo(produce(g))
    return _f

def pipe_gsg(produce, render):
    "fn_g_s, fn_s_g -> fn: w |-> render(produce(w)); n-ary `pipe_gpg`"
    def _f(g):
        return render(produce(g))
    return _f

def fork_stack_decomposed(derive, av):
    """the decomposition identity (for checks): == fork(derive, sync_to_world(av)).

        pipe_gsg(compose_gs(compose_gs(compose_gs(base, dup_top),
                                       map_top(derive)),
                            commit_top(av)),
                peek)

    Trace on w:  base→[w]  dup_top→[w,w]  map_top(derive)→[derive(w),w]
                 commit_top(av)→[sync_to_world(av)(w, derive(w))]  peek→that world.
    """
    return pipe_gsg(
        compose_gs(compose_gs(compose_gs(base, dup_top),
                              map_top(derive)),
                   commit_top(av)),
        peek)

def mk_machine_g(init, step, render):
    "fn, fn, fn -> machine: grid-state machine (no private channel)"
    return ('machine_g', init, step, render)

def mk_machine_gg(init, step, render):
    "fn_g_p, sfn, fn_p_g -> machine: pair_gg-state machine (one private channel)"
    return ('machine_gg', init, step, render)

def unfold_m(g, T, m):
    """grid, int, machine -> mat: thread g through init, iterate step (T-1)x,
    render each frame.  The machine encodes its own state init and projection;
    the interpreter is identical for both state shapes.
    """
    _kind, init, step, render = m
    s = init(g)
    frames = [np.asarray(render(s))]
    for _ in range(T - 1):
        s = step(s)
        frames.append(np.asarray(render(s)))
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
    "fn -> fn: lambda over int, binding var in body"
    def _lam(a):
        body = deepcopy(body_delta)
        _sub_var(body, a)
        return body()
    return _lam

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


def simplify(tree):
    """Collapse spurious nesting in a (normalized) Delta tree, semantics-preserving.

    Rewrite (applied bottom-up, to fixpoint):

        (fork (fork X (sync_to_world v)) (sync_to_world v))
            ->  (fork X (sync_to_world v))

    The inner fork's commit, sync_to_world(v), produces "world with v moved to its
    position in X(world)" — i.e. it already moves only v.  The outer fork then
    re-commits the *same* entity v, re-deriving and re-applying the identical move,
    so it is a no-op wrapper.  (When the two sync values differ the outer commit is
    not a no-op, so the rule requires them equal and leaves such trees untouched.)

    This is a structural rewrite over fork/sync_to_world (file13's DSL); trees from
    other DSLs contain no such nodes and pass through unchanged.
    """
    if not isinstance(tree, Delta):
        return tree

    if tree.tails:
        tree.tails = [simplify(t) for t in tree.tails]

    while (tree.repr == 'fork' and tree.tails and len(tree.tails) == 2):
        derive, commit = tree.tails
        if (commit.repr == 'sync_to_world' and commit.tails and
                derive.repr == 'fork' and derive.tails and len(derive.tails) == 2):
            inner_commit = derive.tails[1]
            if (inner_commit.repr == 'sync_to_world' and inner_commit.tails and
                    inner_commit.tails[0].repr == commit.tails[0].repr):
                tree = derive          # drop the redundant outer fork
                continue
        break

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
