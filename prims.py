"""Primitive sets for the three-phase belief-synthesis curriculum.

One canonical home for every DSL the experiments hand the searcher, replacing the
three divergent `make_core_prims` that used to live in the task files.

    make_core_prims()                  the bare atomic DSL (no symmetric field)
    make_symmetric_prims(decomposed=)  the "cube": core + every symmetric complement
    make_stack_prims()                 phase 3's depth-polymorphic grid-stack

The cube (`make_symmetric_prims`) is what phases 1 and 2 run.  Each complement is
the *other* corner of one symmetry axis (direction / scope / z-order / projection /
bifunctor / pairing / utility / grid-edit).  None help a theory-of-mind task; each
is the natural tool for some non-mental one.  The experiment's claim is that joint
MDL still selects exactly the (read-model, write-world, single-av) corner for
belief while these corners attach to their non-mental families — so the agency
signature is discovered, not gerrymandered into the primitive set.

    decomposed=False  (PHASE 1)  atomic `fork` and `sync_to_world`; every corner is
                                 a single legible fn_p_g / fn / util node.

    decomposed=True   (PHASE 2)  `fork` AND `sync` are spelled out, so belief must
                                 be *rediscovered* as a compound rather than handed
                                 over.  Specifically:
        fork          -> (pipe_gpg (compose_gp dup (mapsnd derive)) commit)
        sync_to_world -> (register (locate av) (place av))
        sync_to_model -> (via_swap (register (locate v) (place v)))    [direction
                          complement is now a compound too — "decompose complements"]
      The SCOPE complements (sync_all / sync_except) fold over an unbounded set of
      shared values, so they have no register/locate/place spelling and stay ATOMIC.
      The z-order (overlay/underlay) and projection (fst_gg/snd_gg) commits likewise
      stay atomic — only the fork/sync plumbing is decomposed.
"""

from ecd import Delta
from dsl import (
    fn, util, direction, coord, cellvalue, fn_p_g, fn_p_p, fn_g_p,
    fn_g_c, fn_gc_g, fn_g_s, fn_s_s, fn_s_g,
    RIGHT, LEFT, UP, DOWN,
    fork, sync_to_world, overlay, then_sync,
    compose, step, optimize, neg_distance, wall_at,
    # symmetric complements (the "cube")
    fst_gg, snd_gg, sync_to_model, sync_all, sync_except, underlay,
    swap, via_swap, mapsnd, mapfst, bimap, dup, pair_blank,
    compose_gp, pipe_gpg, distance, clear_at, erase,
    # sync decomposition (register/locate/place)
    register, locate, place,
    # stack calculus (phase 3's arity generalization)
    base, dup_top, blank_top, map_top, swap_top, zip_top, commit_top, peek,
    compose_gs, pipe_gsg,
)
from tasks_minds import SIZE   # coord terminals match the grid geometry (0..SIZE-1)

# Two disjoint integer-literal terminal pools (see the `coord`/`cellvalue` note in
# dsl.py).  coord terminals get distinct reprs ('c0'..) because the s-expression
# parser resolves terminals by repr alone — a bare '2' could not disambiguate the
# coord-2 from the cellvalue-2.  Their HEAD stays the plain int, so wall_at still
# receives an integer position.  cellvalue reprs are the plain numerals.
_COORDS     = list(range(SIZE))
_CELLVALUES = list(range(10))   # agent/goal ids, step magnitudes (0..9; 3 = wall)


def _value_terminals():
    "coord terminals (grid positions) + cellvalue terminals (cell ids)."
    return ([Delta(k, coord,     repr=f'c{k}') for k in _COORDS]
            + [Delta(v, cellvalue, repr=str(v)) for v in _CELLVALUES])


def _grid_core():
    "the grid-state arrows, utility, and terminals shared by every DSL"
    return [
        # Grid-state primitives (fn = grid -> grid)
        Delta(compose,      fn,   [fn, fn],              repr='compose'),
        Delta(step,         fn,   [cellvalue, direction], repr='step'),
        Delta(optimize,     fn,   [util, cellvalue],     repr='optimize'),
        Delta(wall_at,      fn,   [coord, coord],        repr='wall_at'),

        # Utility
        Delta(neg_distance, util, [cellvalue],           repr='neg_dist'),

        # Direction terminals
        Delta(RIGHT, direction, repr='right'),
        Delta(LEFT,  direction, repr='left'),
        Delta(UP,    direction, repr='up'),
        Delta(DOWN,  direction, repr='down'),

        # coord + cellvalue terminals
        *_value_terminals(),
    ]


def make_core_prims():
    "the bare atomic DSL: fork/sync interface + grid core (no symmetric field)"
    return [
        # Pair interface: fork produces pairs; fn_p_g consumes them.
        Delta(fork,          fn,     [fn, fn_p_g],         repr='fork'),
        Delta(sync_to_world, fn_p_g, [cellvalue],          repr='sync_to_world'),
        Delta(overlay,       fn_p_g,                       repr='overlay'),
        Delta(then_sync,     fn_p_g, [fn_p_g, cellvalue],  repr='then_sync'),
    ] + _grid_core()


def make_symmetric_prims(decomposed=False):
    """make_core_prims + the symmetric complements (the "cube"); see module docstring.

    decomposed=False (phase 1): atomic fork + sync, every corner a single node.
    decomposed=True  (phase 2): fork and sync spelled out (fork plumbing + register/
    locate/place); sync_to_model becomes the compound (via_swap (register …)); the
    scope complements sync_all/sync_except stay atomic.
    """
    prims = make_core_prims() + [
        # projection: keep the world / keep the model channel, bare.  Listed first
        # among the complements because they are nullary (lower cost than the
        # value-parameterised commits below); enumeration walks -logp in coarse
        # bands, so within a band the cheaper nullary corner must precede the
        # int-taking one it ties with extensionally for best-first to pick the
        # genuinely shorter program.
        Delta(fst_gg,        fn_p_g,          repr='fst_gg'),
        Delta(snd_gg,        fn_p_g,          repr='snd_gg'),
        # z-order: world-wins blend (inpainting / underlay)
        Delta(underlay,      fn_p_g,          repr='underlay'),
        # scope: wholesale vs all-but-one (no single value privileged)
        Delta(sync_all,      fn_p_g,              repr='sync_all'),
        Delta(sync_except,   fn_p_g, [cellvalue], repr='sync_except'),
        # utility: repulsion (flee under maximise)
        Delta(distance,      util,   [cellvalue], repr='distance'),
        # grid edit: remove instead of add
        Delta(clear_at,      fn,     [coord, coord], repr='clear_at'),
        Delta(erase,         fn,     [cellvalue], repr='erase'),
    ]

    if not decomposed:
        # direction complement is an atomic commit in phase 1.
        prims.append(Delta(sync_to_model, fn_p_g, [cellvalue], repr='sync_to_model'))
        return prims

    # ── PHASE 2: decompose both fork AND sync ────────────────────────────────────
    # fork and sync_to_world (and the direction complement sync_to_model) must be
    # DISCOVERED from the product-category plumbing, not handed over.
    prims = [d for d in prims if d.repr not in ('fork', 'sync_to_world')]
    prims += [
        # fork plumbing: dup (Δ) ▸ mapsnd (bifunctor second) ▸ commit, wired by the
        # two typed composers.  fork ≡ (pipe_gpg (compose_gp dup (mapsnd derive)) C).
        Delta(pipe_gpg,    fn,     [fn_g_p, fn_p_g],  repr='pipe_gpg'),
        Delta(compose_gp,  fn_g_p, [fn_g_p, fn_p_p],  repr='compose_gp'),
        Delta(dup,         fn_g_p,                    repr='dup'),
        Delta(mapsnd,      fn_p_p, [fn],              repr='mapsnd'),
        # bifunctor / pairing complements: the wrong-channel / fresh-channel corners
        # belief must avoid (it uses mapsnd + dup).
        Delta(swap,        fn_p_p,                    repr='swap'),
        Delta(mapfst,      fn_p_p, [fn],              repr='mapfst'),
        Delta(bimap,       fn_p_p, [fn, fn],          repr='bimap'),
        Delta(pair_blank,  fn_g_p,                    repr='pair_blank'),
        # sync plumbing: read the agent's coordinate off the model channel
        # (locate av) and impose it on the world (place av), wired by the av-free
        # register.  sync_to_world ≡ (register (locate av) (place av)); av now
        # appears 3× (optimize + locate + place).
        Delta(register,    fn_p_g,  [fn_g_c, fn_gc_g], repr='register'),
        Delta(locate,      fn_g_c,  [cellvalue],       repr='locate'),
        Delta(place,       fn_gc_g, [cellvalue],       repr='place'),
        # direction complement decomposed too: sync_to_model ≡ (via_swap (register
        # (locate v) (place v))) — run the commit on the swapped pair.  No atomic
        # sync_to_model node; the searcher must reach for via_swap.
        Delta(via_swap,    fn_p_g, [fn_p_g],          repr='via_swap'),
    ]
    return prims


def make_stack_prims():
    """PHASE 3: the depth-polymorphic grid-stack — channel arity as a free parameter.

    Lifts the cube ops to an n-ary stack (cons/nil of grids) so the number of
    private channels is something the searcher chooses rather than something the
    interpreter stipulates at 1.
    """
    return [
        # atomic depth-1 pair plumbing (phase-1 core)
        Delta(fork,          fn,     [fn, fn_p_g],      repr='fork'),
        Delta(sync_to_world, fn_p_g, [cellvalue],       repr='sync_to_world'),
        Delta(overlay,       fn_p_g,                    repr='overlay'),

        # grid-state primitives (fn = grid -> grid)
        Delta(compose,      fn,   [fn, fn],              repr='compose'),
        Delta(step,         fn,   [cellvalue, direction], repr='step'),
        Delta(optimize,     fn,   [util, cellvalue],     repr='optimize'),
        Delta(wall_at,      fn,   [coord, coord],        repr='wall_at'),
        Delta(neg_distance, util, [cellvalue],           repr='neg_dist'),

        # ── the grid-stack: depth-polymorphic, n-ary lifts of the cube ops ──
        Delta(base,       fn_g_s,                       repr='base'),
        Delta(dup_top,    fn_s_s,                       repr='dup_top'),
        Delta(blank_top,  fn_s_s,                       repr='blank_top'),
        Delta(swap_top,   fn_s_s,                       repr='swap_top'),
        Delta(map_top,    fn_s_s, [fn],                 repr='map_top'),
        Delta(zip_top,    fn_s_s, [fn_p_g],             repr='zip_top'),
        Delta(commit_top, fn_s_s, [cellvalue],          repr='commit_top'),
        Delta(peek,       fn_s_g,                       repr='peek'),
        Delta(compose_gs, fn_g_s, [fn_g_s, fn_s_s],     repr='compose_gs'),
        Delta(pipe_gsg,   fn,     [fn_g_s, fn_s_g],     repr='pipe_gsg'),

        # direction terminals
        Delta(RIGHT, direction, repr='right'),
        Delta(LEFT,  direction, repr='left'),
        Delta(UP,    direction, repr='up'),
        Delta(DOWN,  direction, repr='down'),

        # coord + cellvalue terminals
        *_value_terminals(),
    ]
