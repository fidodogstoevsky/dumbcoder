import numpy as np
from numpy.random import rand, randint
from numpy import array
from collections import defaultdict
from typing import Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

from itertools import chain

import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import trange
from copy import deepcopy
from time import time

from dsl import *
from scenes import Scenes, as_scenes, solves

###############################################################################
######                              library                               #####
###############################################################################

class Deltas:
    def __init__(self, core):
        self.core = core
        self.invented = []

        self.infer()

    def add(self, d: Delta, terminal=True):
        self.invented.append(d)
        self.infer()

    def pop(self, d: Delta):
        self.invented.pop(self.index(d) - len(self.core))
        self.infer()

    def infer(self):
        self.ds = self.core + self.invented
        self.terminals = array([isterminal(d) for d in self.ds])

        self.types = [d.type for d in self.ds]
        self.childtypes = [d.tailtypes for d in self.ds]

        self.bytype_terminal = defaultdict(list)
        self.bytype = defaultdict(list)

        for i, d in enumerate(self.ds):
            if not d.tailtypes:
                self.bytype_terminal[d.type].append(i)

            self.bytype[d.type].append(i)

        for idx, d in enumerate(self.ds):
            d.idx = idx

        # O(1) lookup dicts (first-occurrence wins, matching original linear-scan semantics)
        self._idx_by_repr = {}
        self._idx_by_head_type = {}
        for i, d in enumerate(self.ds):
            if d.repr not in self._idx_by_repr:
                self._idx_by_repr[d.repr] = i
            key = (id(d.head), d.type)
            if key not in self._idx_by_head_type:
                self._idx_by_head_type[key] = i

    def logp(self, Q, d):
        if d.tails is None:
            return Q[self.index(d)]

        out = 0
        for tail in d.tails:
            out += self.logp(Q, tail)

        return Q[self.index(d)] + out

    def __iter__(self):
        return chain(self.core + self.invented)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if (idx := self.index(idx)) is None:
                return None
        
        return deepcopy(self.ds[idx])

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return f"{self.core} + {self.invented}"

    def index(self, d: Union[Delta, str]):
        if isinstance(d, str):
            return self._idx_by_repr.get(d)
        return self._idx_by_head_type.get((id(d.head), d.type))

    def reset(self):
        self.invented = []
        self.infer()

###############################################################################
######                             enumerate                              #####
###############################################################################

def makepaths(D, Q):
    Paths = [[] for i in range(len(D))]
    Paths_terminal = [[] for i in range(len(D))]

    for d in D:
        if not d.tailtypes:
            continue

        for tidx, tailtype in enumerate(d.tailtypes):
            ps = Q.clone()

            # limit by type
            possibles = D.bytype[tailtype]
            for idx in range(len(ps)):
                if idx not in possibles:
                    ps[idx] = -np.inf

            ps = F.log_softmax(ps, -1).tolist()
            Paths[d.idx].append(deepcopy(ps))

            # permit leafs
            possibles_terminal = D.bytype_terminal[tailtype]

            for idx in range(len(ps)):
                if idx not in possibles_terminal:
                    ps[idx] = -np.inf

            Paths_terminal[d.idx].append(ps)

    return Paths, Paths_terminal

def cenumerate(D, Q, tp, budget, maxdepth, cb, deadline=None):
    """Enumerate programs by probability, callback-style.

    Budget window (lo, hi): fires cb when accumulated log-cost falls in [lo, hi].
    Called in expanding windows from solve_enumeration to try cheapest programs first.
    Q: flat log-prob tensor (nd,).
    """
    if budget[1] <= 0 or maxdepth < 0:
        return True

    if deadline is not None and time() > deadline:
        raise _EnumDone()

    for i in D.bytype[tp]:
        qi = Q[i].item() if isinstance(Q[i], th.Tensor) else float(Q[i])
        if -qi > budget[1]:
            continue

        d = D.ds[i]
        logp = qi
        nbudget = (budget[0] + logp, budget[1] + logp)
        tailtypes = list(d.tailtypes) if d.tailtypes is not None else d.tailtypes

        cenumerate_fold(D, Q, d, tailtypes, nbudget, logp, maxdepth - 1, cb, deadline)

def cenumerate_fold(D, Q, d, tailtypes, budget, offset, maxdepth, cb, deadline=None):
    if tailtypes is not None and len(tailtypes) > 0:
        tailtp = tailtypes.pop(0)

        def ccb(tail, tlogp):
            nd = deepcopy(d)
            if nd.tails is None:
                nd.tails = []

            nd.tails.append(tail)
            nbudget = (budget[0] + tlogp, budget[1] + tlogp)
            noffset = offset + tlogp

            cenumerate_fold(D, Q, nd, list(tailtypes), nbudget, noffset, maxdepth, cb, deadline)

        return cenumerate(D, Q, tailtp, (0, budget[1]), maxdepth, ccb, deadline)

    if budget[0] <= 0 and 0 <= budget[1]:
        return cb(d, offset)

    return True

def groom(D, sources, alogp, budget, paths, maxdepth):
    """ fills in the argument list for a node
    input:
    * D: the primitives
    * sources: list of probability dists
    """
    if len(sources) == 0:
        yield alogp, []
        return

    source, *nextsources = sources


    for idx, logp in enumerate(source):
        if budget + logp < 0:
            continue

        for nlogp, tree in penumerate(D, D[idx], logp, budget + logp, paths, maxdepth-1):
            for nnlogp, nntrees in groom(D, nextsources, alogp + nlogp, budget + nlogp, paths, maxdepth-1):
                yield nnlogp, [tree] + nntrees


def penumerate(D, n, nlogp, budget, paths, maxdepth=3):
    """ expands a node
    enumerate programs by probability
    generator-style, single budget ceiling,
    used when a budget > 0 is passed to
    solve_enumeration and also internally in saturate"""
    if budget < 0 or isterminal(n):
        yield nlogp, n
        return

    sources = paths[int(maxdepth <= 1)][n.idx]

    for logp, args in groom(D, sources, nlogp, budget + nlogp, paths, maxdepth-1):
        n.tails = args
        yield logp, deepcopy(n)

def _unsatisfied_tailtypes(tree):
    """Return the list of tailtypes for arg slots not yet filled in tree (BFS).

    When stitch creates a partial-application abstraction — e.g. fn_0 whose body
    is (unfold (grid_expr) (int_expr)) with unfold's fn slot empty — the missing
    slot's type is not captured by typize (which only sees $i holes).  This
    function collects those missing types so they can be appended to tailtypes,
    making the invented primitive usable during enumeration and correctly typed
    during annotation of later abstractions that call it with the extra args.
    """
    result = []
    qq = [tree]
    while qq:
        n = qq.pop(0)
        if n.tailtypes:
            n_filled = len(n.tails) if n.tails else 0
            for i in range(n_filled, len(n.tailtypes)):
                result.append(n.tailtypes[i])
        if n.tails:
            for t in n.tails:
                qq.append(t)
    return result


def _annotate_holes(D, tree):
    """BFS over a parsed abstraction body tree.
    Propagates the expected type (from each parent's tailtypes) down to
    $i placeholder nodes, marking them ishole=True so typize() can find them.
    Modifies tree in-place.
    """
    qq = [(tree, None)]
    while qq:
        n, expected_type = qq.pop(0)
        # $i placeholders come out of todelta() as bare Delta('$i') – untyped, no tails
        if isinstance(n.head, str) and n.head.startswith('$'):
            if expected_type is not None:
                n.type = expected_type
            n.ishole = True
            continue
        if not n.tails:
            continue
        if n.tailtypes:
            for tail, tailtype in zip(n.tails, n.tailtypes):
                qq.append((tail, tailtype))
        else:
            for tail in n.tails:
                qq.append((tail, None))

###############################################################################
######                             compress                               #####
###############################################################################

def saturate_stitch(D, sols, iterations=10, max_arity=6):
    """Learn abstractions via stitch_core (top-down synthesis).

    Replaces saturate(D, sols):
      1. serialise solution trees to s-expression strings
      2. run stitch_core.compress() to discover reusable fragments
      3. parse each abstraction body, infer argument types, register in D
      4. parse stitch's rewritten programs as the new tree corpus
    """
    try:
        import stitch_core
    except ImportError:
        print("stitch_core not installed; no compression performed")
        D.reset()
        D._stitch_result = None          # nothing to rewrite arbitrary programs through
        D._stitch_name_map = {}
        fallback = [simplify(normalize(s)) for s in sols.values() if s]
        return fallback, [str(t) for t in fallback]

    # Always pass fully-expanded (normalized) programs to stitch.
    # Using compressed programs (with fn_0 as an opaque token) causes a
    # naming collision: stitch names its new discoveries fn_0, fn_1, etc.,
    # clashing with the existing fn_0 token already in the programs.
    # Stitch's fn_0 body then references the input token "fn_0", which after
    # the fn_0→fn_1 offset remapping becomes self-referential ("fn_1 uses fn_1"),
    # and expand_skipped loops forever trying to inline-expand it.
    # With normalized programs stitch sees only primitive operations — no fn_i
    # tokens — so its discoveries are always fresh with clean intra-stitch deps.
    n_already_invented = len(D.invented)
    # normalize then collapse spurious nesting, so stitch compresses minimal forms
    # and never invents (or propagates) redundant fork wrappers.
    trees = [simplify(normalize(s)) for s in sols.values() if s]
    D.reset()
    D._stitch_result = None              # (re)set; populated below once compress runs
    D._stitch_name_map = {}

    if not trees:
        return trees, []

    programs = [str(t) for t in trees]

    print(f"running stitch_core.compress on {len(programs)} programs "
          f"(iterations={iterations}, max_arity={max_arity})")

    result = stitch_core.compress(programs, iterations=iterations,
                                  max_arity=max_arity, silent=True)

    print(f"stitch returned {len(result.abstractions)} abstractions")
    if not result.abstractions:
        print("stitch found no useful abstractions")
        return trees, [str(t) for t in trees]

    # map stitch's original abstraction name -> the (offset-remapped) name registered
    # in D, for the successfully-registered abstractions only.  Stashed on D so an
    # arbitrary program can later be rewritten through this exact library — see
    # rewrite_through_library() (used by the MDL-margin experiment to price rivals).
    registered_name_map = {}

    # Register each abstraction in D in discovery order so that later
    # abstractions can reference earlier ones during parsing.
    # Also track skipped partial-application bodies so their references in
    # rewritten programs can be inline-expanded rather than lost entirely.
    skipped_bodies = {}  # name -> stitch body string (with #i holes, not $i)

    def _parse_sexp(s, i):
        """Parse one s-expression from s starting at position i.
        Skips leading spaces.  Returns (token_str, end_pos) or (None, i).
        Handles nested parentheses correctly — unlike regex, won't consume
        closing parens that belong to the outer expression.
        """
        while i < len(s) and s[i] == ' ':
            i += 1
        if i >= len(s) or s[i] == ')':
            return None, i
        if s[i] == '(':
            start, depth = i, 0
            while i < len(s):
                if s[i] == '(':
                    depth += 1
                elif s[i] == ')':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1], i+1
                i += 1
            return s[start:], i   # unbalanced — return what we have
        else:
            start = i
            while i < len(s) and s[i] not in ' ()':
                i += 1
            return s[start:i], i

    def expand_skipped(prog_str):
        """Inline-expand references to skipped abstractions.

        Stitch may skip fn_X but still rewrite programs using it.  Two cases:

        1. Template hole: fn_1: (#0 1 0), used as (fn_1 place_wall)
           → substitute #0 → (place_wall 1 0)

        2. Partial application: fn_6: (place_ag blank), used as (fn_6 0 3 3 1)
           → no #i holes, extra args appended inside body's outermost parens
           → (place_ag blank 0 3 3 1)

        3. Bare reference: fn_3: (fn_0 2 1), used as a value, e.g.
           (compose (wall_at c0 c3) fn_3)
           → no parens, no args → substitute the body in place
           → (compose (wall_at c0 c3) (fn_0 2 1))

        Uses a proper s-expression parser so nested parens in arguments are
        handled correctly (unlike regex which misidentifies ) as a non-whitespace
        argument token and swallows the tail of the enclosing expression).
        """
        def _subst(body, args):
            """Substitute a skipped abstraction's body for a use site.

            #i holes are filled from args positionally; any args beyond the
            body's #-arity are appended inside the body's outermost parens
            (partial-application case).  A bare reference passes args=[] and so
            yields the body unchanged.
            """
            arity = 0
            while f'#{arity}' in body:
                arity += 1
            expanded = body
            for j in range(min(arity, len(args))):
                expanded = expanded.replace(f'#{j}', args[j])
            extra = args[arity:]
            if extra and expanded.startswith('(') and expanded.endswith(')'):
                expanded = expanded[:-1] + ' ' + ' '.join(extra) + ')'
            return expanded

        def expand_once(s):
            result = []
            i = 0
            while i < len(s):
                c = s[i]
                if c == '(':
                    # check whether this is a call to a skipped abstraction
                    matched = False
                    for name, body in skipped_bodies.items():
                        prefix = '(' + name
                        end_name = i + len(prefix)
                        if (s[i:end_name] == prefix and
                                (end_name >= len(s) or s[end_name] in ' )')):
                            # parse arguments until the matching close paren
                            pos = end_name
                            args = []
                            while pos < len(s) and s[pos] != ')':
                                arg, pos = _parse_sexp(s, pos)
                                if arg is None:
                                    break
                                args.append(arg)
                            if pos < len(s) and s[pos] == ')':
                                result.append(_subst(body, args))
                                i = pos + 1
                                matched = True
                                break
                    if not matched:
                        result.append(c)
                        i += 1
                elif c in ' )':
                    result.append(c)
                    i += 1
                else:
                    # bare atom token: a skipped abstraction can appear as a
                    # value (no parens, no args), e.g. (compose f fn_3).  The
                    # call-form scan above never sees these, so match the whole
                    # atom and substitute its body with no args.
                    atom, j = _parse_sexp(s, i)
                    if atom in skipped_bodies:
                        result.append(_subst(skipped_bodies[atom], []))
                    else:
                        result.append(atom)
                    i = j
            return ''.join(result)

        changed = True
        iters = 0
        while changed and iters < 50:
            new = expand_once(prog_str)
            changed = new != prog_str
            prog_str = new
            iters += 1
        return prog_str

    for abs_result in result.abstractions:
        # Remap stitch's fn_i to fn_{i + n_already_invented} so names are
        # globally unique across ECD iterations and don't shadow prior inventions.
        stitch_i = int(abs_result.name.split('_')[1])
        name = f"fn_{stitch_i + n_already_invented}"

        # stitch uses #i for argument holes; todelta() expects $i
        body_str = abs_result.body
        body_str_dollar = body_str  # preserve #i version for skipped_bodies

        # Inline-expand any skipped abstractions referenced inside this body
        # before attempting to register it.  Needed when a later abstraction's
        # body references an earlier skipped one (e.g. fn_3 body uses fn_1).
        body_str = expand_skipped(body_str)

        # Remap fn_i references inside the body to use the global offset.
        # Must be done longest-name-first to avoid fn_1 matching inside fn_10.
        import re as _re
        for si in sorted(range(len(result.abstractions)), reverse=True):
            old = f'fn_{si}'
            new_n = f'fn_{si + n_already_invented}'
            body_str = _re.sub(rf'\b{old}\b', new_n, body_str)

        body_str_dollar = body_str  # save expanded #i form for future expansions

        for i in range(abs_result.arity):
            body_str = body_str.replace(f'#{i}', f'${i}')

        try:
            hiddentail = tr(D, body_str)
        except Exception as e:
            print(f"skipping abstraction '{name}' — "
                  f"could not parse body '{body_str}': {e}")
            skipped_bodies[name] = body_str_dollar
            continue

        # Mark $i placeholders as typed holes so typize() can collect them
        _annotate_holes(D, hiddentail)

        # Inject explicit $i isarg nodes for any partial-application slots.
        # Stitch uses MDL-optimal encoding: when the rest arg is a single unique
        # token per program (_rest_i), adding #3 to the body costs 1 token but
        # saves 0 per rewrite, so stitch omits it. The result is a 3-arity body
        # `(mk_agent_scene 1 (set_at #2 #1 3) (seek 1 #0))` with the 4th arg
        # passed as an "extra" beyond the explicit arity. Without an explicit $3
        # in the hiddentail, Delta.__call__ drops it via replace_hidden (no match).
        # Injecting isarg nodes fixes evaluation so fn_0(gv,c,r,rest) correctly
        # produces mk_agent_scene(1,set_at(r,c,3),seek(1,gv),rest).
        _partial_types = _unsatisfied_tailtypes(hiddentail)
        if _partial_types:
            hole_idx = len(typize(hiddentail))   # next available $i index
            remaining = list(_partial_types)
            qq = [hiddentail]
            while qq and remaining:
                n = qq.pop(0)
                if n.tailtypes:
                    n_filled = len(n.tails) if n.tails else 0
                    while n_filled < len(n.tailtypes) and remaining:
                        new_hole = Delta(f'${hole_idx}', ishole=True, type=n.tailtypes[n_filled])
                        if n.tails is None:
                            n.tails = []
                        n.tails.append(new_hole)
                        n_filled += 1
                        hole_idx += 1
                        remaining.pop(0)
                if n.tails:
                    for t in n.tails:
                        qq.append(t)

        tailtypes = typize(hiddentail)
        if len(tailtypes) == 0:
            # 0-arity stitch abstractions are either partial applications of
            # multi-arg primitives (e.g. (iterate 1 (gset ...)) with only 2 of
            # 4 args) or constants whose numpy-array head breaks D.index after
            # deepcopy.  Skip them entirely, but save the body for expansion.
            print(f"skipping abstraction '{name}': {abs_result.body} — no typed holes found "
                  f"(partial application or unannotatable constant)")
            skipped_bodies[name] = body_str_dollar
            continue
        else:
            df = Delta(name, type=hiddentail.type, tailtypes=tailtypes,
                       hiddentail=hiddentail, repr=name)

        freeze(df)
        D.add(df)
        registered_name_map[abs_result.name] = name
        _argtypes_str = ', '.join(str(t) for t in (df.tailtypes or []))
        print(f"added abstraction {name}: {abs_result.body}  [{_argtypes_str}] -> {df.type}")

    def remap_names(s):
        "Remap stitch's fn_i references to globally-unique fn_{i+offset} names."
        for si in sorted(range(len(result.abstractions)), reverse=True):
            s = _re.sub(rf'\bfn_{si}\b', f'fn_{si + n_already_invented}', s)
        return s

    # Parse stitch's rewritten programs as the new compressed tree corpus.
    # For programs referencing skipped abstractions, inline-expand them first.
    #
    # Two lists with DIFFERENT contracts:
    #   * new_trees    — successfully-parsed trees for Q training; order irrelevant,
    #                    so parse failures are simply dropped.
    #   * rewritten_out— ONE ENTRY PER INPUT, positionally aligned with `trees`
    #                    (hence with both callers' `sol_keys`, which zip this list
    #                    against them by position).  A parse failure must NOT shorten
    #                    this list or every later key silently binds to the wrong
    #                    program; on failure we keep the base-primitive input form
    #                    (a valid identity rewrite referencing no abstraction token),
    #                    which preserves alignment and stays parseable for the
    #                    usage-census / constructor-specificity consumers.
    assert len(result.rewritten) == len(programs), (
        f"stitch returned {len(result.rewritten)} rewritten programs for "
        f"{len(programs)} inputs — cannot align rewrites to solution keys")
    new_trees = []
    rewritten_out = []
    for prog_str, base_str in zip(result.rewritten, programs):
        expanded = expand_skipped(remap_names(prog_str))
        try:
            tree = tr(D, expanded)
            freeze(tree)
            new_trees.append(tree)
            rewritten_out.append(str(tree))
        except Exception as e:
            if expanded != prog_str:
                print(f"could not parse rewritten program '{prog_str}' "
                      f"(expanded: '{expanded}'): {e}")
            else:
                print(f"could not parse rewritten program '{prog_str}': {e}")
            rewritten_out.append(base_str)

    # Fall back to original trees for Q training if most rewrites failed to parse
    # (stitch often references abstractions that were skipped, making rewrites unparseable)
    training_trees = new_trees if len(new_trees) * 2 >= len(trees) else trees
    # expose the exact compress result + name remap so callers can rewrite arbitrary
    # programs through this same library (rewrite_through_library).
    D._stitch_result = result
    D._stitch_name_map = registered_name_map
    return training_trees, rewritten_out


def rewrite_through_library(D, program_strs):
    """Rewrite base-primitive programs through the abstractions the most recent
    `saturate_stitch` registered in D — applying existing abstractions only, never
    discovering new ones.  This prices an arbitrary program (e.g. a non-mental rival
    spelling) under the SAME final library the corpus was compressed into, so a
    `_dl` comparison against a library-rewritten solution is apples-to-apples.

    Returns one rewritten s-expression per input (referencing D's invented tokens
    where they apply).  A program that shares no abstraction comes back normalized
    but otherwise unchanged.  Falls back to the normalized input if stitch_core is
    unavailable or no abstractions were registered.
    """
    norm = [str(simplify(normalize(tr(D, s) if isinstance(s, str) else s)))
            for s in program_strs]
    result = getattr(D, '_stitch_result', None)
    name_map = getattr(D, '_stitch_name_map', {}) or {}
    if result is None or not name_map:
        return norm
    try:
        import stitch_core
    except ImportError:
        return norm
    keep = [a for a in result.abstractions if a.name in name_map]
    if not keep:
        return norm
    import re as _re
    rewritten = stitch_core.rewrite(norm, keep).rewritten
    out = []
    for s in rewritten:
        # remap stitch's original fn_i -> the name registered in D (longest first so
        # fn_1 never matches inside fn_10); then verify it parses under D, else keep base.
        for orig in sorted(name_map, key=lambda n: -int(n.split('_')[1])):
            s = _re.sub(rf'\b{orig}\b', name_map[orig], s)
        out.append(s)
    safe = []
    for base, lib in zip(norm, out):
        try:
            tr(D, lib)
            safe.append(lib)
        except Exception:
            safe.append(base)     # rewrite referenced a skipped token — price in base prims
    return safe


def needle(D, n, paths, paths_terminal, depth=0):
    if n.tailtypes is None:
        return
    if depth < 0:
        return

    source = paths_terminal if depth <= 1 else paths
    n.tails = []

    for path in source[n.idx]:
        nn = deepcopy(D[sample(path)])

        n.tails.append(nn)
        needle(D, nn, paths, paths_terminal, depth - 1)


def newtree(D, type, paths, paths_terminal, depth=6, q=None):
    if q is None:
        q = th.ones(len(D))

    if q.requires_grad:
        q = q.detach()

    q = q.flatten()
    qroot = deepcopy(q)

    for i in range(len(q)):
        if i not in D.bytype[type]:
            qroot[i] = -np.inf

    qroot = F.softmax(qroot, -1)

    root = D[sample(qroot)]
    tree = deepcopy(root)

    needle(D, tree, paths, paths_terminal, depth=depth)

    return tree

class _EnumDone(Exception):
    pass

def solve_enumeration(Xs, D, Q, solutions=None, maxdepth=10, timeout=60, budget=0,
                      root_type=None, templates=None):
    """Enumerate programs from the DSL and match against tasks.

    A task is a `Scenes` — k trajectories sharing one latent program — and a
    program solves it only by reproducing EVERY scene (see scenes.solves).  A
    bare (T, H, W) trajectory is accepted as a 1-scene task.

    root_type=fn       : programs are grid->grid fns; each scene is wrapped as
                         unfold(scene[0], T_scene, f).
    root_type=fn_p_g   : programs are pair->grid commits; each scene is wrapped as
                         unfold_with_template(scene[0], template_i, T_scene, c)
                         The second grid is a given template, not a
                         derived model, so templates[key] is a list of k templates
                         — one per scene, in scene order.
    """
    import dsl as _dsl
    if root_type is None:
        root_type = fn

    _enum_type = root_type
    _is_pair = (root_type == _dsl.fn_p_g)
    templates = templates or {}

    cnt = 0
    all_cnt = 0
    stime = time()

    LOGPGAP = 2
    done = False
    # (key, task, per-scene templates) — the templates are None for fn-rooted runs.
    targets = [(mat_key(x), as_scenes(x),
                templates.get(mat_key(x)) if _is_pair else None)
               for x in Xs]

    def cb(tree, logp):
        """called once per enumerated program."""
        nonlocal cnt, all_cnt, done

        all_cnt += 1
        if not(all_cnt % 10000) and time() - stime > timeout:
            done = True
            raise _EnumDone()

        try:
            val = tree()
        except Exception:
            return
        if not callable(val):
            return
        cnt += 1

        if not(cnt % 100) and time() - stime > timeout:
            done = True
            raise _EnumDone()

        for tkey, task, tmpl in targets:
            # `solves` short-circuits on the first scene that disagrees, so the
            # k-scene format costs ~1 unfold per task for the overwhelming
            # majority of enumerated programs — the same as the old single-scene
            # matcher.  Only a program that already reproduced scene 1 pays for
            # the remaining k-1, and that is exactly the coincidence we want
            # tested rather than accepted.
            if not solves(val, task, tmpl):
                continue
            # collapse spurious nesting before storing, so the length tiebreak
            # below compares minimal forms and stitch never sees redundant
            # wrappers (which it would otherwise abstract and propagate).
            stree = simplify(deepcopy(tree))
            if tkey not in solutions:
                print(f'[{cnt:6d}] caught {stree}', flush=True)
            if tkey not in solutions or length(stree) < length(solutions[tkey]):
                solutions[tkey] = stree

        if all(tkey in solutions for tkey, _, _ in targets):
            done = True

        if done:
            raise _EnumDone()

    if budget == 0:
        idx = 0
        deadline = stime + timeout
        while not done and time() < deadline:
            try:
                cenumerate(D, Q, _enum_type, (LOGPGAP * idx, LOGPGAP * (idx+1)), maxdepth, cb, deadline)
            except _EnumDone:
                pass
            idx += 1
    else:
        ephermal = Delta('root', ishole=True, tailtypes=[_enum_type])
        D.add(ephermal)
        Q = th.hstack((Q, tensor([0])))

        try:
            for logp, wrapper in penumerate(D, ephermal, 0, budget, makepaths(D, Q), maxdepth=maxdepth+1):
                tree = wrapper.tails[0]
                cb(tree, logp)
        except _EnumDone:
            pass

        D.pop(ephermal)

    return solutions



def _worker_init():
    """Run in each worker process: limit to 1 thread to prevent OpenMP oversubscription.
    On Linux, fork inherits the parent's thread pool state; without this every worker
    spawns a full OpenMP team and all workers compete for the same cores."""
    import torch
    torch.set_num_threads(1)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'


def _n_cpus_available():
    """CPUs actually allocated to this process.
    Uses sched_getaffinity on Linux (respects SLURM cgroups), falls back to cpu_count on macOS."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def _solve_one_task(args):
    """Worker for parallel task enumeration. Must be module-level for pickling.

    Returns (key, solution_or_None, elapsed_seconds) — the wall time is measured
    inside the worker so callers can calibrate per-task timeouts from real runs.
    """
    import time as _time
    x, D, q, sols_snapshot, timeout, budget, root_type = args
    _t0 = _time.perf_counter()
    result = solve_enumeration([x], D, q, dict(sols_snapshot),
                               maxdepth=10, timeout=timeout, budget=budget,
                               root_type=root_type)
    return mat_key(x), result.get(mat_key(x)), _time.perf_counter() - _t0


def ECD(Xs, D, timeout=60, per_task_timeout=None, budget=0, max_iterations=10, seeds=None,
        run_dream=True, max_arity=6, stitch_iterations=None, root_type=None, n_workers=None,
        content_aware_q=True):
    # when ECD is first run, reset the DSL
    D.reset()
    if root_type is None:
        root_type = fn

    _run_dream = run_dream and root_type == fn
    _n_workers = n_workers if n_workers is not None else _n_cpus_available()
    print(f'ECD: using {_n_workers} workers (allocated CPUs: {_n_cpus_available()})', flush=True)

    Qmodel = None  # no model on the first iteration; use uniform Q
    idx = 0
    sols = dict(seeds) if seeds else {}

    def all_solved():
        return all(mat_key(x) in sols for x in Xs)

    def uniform_type_q():
        "type-conditioned uniform Q: logp[i] = -log(count of symbols sharing type with i)"
        import math
        q = th.zeros(len(D))
        for tp, indices in D.bytype.items():
            lp = -math.log(len(indices))
            for i in indices:
                q[i] = lp
        return q

    def task_Q(x):
        "return a flat log-prob tensor for this task's matrix"
        if Qmodel is None:
            q = uniform_type_q()
            if content_aware_q:
                # Boost integer literals visible in the task's initial grids to
                # log-prob 0 (cost-free). For an av=4 task this drops the target
                # program from window 7 (~14 nats) to window ~9 nats, so both av=1
                # and av=4 tasks are solved quickly and proportionally in iter 1.
                # The k scenes of a task share the program's literals by
                # construction, so the union over scenes is the same set the
                # single-scene version computed — just read off more evidence.
                visible = as_scenes(x).values
                for d in D.ds:
                    if d.tailtypes is None and d.type == cellvalue and d.head in visible:
                        q[D.index(d)] = 0.0
        else:
            with th.no_grad():
                # the grid encoder reads ONE trajectory; give it a representative
                # scene (the task's scenes differ only in placement, not program).
                logits = Qmodel(tc_mat(as_scenes(x).rep)[None])  # (1, nd)
                q = F.log_softmax(logits.squeeze(0), dim=-1)     # (nd,)

        return q

    with ProcessPoolExecutor(max_workers=_n_workers, initializer=_worker_init) as pool:
        while idx < max_iterations:
            unsolved = [x for x in Xs if mat_key(x) not in sols]
            # use fixed per_task_timeout if given, else divide global budget evenly
            t = per_task_timeout if per_task_timeout is not None else timeout / len(unsolved)

            # Compute Q tensors in the main process (avoids sending Qmodel to workers),
            # then dispatch all unsolved tasks in parallel.
            # After several iterations fall back to uniform Q for persistently unsolved tasks.
            _uniform_q = uniform_type_q()
            _args = [
                (x, D, task_Q(x) if idx < 3 else _uniform_q, dict(sols), t, budget, root_type)
                for x in unsolved
            ]
            for k, sol, _elapsed in pool.map(_solve_one_task, _args):
                if sol is not None:
                    sols[k] = sol

            soltrees = [s for s in sols.values() if s is not None]
            _si = stitch_iterations if stitch_iterations is not None else 2
            if len(soltrees) > 0:
                trees, rewritten_strs = saturate_stitch(D, sols, iterations=_si, max_arity=max_arity)
            else:
                trees, rewritten_strs = [], []

            idx += 1

            Qmodel = dream(D, trees, training_Xs=Xs, root_type=root_type) if _run_dream else None

            if all_solved():
                break

            unsolved = [x for x in Xs if mat_key(x) not in sols]
            print(f'--- ECD iteration {idx}, {len(unsolved)}/{len(Xs)} unsolved ---', flush=True)

    full_keys = {mat_key(x) for x in Xs}

    sol_keys = [k for k, v in sols.items() if v is not None]
    rewritten_map = dict(zip(sol_keys, rewritten_strs)) if rewritten_strs else {}

    return ({k: v for k, v in sols.items() if k in full_keys},
            {k: v for k, v in rewritten_map.items() if k in full_keys})

def mat_key(x):
    """Task identity.  A `Scenes` hashes over ALL its scenes, so two tasks are the
    same only if every scene agrees; a bare trajectory keeps the old single-matrix
    key, which is what the output-side comparisons still use."""
    if isinstance(x, Scenes):
        return x.key
    return (x.shape, x.tobytes())


def mat_key_id(x):
    "Stable hex id for a task — a JSON-serialisable form of `mat_key` (same task → same id)."
    import hashlib
    if isinstance(x, Scenes):
        return x.id
    return hashlib.sha1(repr(x.shape).encode() + x.tobytes()).hexdigest()


class MatRecognitionModel(nn.Module):
    """Recognition model: flat matrix-conditioned Q.

    Grid encoder (roles identified by MOTION, not by hardcoded values — the corpus
    uses diverse agent/goal ids, so "agent"=the cell vacated between frame 0 and the
    last frame, "goal"=the non-bg cell occupied at both ends):
      h_agent0    = mean(row_embed + col_embed) over FRAME-0 agent (mover) start cell
      h_goal0     = mean(row_embed + col_embed) over the stationary (goal) cell
      h_wall      = mean(row_embed + col_embed) over ALL-frame wall cells  (wall pos)
      h_agent_all = mean(row_embed + col_embed) over ALL-frame non-goal entity cells
                                                                            (traj mean)
      h_T         = t_embed[T-1]                                           (path length)
      h_matrix = concat([h_agent0, h_goal0, h_wall, h_agent_all, h_T])  — (B, 5*nembd)

      Frame-0 gives unambiguous start coordinates.
      h_agent_all captures trajectory shape: a detour path yields a different mean
      agent position from a straight-line path even when T and start/goal are identical
      (e.g. detour (0,3)→(0,2)→(1,2)→(2,2) has mean (0.75,2.25) while straight
      (0,3)→(1,3)→(2,3)→(2,2) has mean (1.25,2.75)).  This is the key signal for
      distinguishing false-belief tasks where T = Manhattan distance.

    Prediction (flat, no tree context):
      dsl_logits = head(h_matrix)   — (nd,)

    This flat Q is computed ONCE per task and reused for all enumeration decisions.
    Eliminating the per-node GRU passes avoids both the 20× enumeration slowdown
    and the depth-generalization failure of tree-conditioned Q.
    """
    def __init__(self, nd, max_coord=16, vocabsize=10, nembd=64, max_t=24):
        super().__init__()
        self.nembd = nembd
        # Per-ENTITY layout (value-agnostic), so two-mover / two-goal scenes (witness,
        # dual belief) are representable rather than blurred into one mean:
        #   2 mover slots × [start-pos ‖ trajectory]  = 2·(2·nembd)
        #   2 goal  slots × [pos]                      = 2·nembd
        #   wall pool                                  = nembd
        #   T (path length)                            = nembd
        self.mat_emb = 8 * nembd
        self.nd = nd
        self.vocabsize = vocabsize

        # symbolic grid encoder
        self.row_embed = nn.Embedding(max_coord, nembd)
        self.col_embed = nn.Embedding(max_coord, nembd)

        # path-length embedding
        self.t_embed = nn.Embedding(max_t, nembd)

        # flat DSL logits: h_matrix -> nd
        self.head = nn.Linear(self.mat_emb, nd)

        print(f'{sum(p.numel() for p in self.parameters()) / 2**20:.2f}M params')

    def encode_matrix(self, x):
        """(B, T, H, W) long tensor -> (B, 8*nembd) grid+path-length embedding.

        PER-ENTITY pooling (value-agnostic): a two-mover / two-goal scene (witness,
        dual belief) is represented in separate slots instead of collapsed into a
        single blurred mean, which a one-agent pooling cannot distinguish from a
        one-mover scene.  Returns (h_matrix, []).
        """
        B = x.shape[0]
        outs = [self._encode_one(x[b]) for b in range(B)]
        return th.stack(outs, dim=0), []

    def _encode_one(self, xb):
        """(T, H, W) long tensor -> (8*nembd,).

        Roles by MOTION, not hardcoded values (the corpus uses diverse agent/goal ids):
        an entity cell occupied in frame 0 but VACATED by the last frame is a mover; one
        occupied at both ends is a stationary goal (its agent ends on it).  Distinct
        movers/goals are separated by VALUE (read off frame 0, never assumed) and slotted
        in canonical POSITION order, so the layout is permutation- and value-independent:

          mover slot k<2 : [ start-pos pool ‖ trajectory pool (cells with that value) ]
          goal  slot k<2 : [ pos pool ]
          wall           : pool over all wall cells
          T              : path-length embedding
        A single-mover scene fills slot 0 only; empty slots are zero.  The per-mover
        trajectory pool is the detour signal — a detour path has a different mean
        position from a straight one even when T and start/goal coincide, and now that
        signal is kept SEPARATE per mover (dual belief detours BOTH).
        """
        T, H, W = xb.shape
        device = xb.device
        WALL, EMPTY = 3, 0
        z = th.zeros(self.nembd, device=device)

        r_idx = th.arange(H, device=device)[:, None].expand(H, W)
        c_idx = th.arange(W, device=device)[None, :].expand(H, W)
        pos = self.row_embed(r_idx) + self.col_embed(c_idx)     # (H,W,nembd)

        def pool(mask):
            "mean position embedding over a (H,W) or (T,H,W) boolean mask"
            m = mask.float()
            if m.dim() == 3:
                num = (m[..., None] * pos[None]).sum(dim=(0, 1, 2))
            else:
                num = (m[..., None] * pos).sum(dim=(0, 1))
            return num / m.sum().clamp(min=1)

        f0, fl = xb[0], xb[-1]
        ent0 = (f0 != EMPTY) & (f0 != WALL)
        entl = (fl != EMPTY) & (fl != WALL)
        mover_cells = sorted(th.nonzero(ent0 & ~entl, as_tuple=False).tolist())  # vacated
        goal_cells  = sorted(th.nonzero(ent0 &  entl, as_tuple=False).tolist())  # stationary

        parts = []
        for k in range(2):                                       # 2 mover slots
            if k < len(mover_cells):
                r, c = mover_cells[k]
                parts.append(pos[r, c])                          # start pos
                parts.append(pool(xb == int(f0[r, c])))         # this mover's trajectory
            else:
                parts += [z, z]
        for k in range(2):                                       # 2 goal slots
            if k < len(goal_cells):
                r, c = goal_cells[k]
                parts.append(pos[r, c])
            else:
                parts.append(z)
        parts.append(pool(xb == WALL))                           # wall pos (empty→invisible)
        t_idx = min(T - 1, self.t_embed.num_embeddings - 1)
        parts.append(self.t_embed(th.tensor(t_idx, device=device)))
        return th.cat(parts, dim=-1)                             # (8*nembd,)

    def forward(self, x):
        "(B,T,H,W) -> dsl_logits (B,nd)"
        matrix_h, _ = self.encode_matrix(x)
        return self.head(matrix_h)

def sample(ps):
    # ps is either a softmax prob vector (newtree) or a log_softmax list with -inf-masked
    # non-candidates (needle's paths).  Detect log-probs by ANY negative entry — the old
    # `ps[0] < 0` test misfired when the first candidate had log-prob exactly 0.0 (a tail
    # type with a single candidate at index 0), leaving ps a Python list that `/=` can't
    # divide.  Coercing to an array up front makes both branches robust.
    ps = np.asarray(ps, dtype=float)
    if np.any(ps < 0):
        ps = np.exp(ps)

    # Guard a zero (or non-finite) total: an all-`-inf` log-prob row (every candidate
    # masked out) exponentiates to all-zeros, and `ps / 0` gives the NaN cdf that
    # trained both Jul-12 phase2 recognition models through warnings.  Fall back to a
    # uniform draw over the candidates so sampling stays well-defined.
    total = ps.sum()
    if not np.isfinite(total) or total <= 0:
        ps = np.ones_like(ps)
        total = ps.sum()
    ps = ps / total
    cdf = ps.cumsum(-1)
    x = rand()
    for i in range(len(ps)):
        if cdf[i] > x:
            return i

    return len(ps)-1

def tc_mat(x, vocabsize=10):
    "convert a numpy matrix to a (T, H, W) long tensor, clamping to valid range"
    return th.from_numpy(np.clip(x, 0, vocabsize - 1).astype(np.int64))

def model_q(qmodel, x):
    """Per-task GLOBAL log-prob tensor from a recognition model (ECD.task_Q's form).

    One matrix-conditioned forward pass gives a flat (nd,) distribution over the
    whole DSL.  NOTE: this is normalized GLOBALLY, not per type, so its costs are on
    a different scale from uniform_type_q / content_q.  The phases use dreamed_q()
    instead, which puts the model on the enumeration cost model's own (type-
    conditional) scale; this raw form is kept for ECD-compatibility/reference."""
    with th.no_grad():
        logits = qmodel(tc_mat(x)[None])                 # (1, nd)
        return F.log_softmax(logits.squeeze(0), dim=-1)  # (nd,)


def uniform_type_q(D):
    "type-conditioned uniform log-prob tensor: logp[i] = -log(#symbols of i's type)."
    import math
    q = th.zeros(len(D))
    for _tp, idxs in D.bytype.items():
        lp = -math.log(len(idxs))
        for i in idxs:
            q[i] = lp
    return q


def content_boost(D, q, x):
    """Force to cost 0 the terminals the task's own frames make visible.  Mutates q.

    Shared by experiment.content_q and dreamed_q so the two cannot drift.  They were
    separate copies of this trick, and the coord branch below existed only in
    content_q: any still-unsolved task with a REAL wall silently lost the boost the
    moment its family entered a dreamed round and switched Qs.

      * cellvalue literals visible anywhere in the task's scenes.  The k scenes of a
        task share the program's literals by construction, so the union over scenes
        is the same set a single scene gave — just evidenced more than once.
      * coord terminals naming a REAL, visible wall (value 3) — the obstacle family's
        wall, not belief's phantom one — so obstacle need not brute-force all
        SIZE×SIZE wall positions (~5k programs -> ~tens).  We scan ALL frames, not
        just frame 0: the wall is stamped during the trajectory (unfold starts on a
        wall-free grid), so it is absent from x[0].  Belief never renders its wall
        into the world frames, so this never fires there — belief's wall coordinate
        stays a uniform-cost latent, as it must, and the cheap-obstacle path can
        never leak into the belief search.

    (ECD.task_Q keeps its own inlined cellvalue-only boost: it is the reference /
    ECD-compatibility driver, not the path the phases enumerate on.)
    """
    task = as_scenes(x)
    visible = task.values
    wall_coords = {int(k) for s in task
                   for trc in np.argwhere(s == 3) for k in trc[1:]}
    for d in D.ds:
        if d.tailtypes is not None:
            continue
        if d.type == cellvalue and d.head in visible:
            q[D.index(d)] = 0.0
        elif d.type == coord and d.head in wall_coords:
            q[D.index(d)] = 0.0
    return q


# Weight on the uniform prior in dreamed_q's mixture — also the damage cap: no
# primitive can cost more than log(1/DREAM_PRIOR_W) nats above its uniform cost, so
# a family the recognition model has never seen stays reachable within a bounded
# number of extra budget windows.
DREAM_PRIOR_W = 0.5


def dreamed_q(qmodel, D, x):
    """Recognition-model Q for one task, on the SAME cost scale as content_q.

    A bare model_q is mis-scaled for the budget enumerator and, trained only on the
    families solved so far, can suppress the rare primitives a still-unsolved family
    (belief) needs.  dreamed_q fixes both:

      * TYPE-CONDITIONAL: the model's logits are re-softmaxed within each type group,
        so a program's summed cost is comparable to the uniform/content prior rather
        than living on the model's global scale (a ~10-node program would otherwise
        land many budget windows later and time out before it's reached).
      * MIXED WITH UNIFORM, NOT FLOORED AT IT.  This was q = max(model, uniform_type_q),
        justified as "the model can make a primitive cheaper but never less reachable,
        so dreaming can only help".  The floor does preserve reachability — but not
        TIME-to-reach, and time-to-reach is what a timed enumerator spends.  Both
        arguments are per-type distributions summing to 1, so their elementwise max
        sums to 1 + TV(model, uniform) > 1: every primitive gets cheaper and none gets
        dearer, which deflates the whole cost scale instead of reordering it.
        solve_enumeration sweeps fixed-width budget windows [2i, 2i+2) under a single
        wall clock, so deflation does not pull the model's guesses to the front — it
        pulls EVERYTHING into the early windows and fattens each one.  A confident
        model takes up to ~log 2 nats off per node; over a ~10-node belief program
        that is ~7 nats, i.e. ~3.5 extra windows of junk enumerated ahead of it.
        Measured on the 2026-07-26 runs: dreamed rounds solved FEWER tasks per round
        than uniform ones (phase-1 round 2: +88 dreamed vs +103 uniform; 290 vs 291
        final).  The mixture keeps each type group's mass at exactly 1 while capping
        how far the model can push any primitive ABOVE its uniform cost at
        log(1/DREAM_PRIOR_W) nats — which is what the floor was really buying.
      * CONTENT BOOST: via the shared content_boost, so this Q prices the task's own
        visible terminals exactly as the uniform/content baseline does.  The coord
        half of that boost used to be missing here.
    """
    import math
    with th.no_grad():
        logits = qmodel(tc_mat(as_scenes(x).rep)[None]).squeeze(0)  # (nd,)
    model = th.full((len(D),), -float('inf'))
    for _tp, idxs in D.bytype.items():                     # per-type renormalize
        idxs_t = th.tensor(idxs)
        model[idxs_t] = F.log_softmax(logits[idxs_t], dim=-1)
    # log((1-w)·model + w·uniform): still sums to 1 within each type group
    q = th.logaddexp(model + math.log(1.0 - DREAM_PRIOR_W),
                     uniform_type_q(D) + math.log(DREAM_PRIOR_W))
    return content_boost(D, q, x)

def dream(D, soltrees=[], training_Xs=None, root_type=None, n_iters=600):
    import dsl as _dsl
    if root_type is None:
        root_type = fn

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    qmodel = MatRecognitionModel(len(D)).to(device)

    opt = th.optim.Adam(qmodel.parameters())
    paths, paths_terminal = makepaths(D, th.ones(len(D)))

    _fantasy_type = root_type

    # The recognition model conditions on ONE trajectory, so a multi-scene task
    # contributes each of its scenes separately: same programs, k× the grid
    # evidence, and the replay source below can draw any scene of a solved task.
    _train_scenes = [s for x in (training_Xs or []) for s in as_scenes(x)]

    # Pre-compute grid parameters for fantasy generation.
    # Infer grid size and goal values from training tasks so nothing is hardcoded.
    _av_list = [1]
    _sz      = _train_scenes[0].shape[1] if _train_scenes else 5
    _gv_list = sorted({int(v) for x in _train_scenes
                       for v in np.unique(x[0])
                       if v not in (0, 3) and int(v) not in _av_list}) or [2]
    # Corpus scene statistics, so fantasy grids match the mix the model must encode.
    # `_val_pool` — every entity value the corpus uses (value-agnostic sampling);
    # `_multi_frac` — fraction of scenes with ≥4 entities (the two-mover + two-goal
    # regime of witness/dual belief).  Programs are still library-sampled (not injected),
    # so a multi-entity grid only *lets* a sampled program drive two movers — it exercises
    # the encoder's second mover/goal slots rather than manufacturing belief scenes.
    _ent_counts = [int(((np.unique(x[0]) != 0) & (np.unique(x[0]) != 3)).sum())
                   for x in _train_scenes]
    _val_pool   = sorted({int(v) for x in _train_scenes
                          for v in np.unique(x[0]) if v not in (0, 3)}) or [1, 2]
    _multi_frac = (sum(c >= 4 for c in _ent_counts) / len(_ent_counts)
                   if _ent_counts else 0.0)

    def _fresh_ig():
        """Fresh fantasy grid: entity values at random non-overlapping positions, so a
        sampled program has real entities to interact with.  With probability
        `_multi_frac` the grid is a corpus-matching FOUR-entity scene (two mover + two
        goal candidates drawn value-agnostically from `_val_pool`); otherwise the plain
        single-mover grid.  Roles are still decided by the program's motion — this only
        provides scenes the two-slot encoder can represent."""
        ig = np.zeros((_sz, _sz), dtype=int)
        cells = [(r, c) for r in range(_sz) for c in range(_sz)]
        np.random.shuffle(cells)
        if np.random.rand() < _multi_frac and len(_val_pool) >= 4:
            vals = [int(v) for v in np.random.choice(_val_pool, size=4, replace=False)]
        else:
            vals = _av_list + _gv_list
        for _i, _v in enumerate(vals):
            ig[cells[_i][0], cells[_i][1]] = int(_v)
        return ig

    tbar = trange(n_iters)
    for _dream_iter in tbar:
        n_replay  = min(4, len(soltrees))
        replays   = [soltrees[i] for i in randint(len(soltrees), size=n_replay)] if n_replay else []
        fantasies = [newtree(D, _fantasy_type, paths, paths_terminal, depth=10)
                     for _ in range(8 - n_replay)]
        tagged    = [(t, False) for t in replays] + [(t, True) for t in fantasies]

        opt.zero_grad()

        node_loss = tensor(0.0, device=device)
        n_nodes = 0

        for tree, is_fantasy in tagged:
            try:
                val = tree()
                if not callable(val):
                    continue
                if is_fantasy:
                    ig = _fresh_ig()
                    T  = int(randint(3, 8))
                else:
                    src = _train_scenes[randint(len(_train_scenes))] if _train_scenes else None
                    if src is None:
                        continue
                    ig = src[0]
                    T  = src.shape[0]
                out = _dsl.unfold(ig, T, val)
            except Exception:
                continue

            if not isinstance(out, np.ndarray) or 0 in out.shape or out.shape[0] < 2:
                continue

            x = tc_mat(out)[None].to(device)   # (1, T, H, W)
            dsl_logits = qmodel(x)             # (1, nd)

            # flat Q: predict each node in the tree using the same matrix embedding.
            # No tree-context GRU — avoids depth-generalization failure and per-node
            # forward-pass overhead during enumeration.
            accum = [node_loss, n_nodes]

            def visit(node):
                idx = D.index(node)
                if idx is not None:
                    accum[0] = accum[0] + F.cross_entropy(
                        dsl_logits, tensor([idx], device=device)
                    )
                    accum[1] += 1
                if node.tails:
                    for child in node.tails:
                        visit(child)

            visit(tree)
            node_loss = accum[0]
            n_nodes   = accum[1]

        if n_nodes == 0:
            continue

        loss = node_loss / n_nodes
        loss.backward()
        opt.step()

        tbar.set_description(f'{loss=:.2f} nodes={n_nodes}')

    return qmodel.to('cpu')