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


from tqdm import trange
from copy import deepcopy
from time import time

from dsl import *

###################################################################################
##### 

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

    def __contains__(self, d):
        if d in self.ds:
            return True

        if isinstance(d, Delta):
            outd = d()
        else:
            outd = d

        od = [d() for d in self.ds if not d.tailtypes]

        return outd in od


    def index(self, d: Union[Delta, str]):
        if isinstance(d, str):
            return self._idx_by_repr.get(d)
        return self._idx_by_head_type.get((id(d.head), d.type))

    def reset(self):
        self.invented = []
        self.infer()


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

class TreeQ:
    """Wraps MatRecognitionModel for top-down tree-conditioned Q during enumeration.

    logits(h_parent) -> (nd,) log-prob tensor conditioned on parent hidden state.
    node_state(d_idx, slot, h_parent) -> (1, nembd) child hidden state.
    h0: zeros root hidden state.
    """
    def __init__(self, model, mat_h):
        self.model = model
        self.mat_h = mat_h          # (1, nembd), already on cpu, no grad
        self.h0 = th.zeros(1, model.nembd)

    def logits(self, h_parent=None):
        if h_parent is None:
            h_parent = self.h0
        with th.no_grad():
            raw = self.model.head(th.cat([self.mat_h, h_parent], dim=-1))
            return F.log_softmax(raw, -1).squeeze(0)   # (nd,)

    def node_state(self, d_idx, slot, h_parent):
        with th.no_grad():
            return self.model.node_state(
                tensor([d_idx]), tensor([slot]), h_parent
            )   # (1, nembd)


def cenumerate(D, Q, tp, budget, maxdepth, cb, deadline=None, h_parent=None):
    """ enumerate programs by probability
    callback-style, budget window used in
    the main solve_enumeration loop with expanding windows.

    Q: flat log-prob tensor  OR  TreeQ instance.
    h_parent: (1, nembd) hidden state of the slot being filled, or None for root.
    """
    if budget[1] <= 0 or maxdepth < 0:
        return True

    if deadline is not None and time() > deadline:
        raise _EnumDone()

    # Compute local log-prob vector, conditioned on h_parent when using TreeQ
    if isinstance(Q, TreeQ):
        q = Q.logits(h_parent)
    else:
        q = Q

    for i in D.bytype[tp]:
        qi = q[i].item() if isinstance(q[i], th.Tensor) else float(q[i])
        if -qi > budget[1]:
            continue

        d = D.ds[i]
        logp = qi
        nbudget = (budget[0] + logp, budget[1] + logp)
        tailtypes = list(d.tailtypes) if d.tailtypes is not None else d.tailtypes

        # pass d's own index and the h_parent used to choose it so
        # cenumerate_fold can derive each child slot's h_parent
        cenumerate_fold(D, Q, d, tailtypes, nbudget, logp, maxdepth - 1, cb, deadline,
                        d_idx=i, h_parent_of_d=h_parent)

def cenumerate_fold(D, Q, d, tailtypes, budget, offset, maxdepth, cb, deadline=None,
                    d_idx=None, h_parent_of_d=None):
    if tailtypes is not None and len(tailtypes) > 0:
        current_slot = len(d.tails) if d.tails else 0
        tailtp = tailtypes.pop(0)

        # h_parent for nodes chosen to fill current_slot of d
        if isinstance(Q, TreeQ) and d_idx is not None and h_parent_of_d is not None:
            h_child = Q.node_state(d_idx, current_slot, h_parent_of_d)
        else:
            h_child = None

        def ccb(tail, tlogp):
            nd = deepcopy(d)
            if nd.tails is None:
                nd.tails = []

            nd.tails.append(tail)
            nbudget = (budget[0] + tlogp, budget[1] + tlogp)
            noffset = offset + tlogp

            # same d, same h_parent_of_d — cenumerate_fold will compute the
            # next slot's h_child from those on the next iteration
            cenumerate_fold(D, Q, nd, list(tailtypes), nbudget, noffset, maxdepth, cb, deadline,
                            d_idx=d_idx, h_parent_of_d=h_parent_of_d)

        return cenumerate(D, Q, tailtp, (0, budget[1]), maxdepth, ccb, deadline, h_parent=h_child)

    if budget[0] < 0 and 0 <= budget[1]:
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


def saturate_stitch(D, sols, iterations=10, max_arity=6):
    """Learn abstractions via stitch_core (top-down synthesis).

    Replaces saturate(D, sols):
      1. serialise solution trees to s-expression strings
      2. run stitch_core.compress() to discover reusable fragments
      3. parse each abstraction body, infer argument types, register in D
      4. parse stitch's rewritten programs as the new tree corpus

    Falls back to saturate() if stitch_core is not installed.
    """
    try:
        import stitch_core
    except ImportError:
        print("stitch_core not installed; no compression performed")
        D.reset()
        fallback = [normalize(s) for s in sols.values() if s]
        return fallback, [str(t) for t in fallback]

    ghosttime = time()
    trees = [normalize(s) for s in sols.values() if s]
    D.reset()

    if not trees:
        return trees, []



    # If all solutions are singleton(grid_expr), strip singleton so stitch
    # discovers grid-level abstractions that can be composed with more gset calls.
    # We'll re-wrap with singleton when reconstructing the tree corpus.
    singleton_d = next((d for d in D.core if d.repr == 'singleton'), None)
    strip_singleton = (
        singleton_d is not None and
        all(t.head is singleton and t.tails and len(t.tails) == 1 for t in trees)
    )
    if strip_singleton:
        print("stripping singleton for grid-level compression")
        inner_trees = [t.tails[0] for t in trees]
        programs = [str(t) for t in inner_trees]
    else:
        programs = [str(tree) for tree in trees]

    print(f"running stitch_core.compress on {len(programs)} programs "
          f"(iterations={iterations}, max_arity={max_arity})")

    result = stitch_core.compress(programs, iterations=iterations,
                                  max_arity=max_arity, silent=True)

    if not result.abstractions:
        print("stitch found no useful abstractions")
        return trees, [str(t) for t in trees]

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

        Uses a proper s-expression parser so nested parens in arguments are
        handled correctly (unlike regex which misidentifies ) as a non-whitespace
        argument token and swallows the tail of the enclosing expression).
        """
        def expand_once(s):
            result = []
            i = 0
            while i < len(s):
                if s[i] == '(':
                    # check whether this is a call to a skipped abstraction
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
                                end = pos + 1
                                # count #i holes in body
                                arity = 0
                                while f'#{arity}' in body:
                                    arity += 1
                                # substitute #i → args[i]
                                expanded = body
                                for j in range(min(arity, len(args))):
                                    expanded = expanded.replace(f'#{j}', args[j])
                                # extra args beyond arity: append into body's
                                # outermost (...) — handles partial applications
                                # like (fn_6 a b c) where fn_6 = (place_ag blank)
                                extra = args[arity:]
                                if extra and expanded.startswith('(') and expanded.endswith(')'):
                                    expanded = expanded[:-1] + ' ' + ' '.join(extra) + ')'
                                result.append(expanded)
                                i = end
                                break
                    else:
                        result.append(s[i])
                        i += 1
                else:
                    result.append(s[i])
                    i += 1
            return ''.join(result)

        changed = True
        while changed:
            new = expand_once(prog_str)
            changed = new != prog_str
            prog_str = new
        return prog_str

    for abs_result in result.abstractions:
        # stitch uses #i for argument holes; todelta() expects $i
        body_str = abs_result.body
        body_str_dollar = body_str  # preserve #i version for skipped_bodies

        # Inline-expand any skipped abstractions referenced inside this body
        # before attempting to register it.  Needed when a later abstraction's
        # body references an earlier skipped one (e.g. fn_3 body uses fn_1).
        body_str = expand_skipped(body_str)
        body_str_dollar = body_str  # save expanded #i form for future expansions

        for i in range(abs_result.arity):
            body_str = body_str.replace(f'#{i}', f'${i}')

        try:
            hiddentail = tr(D, body_str)
        except Exception as e:
            print(f"skipping abstraction '{abs_result.name}' — "
                  f"could not parse body '{body_str}': {e}")
            skipped_bodies[abs_result.name] = body_str_dollar
            continue

        # Mark $i placeholders as typed holes so typize() can collect them
        _annotate_holes(D, hiddentail)

        tailtypes = typize(hiddentail)

        name = abs_result.name  # e.g. "fn_0", "fn_1", ...
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
        print(f"added abstraction {name}: {abs_result.body}  [{df.type}]")

    # Parse stitch's rewritten programs as the new compressed tree corpus.
    # For programs referencing skipped abstractions, inline-expand them first.
    new_trees = []
    for prog_str in result.rewritten:
        expanded = expand_skipped(prog_str)
        try:
            tree = tr(D, expanded)
            if strip_singleton:
                # re-wrap the grid tree with singleton
                wrapper = deepcopy(singleton_d)
                wrapper.tails = [tree]
                tree = wrapper
            freeze(tree)
            new_trees.append(tree)
        except Exception as e:
            if expanded != prog_str:
                print(f"could not parse rewritten program '{prog_str}' "
                      f"(expanded: '{expanded}'): {e}")
            else:
                print(f"could not parse rewritten program '{prog_str}': {e}")

    print(f"stitch compression took {(time() - ghosttime)/60:.2f}m")
    # Fall back to original trees for Q training if most rewrites failed to parse
    # (stitch often references abstractions that were skipped, making rewrites unparseable)
    training_trees = new_trees if len(new_trees) * 2 >= len(trees) else trees
    print(f"using {len(training_trees)}/{len(trees)} trees for Q training "
          f"({'rewritten' if training_trees is new_trees else 'original'})")
    return training_trees, result.rewritten


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

def solve_enumeration(Xs, D, Q, solutions=None, maxdepth=10, timeout=60, budget=0):
    """runs for each unsolved task
    systematically enumerates programs from the DSL, evaluating each one.
    If a program evaluates to a full task matrix in Xs, it's saved in solutions.
    The enumeration is budget-based: it iterates over increasingly wide
    probability budgets (LOGPGAP * idx to LOGPGAP * (idx+1)), so it tries
    the most probable programs first and fans out.
    Stops when all matrices in Xs are solved or timeout is hit."""
    print(f'{len(D)=}')

    cnt = 0
    all_cnt = 0
    stime = time()

    LOGPGAP = 2
    done = False
    targets = {mat_key(x): x for x in Xs}

    def cb(tree, logp):
        """called once per enumerated program.
        checks if the output matches any full task matrix in Xs."""
        nonlocal cnt, all_cnt, done, stime

        all_cnt += 1
        if not(all_cnt % 10000) and time() - stime > timeout:
            done = True
            raise _EnumDone()

        try:
            out = tree()
        except Exception as e:
            return

        if not isinstance(out, np.ndarray) or 0 in out.shape:
            return

        cnt += 1

        if not(cnt % 10000) and cnt > 0:
            print(f'! {cnt} trees, {cnt/(time()-stime):.0f}/s', flush=True)

        if not(cnt % 100) and time() - stime > timeout:
            done = True
            raise _EnumDone()

        okey = mat_key(out)
        if okey in targets:
            if okey not in solutions:
                print(f'[{cnt:6d}] caught {tree}', flush=True)

            if okey not in solutions or length(tree) < length(solutions[okey]):
                solutions[okey] = deepcopy(tree)

            if all(mat_key(x) in solutions for x in Xs):
                done = True

        if done:
            raise _EnumDone()

    if budget == 0:
        idx = 0
        deadline = stime + timeout
        while not done and time() < deadline:
            try:
                cenumerate(D, Q, mat, (LOGPGAP * idx, LOGPGAP * (idx+1)), maxdepth, cb, deadline)
            except _EnumDone:
                pass
            idx += 1
    else:
        ephermal = Delta('root', ishole=True, tailtypes=[mat])
        D.add(ephermal)
        Q = th.hstack((Q, tensor([0])))

        try:
            for logp, wrapper in penumerate(D, ephermal, 0, budget, makepaths(D, Q), maxdepth=maxdepth+1):
                tree = wrapper.tails[0]
                cb(tree, logp)
        except _EnumDone:
            pass

        D.pop(ephermal)

    took = time() - stime
    print(f'total: {cnt}, took: {took/60:.1f}m, iter: {cnt/(took+1e-9):.0f}/s', flush=True)
    print(f'solved: {sum(mat_key(x) in solutions for x in Xs)}/{len(Xs)}', flush=True)
    return solutions


def bootstrap_nav_solutions(Xs, D, max_walls=1):
    """Construct nav solutions analytically for tasks with at most max_walls visible walls.

    Tasks with more walls (and false-belief tasks) are left for ECD.
    The Q model, trained on these seeds, learns to read entity and wall positions
    from the matrix encoder, concentrating probability on the visible coordinates
    and reducing effective search window from ~10 to ~3 for harder tasks.

    max_walls: bootstrap tasks with this many walls or fewer (default 1).
    """
    def find_d(repr_name):
        for d in D.ds:
            if getattr(d, 'repr', '') == repr_name:
                return d
        return None

    def int_d(v):
        for d in D.ds:
            if d.type == int and d.tailtypes is None and d.head == v:
                return deepcopy(d)
        return None

    d_nav   = find_d('nav_unfold')
    d_ag    = find_d('place_ag')
    d_wall  = find_d('place_wall')
    d_blank = find_d('blank')

    sols = {}
    for x in Xs:
        frame = x[0]
        T = x.shape[0]
        agent = [(r,c) for r in range(frame.shape[0]) for c in range(frame.shape[1]) if frame[r,c] == 1]
        goal  = [(r,c) for r in range(frame.shape[0]) for c in range(frame.shape[1]) if frame[r,c] == 2]
        walls = [(r,c) for r in range(frame.shape[0]) for c in range(frame.shape[1]) if frame[r,c] == 3]

        # Only bootstrap up to max_walls; leave harder tasks for ECD
        if len(walls) > max_walls:
            continue

        if not agent or not goal or int_d(T) is None:
            continue
        ar, ac = agent[0];  gr, gc = goal[0]

        ag_node = deepcopy(d_ag)
        ag_node.tails = [deepcopy(d_blank), int_d(ar), int_d(ac), int_d(gr), int_d(gc)]
        grid_node = ag_node

        for wr, wc in walls:
            if int_d(wr) is None or int_d(wc) is None or d_wall is None:
                grid_node = None; break
            pw = deepcopy(d_wall)
            pw.tails = [grid_node, int_d(wr), int_d(wc)]
            grid_node = pw

        if grid_node is None:
            continue

        root = deepcopy(d_nav)
        root.tails = [grid_node, int_d(T)]
        try:
            if np.array_equal(root(), x):
                sols[mat_key(x)] = root
        except Exception:
            pass

    return sols


def ECD(Xs, D, timeout=60, per_task_timeout=None, budget=0, max_iterations=10, seeds=None):
    # when ECD is first run, reset the DSL
    D.reset()

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
            return uniform_type_q()
        with th.no_grad():
            logits = Qmodel(tc_mat(x)[None])          # (1, nd)
            return F.log_softmax(logits.squeeze(0), dim=-1)   # (nd,)

    while idx < max_iterations:
        unsolved = [x for x in Xs if mat_key(x) not in sols]
        # use fixed per_task_timeout if given, else divide global budget evenly
        t = per_task_timeout if per_task_timeout is not None else timeout / len(unsolved)

        for x in unsolved:
            if mat_key(x) in sols:
                continue  # may have been solved by an earlier task in this round
            sols = solve_enumeration([x], D, task_Q(x), sols,
                                     maxdepth=10, timeout=t, budget=budget)

        soltrees = [s for s in sols.values() if s is not None]
        if len(soltrees) > 0:
            trees, rewritten_strs = saturate_stitch(D, sols)
        else:
            trees, rewritten_strs = [], []

        idx += 1

        Qmodel = dream(D, trees)

        if all_solved():
            break

        unsolved = [x for x in Xs if mat_key(x) not in sols]
        print(f'--- ECD iteration {idx}, {len(unsolved)}/{len(Xs)} unsolved, Q task-specific ---', flush=True)

    full_keys = {mat_key(x) for x in Xs}

    sol_keys = [k for k, v in sols.items() if v is not None]
    rewritten_map = dict(zip(sol_keys, rewritten_strs)) if rewritten_strs else {}

    return ({k: v for k, v in sols.items() if k in full_keys},
            {k: v for k, v in rewritten_map.items() if k in full_keys})

def mat_key(x):
    return (x.shape, x.tobytes())


class MatRecognitionModel(nn.Module):
    """Recognition model: flat matrix-conditioned Q.

    Grid encoder:
      h_agent0    = mean(row_embed + col_embed) over FRAME-0 agent cells  (start pos)
      h_goal0     = mean(row_embed + col_embed) over FRAME-0 goal cells   (start pos)
      h_wall      = mean(row_embed + col_embed) over ALL-frame wall cells  (wall pos)
      h_agent_all = mean(row_embed + col_embed) over ALL-frame agent cells (traj mean)
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
        self.mat_emb = 5 * nembd   # frame0_agent + frame0_goal + all_wall + all_agent + T
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
        """(B, T, H, W) long tensor -> (B, 4*nembd) grid+path-length embedding.

        Agent and goal are pooled from FRAME 0 ONLY (clean start positions).
        Walls are pooled from ALL frames (static across frames anyway).
        T is embedded directly as a path-length feature.
        Returns (h_matrix, []).
        """
        B, T, H, W = x.shape
        device = x.device

        r_idx = th.arange(H, device=device)[None, None, :, None].expand(B, T, H, W)
        c_idx = th.arange(W, device=device)[None, None, None, :].expand(B, T, H, W)
        pos_e = self.row_embed(r_idx) + self.col_embed(c_idx)   # (B,T,H,W,nembd)

        # frame-0 position embeddings (B, 1, H, W, nembd)
        pos_e0 = pos_e[:, :1, :, :, :]
        x0 = x[:, :1, :, :]   # (B, 1, H, W)

        parts = []
        # agent and goal: pool from frame 0 only (unambiguous start positions)
        for val in (1, 2):
            mask = (x0 == val).float().unsqueeze(-1)             # (B,1,H,W,1)
            h = (pos_e0 * mask).sum(dim=(1,2,3)) / mask.sum(dim=(1,2,3)).clamp(min=1)
            parts.append(h)                                       # (B, nembd)

        # walls: pool from all frames (walls are static; all-frame sum just reinforces)
        mask_w = (x == 3).float().unsqueeze(-1)                  # (B,T,H,W,1)
        h_wall = (pos_e * mask_w).sum(dim=(1,2,3)) / mask_w.sum(dim=(1,2,3)).clamp(min=1)
        parts.append(h_wall)                                      # (B, nembd)

        # all-frame agent mean: encodes trajectory shape (differs between detour and
        # straight path even when T and start/goal positions are the same).
        # e.g. detour (0,3)→(0,2)→(1,2)→(2,2) has mean (0.75, 2.25) while
        # straight (0,3)→(1,3)→(2,3)→(2,2) has mean (1.25, 2.75).
        mask_a = (x == 1).float().unsqueeze(-1)                  # (B,T,H,W,1)
        h_agent_all = (pos_e * mask_a).sum(dim=(1,2,3)) / mask_a.sum(dim=(1,2,3)).clamp(min=1)
        parts.append(h_agent_all)                                 # (B, nembd)

        # T embedding: direct path-length signal for detour detection
        t_idx = th.full((B,), T - 1, dtype=th.long, device=device).clamp(
            max=self.t_embed.num_embeddings - 1)
        parts.append(self.t_embed(t_idx))                         # (B, nembd)

        h_matrix = th.cat(parts, dim=-1)   # (B, 5*nembd)
        return h_matrix, []

    def forward(self, x):
        "(B,T,H,W) -> dsl_logits (B,nd)"
        matrix_h, _ = self.encode_matrix(x)
        return self.head(matrix_h)

    @property
    def mat_embed_size(self):
        return self.mat_emb

def sample(ps):
    if ps[0] < 0:
        ps = np.exp(ps)

    ps /= ps.sum()
    cdf = ps.cumsum(-1)
    x = rand()
    for i in range(len(ps)):
        if cdf[i] > x:
            return i

    return len(ps)-1

def tc_mat(x, vocabsize=10):
    "convert a numpy matrix to a (T, H, W) long tensor, clamping to valid range"
    return th.from_numpy(np.clip(x, 0, vocabsize - 1).astype(np.int64))

def dream(D, soltrees=[]):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    qmodel = MatRecognitionModel(len(D)).to(device)

    opt = th.optim.Adam(qmodel.parameters())
    paths, paths_terminal = makepaths(D, th.ones(len(D)))

    tbar = trange(600)
    for _ in tbar:
        if len(soltrees) > 0:
            # Train purely on solution trees so Q concentrates on task structure.
            # Random trees add noise that competes with the solution signal.
            trees = [soltrees[i] for i in randint(len(soltrees), size=8)]
        else:
            trees = [newtree(D, mat, paths, paths_terminal, depth=10) for _ in range(8)]

        opt.zero_grad()

        node_loss = tensor(0.0, device=device)
        n_nodes = 0

        for tree in trees:
            try:
                out = tree()
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


###################################################################################
##### task production

def make_task(path, size=4):
    "create a (T, size, size) matrix with a single 1-cell moving along path"
    x = np.zeros((len(path), size, size), dtype=int)
    for t, (r, c) in enumerate(path):
        x[t, r, c] = 1
    return x

    # Xs = [
    #     make_task([(0,0),(0,1),(0,2),(0,3)]),  # right, row 0
    #     make_task([(1,0),(1,1),(1,2),(1,3)]),  # right, row 1
    #     make_task([(2,0),(2,1),(2,2),(2,3)]),  # right, row 2
    #     make_task([(3,0),(3,1),(3,2),(3,3)]),  # right, row 3
    #     make_task([(0,3),(0,2),(0,1),(0,0)]),  # left, row 0
    #     make_task([(1,3),(1,2),(1,1),(1,0)]),  # left, row 1
    #     make_task([(2,3),(2,2),(2,1),(2,0)]),  # left, row 2
    #     make_task([(3,3),(3,2),(3,1),(3,0)]),  # left, row 3
    #     make_task([(0,0),(1,0),(2,0),(3,0)]),  # down, col 0
    #     make_task([(0,1),(1,1),(2,1),(3,1)]),  # down, col 1
    #     make_task([(0,2),(1,2),(2,2),(3,2)]),  # down, col 2
    #     make_task([(0,3),(1,3),(2,3),(3,3)]),  # down, col 3
    #     make_task([(3,0),(2,0),(1,0),(0,0)]),  # up, col 0
    #     make_task([(3,1),(2,1),(1,1),(0,1)]),  # up, col 1
    #     make_task([(3,2),(2,2),(1,2),(0,2)]),  # up, col 2
    #     make_task([(3,3),(2,3),(1,3),(0,3)]),  # up, col 3
    # ]

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


def make_false_belief_task(size=6, n_phantoms=1, seed=None, min_dist=None, return_meta=False):
    """Generate a false belief navigation task.

    The agent navigates optimally on their BELIEVED grid (which has phantom
    walls not present in the actual world).  The observed trajectory X shows
    only agent (1) and goal (2) — phantom walls are hidden — so the path
    appears suboptimal on the blank actual grid.

    Solution program: believed_nav_unfold(place_wall(place_ag(blank,...),pwr,pwc), T)

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


def bootstrap_false_belief_solutions(task_phantoms, D):
    """Construct believed_nav_unfold solutions analytically for false-belief tasks.

    task_phantoms: list of (x, meta) pairs where x is (T,H,W) and
                   meta = {'agent', 'goal', 'phantom_walls'} from make_false_belief_task.
    Returns dict of mat_key -> solution tree for tasks that validate correctly.
    """
    def find_d(repr_name):
        for d in D.ds:
            if getattr(d, 'repr', '') == repr_name:
                return d
        return None

    def int_d(v):
        for d in D.ds:
            if d.type == int and d.tailtypes is None and d.head == v:
                return deepcopy(d)
        return None

    d_bnav  = find_d('believed_nav_unfold')
    d_ag    = find_d('place_ag')
    d_wall  = find_d('place_wall')
    d_blank = find_d('blank')

    if any(d is None for d in [d_bnav, d_ag, d_wall, d_blank]):
        print("bootstrap_false_belief_solutions: missing required primitives")
        return {}

    sols = {}
    for x, meta in task_phantoms:
        T = x.shape[0]
        ar, ac = meta['agent']
        gr, gc = meta['goal']
        phantom_walls = meta['phantom_walls']

        if int_d(T) is None or int_d(ar) is None or int_d(ac) is None:
            continue
        if int_d(gr) is None or int_d(gc) is None:
            continue

        ag_node = deepcopy(d_ag)
        ag_node.tails = [deepcopy(d_blank), int_d(ar), int_d(ac), int_d(gr), int_d(gc)]
        grid_node = ag_node

        ok = True
        for wr, wc in phantom_walls:
            if int_d(wr) is None or int_d(wc) is None:
                ok = False; break
            pw = deepcopy(d_wall)
            pw.tails = [grid_node, int_d(wr), int_d(wc)]
            grid_node = pw

        if not ok:
            continue

        root = deepcopy(d_bnav)
        root.tails = [grid_node, int_d(T)]
        try:
            if np.array_equal(root(), x):
                sols[mat_key(x)] = root
        except Exception:
            pass

    print(f"bootstrap_false_belief: seeded {len(sols)}/{len(task_phantoms)} tasks")
    return sols

###################################################################################

if __name__ == '__main__':
    """create an instance of the Deltas class, named D
    a Deltas object D is an instance of a DSL
    it's initiated with core primitives
    """
    # 4x4 grids: coordinates 0-3, path lengths up to ~7, so ints 0-7 suffice.
    # Smaller int space means enumeration reaches window 5-7 in 120s.
    # Once Q trains on 0-wall seeds it learns to read entity positions from the
    # matrix, collapsing the visible coords (ar,ac,gr,gc,wr,wc) to near-zero
    # effective cost and leaving only T unknown — dropping to window ~3.
    D = Deltas([
        # mat construction
        Delta(believed_nav_unfold, mat, [grid, int],           repr='believed_nav_unfold'),
        Delta(hide_walls,      mat,  [mat],                    repr='hide_walls'),
        Delta(nav_unfold,      mat,  [grid, int],              repr='nav_unfold'),
        Delta(unfold,          mat,  [grid, int, fn],          repr='unfold'),
        # grid primitives
        Delta(blank44,         grid,                            repr='blank'),
        Delta(gset,            grid, [grid, int, int, int],    repr='gset'),
        Delta(place_agent_goal,grid, [grid, int, int, int, int], repr='place_ag'),
        Delta(place_wall,      grid, [grid, int, int],         repr='place_wall'),
        # intentional motion
        Delta(approach,        fn,   [int, int],               repr='approach'),
        Delta(approach(1, 2),  fn,                             repr='navigate'),
        # goal algebra: if(exists(3), be_at(3), be_at(2)) etc.
        Delta(exists,          fn_pred, [int],                 repr='exists'),
        Delta(be_at,           fn,      [int],                 repr='be_at'),
        Delta(if_goal,         fn,      [fn_pred, fn, fn],     repr='if'),
        # goal type + optimize: declarative goals compiled to step functions
        Delta(at,              goal,    [int],                 repr='at'),
        Delta(if_else,         goal,    [fn_pred, goal, goal], repr='if_else'),
        Delta(optimize,        fn,      [goal],                repr='optimize'),
        # int terminals: 0-7 covers 4x4 coords (0-3) and typical path lengths
        Delta(0,  int, repr='0'),
        Delta(1,  int, repr='1'),
        Delta(2,  int, repr='2'),
        Delta(3,  int, repr='3'),
        Delta(4,  int, repr='4'),
        Delta(5,  int, repr='5'),
        Delta(6,  int, repr='6'),
        Delta(7,  int, repr='7'),
        Delta(8,  int, repr='8'),
        Delta(9,  int, repr='9'),
    ])

    # Generate false-belief tasks with metadata so we can bootstrap the first half.
    # The second half (fb_tasks[3:]) is left for ECD to discover.
    fb_tasks_meta = make_false_belief_tasks(n=6, size=4, n_phantoms=1, seed=4,
                                            return_meta=True)
    Xs_fb = [x for x, _ in fb_tasks_meta]

    Xs = (make_nav_tasks(n=6, size=4, n_walls=0, seed=0) +
          make_nav_tasks(n=6, size=4, n_walls=1, seed=1) +
          make_nav_tasks(n=6, size=4, n_walls=2, seed=2) +
          Xs_fb)
    print(f"{len(Xs)} tasks, shapes: {[x.shape for x in Xs[:4]]}")

    # Bootstrap all 18 nav tasks (0+1+2 wall) analytically — gives Q position
    # embeddings at every nav tree depth.  Then seed 3/6 false-belief tasks so Q
    # learns to assign probability to believed_nav_unfold.  ECD must find the
    # remaining 3 false-belief tasks, driving at least a few ECD iterations.
    nav_seeds = bootstrap_nav_solutions(Xs, D, max_walls=2)
    fb_seeds  = bootstrap_false_belief_solutions(fb_tasks_meta[:4], D)
    seeds = {**nav_seeds, **fb_seeds}
    print(f"bootstrapped {len(seeds)}/{len(Xs)} solutions")

    Z, rewritten_map = ECD(Xs, D, per_task_timeout=120, max_iterations=10, seeds=seeds)
    for k, v in Z.items():
        if v is not None:
            print(f'solution: {v}')
            if k in rewritten_map:
                print(f'rewritten: {rewritten_map[k]}')
