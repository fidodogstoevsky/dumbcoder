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
    for abs_result in result.abstractions:
        # stitch uses #i for argument holes; todelta() expects $i
        body_str = abs_result.body
        for i in range(abs_result.arity):
            body_str = body_str.replace(f'#{i}', f'${i}')

        try:
            hiddentail = tr(D, body_str)
        except Exception as e:
            print(f"skipping abstraction '{abs_result.name}' — "
                  f"could not parse body '{body_str}': {e}")
            continue

        # Mark $i placeholders as typed holes so typize() can collect them
        _annotate_holes(D, hiddentail)

        tailtypes = typize(hiddentail)

        name = abs_result.name  # e.g. "fn_0", "fn_1", ...
        if len(tailtypes) == 0:
            # 0-arity stitch abstractions are either partial applications of
            # multi-arg primitives (e.g. (iterate 1 (gset ...)) with only 2 of
            # 4 args) or constants whose numpy-array head breaks D.index after
            # deepcopy.  Skip them entirely.
            print(f"skipping abstraction '{name}': {abs_result.body} — no typed holes found "
                  f"(partial application or unannotatable constant)")
            continue
        else:
            df = Delta(name, type=hiddentail.type, tailtypes=tailtypes,
                       hiddentail=hiddentail, repr=name)

        freeze(df)
        D.add(df)
        print(f"added abstraction {name}: {abs_result.body}  [{df.type}]")

    # Parse stitch's rewritten programs as the new compressed tree corpus
    new_trees = []
    for prog_str in result.rewritten:
        try:
            tree = tr(D, prog_str)
            if strip_singleton:
                # re-wrap the grid tree with singleton
                wrapper = deepcopy(singleton_d)
                wrapper.tails = [tree]
                tree = wrapper
            freeze(tree)
            new_trees.append(tree)
        except Exception as e:
            print(f"could not parse rewritten program '{prog_str}': {e}")

    print(f"stitch compression took {(time() - ghosttime)/60:.2f}m")
    return new_trees if new_trees else trees, result.rewritten


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


def bootstrap_nav_solutions(Xs, D):
    """Construct nav solutions analytically by reading entity positions from frame 0.

    For each task, builds nav_unfold(place_wall*(place_ag(blank,ar,ac,gr,gc), ...), T)
    directly from the task grid, bypassing enumeration.
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

    d_nav  = find_d('nav_unfold')
    d_ag   = find_d('place_ag')
    d_wall = find_d('place_wall')
    d_blank = find_d('blank')

    sols = {}
    for x in Xs:
        frame = x[0]
        T = x.shape[0]
        agent = [(r,c) for r in range(frame.shape[0]) for c in range(frame.shape[1]) if frame[r,c] == 1]
        goal  = [(r,c) for r in range(frame.shape[0]) for c in range(frame.shape[1]) if frame[r,c] == 2]
        walls = [(r,c) for r in range(frame.shape[0]) for c in range(frame.shape[1]) if frame[r,c] == 3]

        if not agent or not goal or int_d(T) is None:
            continue
        ar, ac = agent[0];  gr, gc = goal[0]

        ag_node = deepcopy(d_ag)
        ag_node.tails = [deepcopy(d_blank), int_d(ar), int_d(ac), int_d(gr), int_d(gc)]
        grid = ag_node

        for wr, wc in walls:
            if int_d(wr) is None or int_d(wc) is None or d_wall is None:
                grid = None; break
            pw = deepcopy(d_wall)
            pw.tails = [grid, int_d(wr), int_d(wc)]
            grid = pw

        if grid is None:
            continue

        root = deepcopy(d_nav)
        root.tails = [grid, int_d(T)]

        try:
            result = root()
            if np.array_equal(result, x):
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
        "return a TreeQ (or flat uniform tensor) for tree-conditioned enumeration"
        if Qmodel is None:
            return uniform_type_q()
        with th.no_grad():
            mat_h, _ = Qmodel.encode_matrix(tc_mat(x)[None])
        return TreeQ(Qmodel, mat_h)

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

    # Print all solutions rewritten with the newest abstractions
    if rewritten_strs:
        print("\n--- solutions rewritten with newest abstractions ---")
        for s in rewritten_strs:
            print(s)

    return {k: v for k, v in sols.items() if k in full_keys}

def mat_key(x):
    return (x.shape, x.tobytes())


class MatRecognitionModel(nn.Module):
    """Recognition model with symbolic grid encoder and top-down Tree GRU program encoder.

    Grid encoder (entity-based):
      Treats each non-zero cell as a (row, col, value) entity.
      h_entity = row_embed[r] + col_embed[c] + val_embed[v]
      h_matrix = mean over all entities across all frames.
      No CNN — the grid content is already symbolic.

    Program encoder (top-down Tree GRU):
      h_node = tree_rnn(prog_embed[d] + slot_embed[k], h_parent)
      where k is the child-slot index this node fills, h_parent is the
      parent node's hidden state (zeros at the root).

    Prediction:
      dsl_logits = head(concat(h_matrix, h_parent))
    """
    def __init__(self, nd, max_coord=16, vocabsize=10, nembd=64, max_slots=8):
        super().__init__()
        self.nembd = nembd
        self.nd = nd
        self.vocabsize = vocabsize

        # symbolic grid encoder
        self.row_embed = nn.Embedding(max_coord, nembd)
        self.col_embed = nn.Embedding(max_coord, nembd)
        self.val_embed = nn.Embedding(vocabsize, nembd)

        # top-down tree GRU
        self.prog_embed = nn.Embedding(nd, nembd)
        self.slot_embed = nn.Embedding(max_slots, nembd)
        self.tree_rnn = nn.GRUCell(nembd, nembd)

        # DSL logits from concat(h_matrix, h_parent)
        self.head = nn.Linear(2 * nembd, nd)

        print(f'{sum(p.numel() for p in self.parameters()) / 2**20:.2f}M params')

    def encode_matrix(self, x):
        """(B, T, H, W) long tensor -> (B, nembd) grid embedding.

        Each non-zero cell contributes row_embed[r] + col_embed[c] + val_embed[v];
        these are mean-pooled across all cells and frames.
        Returns (h_matrix, []) — empty list for API compat with dream.
        """
        B, T, H, W = x.shape
        device = x.device

        r_idx = th.arange(H, device=device)[None, None, :, None].expand(B, T, H, W)
        c_idx = th.arange(W, device=device)[None, None, None, :].expand(B, T, H, W)

        e = (self.row_embed(r_idx) +
             self.col_embed(c_idx) +
             self.val_embed(x.clamp(0, self.vocabsize - 1)))   # (B, T, H, W, nembd)

        mask = (x != 0).float().unsqueeze(-1)                  # (B, T, H, W, 1)
        h_matrix = (e * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1)
        return h_matrix, []

    def node_state(self, node_idx, slot, h_parent):
        """Compute child hidden state from parent node primitive and slot index.

        node_idx: (B,) long  — DSL index of the parent node
        slot:     (B,) long  — which child-slot of the parent we're filling
        h_parent: (B, nembd) — parent's hidden state (zeros at root)
        Returns:  (B, nembd)
        """
        e = self.prog_embed(node_idx) + self.slot_embed(slot.clamp(max=self.slot_embed.num_embeddings - 1))
        return self.tree_rnn(e, h_parent)

    def forward(self, x, h_parent=None):
        "(B,T,H,W), (B,nembd)|None -> dsl_logits (B,nd), []"
        matrix_h, _ = self.encode_matrix(x)
        B = x.shape[0]
        if h_parent is None:
            h_parent = th.zeros(B, self.nembd, device=x.device)
        dsl_logits = self.head(th.cat([matrix_h, h_parent], dim=-1))
        return dsl_logits, []

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

    tbar = trange(100)
    for _ in tbar:
        trees = [newtree(D, mat, paths, paths_terminal, depth=10) for _ in range(4)]
        if len(soltrees) > 0:
            for i in randint(len(soltrees), size=4):
                trees.append(soltrees[i])

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

            x = tc_mat(out)[None].to(device)          # (1, T, H, W)
            matrix_h, _ = qmodel.encode_matrix(x)     # (1, nembd)

            # top-down DFS: predict each node given (matrix_h, h_parent),
            # then compute h_child = tree_rnn(embed(d) + slot(k), h_parent)
            # for each child slot k before recursing into it.
            h0 = th.zeros(1, qmodel.nembd, device=device)
            accum = [node_loss, n_nodes]   # mutable refs threaded through visit

            def visit(node, h_parent, slot):
                idx = D.index(node)
                if idx is not None:
                    dsl_logits = qmodel.head(th.cat([matrix_h, h_parent], dim=-1))
                    accum[0] = accum[0] + F.cross_entropy(
                        dsl_logits, tensor([idx], device=device)
                    )
                    accum[1] += 1

                if node.tails:
                    if idx is not None:
                        # compute this node's own hidden state from its primitive
                        # and the slot it fills in *its* parent, then use it as
                        # h_parent for each child
                        idx_t  = tensor([idx],  device=device)
                        slot_t = tensor([slot], device=device)
                        h_node = qmodel.node_state(idx_t, slot_t, h_parent)
                    else:
                        h_node = h_parent
                    for k, child in enumerate(node.tails):
                        visit(child, h_node, k)

            visit(tree, h0, 0)
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


def make_false_belief_task(size=6, n_phantoms=1, seed=None, min_dist=None):
    """Generate a false belief navigation task.

    The agent navigates optimally on their BELIEVED grid (which has phantom
    walls not present in the actual world).  The observed trajectory X shows
    only agent (1) and goal (2) — phantom walls are hidden — so the path
    appears suboptimal on the blank actual grid.

    Solution program: hide_walls(nav_unfold(believed_grid, T))

    Bootstrap fails: reading frame 0 finds no walls, so the bootstrapped
    0-wall solution produces the optimal path, which doesn't match X.
    ECD must search over phantom wall positions to explain the detour.

    Returns (T, size, size) int array, or None if no valid layout found.
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

        return x

    return None


def make_false_belief_tasks(n=6, size=6, n_phantoms=1, seed=0):
    "generate n false belief tasks, retrying on failures"
    rng = np.random.default_rng(seed)
    tasks = []
    attempt = 0
    while len(tasks) < n:
        t = make_false_belief_task(size=size, n_phantoms=n_phantoms,
                                   seed=int(rng.integers(1<<31)))
        attempt += 1
        if t is not None:
            tasks.append(t)
    print(f"generated {n} false-belief tasks ({attempt} attempts, {size}x{size}, "
          f"{n_phantoms} phantom wall(s))")
    return tasks

###################################################################################

if __name__ == '__main__':
    """create an instance of the Deltas class, named D
    a Deltas object D is an instance of a DSL
    it's initiated with core primitives
    """
    D = Deltas([
        # mat construction
        Delta(hide_walls,      mat,  [mat],                    repr='hide_walls'),
        Delta(nav_unfold,      mat,  [grid, int],              repr='nav_unfold'),
        Delta(unfold,          mat,  [grid, int, fn],          repr='unfold'),
        # grid primitives
        Delta(blank66,         grid,                            repr='blank'),
        Delta(gset,            grid, [grid, int, int, int],    repr='gset'),
        Delta(place_agent_goal,grid, [grid, int, int, int, int], repr='place_ag'),
        Delta(place_wall,      grid, [grid, int, int],         repr='place_wall'),
        # intentional motion
        Delta(approach,        fn,   [int, int],               repr='approach'),
        Delta(approach(1, 2),  fn,                             repr='navigate'),
        # int terminals
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

    Xs = (make_nav_tasks(n=6, size=6, n_walls=0, seed=0) +
          make_nav_tasks(n=6, size=6, n_walls=1, seed=1) +
          make_nav_tasks(n=6, size=6, n_walls=2, seed=2) +
          make_false_belief_tasks(n=6, size=6, n_phantoms=1, seed=4))
    print(f"{len(Xs)} tasks, shapes: {[x.shape for x in Xs]}")

    seeds = bootstrap_nav_solutions(Xs, D)
    print(f"bootstrapped {len(seeds)}/{len(Xs)} solutions")

    Z = ECD(Xs, D, per_task_timeout=60, max_iterations=10, seeds=seeds)
    for k, v in Z.items():
        if v is not None:
            print(f'solution: {v}')
            print(f'evaluates to:\n{v()}')
