import numpy as np
from numpy.random import rand, randint
from numpy import array
from collections import Counter, defaultdict
from typing import Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, as_tensor, from_numpy

from itertools import chain

import pickle
from tqdm import trange
from copy import deepcopy
from time import time

from dsl import *

class Deltas:
    # a collection of Delta primitives forming the DSL
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

def cenumerate(D, Q, tp, budget, maxdepth, cb, deadline=None):
    """ enumerate programs by probability
    callback-style, budget window used in
    the main solve_enumeration loop with expanding windows"""
    if budget[1] <= 0 or maxdepth < 0:
        return True

    if deadline is not None and time() > deadline:
        raise _EnumDone()

    for i in D.bytype[tp]:
        if -Q[i] > budget[1]:
            continue

        d = D.ds[i]
        logp = Q[i]
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
        return [normalize(s) for s in sols.values() if s]

    ghosttime = time()
    trees = [normalize(s) for s in sols.values() if s]
    D.reset()

    if not trees:
        return trees

    print(f"size of the forest: {len(pickle.dumps(trees)) >> 10}M")

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
        return trees

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
    return new_trees if new_trees else trees


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


def ECD(Xs, D, timeout=60, budget=0, max_iterations=10):
    # when ECD is first run, reset the DSL
    D.reset()

    Qmodel = None  # no model on the first iteration; use uniform Q
    idx = 0
    sols = {}

    def all_solved():
        return all(mat_key(x) in sols for x in Xs)

    def task_Q(x):
        "return a task-specific log-prob vector, or uniform if no model yet"
        if Qmodel is None:
            return F.log_softmax(th.ones(len(D)), -1)
        return F.log_softmax(Qmodel(tc_mat(x)[None])[0].flatten().detach(), -1)

    while idx < max_iterations:
        unsolved = [x for x in Xs if mat_key(x) not in sols]
        # allocate timeout evenly across unsolved tasks
        per_task_timeout = timeout / len(unsolved)

        for x in unsolved:
            if mat_key(x) in sols:
                continue  # may have been solved by an earlier task in this round
            sols = solve_enumeration([x], D, task_Q(x), sols,
                                     maxdepth=10, timeout=per_task_timeout, budget=budget)

        soltrees = [s for s in sols.values() if s is not None]
        if len(soltrees) > 0:
            trees = saturate_stitch(D, sols)
        else:
            trees = []

        idx += 1

        Qmodel = dream(D, trees)

        if all_solved():
            break

        unsolved = [x for x in Xs if mat_key(x) not in sols]
        print(f'--- ECD iteration {idx}, {len(unsolved)}/{len(Xs)} unsolved, Q task-specific ---', flush=True)

    full_keys = {mat_key(x) for x in Xs}
    return {k: v for k, v in sols.items() if k in full_keys}

def mat_key(x):
    return (x.shape, x.tobytes())

def mat_eq(a, b):
    return a.shape == b.shape and np.array_equal(a, b)

def mat_type(X):
    return mat


class MatRecognitionModel(nn.Module):
    """Recognition model conditioned on both the task matrix and a partial program tree.

    Matrix encoder:
      1. encode frame t with a 2d cnn
      2. update a recurrent hidden state
      3. predict frame t+1 from hidden state (auxiliary loss)
      4. pool to a single matrix embedding

    Program encoder:
      - embed each already-placed primitive by its DSL index
      - mean-pool to a single program embedding

    Output:
      - concat(matrix_embedding, program_embedding) -> DSL logits

    forward(x, prog_ctx=None)
      x:        (B, T, H, W) int tensor  — the task matrix
      prog_ctx: (B, K) int tensor        — indices of primitives already placed
                                           (None or K=0 means empty partial program)
    returns (dsl_logits, frame_pred_logits)
    """
    def __init__(self, nd, vocabsize=10, nembd=64):
        super().__init__()
        self.nembd = nembd
        self.vocabsize = vocabsize

        # matrix encoder
        self.embed = nn.Embedding(vocabsize, nembd)
        self.encoder = nn.Sequential(
            nn.Conv2d(nembd, nembd, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(nembd, nembd, 3, padding=1),
            nn.GELU(),
        )
        self.rnn = nn.GRUCell(nembd, nembd)
        self.decoder = nn.Sequential(
            nn.Conv2d(nembd, nembd, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(nembd, vocabsize, 1),
        )

        # program encoder: embed each placed primitive, then mean-pool
        self.prog_embed = nn.Embedding(nd, nembd)

        # DSL logits from concat(matrix_hidden, program_hidden)
        self.head = nn.Linear(2 * nembd, nd)

        print(f'{sum(p.numel() for p in self.parameters()) / 2**20:.2f}M params')

    def encode_frame(self, frame):
        "frame: (B, H, W) int tensor -> (B, nembd, H, W)"
        emb = self.embed(frame)            # (B, H, W, nembd)
        emb = emb.permute(0, 3, 1, 2)     # (B, nembd, H, W)
        _, _, h, w = emb.shape
        pad = [0, max(0, 3 - w), 0, max(0, 3 - h)]
        if any(p > 0 for p in pad):
            emb = F.pad(emb, pad)
        return emb + self.encoder(emb)     # residual

    def forward(self, x, prog_ctx=None):
        # x: (B, T, H, W) int tensor
        B, T, H, W = x.shape
        h = th.zeros(B, self.nembd, device=x.device)

        frame_preds = []
        for t in range(T):
            enc = self.encode_frame(x[:, t])      # (B, nembd, H', W')
            pooled = enc.mean(dim=(2, 3))          # (B, nembd)
            h = self.rnn(pooled, h)

            if t < T - 1:
                h_spatial = h[:, :, None, None].expand(-1, -1, H, W)
                pred = self.decoder(h_spatial)     # (B, vocabsize, H, W)
                frame_preds.append(pred)

        # encode partial program: mean-pool primitive embeddings
        if prog_ctx is not None and prog_ctx.shape[1] > 0:
            prog_h = self.prog_embed(prog_ctx).mean(dim=1)  # (B, nembd)
        else:
            prog_h = th.zeros(B, self.nembd, device=x.device)

        dsl_logits = self.head(th.cat([h, prog_h], dim=-1))  # (B, nd)
        return dsl_logits, frame_preds

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

        # Build autoregressive training examples.
        # For each node at DFS position i in a tree, the partial-program context
        # is the indices of all nodes placed before it (positions 0..i-1).
        # Training target: predict node i given (matrix, context).
        Xy = []  # list of (matrix_tensor, target_idx, ctx_indices_list)
        for tree in trees:
            try:
                out = tree()
            except Exception:
                continue

            if not isinstance(out, np.ndarray) or 0 in out.shape or out.shape[0] < 2:
                continue

            xtc = tc_mat(out)
            nodes = alld(tree)
            node_indices = [D.index(d) for d in nodes]

            for i, target_idx in enumerate(node_indices):
                if target_idx is None:
                    continue
                ctx = [j for j in node_indices[:i] if j is not None]
                Xy.append((xtc, target_idx, ctx))

        if len(Xy) == 0:
            continue

        opt.zero_grad()

        dsl_loss = tensor(0.0, device=device)
        frame_loss = tensor(0.0, device=device)
        nframes = 0

        for xtc, target_idx, ctx in Xy:
            x = xtc[None].to(device)                          # (1, T, H, W)

            if ctx:
                prog_ctx = tensor([ctx], device=device)       # (1, K)
            else:
                prog_ctx = None

            dsl_logits, frame_preds = qmodel(x, prog_ctx)
            dsl_loss = dsl_loss + F.cross_entropy(dsl_logits, tensor([target_idx], device=device))

            for t, pred in enumerate(frame_preds):
                target_frame = x[:, t + 1]
                frame_loss = frame_loss + F.cross_entropy(pred, target_frame)
                nframes += 1

        loss = dsl_loss / len(Xy)
        if nframes > 0:
            loss = loss + frame_loss / nframes

        loss.backward()
        opt.step()

        tbar.set_description(f'{loss=:.2f} batchsize={len(Xy)}')

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

def make_static_nav(size=6, n_walls=3, seed=None, min_dist=None):
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


def make_static_tasks(n=12, size=6, n_walls=3, seed=0):
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

###################################################################################

if __name__ == '__main__':
    """create an instance of the Deltas class, named D
    a Deltas object D is an instance of a DSL
    it's initiated with core primitives
    """
    D = Deltas([
        # mat construction
        Delta(unfold,    mat,       [grid, int, fn],                  repr='unfold'),
        Delta(singleton, mat,       [grid],                           repr='singleton'),
        # grid primitives
        # blank44 is a terminal (saves 2 int-holes vs zeros(4,4))
        Delta(blank44,   grid,                                         repr='blank'),
        #Delta(blank66,   grid,                                         repr='blank'),
        Delta(gset,      grid,      [grid, int, int, int],            repr='gset'),
        Delta(place_agent_goal, grid, [int, int, int, int],          repr='place_ag'),
        # fn constructors
        Delta(step_fn,   fn,        [int, direction],                 repr='step'),
        # direction terminals
        Delta(RIGHT,     direction,                                    repr='right'),
        Delta(LEFT,      direction,                                    repr='left'),
        Delta(UP,        direction,                                    repr='up'),
        Delta(DOWN,      direction,                                    repr='down'),
        # int terminals
        Delta(0,         int,                                          repr='0'),
        Delta(1,         int,                                          repr='1'),
        Delta(2,         int,                                          repr='2'),
        Delta(3,         int,                                          repr='3'),
        Delta(4,         int,                                          repr='4'),
        Delta(5,         int,                                          repr='5'),
        Delta(6,         int,                                          repr='6'),
    ])

    Xs = make_static_tasks(n=6, size=4, n_walls=0) + make_static_tasks(n=6, size=4, n_walls=1)


    #Xs = Xs + make_nav_tasks(n=12, size=6, n_walls=6, seed=0)
    print(f"{len(Xs)} tasks, shapes: {[x.shape for x in Xs]}")

    Z = ECD(Xs, D, timeout=600, max_iterations=10)
    for k, v in Z.items():
        if v is not None:
            print(f'solution: {v}')
            print(f'evaluates to:\n{v()}')
