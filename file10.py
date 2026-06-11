"""file10.py: hierarchical Bayesian model selection over DSL hypothesis classes.

Adds a level-2 model above ECD. Instead of the user hand-selecting which DSL
to run (file3=desire, file6=false-belief, file9=joint), the system maintains
a posterior over competing DSL hypotheses and updates it with each round of
enumeration.

Level-2 model: HierarchicalBeliefs

For each task key it accumulates log-likelihoods per class. The MDL approximation is log p(x | H_k) ≈ -cost_k where cost_k = -D_k.logp(Q_k, sol). When no solution is found, it charges a flat 80-nat penalty. The posterior is log p(H_k | X) ∝ log_prior_k + Σ_i log_lik_k(x_i). shared_agent=True aggregates evidence across tasks (same agent type produced all of them); shared_agent=False gives each task its own posterior — useful for the mixed-corpus demo.

The gating logic in ECD_hierarchical

The core loop is: enumerate under each D_k → update beliefs → pick the MAP class → run stitch + dream only on the MAP class's solutions. This is the key hierarchical coupling: the level-2 posterior gates which primitives get compressed and which Q model gets refined. Minority-class Q models are frozen — they can still beat the MAP class in future rounds if evidence shifts.

Three things still rough in this sketch:

1. _model_q assumes the stored Q was from a previous round with the same D size. The _flat_q hack on the dream model is ugly — cleaner would be a wrapper that stores (model, D_snapshot) and reindexes properly when D grows via stitch.
2. The cross-class primitive propagation (invented terms from MAP class → simpler classes) is a placeholder with a rough type filter. The right semantics is: if stitch discovers fn_desire($ig, $gv) under H_desire, H_nav shouldn't inherit it — but a future H_nav+bonus run might usefully see it as a fixed terminal.
3. dream isn't wired for scene_model root type (only for mat/fn), so the belief class's Q stays uniform. This is probably fine initially — the belief solutions are structurally distinctive enough that uniform Q still wins on cost.

─────────────────────────────────────────────────────────────────────────────
Architecture:

  Level 1 (ECD, unchanged per class):
    hypothesis = a program p in DSL D_k
    prior      = Q_k (recognition model, updated each iteration by dream)
    evidence   = does p simulate to the observed trajectory x?

  Level 2 (new):
    hypothesis = which DSL class H_k is operative
    prior      = log_class_prior  (bias toward simpler theories)
    evidence   = p(x | H_k) ≈ exp(-cost_k(x))   [MDL approximation]
                 cost_k(x) = -D_k.logp(Q_k, best_sol_under_H_k)
    posterior  = log p(H_k | X) ∝ log p(H_k) + Σ_i log p(x_i | H_k)

The posterior tells you which theory of mind is needed to explain behavior,
and gates stitch (compression) and dream (Q training) toward the winning class.

─────────────────────────────────────────────────────────────────────────────
Hypothesis classes:

  H_nav    :: root_type=mat   — agent navigates by BFS (unfold + approach)
  H_desire :: root_type=mat   — agent has a goal value (gset + optimize + neg_dist)
  H_belief :: root_type=scene_model — agent has a goal AND a false belief
                                       (mk_agent_scene + set_at + seek)

Simpler is preferred a priori; richer hypotheses are selected only when their
evidence (lower solution cost) outweighs the prior penalty.

─────────────────────────────────────────────────────────────────────────────
Run:
  python file10.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any
import math
import numpy as np
import torch as th
import torch.nn.functional as F

from ecd import (
    Deltas, Delta, solve_enumeration, saturate_stitch, dream,
    task_terminals, normalize, mat_key, tc_mat,
)
from dsl import (
    mat, grid, fn, util, scene_model,
    unfold_auto, approach,
    gset, optimize, neg_distance,
    set_at, id_fn, seek,
    mk_agent_scene, empty_scene,
)
from tasks import (
    make_nav_tasks,
    make_desire_tasks,
    make_false_belief_desire_tasks,
)


# ── DSL hypothesis classes ─────────────────────────────────────────────────────

@dataclass
class DSLHypothesis:
    name:       str
    primitives: list       # core Delta list (no ig_i; those are added per-task)
    root_type:  Any        # mat | scene_model — controls simulation mode
    agents:     list       # [(av, gv), ...] for multi-agent sim; [] for mat
    ig_mode:    str        # 'full' | 'agent' | None — task terminal mode
    log_prior:  float      # log p(H_k); simpler classes get less negative values


_INT_DELTAS = [
    Delta(i, int, repr=str(i)) for i in range(6)
]

def _make_hypotheses() -> list[DSLHypothesis]:
    nav_prims = [
        Delta(unfold_auto, mat, [grid, fn],  repr='unfold'),
        Delta(approach,    fn,  [int, int],  repr='approach'),
    ] + _INT_DELTAS

    desire_prims = nav_prims + [
        Delta(gset,         grid, [grid, int, int, int], repr='gset'),
        Delta(optimize,     fn,   [util, int],           repr='optimize'),
        Delta(neg_distance, util, [int],                 repr='neg_dist'),
    ]

    belief_prims = [
        Delta(mk_agent_scene, scene_model,
              [int, fn, fn, scene_model], repr='mk_agent_scene'),
        Delta(empty_scene, scene_model,   repr='empty_scene'),
        Delta(set_at, fn, [int, int, int], repr='set_at'),
        Delta(id_fn,  fn,                  repr='id_fn'),
        Delta(seek,   fn, [int, int],      repr='seek'),
    ] + _INT_DELTAS

    return [
        DSLHypothesis('nav',    nav_prims,    mat,          [],           'full',  log_prior=-1.0),
        DSLHypothesis('desire', desire_prims, mat,          [],           'agent', log_prior=-2.0),
        DSLHypothesis('belief', belief_prims, scene_model,  [(1, None), (4, None)], None, log_prior=-3.0),
    ]


# ── Level-2 beliefs ────────────────────────────────────────────────────────────

class HierarchicalBeliefs:
    """Per-class log-posterior over DSL hypotheses.

    shared_agent=True: all tasks come from the same agent type, so evidence
    aggregates across tasks: log p(H_k | X) ∝ log p(H_k) + Σ_i log p(x_i | H_k).

    shared_agent=False: each task maintains its own posterior independently.
    Use this when a mixed batch may contain agents of different types.
    """
    NO_SOLUTION_COST = 80.0   # nats penalty when no solution found under H_k

    def __init__(self, hypotheses: list[DSLHypothesis], shared_agent: bool = True):
        self.hypotheses = hypotheses
        self.shared_agent = shared_agent
        # Accumulated log-likelihoods (separate from prior so we can recompute posterior)
        self._task_loglik: dict[tuple, dict[str, float]] = {}
        self._shared_loglik: dict[str, float] = {h.name: 0.0 for h in hypotheses}

    def _ensure(self, key):
        if key not in self._task_loglik:
            self._task_loglik[key] = {h.name: 0.0 for h in self.hypotheses}

    def update(self, task_key: tuple, class_name: str, cost: float | None):
        """Record evidence from one (task, class) enumeration round.

        cost = -D_k.logp(Q_k, sol) in nats.  Pass None when no solution found.
        """
        self._ensure(task_key)
        log_lik = -(cost if cost is not None else self.NO_SOLUTION_COST)
        self._task_loglik[task_key][class_name] += log_lik
        if self.shared_agent:
            self._shared_loglik[class_name] += log_lik

    def log_posterior(self, task_key: tuple | None = None) -> dict[str, float]:
        """Normalized log-posterior over hypothesis classes."""
        if self.shared_agent or task_key is None:
            logliks = self._shared_loglik
        else:
            self._ensure(task_key)
            logliks = self._task_loglik[task_key]

        raw = {h.name: h.log_prior + logliks[h.name] for h in self.hypotheses}
        lse = float(np.logaddexp.reduce(list(raw.values())))
        return {k: v - lse for k, v in raw.items()}

    def map_class(self, task_key: tuple | None = None) -> str:
        lp = self.log_posterior(task_key)
        return max(lp, key=lp.get)

    def report(self, iteration: int):
        lp = self.log_posterior()
        print(f"\n── Level-2 posterior (iteration {iteration}) ─────────────")
        for name, logp in sorted(lp.items(), key=lambda kv: -kv[1]):
            prob = math.exp(logp)
            bar  = '█' * int(prob * 30)
            print(f"  {name:10s}  p={prob:.3f}  {bar}")
        print(f"  MAP: {self.map_class()}")
        print("─────────────────────────────────────────────────────────\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _uniform_q(D: Deltas) -> th.Tensor:
    q = th.zeros(len(D))
    for tp, indices in D.bytype.items():
        lp = -math.log(len(indices))
        for i in indices:
            q[i] = lp
    return q

def _build_D(h: DSLHypothesis, Xs: list, Qmodel) -> tuple[Deltas, th.Tensor]:
    """Construct a fresh Deltas for hypothesis h, add task terminals, return (D, Q)."""
    D = Deltas(h.primitives)

    # mat-mode hypotheses need per-task ig_i grid terminals
    if h.root_type == mat and h.ig_mode is not None:
        igs = task_terminals(Xs, mode=h.ig_mode)
        for ig in igs:
            D.add(ig)

    # scene_model-mode: initial state extracted via ig_map inside solve_enumeration;
    # no explicit terminal needed here.

    Q = _uniform_q(D) if Qmodel is None else _model_q(Qmodel, D)
    return D, Q

def _model_q(model, D: Deltas) -> th.Tensor:
    """Resize stored Q to match current D length (D may have grown with igs/invented)."""
    stored = model._flat_q
    if stored.shape[0] == len(D):
        return stored.clone()
    q = _uniform_q(D)
    q[:stored.shape[0]] = stored
    return q

def _solution_cost(D: Deltas, Q: th.Tensor, sol) -> float:
    """MDL cost of a solution: -log p(program | D, Q)."""
    return -float(D.logp(Q, sol))


# ── ECD_hierarchical ──────────────────────────────────────────────────────────

def ECD_hierarchical(
    Xs,
    hypotheses:        list[DSLHypothesis] | None = None,
    shared_agent:      bool  = True,
    per_class_timeout: float = 30.0,
    max_iterations:    int   = 8,
    max_arity:         int   = 5,
):
    """ECD with a Bayesian level-2 model over DSL hypothesis classes.

    Each iteration, for every unsolved task x:
      1. Enumerate under each D_k for per_class_timeout seconds.
      2. Compute solution cost = -D_k.logp(Q_k, sol); update level-2 posterior.
      3. The MAP class determines which D_k drives stitch + dream this round.
      4. The next-iteration Q_k is trained only on solutions attributed to class k.

    Returns:
      solutions : {mat_key: program tree}  — best sol found under any class
      beliefs   : HierarchicalBeliefs      — final level-2 posterior
      Qmodels   : {class_name: flat Q}     — per-class recognition snapshots
    """
    if hypotheses is None:
        hypotheses = _make_hypotheses()

    beliefs = HierarchicalBeliefs(hypotheses, shared_agent=shared_agent)

    # Per-class state
    Qmodels   = {h.name: None for h in hypotheses}  # None → use uniform Q
    class_sols = {h.name: {} for h in hypotheses}   # {class: {mat_key: sol}}
    all_sols  = {}                                   # {mat_key: (sol, class, cost)}

    for iteration in range(max_iterations):
        unsolved = [x for x in Xs if mat_key(x) not in all_sols]
        if not unsolved:
            break

        print(f"\n{'='*60}")
        print(f"HierarchicalECD  iteration={iteration}  unsolved={len(unsolved)}/{len(Xs)}")
        print(f"{'='*60}")

        # ── Per-class enumeration ──────────────────────────────────────────────
        for h in hypotheses:
            D_k, Q_k = _build_D(h, unsolved, Qmodels[h.name])

            sols_k = dict(class_sols[h.name])  # carry over prior solutions
            sols_k = solve_enumeration(
                unsolved, D_k, Q_k, sols_k,
                maxdepth=10,
                timeout=per_class_timeout,
                root_type=h.root_type,
                agents=h.agents or None,
            )
            class_sols[h.name] = sols_k

            # Update level-2 beliefs with MDL evidence
            for x in unsolved:
                xkey = mat_key(x)
                sol  = sols_k.get(xkey)
                cost = _solution_cost(D_k, Q_k, sol) if sol is not None else None
                beliefs.update(xkey, h.name, cost)

                # Track the cheapest solution found across all classes
                if sol is not None:
                    prev = all_sols.get(xkey)
                    if prev is None or cost < prev[2]:
                        all_sols[xkey] = (sol, h.name, cost)

            n_k = sum(1 for x in Xs if mat_key(x) in sols_k and sols_k[mat_key(x)])
            print(f"  [{h.name:8s}]  solved: {n_k}/{len(Xs)}")

        beliefs.report(iteration)

        # ── Level-2 gating: stitch + dream only for the MAP class ─────────────
        # The posterior is the signal. Stitch compresses the MAP class's solutions
        # into reusable abstractions; dream trains a recognition model Q for that
        # class. Minority-class Qs are kept frozen (they can still win future rounds
        # if evidence shifts).
        map_name = beliefs.map_class()
        map_h    = next(h for h in hypotheses if h.name == map_name)
        D_map, Q_map = _build_D(map_h, Xs, Qmodels[map_name])

        map_k_sols = class_sols[map_name]
        soltrees = [s for s in map_k_sols.values() if s is not None]

        if soltrees:
            print(f"  Compressing {len(soltrees)} solutions under MAP class '{map_name}'…")
            trees, _ = saturate_stitch(D_map, map_k_sols, iterations=2, max_arity=max_arity)

            if map_h.root_type == mat:
                # dream works for mat root type; store Q as a flat tensor snapshot
                trained = dream(D_map, trees)
                # Snapshot: encode the flat Q used for cost computation next round.
                # The model outputs logits; store a dummy forward pass for a blank mat.
                blank_x = tc_mat(np.zeros((2, 4, 4), dtype=int))[None]
                with th.no_grad():
                    logits = trained(blank_x)
                q_snapshot = F.log_softmax(logits.squeeze(0), dim=-1).detach()
                trained._flat_q = q_snapshot
                Qmodels[map_name] = trained
            else:
                # dream not yet wired for scene_model root; keep uniform Q
                print(f"  (dream skipped for root_type={map_h.root_type})")

        # Also promote invented primitives from MAP class into minority class DSLs
        # so that future minority-class enumeration can reuse discovered abstractions.
        # This is the cross-class knowledge transfer step: stitch discoveries made
        # at the MAP class level propagate down to simpler hypotheses as new terminals.
        if D_map.invented:
            print(f"  Propagating {len(D_map.invented)} invention(s) to other classes…")
            for h in hypotheses:
                if h.name != map_name:
                    # Add only type-compatible invented primitives
                    extra = [deepcopy(d) for d in D_map.invented
                             if d.type in {tp for tp in h.primitives[0:1]}]  # rough filter
                    h.primitives = h.primitives + extra

        n_solved = len(all_sols)
        att = {h.name: sum(1 for _,cls,_ in all_sols.values() if cls == h.name)
               for h in hypotheses}
        print(f"  Total solved: {n_solved}/{len(Xs)}  attribution: {att}")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n\n══ Final level-2 posterior ═══════════════════════════════════")
    beliefs.report(iteration=max_iterations)

    print("══ Solution attribution (per class) ══════════════════════════")
    for h in hypotheses:
        h_keys = [k for k, (_,cls,_) in all_sols.items() if cls == h.name]
        prob   = math.exp(beliefs.log_posterior()[h.name])
        print(f"  {h.name:10s}  p={prob:.3f}  {len(h_keys)} tasks attributed")

    final_sols = {k: tree for k, (tree, _, _) in all_sols.items()}
    return final_sols, beliefs, Qmodels


# ── Tasks ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Mix of task types to give the level-2 model a genuine inference problem.
    # A batch from a pure-nav agent should drive p(H_nav) → 1;
    # a batch with false beliefs should drive p(H_belief) → 1.
    # Here we mix all three types so the posterior must aggregate evidence.
    print("Generating tasks…")

    Xs_nav = make_nav_tasks(n=6, size=4, n_walls=0, seed=1)

    raw_des = make_desire_tasks(n_per_goal=4, goal_vals=(2, 4), size=4, seed=0)
    Xs_des  = [x for x, _ in raw_des]

    raw_bel = make_false_belief_desire_tasks(
        n_per_combo=4, agent_goal_combos=[(1, 2), (4, 5)], size=5, seed=0)
    Xs_bel  = [x for x, _ in raw_bel]

    # For a clean level-2 inference demo: use only one type at a time and check
    # which hypothesis wins. Below we run the mixed corpus.
    Xs = Xs_nav + Xs_des + Xs_bel

    print(f"  {len(Xs_nav)} nav tasks")
    print(f"  {len(Xs_des)} desire tasks")
    print(f"  {len(Xs_bel)} false-belief tasks")
    print(f"  {len(Xs)} total\n")

    hypotheses = _make_hypotheses()

    solutions, beliefs, Qmodels = ECD_hierarchical(
        Xs,
        hypotheses=hypotheses,
        shared_agent=False,   # per-task beliefs since tasks come from different agent types
        per_class_timeout=20,
        max_iterations=6,
        max_arity=5,
    )

    print(f"\nSolved: {len(solutions)}/{len(Xs)}")

    # ── Show per-task posteriors ───────────────────────────────────────────────
    # With shared_agent=False each task has its own posterior.
    # This is the level-2 "classification" output: for each observed trajectory,
    # what theory of mind best explains it?
    print("\n── Per-task MAP class ────────────────────────────────────────")
    for label, Xs_type in [('nav', Xs_nav), ('desire', Xs_des), ('belief', Xs_bel)]:
        correct = wrong = unsolved = 0
        for x in Xs_type:
            k = mat_key(x)
            if k not in solutions:
                unsolved += 1
                continue
            map_c = beliefs.map_class(k)
            if map_c == label:
                correct += 1
            else:
                wrong += 1
        print(f"  {label:10s}  correct={correct}  wrong={wrong}  unsolved={unsolved}")
