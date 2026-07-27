#import "@preview/illc-mol-thesis:0.2.0": *

#show raw.where(lang: "python"): it => block(
  fill: rgb("#f6f8fa"),
  stroke: 0.5pt + rgb("#d0d7de"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #set text(font: "DejaVu Sans Mono", size: 9pt)
  #it
]

#set heading(numbering: "1.")
#mol-chapter("Appendix")

This appendix collects the implementation of the model outlined in Chapter 3.

- full list of tasks
- full list of primitives

== The interpreter <app-interpreter>

The interpreter is discussed at the model level in @app-interpreter. What follows is the design argument in the form it was originally worked out, followed by the two interpreters themselves.

Say we have a candidate program $rho$, so the enumerator needs to check for each task $x$ in the corpus whether $rho$ solves task $x$. A program $rho$ solves task $x$ iff running $rho$ on initial frame $x_0$ outputs the entire history $x$. So every solution should be a program that takes as input a $d times d$ frame and returns a $d times d times n$ stack of frames, a program of the form: "iteratively apply transition function $f$, starting with initial frame $x_0$, for $n$ timesteps". So every solution program has the same outer structure, with only $f$ varying between programs.

This presents a problem: this shared structure, present in every program, will be baked-in to every abstraction. For example say ten tasks in a corpus depict some object falling down, so stitch should identify `(step down #0)` as a shared structure and add this "gravity" primitive to the library. But because "apply $f$, starting with $x_0$, for $n$ timesteps" is a structure common to all programs, stitch would add an overspecified abstraction $f_1 := lambda$` 0 1`. _iteratively apply `(step down #0)`, starting with $x_0$, for `#1` timesteps_ rather than the more general $f_2 := lambda$ `0`. `(step down #0)`.

The problem is that $f_1$ doesn't compose, which defeats the purpose of library learning. Say we abstracted $f_1$ as "gravity", and we now want to use it to solve a task involving two objects both dropping to the ground. We can't do that, because $f_1$'s output is of type `matrix`, an entire 3d scene depicting just one falling object. But if we've got an $f_2$ abstraction with output type `fn`, we can create a program like `(compose (fn_2 4) (fn_2 5))` that reuses the abstraction for each component of a scene within that same scene (where `compose` is of type `fn -> fn -> fn`).

Since "iteratively apply transition function $f$, starting with initial frame $x_0$, for $n$ timesteps" is common to every program, what the enumerator is looking for is really just that transition function $f$. The rest is just what's needed to render a scene given a transition function, to check whether $f$ correctly produces a given task scene. So we simply move the rendering machinery from program space to the evaluation framework.

We implement an interpreter that, given a candidate program (transition function) $f$, a task's initial frame $x_0$, and the task's duration ($z$-dimension) $n$, generates a scene $x'$ by applying $f$ iteratively (starting with $x_0$) for $n$ timesteps.

```python
def unfold(g: grid, T: int, f: fn) -> mat:
    frames = [g.copy()]
    for _ in range(T - 1):
        g = f(g)
        frames.append(g.copy())
    return np.stack(frames)
```

`unfold` is the anamorphism that at each timestep applies $f$ to the previous grid to generate the new grid, and appends the new grid to the list of grids. So `unfold`'s state is just a single grid: it doesn't encode a world/model grid pair, nor a registry of agents and goals. A transition function $f$ is a `grid` endomorphism `fn :: grid -> grid`.

For some of the tasks, we use an interpreter for programs of root `fn_p_g :: (grid, grid) -> grid` since those tasks don't involve just one input grid but a pair. That uses `unfold_with_template(g, template, T, c)`

```python
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
```

== Program representation <app-delta>

The hypothesis space is described at the model level in !types. This is the data structure that represents it.

Every program, primitive, and invented abstraction is a `Delta`, i.e. an object that can serve as a node in an expression tree. A `Delta` carries a `head` (the Python value or callable it denotes), a `type` (the type its subtree returns), a list of `tailtypes` (the types of the arguments it expects), and a list of `tails` (the actual argument subtrees, once they are filled in). So a `Delta` with no `tailtypes` is a _terminal_, a leaf whose value is just its head, for example `up :: dir` or the literal `4 :: cellvalue`. A `Delta` with `tailtypes` is an _operator_: an interior node like `step :: cellvalue, dir -> fn`, whose head is applied to its filled-in tails. Then a node's `arrow` is the signature that the enumerator matches against when it fills argument slots. For a terminal, its arrow is just its `type`. For an operator, it's the pair `(tailtypes, type)`, the arrow from its input types to its output types.

The expression tree is directly executable. Calling a `Delta` evaluates the program it represents, so a terminal returns its head while an operator recursively evaluates each of its tails and applies its head to the results. So the program `(step 4 up)` is a `step`-headed node with tails `4` and `up`; calling it evaluates the two leaves to the cell value `4` and the direction `(-1,0)`, then applies `step` to obtain a transition function `fn :: grid -> grid`.

An invented abstraction is represented as a `Delta` as well, but it's distinguished by a `hiddentail` which is the body of the abstraction. The `hiddentail` is itself a `Delta` tree over lower-level primitives, with numbered argument holes `$0, $1, ...` standing in for the abstraction's parameters. When such a node is called, its body is copied and each hole `$i` is replaced by the corresponding tail before the body is evaluated. This is what makes an abstraction cost a single token. The whole subtree it abbreviates is hidden inside the `hiddentail` and does not appear in the program that uses it.

The description length of a program is its node count, computed by `length`. a leaf counts one, and an operator counts one plus the lengths of its tails. So an invented abstraction used as a token adds just one, since its body lives in the un-charged `hiddentail`. This node count is the minimum-description-length quantity the enumerator uses to tiebreak between programs that solve a task within the same budget window (see !enumeration).

=== The library object

The library of primitives is a `Deltas` object. It is instantiated with a list of `core` primitives and grows a parallel list of `invented` abstractions as compression discovers them. When the enumerator draws primitives, it draws from the concatenation of `core` and `invented` which is the full library. Whenever a primitive is added, `Deltas.infer` rebuilds the indices the search depends on — most importantly `bytype`, which groups the alphabet's symbols by their return type. This grouping is what the type-uniform prior is computed from: a symbol of type $tau$ is priced at $-ln$ of the number of symbols sharing type $tau$, so `bytype` gives both the branching factor at each argument slot and the cost of each choice (!enumeration).

== Task generators and necessity filters <app-generators>

The identifiability condition on the mental families is stated in !identifiability. It is enforced by the generators as follows.

one of the challenges of creating such a corpus is making tasks that require belief attribution.

checks in the (belief) task generators to make sure they aren't physically explainable, so they're not underdetermined. For example, in a wall false belief task an agent might have a false belief about a wall being there, but at a location nowhere near the agent's path to its goal. So the agent can still navigate to the goal freely as if it didn't have such a false belief, positing the false belief isn't necessary or helpful to explain the scene. So those scenes are rejected.

we implement a filter that

== Cost-bounded enumeration <app-enumeration>

The search is described at the model level in !enumeration. These are its mechanics.

While a single cost-ordered stream can in principle be checked against every task in the corpus at once, in the experiments each still-unsolved task is instead enumerated as its own cost-ordered stream, parallelized across tasks. This per-task granularity is what lets the prior $Q$ be conditioned on the individual scene rather than shared across the corpus — a fact we rely on in !dreaming, where $Q$ becomes a task-conditional distribution $Q(dot | x)$ emitted by a recognition model.

Program search proceeds by budget window to ensure that cheaper programs are enumerated first. For `(step 4 up)`, the first two windows are skipped by pruning. We start enumerating in window $[0,2)$, but since `step`'s cost is 2.1 nats it exceeds the budget, so it cannot be chosen as the root node of the program tree and we proceed to budget window $[2,4)$. After paying 2.1 nats for `step`, only 1.9 nats remain to fill the two holes in the partial program tree. `step`'s tail types (arguments) are `cellvalue` and `step`. But any primitive of type `cellvalue` costs 2.3 nats, so `step`'s first argument can't be placed. So trees with a `step` node are pruned, since they would never complete within the second budget window. Search is cheapest-first, since more complex programs only become affordable once the window is wide enough.

At budget window $[4,6)$ the ceiling is finally sufficiently wide to admit all three symbols. The tree is assembled incrementally: each enumerated subtree is appended to a copy of its parent node, and a callback fills the remaining argument slots by the same recursion (so programs of any arity or nesting depth are handled by the same procedure). Since enumeration is type-directed, each argument slot draws candidates only from symbols of the matching type, so no ill-typed partial tree is ever constructed. `(step 4 up)` completes at its total cost of $approx 5.8$ nats.

When `(step 4 up)` is found the tree rooted by `step` has no more available holes, so the callback fires to evaluate the found program $rho$. For each task $x$ in the corpus, the interpreter generates the output matrix given the input grid of the task by calling `unfold` on $rho$, initial grid $x_0$, and $x$'s third dimension $n$. Then the resulting output matrix $x'$ is hashed and compared against task $x$'s hash. So for a program to solve a task, it must produce a grid that exactly matches that task's entire trajectory, not just the endpoint of the trajectory. This ensures that programs capture the full behavior rather than just the end result.

If `(step 4 up)` is the first solution found for the task, it's stored as the task's solution. Because the windows are tried in increasing order of cost, the first solution the search returns is in principle the shortest program that solves the task. But enumeration is parallelized so search unfolds simultaneously over program trees, so we need to tiebreak when multiple programs solve a task within the same budget window. To tiebreak we first run `Delta.simplify` on the programs, a semantics-preserving rewrite that collapses spurious nesting which sometimes occurs during enumeration. We then run `Delta.length` to get the node count of the expression trees we need to compare. Every operator and leaf adds to the count, and an invented abstraction used as a token adds one to the count too. The body of the abstraction isn't charged, since the components now count for just application. The program with the lower node count wins since it has the lower minimum description length (since by symbol count it's shorter).

Also, the search space contains lots of junk programs that crash or return nothing even though they type-check. So the enumerator callback wraps both the interpreter and the evaluator in `try/except` blocks and skips any degenerate or non-callable program.

== Compression with Stitch <app-stitch>

The compression objective and the tradeoff it makes are given in !compression. This is the search that optimizes it.

In DreamCoder @ellis_dreamcoder_2020 the compression step is bottom-up: for each found program it enumerates the exponential set of program refactorings (stored as version spaces) and intersects them to grow abstractions up from reoccurring subexpressions. Rather than this bottom-up approach, we delegate the search for abstractions to the top-down Stitch @bowers_top-down_2023 library-learning system. Instead of enumerating refactorings of every program, stitch searches the space of abstractions directly. So stitch doesn't have to materialize every rewrite.

To create an abstraction, stitch starts with a tree that consists of just a root node `??`, an unexpanded hole that the search still needs to refine. At the root `??`, the abstraction _matches_ every subexpression, since it's the most general abstraction possible: any subexpression can be trivially "rewritten"

. As the tree is built up and the search is refined, the set of possible match locations shrinks.

A node of the stitch abstraction search tree is a partial abstraction. It consists of some combination of primitive operations (like `step`) and argument holes `#i`, and at least one unexpanded hole `??` that the search still needs to refine.

At each step it commits part of the abstraction's body to a concrete operator or splits a subtree, branching downward toward more specific abstractions.

upper bound for a partial abstraction $A_(??)$ @bowers_top-down_2023

$
  U_("upperbound")(A_(??)) = sum_(e in "matches"(A_(??))) "size"(e)
$

goal of compression is to minimize the size (cost) of program corpus after rewrite. Compressive utility function:

Each compression phase consists of a fixed number of stitch iterations, determined by the `iterations` parameter. Each iteration searches for the single most compressive abstraction, rewrites the corpus in terms of it, and repeats, so later abstractions may be built out of earlier ones. The `max_arity` parameter bounds how many holes (`#0`, `#1`, …) an abstraction may carry.

=== Normalization before compression

Before passing the found solutions to stitch for abstraction, we first need to fully expand any existing primitives in those solutions. For this we implement `normalize`, the complementary operation to abstraction. It fully inlines every invented `fn_k` back to the bare primitive alphabet, recursively substituting each abstraction's argument holes with its actual arguments until nothing but core primitives remain.

We need to normalize because stitch's top-down compression can only abstract over structure it can see, and an opaque `fn_k` token has no visible interior. So any abstraction that would need to reach inside `fn_k`, or span its boundary and surrounding context, is simply invisble to the search.

the compression step must be handed programs in a common vocabulary — feeding it already-compressed programs, in which a prior round's fn_0 appears as an opaque token, would let those tokens collide with the fresh abstractions the compressor invents under the same names. Normalizing before compression guarantees the compressor sees only primitives and the structure it discovers is genuinely its own. Renaming it to `inv_0` avoids the lexical clash but leaves it just as opaque; stitch still can't see the `(compose (wall_at …) (optimize …))` sitting inside it. Inlining flattens the token back to that primitive structure, and only then can the compressor find abstractions that cut across the old boundaries.

The best syntactic fragment to abstract from round $N$'s solution may carve the primitives differently than round $N+1$'s. For example,

So we can refactor abstractions that were previously ideal but are no longer ideal

== The recognition model <app-recognition>

Amortization is described at the model level in !dreaming. This is the network, its training data, and the rescaling that puts it on the enumerator's cost scale.

In the first enumeration run, primitives are chosen under type-uniform prior distribution $Q$.

The recognition model `MatRecognitionModel` is built around a symbolic grid encoder rather than a generic convolutional one. Its central design commitment is that entity _roles are read off from motion, not from cell values_: because the corpus deliberately uses diverse agent and goal identifiers, the encoder cannot key on "value 1 is the agent." Instead it labels a cell occupied in frame $x_0$ but vacated by the final frame a _mover_, and a non-background cell occupied at both ends a stationary _goal_. It then pools row and column position embeddings into per-entity slots: two mover slots (each carrying both a start-position pool and a trajectory pool), two goal slots, a wall pool, and a path-length embedding of the duration $T$, concatenated into a single matrix embedding $h_"matrix"$. The per-entity layout is what lets the two-mover, two-goal scenes of the witness and dual belief families be represented in separate slots rather than blurred into one mean, and the separate per-mover trajectory pool is the decisive signal for false belief: a detour path and a straight path yield different mean agent positions even when their start, goal, and $T$ coincide, which is exactly the case a belief task presents. A linear head maps $h_"matrix"$ to logits over the whole library $D$.

Crucially the prediction is _flat_: one forward pass per task yields a single distribution over the DSL, computed once and reused for every enumeration decision on that task. We deliberately forgo a tree-context recurrent network that would condition each node's prediction on its partial parent tree. A per-node network is both far more expensive — it turns each of the thousands of enumeration steps into its own forward pass — and prone to failing exactly where it matters: the abstractions belief requires are deep, and a tree-conditioned model trained on shallow solved programs generalizes poorly to the depths it has not yet seen. A flat, matrix-conditioned $Q$ sidesteps both problems.

*Replays and fantasies.* The model is trained by the wake–sleep procedure of a Helmholtz machine, on two sources of program–scene pairs. _Replays_ are the actual solutions enumeration has found so far, run back through the interpreter on their own task grids — supervised examples of "this scene was solved by this program." But replays alone are scarce and biased toward the families already solved, so they are supplemented with _fantasies_: programs sampled directly from the current library prior and executed by `unfold` on freshly generated grids to synthesize new (scene, program) pairs at no labelling cost. Each dreaming iteration draws up to four replays and fills the rest of a batch of eight with fantasies. The fantasy grids are not arbitrary — their size, entity values, and the fraction carrying four or more entities are all read off the corpus's own scene statistics, so the encoder trains on the scene mix it will face at test time (in particular the multi-entity scenes that exercise its second mover and goal slots). Roles remain decided by the sampled program's motion, so a fantasy only ever _lets_ a program drive two movers; it never manufactures a belief scene by hand. For every pair the model encodes the resulting matrix and is trained by cross-entropy to predict each node of the generating program tree, averaged over nodes.

*Putting the model on the enumeration cost scale.* A recognition model trained this way emits a globally-normalized distribution over the library, which is on the wrong scale for the budget-window enumerator and, having been trained only on the families solved so far, could actively suppress the rare primitives an as-yet-unsolved family needs. `dreamed_q` reconciles the two so that dreaming can only ever help. First it re-softmaxes the model's logits _within each type group_, so a program's summed cost is comparable to the uniform and content priors rather than living on the model's global scale — otherwise a ten-node program would land many budget windows too late and time out before it is ever reached. Second it _floors the result at the uniform prior_, $Q = max(Q_"model", Q_"uniform")$: the model may make a primitive cheaper, and so enumerated earlier, but never push one below its uniform reachability, so every program reachable under the baseline stays reachable. Third it preserves the _content-literal boost_ the baseline depends on, forcing any integer literal already visible in frame $x_0$ to cost zero. The model's confident guesses are explored first, while nothing it has not seen is priced out of reach.

*Feeding back into enumeration.* The trained model does not steer every family. Dreamed $Q$ is applied only to families with at least one solved instance — a scene the recognition model was actually trained on as a replay. A zero-replay family, such as belief before its very first solve, is invisible to the model and could only be mispriced by it: it would flood the early budget windows with the non-belief primitives it _has_ seen, each pushed below uniform, and thereby delay that family's own primitives past the timeout. Such families stay on the proven uniform and content baseline and earn the dreamed prior only once they are solved and can contribute replays of their own. In this way the recognition model accelerates the families the curriculum has already reached without ever slowing down the frontier it has not — and its benefit compounds with compression, since each round it is retrained over a library whose new abstractions let it steer toward deeper structure than the round before.

#load-bib(read("chapter1.bib") + read("chapter2.bib"))
