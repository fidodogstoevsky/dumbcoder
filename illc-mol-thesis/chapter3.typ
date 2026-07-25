#import "@preview/illc-mol-thesis:0.2.0": *

#import "world_tape.typ": world-tape, grid-view, arc-colors

#import "viz.typ": task-figure, all-tasks

#let lc = $chevron.l$
#let rc = $chevron.r$

#let terminal(body) = block(
  fill: black,
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  text(
    fill: white,
    font: "DejaVu Sans Mono",
    size: 9pt,
    raw(body),
  )
)

#let pycode(body) = block(
  fill: rgb("#f6f8fa"),
  stroke: 0.5pt + rgb("#d0d7de"),
  radius: 4pt,
  width: 100%,
)[
  #set text(font: "DejaVu Sans Mono", size: 9pt)
  #raw(body, lang: "python")
]

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

#show raw.where(lang: "lisp"): it => block(
  fill: rgb("#f6f8fa"),
  stroke: 0.5pt + rgb("#d0d7de"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #set text(font: "DejaVu Sans Mono", size: 9pt)
  #it
]

#mol-chapter("Implementation")

== Objective

Modularity-nativists argue for the existence of a Theory of Mind Module (ToMM), a natively-endowed domain-specific system for understanding the behavior of intentional agents. In Alan Leslie's characterization the ToMM introduces attitude concepts like _Believes_ and uses a "proprietary representational system" for ascribing attitudes to agents @leslie_pretending_1994. Our goal is to demonstrate that the ability to theorize in mental terms can in principle be _learned_, without the need for a native ToMM. 

We initialize the system with a library of domain-general primitives, none of which posit an agent's intentional state, nor encodes attitude concepts like _believes_. Lacking bespoke ToM capacities, such a system would on Leslie's view not be able to reason about intentional agents, to attribute a belief to an agent, to explain an agent's movement as navigation according to a private intensional representation. We show that the system learns _useful abstractions_ of domain-general functions, abstractions that encode attitude concepts like _Believes_ and attribute an intensional state to an agent.

In our first trial (phase 1) we show that, given the domain-general capacity to represent counterfactual world states, a system can learn a functional abstraction that encodes a notion of belief attribution. The learned function attributes a counterfactual world state to a particular agent, determines what the agent's movement would be based on its own private (non-world) representation, and then renders only the agent's movement as a real-world artifact (so it doesn't render the agent's believed world as the real world). So this use of the capacity for counterfactual representation is discovered in program space. 

In the second trial (phase 2), we show that the system can arrive at an abstracted notion of belief starting from an even more basic language of 2-ary combinators. We decompose the counterfactual representation machinery, initializing the system with a library of product-category combinators that are unambiguously domain-general. We show that enumeration recovers the basic structure of belief attribution even from these atomized primitive operations. 

== Task Corpus

We were inspired by experiments in the developmental psychology literature in which subjects are tasked with explaining a scene or making predictions about it. We created a corpus consisting of various grid-based tasks that depict some dynamic at play. The system is an observer trying to make sense of the depicted scene by finding a program that correctly generates the scene. 

Most tasks in the corpus are _trajectory tasks_, a matrix $A$ of dimensions $s times s times n$ that depicts a visual scene occurring over $n$ time steps. Each frame is a _grid_ of width and height $s$ (we denote the $i$th grid as $t_i$). The solver just receives the initial grid $t_0$, the first frame (an $s times s$ grid) of task $A$. Its goal is to find a program that generates the full $A$ matrix, i.e. a program that posits the underlying data generation process. A correct solution is a function $f$ that, given a grid $t_i$ where $i < n$, returns the next grid in the sequence $f(t_i)=t_(i+1)$. 

Tasks are classified by task type, by the general concept that explains the task. For example, the below is an instance of a _movement_ task, which depicts a cell value (4) moving up steadily in some direction until it reaches an edge. A solution to the task might be program that encodes "at each timestep, value 4 moves one step up". 

#task-figure("physics", caption: none)

Another type of trajectory task is the _desire_ tasks such as the instance seen below which depicts the value 1 getting closer to value 2 with each step. A solution to the instance might be a program that encodes "at each timestep, value 1 moves one step closer to value 2"

#task-figure("desire", caption: none)

The above tasks (and most others in the corpus) can be solved by programs that explain the scene extensionally, a purely behaviorist visual description. But we're interested in studying tasks that depict agent behavior guided by a belief, tasks that can only be efficiently solved by attributing a belief to an agent. The below task is an instance of a _false wall_ task,

#task-figure("belief_wall", caption: none) 

Whereas a trajectory task's input is one initial grid $t_0$, a _template task_'s input is a pair of grids: a _world_ grid and a _template_ grid, both of dimensions $s times s$. Given (_working_, _template_), the task is to produce $t_1$, a version of the working grid modified according to the pattern given in the template grid. This format is loosely inspired by the ARC-AGI-1 task corpus [CITE]. 

#task-figure("registration", caption: none)


Our corpus consists of various kinds of tasks. Most tasks can be solved without belief attribution, but some tasks require belief attribution. 

require belief attribution:

- wall false belief: goal-oriented motion, navigating based on a believed grid which has a wall where one doesn't exist
- witness false belief: two agents, one has a false belief that a wall is there and navigates to avoid it, the other has a true belief about the world so doesn't avoid the wall
- goal-displacement false belief: classic sally-anne,
- false-obstacle belief
- dual belief

don't require belief attribution:

- simple movement
- desire: goal-oriented motion (uses `optimize`)
- overlay: moving object leaves a trail (uses `fork`)
- comet: goal-oriented motion with moving trail (uses `fork`)
- flee: like desire but with maximizing distance
- deletion: grid edit remove one cell
- denoise: grid edit remove a value
- underlay: complement to overlay
- obstacle: goal seeker detours. the grid doesn't have an obstacle there, but the goal seeker detours nonetheless. So the best explanation is `compose( wall_at r c, optimize(neg_dist gv) av )`, that the agent actually is navigating by a real wall that is there, and it's just that I the viewer don't see it
- relocation: move a wall from one location to another
- registration: snap one object onto the template
- perception: registration complement, record a sensation into the private channel
- multi-registration: snap all objects
- registration-except: 
-  



=== necessity filters

one of the challenges of creating such a corpus is making tasks that require belief attribution. 

checks in the (belief) task generators to make sure they aren't physically explainable, so they're not underdetermined. For example, in a wall false belief task an agent might have a false belief about a wall being there, but at a location nowhere near the agent's path to its goal. So the agent can still navigate to the goal freely as if it didn't have such a false belief, positing the false belief isn't necessary or helpful to explain the scene. So those scenes are rejected. 

we implement a filter that 

== interpreter <interpreter>

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

The interpreter is minimal by design. `unfold` is the anamorphism that at each timestep applies $f$ to the previous grid to generate the new grid, and appends the new grid to the list of grids. So `unfold`'s state is just a single grid: it doesn't encode a world/model grid pair, nor a registry of agents and goals.



A transition function $f$ is a `grid` endomorphism `fn :: grid -> grid`




There is nothing representational about the interpreter, since it only builds up one sequence of grids. The ability to create multiple representations of the world, to create a representation that deviates from the real world (and attribute it to a particular agent), is not encoded in the interpreter. So to assume the interpreter as innate to the system does not smuggle in a theory of mind. 

for some of the tasks, we use an interpreter for programs of root `fn_p_g :: (grid, grid) -> grid` since those tasks don't involve just one input grid but a pair. That uses `unfold_with_template(g, template, T, c)`

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

== Program representation and type system

=== Expression trees

Every program, primitive, and invented abstraction is a `Delta`, i.e. an object that can serve as a node in an expression tree. A `Delta` carries a `head` (the Python value or callable it denotes), a `type` (the type its subtree returns), a list of `tailtypes` (the types of the arguments it expects), and a list of `tails` (the actual argument subtrees, once they are filled in). So a `Delta` with no `tailtypes` is a _terminal_, a leaf whose value is just its head, for example `up :: dir` or the literal `4 :: cellvalue`. A `Delta` with `tailtypes` is an _operator_: an interior node like `step :: cellvalue, dir -> fn`, whose head is applied to its filled-in tails. Then a node's `arrow` is the signature that the enumerator matches against when it fills argument slots. For a terminal, its arrow is just its `type`. For an operator, it's the pair `(tailtypes, type)`, the arrow from its input types to its output types. 

The expression tree is directly executable. Calling a `Delta` evaluates the program it represents, so a terminal returns its head while an operator recursively evaluates each of its tails and applies its head to the results. So the program `(step 4 up)` is a `step`-headed node with tails `4` and `up`; calling it evaluates the two leaves to the cell value `4` and the direction `(-1,0)`, then applies `step` to obtain a transition function `fn :: grid -> grid`. 

An invented abstraction is represented as a `Delta` as well, but it's distinguished by a `hiddentail` which is the body of the abstraction. The `hiddentail` is itself a `Delta` tree over lower-level primitives, with numbered argument holes `$0, $1, ...` standing in for the abstraction's parameters. When such a node is called, its body is copied and each hole `$i` is replaced by the corresponding tail before the body is evaluated. This is what makes an abstraction cost a single token. The whole subtree it abbreviates is hidden inside the `hiddentail` and does not appear in the program that uses it.

The description length of a program is its node count, computed by `length`. a leaf counts one, and an operator counts one plus the lengths of its tails. So an invented abstraction used as a token adds just one, since its body lives in the un-charged `hiddentail`. This node count is the minimum-description-length quantity the enumerator uses to tiebreak between programs that solve a task within the same budget window (see @enumeration).

=== The library

The library of primitives is a `Deltas` object. It is instantiated with a list of `core` primitives and grows a parallel list of `invented` abstractions as compression discovers them. When the enumerator draws primitives, it draws from the concatenation of `core` and `invented` which is the full library. Whenever a primitive is added, `Deltas.infer` rebuilds the indices the search depends on — most importantly `bytype`, which groups the alphabet's symbols by their return type. This grouping is what the type-uniform prior is computed from: a symbol of type $tau$ is priced at $-ln$ of the number of symbols sharing type $tau$, so `bytype` gives both the branching factor at each argument slot and the cost of each choice (@enumeration).

=== A simple monomorphic type system

Types are plain strings. A 2d array, i.e. one frame of a task, is of type `grid`, so a transition function is `fn :: grid -> grid`. A $z$-stack of grids, i.e. a sequence of frames (a full task), is of type `mat`. Cell values, e.g. a value representing an entity on the grid like a `4` moving up each timestep, are of type `cellvalue`. Grid positions are of type `coord`. These two int terminals are split by type so that diversifying cell values does not inflate the branching at a position slot (@primitives). 

The type system is deliberately monomorphic: there is no polymorphism, so every arrow over a different shape of data needs its own named type and its own typed composer. This is a cost — a single polymorphic `compose : (b -> c) -> (a -> b) -> (a -> c)` would subsume many of the types below — but it is a principled one: keeping the primitives ground (rather than schematic) is what lets belief emerge as a discovered _composition_ of concrete operations rather than being smuggled in as a polymorphic combinator (@decomposed). The types beyond `grid`/`fn` are:

- `pair_gg` — a pair of grids `(grid, grid)`, the (world, model) channel pair;
- `fn_g_p` — a pair producer `grid -> pair_gg` (e.g. `dup`);
- `fn_p_g` — a pair consumer `pair_gg -> grid` (e.g. `sync_to_world`, `overlay`);
- `fn_p_p` — a pair endomorphism `pair_gg -> pair_gg` (e.g. `mapsnd`);
- `fn_g_c` — a coordinate reader `grid -> coord` (e.g. `locate`);
- `fn_gc_g` — a coordinate placer `(grid, coord) -> grid` (e.g. `place`);

and, for the arity-generalization phase, the recursive grid-stack `gstack` with its lift/endomorphism/render arrows `fn_g_s`, `fn_s_s`, `fn_s_g` (@arity). Because whether a program opens a private second channel is now a matter of which of these types its nodes carry, the presence of a (world, model) representation is a _structural_ property of a synthesized program, not a commitment baked into the interpreter's state.

=== Types and task roots

Which type counts as a solution is fixed by the interpreter the task is enumerated against. A trajectory task's input is a single initial grid, of type `grid`, and its solution is a transition function of type `fn`, since for a grid at timestep $i$ it must return the grid at $i+1$. So only an `fn`-typed program is a valid solution for a trajectory task: the interpreter `unfold` takes the initial grid (`grid`), a transition function (`fn`), and the duration (`int`), and returns a `mat` to compare against the task.

A template task's input is instead a pair `(grid, grid)` — a working grid and a constant external template — and its solution is a _commit policy_ of type `fn_p_g`, answering "what should each frame do with the template?"; these tasks are enumerated against `unfold_with_template`. Enumeration is therefore organized by root type, `fn` and `fn_p_g` being searched as separate cost-ordered streams, while compression pools the solutions of both roots into a single stitch call (@compression) — so an abstraction for belief must earn its place against the whole solved corpus, not just its own root's share of it.




== Primitives <primitives>

=== core primitives

every run's library starts with core primitives used to construct the transformation functions. `step :: cellvalue, dir -> fn` moves the value one step in a specified direction, chosen among `left :: dir`, `up :: dir`, etc. `optimize :: util, cellvalue -> fn` moves the value one greedy (BFS-optimal) step towards improving its utility, chosen among `distance :: cellvalue -> util` or `neg_dist :: cellvalue -> util`. 

#show table.cell.where(y: 0): set text(style: "normal", weight: "bold")
#set table(stroke: (_, y) => if y == 1 { (top: 0.9pt) } else if y > 1 { (top: 0.2pt) })


=== atomic (phase 1)

- `fork :: fn -> (grid, grid) -> fn`
- `sync_to_world :: cellvalue -> (grid, grid) -> grid`

=== decomposed (phase 2) <decomposed>

The product-category combinator family in our type system is

`
dup_g       : grid -> pair_gg
mapsnd      : fn   -> fn_p_p
compose_gp  : fn_g_p, fn_p_p -> fn_g_p
pipe_gpg    : fn_g_p, fn_p_g -> fn
`

which are just the pair-grounded instantiation of universal categorical combinators

- `dup     : a -> pair a a` (the diagonal $triangle$)
- `mapsnd  : (b -> c) -> pair a b -> pair a c` (the bifunctor 'second')
- `compose : (b -> c) -> (a -> b) -> (a -> c)`

The decomposition is polymorphic in principle. Our monomorphic primitives are the ground instances at `a = grid`. So belief _can_ be assembled from standard categorical combinators.

so if we had polymorphism we could collapse the above family into one `compose`.

=== arity (phase 3) <arity>


== ECD architecture 

ECD is a wake-sleep library-learning loop inspired by the DreamCoder @ellis_dreamcoder_2020. Each iteration of the loop consists of an enumeration phase, a compression phase, and a dreaming phase. In enumeration, the system generates candidate programs to solve tasks

=== Enumeration <enumeration>

While a single cost-ordered stream can in principle be checked against every task in the corpus at once, in the experiments each still-unsolved task is instead enumerated as its own cost-ordered stream, parallelized across tasks. This per-task granularity is what lets the prior $Q$ be conditioned on the individual scene rather than shared across the corpus — a fact we rely on in @dreaming, where $Q$ becomes a task-conditional distribution $Q(dot | x)$ emitted by a recognition model.

In principle the enumerator doesn't itself set out to synthesize a solution program for each particular task. Rather it generates candidate programs arbitrarily from the grammar in order from most to least probable (from lowest to highest cost), checking candidate programs against the task corpus  


Say the task to solve depicts the value 3 rising in each successive grid, so the target solution is the program `(step 4 up)`. Its head is `step :: cellvalue, direction -> fn` and its leaves are `4 :: cellvalue` and `up :: direction`. A program's cost is given by the sum of the costs of its constituent primitives. Say $Q$ is the initial flat prior, the type-uniform distribution, so each primitive's cost is the negative natural log of the number of symbols in the grammar that share its type. For example, there are $N=4$ symbols of type `direction` so each gets $p = 1/N = 1/4$. By Shannon, we have cost(`up`) $= -ln p = -ln 1/4 = ln 4 approx 1.386$ nats. By the same process, `step`'s cost is $ln 8 approx 2.1$ nats and `4`'s cost is $ln 10 approx 2.3$. By logarithmic additivity, to find the program's total cost we simply sum the costs of its constituent primitives which yields $approx 5.8$ nats for `(step 4 up)`.

Program search proceeds by budget window to ensure that cheaper programs are enumerated first. For `(step 4 up)`, the first two windows are skipped by pruning. We start enumerating in window $[0,2)$, but since `step`'s cost is 2.1 nats it exceeds the budget, so it cannot be chosen as the root node of the program tree and we proceed to budget window $[2,4)$. After paying 2.1 nats for `step`, only 1.9 nats remain to fill the two holes in the partial program tree. `step`'s tail types (arguments) are `cellvalue` and `step`. But any primitive of type `cellvalue` costs 2.3 nats, so `step`'s first argument can't be placed. So trees with a `step` node are pruned, since they would never complete within the second budget window. Search is cheapest-first, since more complex programs only become affordable once the window is wide enough.

At budget window $[4,6)$ the ceiling is finally sufficiently wide to admit all three symbols. The tree is assembled incrementally: each enumerated subtree is appended to a copy of its parent node, and a callback fills the remaining argument slots by the same recursion (so programs of any arity or nesting depth are handled by the same procedure). Since enumeration is type-directed, each argument slot draws candidates only from symbols of the matching type, so no ill-typed partial tree is ever constructed. `(step 4 up)` completes at its total cost of $approx 5.8$ nats.

When `(step 4 up)` is found the tree rooted by `step` has no more available holes, so the callback fires to evaluate the found program $rho$. For each task $x$ in the corpus, the interpreter generates the output matrix given the input grid of the task by calling `unfold` on $rho$, initial grid $x_0$, and $x$'s third dimension $n$. Then the resulting output matrix $x'$ is hashed and compared against task $x$'s hash. So for a program to solve a task, it must produce a grid that exactly matches that task's entire trajectory, not just the endpoint of the trajectory. This ensures that programs capture the full behavior rather than just the end result. 

If `(step 4 up)` is the first solution found for the task, it's stored as the task's solution. Because the windows are tried in increasing order of cost, the first solution the search returns is in principle the shortest program that solves the task. But enumeration is parallelized so search unfolds simultaneously over program trees, so we need to tiebreak when multiple programs solve a task within the same budget window. To tiebreak we first run `Delta.simplify` on the programs, a semantics-preserving rewrite that collapses spurious nesting which sometimes occurs during enumeration. We then run `Delta.length` to get the node count of the expression trees we need to compare. Every operator and leaf adds to the count, and an invented abstraction used as a token adds one to the count too. The body of the abstraction isn't charged, since the components now count for just application. The program with the lower node count wins since it has the lower minimum description length (since by symbol count it's shorter).

Also, the search space contains lots of junk programs that crash or return nothing even though they type-check. So the enumerator callback wraps both the interpreter and the evaluator in `try/except` blocks and skips any degenerate or non-callable program. 

=== Compression <compression>

While Enumeration minimizes the cost of each program under a fixed grammar, Compression lets the system revise the grammar in light of the solutions found by enumeration. It identifies syntactic structure common across found solutions, and abstracts it as a new primitive to add to the library. The goal of compression is to minimize the joint cost of the library plus the programs written in it, given by:

$
min_"library" ("DL"("library") + sum_"tasks" "DL"("program" | "library"))
$

The choice of whether to abstract a given shared structure is non-trivial. Consider a program tree fragment that recurs across many solutions. Say it gets added to the library as a new primitive `fn_k`, so now each subsequent usage of the fragment collapses from a multi-node subtree to a single leaf, so it costs one token rather than the several that `fn_k` abbreviates. But the library is now larger, and since a symbol's cost is $-ln$ of the number of symbols sharing its type, adding one more `fn_k`-typed symbol raises the cost of every primitive of that type. The abstraction is worthwile only when the fragment is large enough, and recurs often enough, that the tokens saved across the corpus outweigh the price of carrying one more symbol. So accidentally-shared structure is never abstracted, while shared structural motifs do. 

In DreamCoder @ellis_dreamcoder_2020 the compression step is bottom-up: for each found program it enumerates the exponential set of program refactorings (stored as version spaces) and intersects them to grow abstractions up from reoccurring subexpressions. Rather than this bottom-up approach, we delegate the search for abstractions to the top-down Stitch @bowers_top-down_2023  library-learning system. Instead of enumerating refactorings of every program, stitch searches the space of abstractions directly. So stitch doesn't have to materialize every rewrite. 

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

Before passing the found solutions to stitch for abstraction, we first need to fully expand any existing primitives in those solutions. For this we implement `normalize`, the complementary operation to abstraction. It fully inlines every invented `fn_k` back to the bare primitive alphabet, recursively substituting each abstraction's argument holes with its actual arguments until nothing but core primitives remain. 

We need to normalize because stitch's top-down compression can only abstract over structure it can see, and an opaque `fn_k` token has no visible interior. So any abstraction that would need to reach inside `fn_k`, or span its boundary and surrounding context, is simply invisble to the search.

the compression step must be handed programs in a common vocabulary — feeding it already-compressed programs, in which a prior round's fn_0 appears as an opaque token, would let those tokens collide with the fresh abstractions the compressor invents under the same names. Normalizing before compression guarantees the compressor sees only primitives and the structure it discovers is genuinely its own. Renaming it to `inv_0` avoids the lexical clash but leaves it just as opaque; stitch still can't see the `(compose (wall_at …) (optimize …))` sitting inside it. Inlining flattens the token back to that primitive structure, and only then can the compressor find abstractions that cut across the old boundaries.

The best syntactic fragment to abstract from round $N$'s solution may carve the primitives differently than round $N+1$'s. For example, 

So we can refactor abstractions that were previously ideal but are no longer ideal

enumeration is by root types so `fn` and `fn_p_g` are searched separately. but compression is joint over root types, their solutions are pooled into a single stitch call. so an abstraction for belief has to fight over the space of all solved programs, including those from the other root type. so it really is that much more efficient. This preempts the attack that by separating the search to different root types we're biasing abstraction creation towards the belief tasks since they make up a higher percentage of `fn`-rooted tasks rather than `fn_p_g`-rooted. the same objective that could have paid for belief's structure is free to spend its budget on the overlay, registration, and obstacle families instead, but it ends up going to the belief families because those abstractions compress so much structure. 

What it returns is a set of abstractions together with the corpus rewritten to use them. We infer each abstraction's hole type and register the abstraction as a new `fn_k` in the library `D`.

The rewritten corpus stitch returns then becomes the training data for the dreaming phase and the starting library for the next round, in which programs a level deeper — belief compounds too expensive to enumerate from bare primitives — have become reachable because their shared core now costs a single token. 

=== Dreaming <dreaming>

In the first enumeration run, primitives are chosen under type-uniform prior distribution $Q$. 
Enumeration searches under a prior $Q$ that is fixed across the whole corpus: every task is enumerated against the same type-uniform distribution (with the content-literal exception below). But the tasks are not interchangeable. A scene in which the agent takes a straight path to its goal and a scene in which it detours around a wall call for very different transition functions, and a prior that cannot see the scene has no way to tell them apart — it must spend the same budget windows on both.

In the dreaming phase we train a recognition model @ellis_dreamcoder_2020, a neural network that reads a task $x$ and emits a task-conditional prior $Q(dot | x)$, biasing the grammar toward the primitives that the scene is likely to need. The recognition model `MatRecognitionModel` is built around a symbolic grid encoder rather than a generic convolutional one. Its central design commitment is that 


entity _roles are read off from motion, not from cell values_: because the corpus deliberately uses diverse agent and goal identifiers, the encoder cannot key on "value 1 is the agent." Instead it labels a cell occupied in frame $x_0$ but vacated by the final frame a _mover_, and a non-background cell occupied at both ends a stationary _goal_. It then pools row and column position embeddings into per-entity slots: two mover slots (each carrying both a start-position pool and a trajectory pool), two goal slots, a wall pool, and a path-length embedding of the duration $T$, concatenated into a single matrix embedding $h_"matrix"$. The per-entity layout is what lets the two-mover, two-goal scenes of the witness and dual belief families be represented in separate slots rather than blurred into one mean, and the separate per-mover trajectory pool is the decisive signal for false belief: a detour path and a straight path yield different mean agent positions even when their start, goal, and $T$ coincide, which is exactly the case a belief task presents. A linear head maps $h_"matrix"$ to logits over the whole library $D$.

Crucially the prediction is _flat_: one forward pass per task yields a single distribution over the DSL, computed once and reused for every enumeration decision on that task. We deliberately forgo a tree-context recurrent network that would condition each node's prediction on its partial parent tree. A per-node network is both far more expensive — it turns each of the thousands of enumeration steps into its own forward pass — and prone to failing exactly where it matters: the abstractions belief requires are deep, and a tree-conditioned model trained on shallow solved programs generalizes poorly to the depths it has not yet seen. A flat, matrix-conditioned $Q$ sidesteps both problems.

*Replays and fantasies.* The model is trained by the wake–sleep procedure of a Helmholtz machine, on two sources of program–scene pairs. _Replays_ are the actual solutions enumeration has found so far, run back through the interpreter on their own task grids — supervised examples of "this scene was solved by this program." But replays alone are scarce and biased toward the families already solved, so they are supplemented with _fantasies_: programs sampled directly from the current library prior and executed by `unfold` on freshly generated grids to synthesize new (scene, program) pairs at no labelling cost. Each dreaming iteration draws up to four replays and fills the rest of a batch of eight with fantasies. The fantasy grids are not arbitrary — their size, entity values, and the fraction carrying four or more entities are all read off the corpus's own scene statistics, so the encoder trains on the scene mix it will face at test time (in particular the multi-entity scenes that exercise its second mover and goal slots). Roles remain decided by the sampled program's motion, so a fantasy only ever _lets_ a program drive two movers; it never manufactures a belief scene by hand. For every pair the model encodes the resulting matrix and is trained by cross-entropy to predict each node of the generating program tree, averaged over nodes.

*Putting the model on the enumeration cost scale.* A recognition model trained this way emits a globally-normalized distribution over the library, which is on the wrong scale for the budget-window enumerator and, having been trained only on the families solved so far, could actively suppress the rare primitives an as-yet-unsolved family needs. `dreamed_q` reconciles the two so that dreaming can only ever help. First it re-softmaxes the model's logits _within each type group_, so a program's summed cost is comparable to the uniform and content priors rather than living on the model's global scale — otherwise a ten-node program would land many budget windows too late and time out before it is ever reached. Second it _floors the result at the uniform prior_, $Q = max(Q_"model", Q_"uniform")$: the model may make a primitive cheaper, and so enumerated earlier, but never push one below its uniform reachability, so every program reachable under the baseline stays reachable. Third it preserves the _content-literal boost_ the baseline depends on, forcing any integer literal already visible in frame $x_0$ to cost zero. The model's confident guesses are explored first, while nothing it has not seen is priced out of reach.

*Feeding back into enumeration.* The trained model does not steer every family. Dreamed $Q$ is applied only to families with at least one solved instance — a scene the recognition model was actually trained on as a replay. A zero-replay family, such as belief before its very first solve, is invisible to the model and could only be mispriced by it: it would flood the early budget windows with the non-belief primitives it _has_ seen, each pushed below uniform, and thereby delay that family's own primitives past the timeout. Such families stay on the proven uniform and content baseline and earn the dreamed prior only once they are solved and can contribute replays of their own. In this way the recognition model accelerates the families the curriculum has already reached without ever slowing down the frontier it has not — and its benefit compounds with compression, since each round it is retrained over a library whose new abstractions let it steer toward deeper structure than the round before.




#load-bib(read("chapter1.bib") + read("chapter2.bib"))
