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

== Overview

Modularity-nativists argue for the existence of a Theory of Mind Module (ToMM), a natively-endowed domain-specific system for understanding the behavior of intentional agents. In Alan Leslie's characterization the ToMM introduces attitude concepts like _Believes_ and uses a "proprietary representational system" for ascribing attitudes to agents @leslie_pretending_1994. Our goal is to demonstrate that the ability to theorize in mental terms can in principle be _learned_, without the need for a native ToMM. 

In our first trial we show that, given a non-mental representational substrate, a system can learn a functional abstraction that encodes a notion of belief attribution. The learned function `fork(derive, commit)`


We initialize the system with a library of domain-general primitives, none of which posit an agent's intentional state, nor encodes attitude concepts like _believes_. Lacking bespoke ToM capacities, such a system would on Leslie's view not be able to reason about intentional agents, to attribute a belief to an agent, to explain an agent's movement as navigation according to a private intensional representation. We show that the system learns _useful abstractions_ of domain-general functions, abstractions that encode attitude concepts like _Believes_ and attribute an intensional state to an agent.

In the second iteration, we show that the implementation can arrive at an abstracted notion of belief starting from an even more basic language of 2-ary combinators. 

== ECD architecture 

ECD is a wake-sleep library-learning loop inspired by the DreamCoder @ellis_dreamcoder_2020. Each iteration of the loop consists of an enumeration phase, a compression phase, and a dreaming phase. In enumeration, the system generates candidate programs to solve tasks

=== Enumeration

Candidate programs are enumerated in order from most to least probable, i.e. from lowest to highest cost. To illustrate, say we're searching for a program that solves a task depicting the value 3 rising in each successive grid, so our target program is `(step 3 up)`. Its head is `step :: cellvalue, direction -> fn` and its leaves are `3 :: cellvalue` and `up :: direction`. A program's cost is given by the sum of the costs of its constituent primitives. Say $Q$ is the initial flat prior, the type-uniform distribution, so each primitive's cost is the negative natural log of the number of symbols in the grammar that share its type. For example, there are $N=4$ symbols of type `direction` so each gets $p = 1/N = 1/4$. By Shannon, we have cost(`up`) $= -ln p = -ln 1/4 = ln 4 approx 1.386$ nats. By the same process, `step`'s cost is $ln 8 approx 2.1$ nats and `3`'s cost is $ln 10 approx 2.3$. By logarithmic additivity, to find the program's total cost we simply sum the costs of its constituent primitives which yields $approx 5.8$ nats for `(step 3 up)`. 

Program search proceeds by budget window to ensure that cheaper programs are enumerated first. For `(step 3 up)`, the first two windows are skipped by pruning. For example in window $[2,4)$ after paying 2.1 nats for `step`, only 1.9 nats remain to fill the two holes in the partial program tree. `step`'s tail types (arguments) are `cellvalue` and `step`. But any primitive of type `cellvalue` costs 2.3 nats, so `step`'s first argument can't be placed. So trees with a `step` node are pruned, since they would never complete within the second budget window. So search is cheapest-first, since more complex programs only become affordable once the window is wide enough.

At budget window $[4,6)$ the ceiling is finally sufficiently wide to admit all three symbols. The tree is assembled incrementally: each enumerated subtree is appended to a copy of its parent node, and a callback fills the remaining argument slots by the same recursion (so programs of any arity or nesting depth are handled by the same procedure). Since enumeration is type-directed, each argument slot draws candidates only from symbols of the matching type, so no ill-typed partial tree is ever constructed. `(step 3 up)` completes at its total cost of $approx 5.8$ nats. Since the tree has no more available holes, the callback fires: the finished program is run on the task's input grids and its output is compared against the target. Since it's the first solution found, it is stored as the task's program.
Because the windows are tried in increasing order of cost, the first solution the search returns is (up to ties within a window) the shortest program that solves the task — its minimum-description-length explanation under the current grammar.

=== Compression

While Enumeration minimizes the cost of each program under a fixed grammar, Compression lets the system revise the grammar in light of the solutions found by enumeration. It identifies syntactic structure common across found solutions, and abstracts it as a new primitive to add to the library. The goal of compression is to minimize the joint cost of the library plus the programs written in it, given by:

$
min_"library" ("DL"("library") + sum_"tasks" "DL"("program" | "library"))
$

The choice of whether to abstract a given shared structure is non-trivial. Consider a program tree fragment that recurs across many solutions. Say it gets added to the library as a new primitive `fn_k`, so now each subsequent usage of the fragment collapses from a multi-node subtree to a single leaf, so it costs one token rather than the several that `fn_k` abbreviates. But the library is now larger, and since a symbol's cost is $-ln$ of the number of symbols sharing its type, adding one more `fn_k`-typed symbol raises the cost of every primitive of that type. The abstraction is worthwile only when the fragment is large enough, and recurs often enough, that the tokens saved across the corpus outweigh the price of carrying one more symbol. So accidentally-shared structure is never abstracted, while shared structural motifs do. 

In DreamCoder @ellis_dreamcoder_2020 the compression step is bottom-up: for each found program it enumerates the exponential set of program refactorings (stored as version spaces) and intersects them to grow abstractions up from reoccurring subexpressions. Rather than this bottom-up approach, we delegate the search for abstractions to the top-down Stitch @bowers_top-down_2023  library-learning system. Instead of enumerating refactorings of every program, stitch searches the space of abstractions directly.

To create an abstraction, stitch starts with a tree that consists of just a root node `??`, an unexpanded hole that the search still needs to refine. At the root `??`, the abstraction _matches_ every subexpression, since it's the most general abstraction possible: any subexpression can be trivially "rewritten" 

. As the tree is built up and the search is refined, the set of possible match locations shrinks.  


A node of the stitch abstraction search tree is a partial abstraction. It consists of some combination of primitive operations (like `step`) and argument holes `#i`, and at least one unexpanded hole `??` that the search still needs to refine. 






At each step it commits part of the abstraction's body to a concrete operator or splits a subtree, branching downward toward more specific abstractions.




It's a branch-and-bound search with an upper bound on the achievable compression at each node, so whole regions of the abstraction space get pruned without ever being expanded.

Where the DreamCoder line searches version spaces bottom-up, stitch enumerates candidate abstractions top-down and uses a compressive-utility bound to find, without materializing every rewrite, the fragment whose abstraction most reduces the joint description length. One call runs for a fixed number of `iterations`: each iteration commits the single most compressive abstraction, rewrites the corpus in terms of it, and repeats, so later abstractions may be built out of earlier ones; `max_arity` bounds how many holes (`#0`, `#1`, …) an abstraction may carry. The programs we hand it are first _normalized_ — every previously-invented abstraction is inlined back to bare primitives — so that stitch sees only the primitive alphabet and its discoveries never collide with names from an earlier round. What it returns is a set of abstractions (fragments with `#i` holes) together with the corpus `rewritten` to use them; we remap each `#i` hole to a `$i` argument, infer the hole's type, and register the abstraction as a new `fn_k` in the library `D`.

The crux of the design is that this compression is _joint_. Enumeration must proceed per root type — one budget walk produces one kind of arrow, so the trajectory tasks (`fn`) and the registration tasks (`fn_p_g`) are searched separately — but every solution they find, across both root types and across both the mind-attributing and the minds-free families, is pooled into a _single_ stitch call. There is one library and one compression objective, not one silo per family. This is what makes an abstraction for belief an honest MDL win rather than an artefact of searching the belief tasks in isolation: the same objective that could have paid for belief's structure is free to spend its budget on the overlay, registration, and obstacle families instead, and a fragment is abstracted only if it earns its keep against all of them at once.

What the pooled stitch reliably discovers is the research payload of the whole system. Across the phases it invents a single agent constructor, in Phase 1 of the form
```lisp
fn_0 = (fork (compose (wall_at $3 $2) (optimize (neg_dist $1) $0))
             (sync_to_world $0))
```
in which the agent value `$0` appears _twice_ — once in the policy that plans over the private, phantom-walled grid built by `derive`, and again in the `sync_to_world` commit that writes the result back to the world. Nothing in the grammar forced those two argument slots to be filled by the same value; that they collapse into one shared hole is the structural signature of agency — an agent acting on its own model of the world — and it is _discovered_ by the description-length objective, not stipulated by any primitive. The rewritten corpus stitch returns then becomes the training data for the dreaming phase and the starting library for the next round, in which programs a level deeper — belief compounds too expensive to enumerate from bare primitives — have become reachable because their shared core now costs a single token.

=== Dreaming

== Tasks

A _trajectory task_ is a $d times d times n$ matrix that depicts a visual scene unfolding over $n$ time steps, where each frame has width $d$ and height $d$. The enumerator gets the initial frame $t_0$ as input, and the task is to reproduce the rest of the scene's frames ($t_1$ to $t_n$). The enumerator produces programs of type `fn`, a transition function that describes for a given timestep what to do, it's fed to the interpreter/simulator/unfolder. The analogy to the real-life experimental setting is as follows: the enumerator is an observer trying to make sense of a scene, to explain the underlying processes, the function that determines the transitions.

A _template task_ is an 

== interpreter

== Primitives

atomic, decomposed

#load-bib(read("chapter1.bib") + read("chapter2.bib"))
