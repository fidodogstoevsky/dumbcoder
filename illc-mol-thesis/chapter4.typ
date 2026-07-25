#import "@preview/illc-mol-thesis:0.2.0": *

#import "viz.typ": task-figure, all-tasks

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


#mol-chapter("Results and Discussion")

belief/theory-of-mind is a discoverable compound under MDL rather than a built-in primitive 

== Found abstractions

=== atomic primitives

*round 1* enumeration solves obstacle tasks

#terminal("[  1037] caught (compose (optimize (neg_dist 5) 4) (wall_at c1 c0))")

and simple a few (7) simple false belief tasks

#terminal("[102603] caught (fork (compose (wall_at c1 c2) (optimize (neg_dist 1) 7)) (sync_to_world 7))")

so at the end of the round it abstracts a function for placing a wall and navigating around it

#terminal("fn_0: (compose (wall_at #3 #2) (optimize (neg_dist #1) #0))  [cellvalue, cellvalue, coord, coord] -> fn")

and uses `fn_0` to abstract the belief signature `fork(derive, commit)`: on the model grid, run `fn_0` as the derive, then run `sync_to_world` as the commit using the same `#0` (agent value) as passed to `fn_0`. So run the calculation for `#0` based on the modified grid, and then commit the conclusion to the real world. 

#terminal("fn_3: (fork (fn_0 #0 #3 #2 #1) (sync_to_world #0))  [cellvalue, coord, coord, cellvalue] -> fn")

*round 2* enumeration solves the rest of the simple false belief tasks using `fn_3`, requiring just around $1%$ of the original enumeration cost. So it can successfully use the learned abstraction to generalize. It's much easier to solve the false belief tasks when it already has a function and just needs to fill in the values, rather than needing to rediscover the entire structure of belief again

#terminal("[  1053] caught (fn_3 4 c3 c2 5)")

at the end of the round it abstracts the belief signature again, which was `fn_3` in round 1

#terminal("fn_6: (fork (compose (wall_at #3 #2) (optimize (neg_dist #1) #0)) (sync_to_world #0))  [cellvalue, cellvalue, coord, coord] -> fn")

and it abstracts simple desire

#terminal("fn_7: (optimize (neg_dist #0))  [cellvalue, cellvalue] -> fn")

so then in *round 3* it can use those to solve witness-belief tasks

#terminal("[ 11634] caught (compose (fn_6 5 8 c3 c3) (fn_7 2 1))")

*final* abstractions: belief constructor

#terminal("fn_6  [cellvalue, cellvalue, coord, coord] -> fn
body: (fork (compose (wall_at $3 $2) (optimize (neg_dist $1) $0)) (sync_to_world $0))")

=== decomposed primitives

final abstractions

#terminal(
"fn_6  [cellvalue, cellvalue, coord, coord] -> fn
body: (pipe_gpg (compose_gp dup (mapsnd (compose (wall_at $3 $2) (optimize (neg_dist $1) $0)))) sync_all)
  *** AGENT TYPE CONSTRUCTOR (belief — degenerate scope-complement commit) ***

fn_7  [cellvalue, cellvalue] -> fn
body: (optimize (neg_dist $0) $1)
  (desire fragment)

fn_8  [cellvalue, cellvalue, coord, coord] -> fn
body: (compose (wall_at $3 $2) (optimize (neg_dist $1) $0))
  (obstacle/belief policy: stamp wall ▸ navigate — the shared derive)

fn_9  [coord, coord, cellvalue, cellvalue] -> fn
body: (compose (optimize (neg_dist $3) $2) (wall_at $1 $0))
  (obstacle/belief policy: stamp wall ▸ navigate — the shared derive)

fn_10  [coord, cellvalue, cellvalue] -> fn
body: (pipe_gpg (compose_gp dup (mapsnd (compose (wall_at c2 $0) (optimize (neg_dist $1) $2)))) sync_all)
  *** AGENT TYPE CONSTRUCTOR (belief — degenerate scope-complement commit) ***

fn_11  [fn, fn_p_g] -> fn
body: (pipe_gpg (compose_gp dup (mapsnd $0)) $1)"
)

== The structure of theory of mind

explain the structure of ToM, of the agency signature, of belief attribution

belief is a function that takes a proposition and an agent

```lisp
fn_0 = (fork (compose (wall_at $3 $2) (optimize (neg_dist $1) $0))
             (sync_to_world $0))
```

in which the agent value `$0` appears twice. once in the policy that plans over the private, phantom-walled grid built by `derive`, and again in the `sync_to_world` commit that writes the result back to the world.

Nothing in the grammar forced those two argument slots to be filled by the same value. The fact that they collapse into one shared hole is the structural signature of agency, of an agent acting on its own model of the world. This signature is discovered by the description-length objective, it's not stipulated by a primitive. 

Dennet on intentionality: it's not a specific structure of "belief", rather just something that plays that role

== Scaffolding

Piaget's theory of stages of cognitive development [CITE]

Wellman's theory of mind scale [CITE]

show that lower-level solutions to non-mental tasks are building blocks for mental task solutions


== Benefits of concept learning

The guiding motivation of this project is that an intensional theory, attributing a belief to an agent, is the most compressed representation of an agent's behavior. This is obvious, since the agent navigates based on its intensional representation, so a theory that accounts for this will better represent the underlying data generating process. It's certainly possible to describe an agent's movement without attributing a belief to it, but it would be a laborious explanation that accounts for each step in the sequence. And of coruse it doesn't generalize well, it overfits to a specific agent's movement in a particular scene. 

To illustrate this we show that no non-intensional solution is as short as one which stipulates belief attribution. As described in chapter 3 the enumerator processes candidate programs in $-log p$ order and saves just the first (cheapest) solution it finds, call it _found_. For false belief tasks should be a program that attributes a belief to an agent. Since a false belief task can also be solved by a purely extensional program, call any such program a _rival_ since it's an extensional rival to the intensional explanation.

For each rival, we generate a _margin_ score by taking _margin_ = DL(_rival_) $-$ DL(_found_), where DL is the description length (how many nats it takes to write program $p$ under library $L$). So for any rival, if _margin_ $> 0$ then the intensional program is shorter than the extensional, otherwise an extensional program is at least as short. We plot the empirical cumulative distribution over the margin values for all rivals, so the $x$-axis shows possible margin values and the $y$-axis shows the cumulative fraction of rival programs with _margin_ $lt.eq x$. Since description length depends on the given library we plot two ECDFs: dotted for programs written in the base library, and solid for programs written in the final library (enriched with abstractions found through five ECD rounds).

We first give the plot for the higher level DSL in which `fork` and `sync` are given as atomic primitives. 

#figure(
  image("mdl_margin_1.png", width: 80%),
  caption: [ECDF of MDL margin over rival programs (atomic DSL)]
)

Both curves sit entirely to the right of $x=0$, so for each of the 24 false belief tasks, the _found_ (intensional) program is a shorter description length than any _rival_ (extensional) program. In the base (dashed) library, rival programs cost only $approx 1.3$ nats more than the found program. But the margin baloons to $approx 10.9$ in the final (solid) library: once the `fork(derive, commit)` abstraction is added, the enumerator can immediately generate programs that posit an agent's intensional state. Without such a learned abstraction, an intensional program is only slightly less verbose than an extensional one. But with the abstraction, intensional programs are vastly more compressed than extensional ones. 

This effect is even more pronounced in the plot for the lower-level DSL, in which `fork` and `sync` are decomposed to their product category components. 

#figure(
  image("mdl_margin_2.png", width: 80%),
  caption: [ECDF of MDL margin over rival programs (decomposed DSL)]
)

Under the base primitive library, the mental reading is barely shorter than the physical one ($approx 0.1$ nats). Since the intensional machinery of forking and syncing needs to be constructed from lower-level components, length of intensional programs end up virtually tied with that of the verbose extensional programs. But once the belief abstraction `fn_0` is learned and added to the library, the margin grows to $approx + 8.4$ nats. So mental programs go from being just as bad as non-mental ones to being vastly better, through learning a fitting abstraction. Note that the jump to a $approx + 9.96$ nats margin for the last 11 tasks is due to a further specialized `fn_4 = (fn_0 ... c2)` that hard-codes a phantom-wall at a particular coordinate `c2` so such programs are one coordinate cheaper. But the rival cost is constant accross both, so the entire step is a property of how tightly the library compressed the intensional side. 

We can also see this effect when comparing description lengths of different families of tasks: belief, desire, physics, and world. To plot, we fix each corpus (task type) and calculate each corpus's description length given a round as such:

$
  "DL"(C|r) = sum_(c in C) "DL"(p_c|L_r) + "DL"(L_r)
$

where $C$ is a task corpus (e.g. belief tasks), $c$ is an instance of a task in that corpus, $r$ is an ECD round, and $p_c$ is the found (minimal) program that solves task $c$. So calculate the description length of a corpus at a given round, for each task in the corpus $c in C$ we sum the DL of each program $p_c$ that solves it, given under that round's library $L_r$. And we add the description length of the library $L_r$ itself, a structure term to account for the cost of a more complicated library. 

Under the base library, total description length (black line) of the entire task corpus (every $C$) is 3209 nats. After four ECD rounds it drops to 2076 nats, an 1134 nat compression. We see the biggest drop in the belief corpus, once the system learns to efficiently represent intentional agents using the `fork(derive, commit)` abstraction. Once that constructor is in the library, every belief task rewrites through it and the corpus gets dramatically shorter at once. 

#figure(
  image("corpus_dl_1.png", width: 100%),
  caption: [Description length per task family (atomic primitives)]
)

Unlike the belief corpus, the description length of the other (non-mental) tasks remains mostly stable as new abstractions are learned. They're solved just as easily under the base primitives as they are under the final library (a bit worse actually since the library is more complex). If the initial primitives _did_ encode mental notions, if there were a native ToMM, we'd see similar results for the belief tasks: they'd be solved just as easily from the start. But since iterated abstraction yields a significant improvement, it's clear that the initial primitives are not mental but that the ECD loop enriches the library with abstractions that _do_ encode mental concepts (the concept of belief attribution). 

#figure(
  image("corpus_dl_2.png", width: 100%),
  caption: [Description length per task family (decomposed primitives)]
)

Belief tasks are reached late and cheaply. They only become solvable once the ECD loop has abstracted the reusable parts, and the moment that abstraction is added to the library, belief tasks flip from timing out to being solved in seconds. See the following plots of cumulative task solutions per round, and per-task solve time. 

#figure(
  image("solve_dynamics_1.png", width: 100%),
  caption: [Cumulative solves by task family (atomic primitives)]
)

Tasks other than belief get solved in the first round, they don't need abstractions. But belief can't be enumerated from base primitives, so it exhibits an S-curve. There are very few solves early on, since each solution must be laboriously composed of deep nested primitives. Then there's a sharp step up when the stitch finally abstracts the agency signature, the shared-model structure belief reuses. Then the learned abstraction is used to solve the remaining belief tasks, so the solved tasks plataeu for the last round. 

#figure(
  image("solve_dynamics_2.png", width: 100%),
  caption: [Cumulative solves by task family (decomposed primitives)]
)

Starting with the atomic library, the abstraction jump occurs at round 2 where we abruptly see $+38$ belief solves. If we start with the decomposed library, the abstraction takes an extra round to assemble so the step is at round 3 where we see $+30$ belief solves. 


*Plot 2*: per-task solve time, round n vs round n+1, as a paired slope-graph or log-scale scatter — the 1100 s → 10 s collapse is currently buried in log text.

=== Per-variant belief coverage + combo

*Plot 1:* Bar chart of solved/total for wall/witness/goal/dual



*Plot 2*: heatmap of (gv,av) $times$ abstraction-used showing fn_8 tiling every cell.

Rows are the (goal value, agent value) combinations in a task, columns are the invented abstractions added to the library. The count indicates how many solutions use that abstraction. 

INSERT FIGURE
// #figure(
//   image("agent_tiling.png", width: 100%),
//   caption: [Usage of each abstraction across instances of belief tasks]
// )

So the belief abstraction generalizes across the (goal, agent) combinations it was trained on.

=== Behavioral probe: Passing the false-belief test

The analyses so far are about description length, showing that the intensional program is the shorter account of the data thanks to the abstraction that makes it easier to express more complex programs. We now show that the system passes the false belief test, reproducing the classic Sally-Anne false-belief setup @wimmer_beliefsabout_nodate. The test simply asks where the system predicts the agent will look. An object is where the agent last saw it, but the agent's belief about its location is now false; a child who attributes belief points to the believed location, while a younger child who cannot yet attribute belief points to the object's true location.

We use the goal-displacement family of tasks for this. These tasks depict a value's trajectory that changes direction 90$degree$

#task-figure("belief_goal")

, because unlike the wall and witness families it lets the two readings come apart behaviorally. In a wall-belief scene the extensional rival posits a transient wall in the actual world rather than positing a wall in the agent's private believed world. That rival reproduces the entire trajectory, which is why the MDL margin, not behavior, is what separates them there. But in a goal-displacement scene the agent walks to where it believes the goal is (one cell from the true goal, which never moves) so a program that does not represent the agent's belief cannot reproduce the walk. The two readings predict the agent settling on different cells, which is the false-belief test. 

The probe reuses the phase 1 run and re-stitches its solutions to rebuild exactly the library of abstractions it converged to (this is the same reconstruction that the MDL-margin experiment performs). Onto a held-out scene (drawn from a fresh seed that never entered the run, and checked to be absent from the training corpus) we instantiate the discovered belief compound (the `fork(derive, commit)` abstraction the library reprices belief through) and, as its foil, the shortest program expressible in the same library. @fig-behavioral-probe shows one such scene.

#figure(
  image("behavioral_probe.png", width: 100%),
  caption: [Behavioral probe on a held-out Sally-Anne scene. The learned belief compound sends the agent to the believed cell (passes); the shortest non-mental program under the same library sends it to the true goal (the naive answer).]
) <fig-behavioral-probe>

With `fn_1` as the abstraction

```lisp
(compose
  (optimize (neg_dist #1) #0)
  #2)```

The learner equipped with the belief compound writes the program

```lisp
(fork
  (fn_1 1 2 (step 2 down)) 
  (sync_to_world 1))```

Which is normalized as

```lisp
(fork
  (compose
    (optimize (neg_dist 2) 1)
     (step 2 down))
  (sync_to_world 1))```

the discovered `fork(derive, commit)` abstraction applied to a derive that privately displaces the goal and seeks it, committing only the agent's own move back to the world via `sync_to_world`. Its final frame puts the agent on the believed cell, one step past the true goal. It searches where the agent thinks the goal is. The learner restricted to the non-mental fragment writes the shortest thing that fragment can say, plain desire `(optimize (neg_dist 2) 1)`, and its agent walks straight to the true goal, where the goal really is. Across all 24 held-out scenes the belief compound lands on the believed cell every time, the shortest non-mental program on the true cell every time.

Note that here the non-mental program is shorter than the intensional one (7.9 nats against 20.5), which is the opposite of the MDL-margin result, and this is the point. Where the transient-wall rival reproduces the scene, it does so only at a longer description length, so MDL rules it out. Where the pure-desire rival is cheaper, it does so only by predicting the wrong behavior, so the data rule it out. There is no non-mental program that is both as short and as accurate as the belief compound. The naive reading is available and parsimonious but it fails the test. A learner that has compressed its experience into a belief compound passes the false-belief task. a learner confined to the non-mental fragment fails by predicting the true location rather than the believed one. The abstraction the loop discovers for the sake of compression what licenses the mentalistic prediction.

== Ablation experiments

=== Silo vs joint stitch

compress belief solutions alone vs pooled with the non-mental corpus. Is the constructor still found, and at what DL?

*Plot*: library-DL and constructor-invention bar pairs.

=== Corpus-mix dose-response

vary the non-mental fraction (0%, 25%, 50%, …); measure constructor invention and belief solve rate. Shows the non-belief tasks are the scaffolding that gets us to abstract belief

*Plot*: solve rate / invention frequency vs mix.

=== Budget dose-response

--t-fn at 600/1200/2400 s; belief solve rate per variant. Supports argument that misses are budget-bound, not due to representational limitation

*Plot*: solve rate vs budget, per variant (dual should be the steepest riser).

=== Dreaming ablation

--no-dream vs dreamed Q: cumulative solves vs wall-clock. Round 3-vs-4 in the log already hints the dreamed Q matters.

*plot* a curve that shows Q bias contribution

=== curriculum ablation

remove certain tasks, show that low-level abstractions don't get abstracted and then that prevents us from reaching high level abstractions

=== Seed robustness

5 corpus seeds $times$ the main run; report every verdict metric as k/5 with solve-rate error bars

== Discussion and rebuttals


question: to what extent is the ToM already present in the input data and DSL and structure

potential argument: `fork` and `sync_to_world` are just decomposed `believes`. By your definition you'be we've been using from the literature, ToM/mentalizing is to be able to attribute a belief to an agent. A belief is an intensional state, a grid representation that may differ from the world. So `fork(derive, commit)` is already belief attribution, you've just broken it apart into components and now they're put back together.

Responses: 
1. `fork` and `sync` have non-mental uses, and they're often used independently of each other to solve other things, so they're general-purpose
2. there are also lots of other primitives that are the duals (cube axis), and those don't get selected

cube of symmetric complements, and general pair interface (`overlay`, `registration`) that stops `fork`/`sync` from being a disguised `believe`. 

potential argument: the `fork` primitive smuggles in ToM because creating a copy of the world grid is a representational capacity

response: ToM isn't just the representational capacity. creating a copy of the world that differs from it is just counterfactual reasoning, the what-if ability. it's super domain-general. infants have been shown to have it, disjunctive reasoning. The ToM is an abstraction that _uses_ this domain-general counterfactual representaion capacity to attribute a counterfactual world to a particular agent - that this modified world is _that agent's_ world. 

potential argument: `optimize` is basically ToM because BFS navigation requires the agent to recognize a 

out of the full 2×2×… cube of channel-direction / channel-target / z-order / scope variants, joint MDL still selects exactly the read-model→write-world, single-av corner for belief, and leaves the symmetric variants attached to their non-mental tasks.

pair apparatus was available to every task, but recruited by only the false belief ones. Those are the trajectories that can't be accounted for by only a single grid, so the search pays +~10 nats for the second grid only when nothing cheaper explains the data. Representationalist's criterion: posit the hidden world only if it pays for itself, only if its necessary. Availability is not use. The inference "this agent needs a belief" is made, per task, by cost-driven model selection. 

The `(grid,grid)` and arrow types aren't smuggled-in ToM, they're just the logical form that ToM requires. Intensionality is the coexistence of world and model. Any substrate expressive enough to even state "act on the world via a transformed model of it" must be able to hold two representations at once. And "two things held at once" is a product, in some guise (a pair, two registers, two tape regions, two variables). You cannot discover ToM in a substrate that can't hold two representations [for the same reason you can't discover addition in a language with no notion of "two numbers." The pair is to theory of mind what the integers are to arithmetic.]
And the pair is symmetric and content-free: `dup` makes `(w, w)`. There's nothing in it that says "the second one is the model, the believed world." The asymmetry that makes one copy real and the other believed is: transform the second (`mapsnd`, not `mapfst`), commit from model to world (`sync_to_world`). This asymmetry is entirely in the composition, which is the discovered part. The medium is neutral, it could go any way, as seen in the other tasks which do use those other primitives and wire them in other ways. But the system is able to discover this particular composition of these particular primitives to solve belief tasks, via programs that constitute a theory of mind.

Sure you can keep going. Decompose `dup` from a flat memory/tape with allocate/read/write, so "hold a second world-model" becomes a discovered pattern of buffer use. But be clear-eyed: that relocates the regress, it doesn't end it. "Allocate a buffer" is then the primitive, and it's no more or less mental than "form a pair" — the capacity-for-plurality is presupposed by intensionality at every level, because it's constitutive of intensionality. There is no substrate that discovers ToM "from nothing"; there's only substrates whose primitives are, or aren't, individually non-mental and general.

generality by reuse is what shows it
1. the same combinators used to create the belief abstraction are used in non-mental tasks
2. the library is also initialized with symmetric primitives to the ones used in the belief abstraction

so it's a domain-general library that the search happened to recruit for belief. 

#load-bib(read("chapter1.bib") + read("chapter2.bib"))