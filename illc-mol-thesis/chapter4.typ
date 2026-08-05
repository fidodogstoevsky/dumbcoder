#import "@preview/illc-mol-thesis:0.2.0": *

#import "viz.typ": task-figure, scenes-figure, all-tasks

#set heading(numbering: "1.")

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

#show table.cell.where(y: 0): set text(style: "normal", weight: "bold")
#show table: set par(justify: false)
#set table(stroke: (_, y) => if y == 1 { (top: 0.9pt) } else if y > 1 { (top: 0.2pt) })

#mol-chapter("The Model")

== The learning problem <problem>

In our model the learner is an observer. It is shown scenes, which are visual episodes that unfold over discrete time steps on a small grid. For each set of associated scenes the learner must recover the underlying process that generated what it saw, by positing candidate hypotheses until it finds one that works. Hypotheses are programs, so a hypothesis is correct only when executing it reproduces the entire sequence of frames it was given. Concepts are primitives, so a concept is learned when the learner enriches its library of primitives with a new abstraction. We ask whether such a learner, shown scenes whose shortest explanation requires belief attribution, comes to learn the attribution structure as a new abstraction. 

== The task corpus <corpus>

We created a corpus of sets of grid-based tasks. Most tasks are _trajectory tasks_. A trajectory is a sequence of 2d grids over the time dimension, that depicts some dynamic at play (e.g. a body that falls, an agent that approaches something, etc.). A task consists of $k$ trajectories that all depict the same dynamic but vary in setup (where the entities start, how long the scene takes, what else is lying on the grid, etc). Given a task, the learner's goal is to find a transition function that, for any trajectory in the task, returns the next grid for any grid in the sequence. So the transition function should be a generalized explanation of the underlying dynamic. 

For example, the task below consists of four scenes that each depict the value 4 moving up. The grids of each timestep are collapsed into one, so each arrow indicates one timestep forward. A solution to the below task is a program that encodes "at each time step value 4 moves one step up". Such a program solves the task because for each scene, given that scene's starting positions, the program reproduces the rest of the trajectory. 

#figure(
scenes-figure("physics"),
kind: image,
caption: [an instance of a directional movement task]
)

The following task is solved by a program that encodes "at each time step, value 1 moves one step closer to value 2".

#figure(
scenes-figure("desire"),
kind: image,
caption: [an instance of a goal-seeking task]
)

Both of these, and most of the corpus, can be solved by programs that describe the scene extensionally via a purely behaviourist visual description. The task families we care about are those that can only be solved by attributing a belief to an agent.

Below is an instance of a _goal-displacement_ task, inspired by the false-belief task @wimmer_beliefsabout_nodate. The dark blue value (1) comes to rest on a cell a fixed step away from the green value (2), rather than on it (the teal 7 is an inert bystander, whose role @app-generators explains). A solution to this task is a program that encodes "at each time step value 1 moves one step closer to value 2, given that value 2 is two steps above its actual location". So solving this task requires attributing to an agent a model that deviates from the actual one.

#figure(
  scenes-figure("belief_goal"),
  kind: image,
  caption: [An instance of a goal displacement task],
) <fig-belief-goal>

Another task family that requires attribution to solve are the _false wall_ tasks. The dark blue value (1) bends around a cell it wrongly believes is blocked, on its way to the goal (green). The believed wall is nowhere in the picture: it is private to the agent, and no frame ever renders it --- a real wall would be a dark grey 3. The cell the agent bends around is instead occupied, for the whole trajectory, by an inert bystander (the purple 5). The bystander is what keeps the belief attributable at all: without it, a program could stamp a real wall into the world, walk the agent a step, and erase the wall again before the frame is rendered, reproducing every frame while attributing nothing. With the bystander in place, that _transient-wall_ spelling overwrites an object the frames show, and nothing in the language can put it back (@app-generators).

#figure(
  scenes-figure("belief_wall"),
  kind: image,
  caption: [An instance of a false wall task],
) <fig-belief-wall>

Three further belief families vary the attribution rather than its content. In a _witness_ task, the false-wall scene gains a second agent that is not misled: it seeks a goal of its own and walks straight across the cell the first agent bends around, which no real wall could permit. In a _two-observer_ task, two agents seek the same goal and only one of them is wrong about where it is, so a solution must attribute a belief to one mover and withhold it from the other. And in a _false-obstacle_ task the agent is wrong about two things at once, detouring around a phantom wall on its way to a displaced goal, while a real wall stands elsewhere in the scene --- so a solution must hold apart, in one program, a wall that exists and a wall that is merely believed. Together with goal displacement these are the five belief families of the corpus; @tab-families gives the program behind each, and @criteria says which family carries which criterion.

A second, smaller group are _template tasks_. Where a trajectory task's input is a single initial grid, a template task's input is a pair of grids: a _working_ grid and a _template_ grid, both $s times s$. The task is to produce the working grid modified according to the pattern given in the template. This format is loosely inspired by the ARC-AGI-1 corpus [CITE].

#task-figure("registration", caption: none)

// We include the template families because they populate the pair interface (@types) from the non-mental side. A program that consumes a (working, template) pair with `sync_to_world` is performing image registration, not holding a belief, and it is doing so with exactly the primitives that belief uses (@interpreter).

We designed our corpus to exercise a variety of domains. See @tab-families in the appendix for the full list of task families in the corpus.

// The first is $k$ itself. The likelihood of @sec-bayes eliminates rather than ranks, so requiring $k$ scenes of one latent program conjoins $k$ eliminations: a rival that fit one scene by luck is removed rather than down-weighted. That is most of the work. Every task in the goal-displacement family admits a spelling in which the agent's bend is explained by a phantom wall rather than a displaced goal --- for _some single scene_ of the task, and for none of them across all four, since a single wall coordinate cannot follow a believed goal that sits somewhere different in each scene.

// The second and third are conditions on the scene, which $k$ does not deliver: the falsely believed wall must lie on the route the agent would otherwise take, so that the belief is doing work; and the scene must make a world-level stamp of that wall detectable, so that "believes a wall is there" is distinguishable from "a real wall stood there and was taken down before the frame was rendered". Both are enforced by the generators, and @app-generators states them precisely, together with the handful of rivals that have to be excluded by argument rather than by the scenes because they are extensional identities and no number of scenes separates them.

// Template tasks are evaluated against a second interpreter, which threads a working grid while pairing each frame with a constant external template and applying a commit policy. Here the second channel is a given input rather than a privately derived model, so a program that consumes the pair is aligning two images, not holding a belief. The same vocabulary therefore does mental work in one interpreter and non-mental work in the other, which is what lets the corpus price belief against genuinely non-mental competitors built from the very same parts.

== The hypothesis space <space>

=== Programs as typed expression trees

A hypothesis is an expression tree over the library of primitives. Each node carries a type together with the types of the arguments it expects. A node with no arguments is a terminal whose value is itself, like `up` of type $D$ or the literal `4` of type $V$; a node with arguments is an operator, like `step` of type $V times D arrow.r (G arrow.r G)$, applied to its filled-in arguments. The tree is directly executable, and evaluating it bottom-up yields the object its root type names. So `(step 4 up)` evaluates its two leaves to the cell value `4` and the direction `(-1,0)` and applies `step` to obtain a transition function in $G arrow.r G$.

A program's syntactic size is its node count: one for a leaf, one plus its arguments for an operator. An abstraction used as a token adds just one, because the subtree it abbreviates lives inside the body and does not appear in the program that uses it. An abstraction converts many charged nodes into one. (See @app-delta for the representation that implements expression trees). 

=== Type system <types>

$G$ is the type of 2d grids of cell values, the frames of a scene. Write $G^*$ for stacks of grids, i.e. a full scene composed of a sequence of grids. So a transition function is then an element of $G arrow.r G$. The cell values are members of the ground set $V = {0, dots, 9}$, where each is an entity on the grid (e.g. the value $4$ moving up each time step). The grid coordinates are $C = {0, dots, s-1}$, a row or column index, from which a _position_ in $C times C$ is built. The remaining sets are $D$, the four movement directions, and $U subset.eq (C times C arrow.r RR)$, the positional utilities that the `optimize` primitive (@primitives) scores.

Everything else the language names is an arrow between these. The arrows relevant to our argument are the ones into and out of the product $G times G$, which serves as the (world, model) channel pair:

$
& G arrow.r G times G && quad "build a pair from a grid" \
& G times G arrow.r G times G && quad "transform a pair" \
& G times G arrow.r G && quad "collapse a pair back to a grid" \
& G arrow.r C times C && quad "read a position off a grid" \
& G times (C times C) arrow.r G && quad "impose a position on a grid" \
$

So there are four jobs to be done with a pair: building a pair, transforming a pair, collapsing a pair, and reading and writing the positions that give a collapse something to go off of. These operations are performed by the product primitives (@primitives).

Whether a program opens a private second channel is a matter of which of these arrows its nodes inhabit, so a (world, model) representation is a structural property of the program. The fixed product does carry the commitment that a learner holding a private representation holds exactly one (generalizing this is left as further work, see @sec-arity).

A program is checked by running it through an interpreter (@app-interpreter). The solution to a trajectory task (@corpus) is a program of type $G arrow G$, a transition function that given the task's initial grid at step $i$ should return the next grid at step $i+1$. So the interpreter iterates the program $G arrow G$ on the task's initial grid $G$, yielding the stack of grids $G^*$ which can be compared to the target scene. So the interpreter's state at any step is just a single grid $G$, not a (world, model) pair $G times G$, and certainly not a registry of agents, goals, or entities. No pair is handed from one step to the next: a program that opens a second channel (a model channel that may diverge from the world) must build it, use it, and collapse it again inside that step. 

A template task starts from a pair in $G times G$, a working grid and a constant external template, so its solution is a _commit_ $G times G arrow.r G$ (which is one of the collapsing arrows above). The commit answers what each frame should do with the template. In a template task the pair is given rather than built by the program, which is the one case in which a program is handed a second channel it did not open itself. Since a type-directed enumeration yields programs of one type only, trajectory tasks ($G arrow G$-rooted) and template tasks ($G times G arrow G$-rooted) are searched separately (@enumeration). Though compression pools the solutions of both (@compression).

== The primitive library <primitives>

The learner starts with a library of primitive operations, listed in full in @app-primitives.

Some of the primitives are for assembling transition functions, e.g. `step`, of type $V times D arrow.r (G arrow.r G)$, moves every cell of a given value one step in a given direction. `optimize`, of type $U times V arrow.r (G arrow.r G)$, moves a value one step towards or away from a value according to the given utility. There are two utilities to give it, both of type $V arrow.r U$: `neg_dist`, minus the distance to the nearest cell of a target value, and `distance`, plus that distance, so that maximising the first approaches a target and maximising the second flees it. Three primitives edit the grid outright: `wall_at` stamps a wall value (3) at a position, `clear_at` clears a single cell, both are of type $C times C arrow.r (G arrow.r G)$. And `erase` is of type $V arrow.r (G arrow.r G)$, it removes every cell of a value. Finally `compose` runs one transition function then another, so it's of type $(G arrow.r G) arrow.r (G arrow.r G) arrow.r (G arrow.r G)$. The terminals are the four directions, the cell values $0 dots 9$ of type $V$, and the coordinates `c0 … c4` of type $C$, a pair of which names a position.

Since we're interested in how transition functions $G arrow.r G$ interact with (world, model) pairs $G times G$, the rest of our primitives are the structure of the product. They are defined for grids $w, m, g in G$, transition functions $f, f' in G arrow.r G$, values $v in V$ and positions $p in C times C$. There are four kinds, one for each arrow of @types: those that build a pair out of a grid, those that transform a pair, those that collapse a pair back to a grid, and those that read and write a position.

We have two ways of _building_ a pair, both of type $G arrow.r G times G$. The diagonal $Delta(w) = (w,w)$, which the library calls `dup`, copies a grid into both channels. Its complement `pair_blank`, $w mapsto (w, bold(0))$, pairs the grid with an empty one instead, so the second channel starts as scratch rather than as a copy of the world.

We have four ways of _transforming_ a pair, all of type $G times G arrow.r G times G$. Three are the functorial actions, which lift transition functions to act on the channels: `mapsnd` acts on the second factor, $(id times f)(w,m) = (w,f(m))$, `mapfst` on the first, $(f times id)(w,m) = (f(w),m)$, and `bimap` on both at once with one function for each, $(f times f')(w,m) = (f(w),f'(m))$. The fourth is the twist $(w,m) mapsto (m,w)$, `swap`, which exchanges the channels, so anything defined on one factor can be conjugated into a version acting on the other.

We have several ways of _collapsing_ a pair to a single grid, of type $G times G arrow.r G$. The projections `fst` and `snd` return one channel and discard the other, $(w,m) mapsto w$ and $(w,m) mapsto m$. `overlay` and `underlay` union the two channels, differing only in which one wins where they disagree. `sync_all` moves every value the two channels share to the position it holds in the second, and `sync_except v` does the same for every shared value but $v$. These two are _scope_ collapses: each adopts the second channel's account of where things stand, and they differ only in how much of it they adopt --- all of it, or all of it minus one named value. `sync_except` thus publishes by blacklist: the value it names is the one it holds _back_, and everything unnamed goes through. The remaining point on that scale --- adopting the second channel's account of exactly one named value, a whitelist of one --- is deliberately not on the list, and assembling it is what the accessors below are for.

Finally we have the accessors on positions. `locate`, of type $V arrow.r (G arrow.r C times C)$, gives $"loc"_v (g)$, the position of $v$ in $g$; `place`, of type $V arrow.r (G times (C times C) arrow.r G)$, gives $"put"_v (g,p)$, the grid $g$ with $v$ moved to $p$. A reader and a writer are joined by `register`, of type $(G arrow.r C times C) times (G times (C times C) arrow.r G) arrow.r (G times G arrow.r G)$, which reads a position off the second channel and imposes it on the first: $"register"("loc", "put")(w,m) = "put"(w, "loc"(m))$. So `register (locate v) (place v)` is the collapse that moves the single value $v$ to wherever the second channel puts it, and it is a collapse the learner has to assemble from three parts rather than one it is handed. Why the whole-scope collapses are handed over in one token and this one is not is taken up in @granularity.

Each of these arrows needs its own composer, since `compose` joins two transition functions only, it's of type $(G arrow G) arrow (G arrow G) arrow (G arrow G)$. So `compose_gp`, of type $(G arrow.r G times G) times (G times G arrow.r G times G) arrow.r (G arrow.r G times G)$, extends a builder by a transform. `pipe_gpg`, of type $(G arrow.r G times G) times (G times G arrow.r G) arrow.r (G arrow.r G)$, follows a builder with a collapse, which is the composer that returns to $G arrow.r G$. The interpreter only ever iterates a transition function, so a program that opens a second channel must close it again within the same time step. `compose_pg`, of type $(G times G arrow.r G times G) times (G times G arrow.r G) arrow.r (G times G arrow.r G)$, follows a transform with a collapse, and is the only one that acts on a pair the program did not itself build, which is what a template task's given pair calls for.

The primitives are just the product structure, so nothing in them distinguishes a world from a model. The pair is symmetric: both factors are grids, every structure map is present in both directions, and no primitive marks either channel as anyone's representation. Which channel is the world and which is a belief is settled only by how a program assembles these pieces together, which is found by search. 

=== The target compound <signature>

@sec-false-belief wrote belief attribution as a two-step computation: derive the counterfactual model from the observer's own, $m' = "derive"(m)$, then run the planner on it unchanged. Transcribed into this setting the two steps keep their names. The observer must first _derive_ a private grid that differs from the world in whatever the agent is wrong about, then _run_ the agent's movement policy in the private grid, so that its move is optimal with respect to its own belief rather than to the real world. But this architecture adds a third step with no counterpart in Chapter 3: _commit_ the result back to the world, publishing where the agent went and nothing else. The commit is owed to the interpreter rather than to the theory --- it renders one grid per step, so motion on a private grid doesn't affect the rendered grid unless an agent's motion is explicitly committed from it.

Only the _run_ step is done by a primitive, namely `optimize` which builds the agent's movement policy. Nothing in the library opens a private grid, and nothing publishes a single value's position back out of one. Both have to be assembled from the product structure above. So for _derive_ and _commit_, we have two target structures:

$
"fork"("derive", "commit") &= "commit" compose ("id" times "derive") compose Delta \
"sync"_v &= "put"_v compose ("id" times "loc"_v) \
$

$"fork"$ is the derive-and-commit frame, taking a $"derive" in G arrow.r G$ and a $"commit" in G times G arrow.r G$. It copies the world into a pair, applies the derive to the second channel to get the private grid, and hands the pair to the commit, which collapses it back to one grid; the private grid exists only for the duration of that call. $"sync"_v$ is the commit belief needs: it reads where $v$ sits in the private grid and moves the $v$ in the world to that position, so exactly one value's position is published and everything else stands as the world had it.

Belief is the composition of fork and sync, with the derive doing the agent's error and the commit doing the attribution. Taking a phantom wall as the error, it is $"fork"(pi_a compose "wall"_(r,c), "sync"_a)$, which unfolds to the map

$
w mapsto "put"_a (w, "loc"_a (pi_a ("wall"_(r,c) (w))))
$

for an agent value $a$, a seek policy $pi_a$, and a wall stamped at row $r$ and column $c$, the three steps in order are: copy the world and put a wall on that copy, plan based on the copy of the world which has a wall in it, then finally write to the rendered world only where the agent ended up.

Note that the agent value $a$ occupies three slots in that term: the movement policy $pi_a$ that plans over the privately walled grid, the read $"loc"_a$ that finds where the agent ended up, and the write $"put"_a$ that publishes it. So to find this compound, the enumerator needs to independently fill the argument slots of the three primitives with the same value placeholder.

=== Where the decomposition stops <granularity>

Even twelve-month-old infants encode an action as directed at a goal state, and expect an agent to use the most efficient means available to it @timeline. They employ the teleological stance, which relates an action, a goal state, and the constraints of the situation but without attributing any representation to agents @gergely_taking_1995 @gergely_teleological_2003. Under the teleological stance, the reality constraints that the observer consults are their own, not the agent's.  

We return to this distinction in @rationality, when we defend our choice to hand our model learner a primitive planner `optimize`, which moves a value one step along the best available path toward improving some utility, even though the planner is algorithmically complex. From the developmental literature we know that efficient goal-directed interpretation is in place around a year of age, three years before anything recognizable as false-belief attribution (@timeline). And on Gergely and Csibra's analysis it isn't yet ToM at all, it just says how a body moves given a goal and a layout.

The library of @primitives cuts the language at conspicuously uneven depths. The commit that publishes one value must be assembled from a reader, a writer and a joiner, while `sync_all`, which resolves an entire pair, is a single token; a reader is entitled to ask why the knife fell where it did, since a granularity chosen freely is a granularity that could have been chosen to flatter the result. Two rules fix it, and neither leaves room for tuning.

The first rule covers the builders, the transforms and the projections: they are the canonical structure maps of the product $G times G$, and the kit is complete rather than curated. The diagonal, the pairing with the empty grid, the two functorial actions and their joint form, the twist and the two projections are what having a product _is_, and each map comes with its complement --- `dup` with `pair_blank`, `mapsnd` with `mapfst`, `fst` with `snd` --- so no route through the pair is cheaper than its mirror image. At this layer there is nothing to decompose, and nothing was chosen.

The second rule covers the collapses, where choices did have to be made, and it can be stated in one sentence: _the decomposition reaches exactly as far as the single-value vocabulary of `locate` and `place` can spell, and what that vocabulary cannot spell stays atomic._ The commit belief needs --- publish one named value's position out of the private channel --- has a spelling in that vocabulary, so it is withheld as a token and the learner must build `register (locate v) (place v)` out of parts that each do non-mental work elsewhere in the corpus. The scope collapses have no such spelling: `sync_all` and `sync_except` fold over _every_ value the two channels share, a set fixed by the scene rather than by the program, and writing them out would require granting iteration over $V$ --- a strictly more powerful construct than anything the language contains, and a far less innocent gift than the two collapses it would be used to spell. `overlay` and `underlay`, which union whole grids cell by cell, are atomic on the same grounds. So the rule is not "decompose everything"; it is that the compound under test must be assembled from parts, while its neighbours may be bought whole.

That policy has a consequence that should be stated here rather than discovered in the results: it prices the boundary belief draws asymmetrically. A commit can name what it publishes or name what it withholds --- the whitelist `register (locate v) (place v)` and the blacklist `sync_except v` draw the same kind of line from opposite sides --- and the blacklist costs one node where the whitelist costs three. Nor are the two always distinguishable extensionally. On a scene where the agent is the only shared value the derive has moved, `sync_all` denotes exactly $"sync"_a$; where the derive has also displaced one shared value, a believed-in goal, `sync_except` of that value does. Wherever that coincidence holds, the cheap spelling and the assembled one compute the same collapse, and a description-length learner should be expected to buy the cheap one. The asymmetry does not bias the search toward belief --- the atomically priced scope collapses are the commits of non-mental families (`multi_registration` and `registration_except` in @tab-families), and within the mental families it taxes precisely the whitelist form of the agency signature --- but it does mean that the _spelling_ of a discovered commit is a prediction of the price list rather than a free observation. Chapter 5 reads it as such (@sec-selective), with the control run of @sec-atomic-run, in which both spellings cost one token, separating what the price explains from what the denotation does.

== Inference <inference>

Our procedure runs in alternating rounds of exploration and compression, the loop of @dechter_bootstrap_nodate as it is realised in DreamCoder @ellis_dreamcoder_2020. The exploration phase enumerates programs for the tasks not yet solved, under the current library. The compression phase revises the library to compress the programs found. We follow @ellis_dreamcoder_2020 in the revision step --- the objective is the joint description length of library and corpus, and what is invented is a $lambda$-abstraction rather than a reused subexpression --- and depart from it in three ways: the abstraction search is delegated to the top-down Stitch @bowers_top-down_2023 rather than run bottom-up (@compression); the likelihood is evaluated against $k$ observations at once rather than one (@corpus); and no recognition model is trained, so there is no third phase (@no-amortization). The two phases divide the inference of Chapter 2 between them: exploration performs the per-task inference of @sec-bayes, compression the revision of the language of @sec-hbm.

=== Search <enumeration>

The exploration phase finds a hypothesis that correctly explains a scene, by enumerating candidate programs in decreasing order of prior probability and checking if the program produces the scene. A hypothesis is a program (@sec-programs), specifically a typed expression tree over the current library (@space). Since enumeration in decreasing prior probability is enumeration in increasing description length, the search walks outward in expanding bands of cost (expanding $tilde e^ell$, see @sec-search). The first program it finds that reproduces the data is the shortest one that does, which is the maximum a posteriori hypothesis ($h^*$, see @sec-programs). The enumeration runs under a time budget, and a hypothesis whose cost band is not reached before the budget expires is simply not found.

The library fixes the prior. The prior $P(h) prop e^(-"DL"(h))$ (see @sec-programs) is realised as a distribution over the library's symbols, under which a program's cost is the summed $-log p$ of its nodes. Symbols are grouped by return type, and a symbol of type $tau$ is priced at $-ln$ of the number of symbols sharing $tau$, i.e. a type-uniform distribution#footnote[One exception to the uniform distribution is that a cell value which is visible in the task's first frame costs nothing, since naming what is already on display is not a hypothesis about anything. But a placed wall is full-price, since it's positing something that isn't there]. So the types give the branching factor at each argument slot, and the cost of each choice at that slot. Enumeration is type-directed: an argument slot draws only from symbols whose type matches, so no ill-typed tree is ever built.

// This equivalence between probability and code length is what the rest of the thesis measures things in, and it's worth naming. Solomonoff's theory of inductive inference is its ancestor: hypotheses are programs, and each gets prior weight falling exponentially in its length @solomonoff_formal_1964. But that weighting is over programs for a universal machine, so the resulting prior is uncomputable and a learner cannot use it. What a learner can use is the same idea relativized to a particular language, which is the minimum description length principle: inductive inference as the search for the encoding that describes the data in the fewest bits @rissanen_modeling_1978 @grunwald_minimum_2007. Because our likelihood is an indicator, the encoding of the data given the hypothesis costs nothing beyond the hypothesis itself, and MDL collapses into "shortest correct program". It doesn't stay collapsed. When @compression asks whether the learner should adopt a new primitive, the thing being encoded is no longer one trajectory but a whole corpus of solutions, and the code has two parts to pay for — the vocabulary and the corpus written in it — which is MDL in its usual two-part form. So the same principle stated here for a single hypothesis is what will later license changing the language. All quantities in what follows are reported in nats, i.e. description lengths under natural logarithms.

The budget is what makes the double role of the prior, argued for abstractly at the end of @sec-search, into something concrete. There the point was that a long hypothesis is not merely improbable but unreachable; here that is a fact about a clock. The two readings of description length cannot come apart, because in this procedure making a hypothesis more probable and making it cheaper to find are the same operation. And that points at the only remedy available: not to enlarge the budget, but to shorten the target, which is what @compression does.

Two details are specific to this setting. First, the success criterion, which is where the indicator likelihood $bb(1)[h "solves" d]$ of @sec-bayes is cashed out as an execution: a candidate solves a task only if the interpreter, started from each scene's own first frame, reproduces that scene's every frame exactly, for all $k$ scenes (@corpus). Second, each still-unsolved task is enumerated as its own cost-ordered stream rather than checked against one stream shared across the corpus, which is what lets the search be parallelized across tasks. The mechanics of cost-bounded search, including the pruning and the tie-break within a cost window, are given in @app-enumeration.

=== Library revision <compression>

The compression phase does the shortening, and it is here that the learned prior of @sec-hbm becomes an algorithm. The library is that prior made concrete: a set of named, reusable abstractions which, by fixing what symbols exist and how the probability mass is divided among them, is the language under which the next round's tasks are inferred. Naming is $lambda$-abstraction used as @sec-programs advertised it, so the learner comes to possess a term whose content is fixed entirely by what it does in combination with the others.

Where exploration minimises description length one hypothesis at a time, against a library held fixed, compression minimises the _total_ description length: the length of the library itself plus the summed cost of every solved program written in it. That quantity is @sec-hbm's posterior over languages in computable form @ellis_dreamcoder_2020 --- negative logarithms taken, so that maximising the posterior becomes minimising a two-part code @rissanen_modeling_1978, with the sum over every program the library can express replaced by the best solution actually found for each task. And it is what makes the revision inference rather than convenience: the vocabulary is adjusted by an objective, not by a modeller who finds the search going badly, which is the objection @sec-hbm raised against itself.

$
min_"library" ("DL"("library") + sum_"tasks" "DL"("program" | "library"))
$

Both terms matter, and the tension between them is what makes the choice non-trivial. Naming a recurring fragment collapses it, at each site where it is used, from a subtree to a single token, which shrinks the second term. But the library is now larger, and since a symbol's probability falls as the number of symbols competing with it rises, every program written in the enlarged library gets a little more expensive, which grows the first. An abstraction is kept only when the tokens it saves across the whole corpus outweigh the price of carrying it. Structure that recurs by accident is not abstracted; structure that recurs because it is genuinely shared is. The constraint doing the work is conservation of probability mass, now operating on the language rather than on the data: a library that could express anything cheaply would be a library in which nothing is cheap.

The objective is indifferent to which tasks the shared structure came from. It sees a pooled corpus of solutions and asks what would compress it, so an abstraction serving one family of problems must win against every abstraction that would serve any other family. This is also reason to expect the language to be better constrained than any single hypothesis written in it: a solution is answerable only to its own task, whereas a candidate abstraction is priced against every solution in the corpus at once, pooling evidence across tasks that are, from the point of view of any one of them, unrelated. A child who has watched many people pursue many different goals has seen very little evidence about what any one of them wants, but a great deal about the general shape of the relationship between wanting, seeing and doing. Two things about how the objective is optimised here matter to the argument.

The first is the search procedure. Where DreamCoder looks for abstractions bottom-up, enumerating refactorings of each found program and intersecting them, we delegate the search to the top-down Stitch @bowers_top-down_2023 library-learning system, which searches the space of abstractions directly and so never has to materialize every rewrite. This is a difference in tractability rather than in objective; the procedure and its utility bound are given in @app-stitch.

The second is the scope over which it runs. Search is divided by solution type (roots), but compression is *joint*: it pools the solutions of both types and revises one library over all of them, so an abstraction has to earn its place against the whole solved corpus rather than against its own share of it (@positive-differences).

What compression returns is a set of abstractions together with the corpus rewritten to use them. Each abstraction is registered as a new symbol, on the same footing as a primitive that was given --- what was a subtree found in round $N$ is an atom in round $N+1$ --- and the rewritten corpus becomes the starting point for the next round --- in which programs a level deeper, belief compounds too expensive to enumerate from bare primitives, have become reachable because their shared core now costs a single token. This is the cumulative structure @sec-hbm called bootstrapping: each round changes what the next round can find.


== The model does not encode a Bayesian theory of mind <no-btom>

The charge to answer is that Chapter 3's stipulations have been re-encoded in a different notation and the learner then congratulated for finding them: if the interpreter, the types, the library, the corpus or the prior already carries the world/model distinction or a notion of agency, a discovered belief abstraction shows nothing. The audit below takes the five in the order of @stipulated, and then two confounds of our own. Each item has the same shape: something is given, and the thing that would turn it into an attribution is withheld, so that the connection between them is left for the search to make.

=== The decomposition into agent, goal and belief <no-decomposition>

BToM hands agent, goal and belief to a planner as separately inferred arguments. Here there is no planner with argument slots to inherit that from, and no type denotes an agent, a goal or an attitude: there are grids, cell values, coordinates and arrows between them. A cell value is an integer between 0 and 9, and the corpus rotates through eight (agent, goal) pairs so that no integer is privileged --- an abstraction specialised to "value 1 is the agent" would fail to recur and would not be kept. Role is positional: the thing we call the agent is whatever value occupies the policy slot and the commit slot at once. That shape, `(fork <derive> <commit>)` with a value shared between a policy inside the derive and the commit outside it, is not a primitive but a compound the searcher assembles and compression decides to keep, so an invented abstraction binding one hole to both is a discovery of the structure rather than of a parameter inside a given one.

=== The separation of the believed world from the actual world <no-separation>

This is the stipulation that makes false belief representable at all, and where the model is most exposed: BToM evaluates the planner against a supplied world model $m'$ and permits $m eq.not m'$. What we give instead is a product type and general combinators that inhabit it. The pair is symmetric and content-free --- `dup` produces $(w, w)$, and nothing in it says the second copy is a _model_ or that anyone _holds_ it. Belief is a matter of wiring: `mapsnd` rather than `mapfst`, the direct commit rather than its conjugate through `swap`, `register (locate av) (place av)` rather than `sync_all`. Each of those choices has its opposite in the library, exercised by some non-mental family in @tab-families; the medium is neutral, and the asymmetry is made in program space by the search.

The interpreter does not supply it either. In BToM the divergent model is an input to a planner the observer runs _inside itself_; ours has one grid of state, no second channel to hand anybody, and no mode distinguishing evaluating a program from evaluating a program-as-an-agent-would. A private grid exists only because a program built it, and the interpreter cannot tell it from any other. Nor are `fork` and `sync_to_world` given as atoms --- that is the control run of @sec-atomic-run --- since the searcher here is handed `dup`, `mapsnd`, `compose_gp`, `pipe_gpg`, `locate`, `place` and `register`, product-category combinators no one would describe as mental.

A stronger form of the worry puts the concession one level down: the ability to build a copy of the world that differs from the world just _is_ the representational capacity theory of mind consists in. It is not. Building a world state that differs from the actual one is counterfactual reasoning --- the capacity to consider what would be the case if things were otherwise --- and it is domain-general, with a developmental record of its own well before false-belief attribution. Theory of mind is an abstraction that _uses_ that capacity, by attributing the divergent world to a particular agent, so that it is _that agent's_ world. That attribution is the conjunct the shared hole encodes, and it is exactly what the pair does not supply: `fork` will derive a private grid, compute over it, and publish an unrelated value, and the language will let you write that. What it will not do is tell you not to.

The regress does not end, and we do not claim otherwise. `dup` could itself be decomposed into reads and writes over a flat memory, so that holding a second world model becomes a discovered pattern of buffer use; but "allocate a buffer" is then the primitive, and no more mental than "form a pair", because intensionality just _is_ the coexistence of a world and a model of it, so any substrate expressive enough to state the structure can hold two things at once. The pair is to theory of mind roughly what the integers are to arithmetic: not a disguised form of the thing to be learned, but the medium without which it cannot be stated. There is no substrate in which theory of mind is discovered from nothing, only substrates whose primitives are, or are not, individually non-mental and general.

=== The space of admissible divergences <no-range>

BToM must also fix which divergences are admissible, and the scenarios that produce them. Here there is no range to fix: the derive slot of a fork accepts _any_ program of arrow $G arrow.r G$, so the belief families differ in what the belief is _about_ rather than in some parameter of a common schema --- a phantom wall is `wall_at r c`, a displaced goal is `step gv d`, being wrong about both at once is a sequence of the two. The goal-displacement family varies its content across four shove shapes, so an abstraction specialised to one of them would not recur and compression must keep the content subtree as a hole: the productivity and systematicity @explanandum demanded.

The concession is that what the learner finds is a divergent world model, not a belief that is _formed_ and could be _revised_. Nothing represents how the agent came to be wrong, and nothing updates when it looks again. In BToM that lack is a stipulation about admissible divergences; here it is a limit on what the corpus depicts and what a single-frame transition function can express, and the most important respect in which the discovered structure is thinner than the thing it is a model of (@sec-persistent).

=== The space of goals <no-goal-space>

BToM's goal space, and the prior on it, are given; Baker et al. compare several goal priors precisely because the choice matters and is the modeller's to make @baker_action_2009. Here a goal is whichever cell value happens to fill the argument of `neg_dist`, drawn from the same ten integer terminals that supply agent identifiers and step magnitudes. There is no goal type and no goal prior --- there is inference over programs, and a program seeks something because a symbol landed in that slot. Even the polarity is a search choice: `distance` sits alongside `neg_dist`, so maximising utility flees a value as readily as it approaches one, and the `flee` family exists to keep that corner inhabited. The only prior over any of it is the type-uniform distribution, which prices a goal value with the rule that prices `up` and `compose`.

=== Rationality, and the one thing we do grant <rationality>

Rationality is the one stipulation of BToM our learner also receives, in the form of the `optimize` primitive, on the grounds Chapter 3 gave: it is the only item on the list with independent developmental support, in place well before the first birthday and some three years before anything recognizable as false-belief attribution (@sec-planner, @timeline) @gergely_taking_1995, and on the best analysis of it not mentalistic, since the teleological stance relates an action, a goal state and the constraints of a situation without attributing any representation to the agent @gergely_teleological_2003.

Its signature makes the point formally. `optimize` has type $U times V arrow.r (G arrow.r G)$: it receives one grid, has no parameter identifying whose grid it is and no access to any other channel, and treats whatever grid it is handed as the layout. It cannot represent a divergence, because it cannot see two things to diverge, and it cannot attribute anything, because it has no argument for an attributee. Belief arises only when something _else_ hands it a grid that is not the world and then publishes only part of the result --- and that something else is the composition, which is what the learner has to find. Nor is the planner reserved for the mental families: `desire` uses it to walk to a goal, `flee` to run from a hazard, `comet` to bend a trail, and `obstacle` to detour around a real wall, on scenes whose rendered trajectories closely resemble the belief ones. Availability is not use.

The residual cost is real and we do not pretend otherwise: `optimize` packs a full breadth-first search into one token, and a more elementary treatment would decompose it into memory, conditionals and interaction with the environment (@sec-optimize). What we claim is only that granting a planner is not granting a theory of mind.

=== Two confounds that are ours rather than BToM's <confounds>

Two further items answer to nobody on Chapter 3's list; they are properties of how the experiment is run, and the two knobs a sceptical reader would reach for first.

*The success criterion admits no partial credit.* A candidate program either reproduces all $k$ scenes of a task cell for cell or it does not solve the task (@corpus, @sec-bayes). A graded likelihood is exactly where a thumb could rest on the scale: under partial credit a mental program that is nearly right could be preferred to an extensional rival that is exactly right, and the comparison of @criteria --- whether the mental program wins on description length _among programs that all reproduce the data_ --- would no longer be the one being made.

*Nothing in the loop is conditioned on what a scene looks like.* We omit DreamCoder's recognition model, for the reason @no-amortization gives at length: a proposal distribution trained on solved tasks has its prior over the library fit to the data, and belief is precisely the structure the corpus is meant to leave unfitted. Enumeration runs under the type-uniform prior over the current library and nothing else, so the only thing that can make a belief compound reachable is that earlier compression made it short --- the mechanism the ordering claim of @criteria depends on. The choice was settled by measurement rather than by design, and @sec-limits reports what the loop gave when it was run both ways.

=== Two structural differences worth naming <positive-differences>

*Inference here is not agent-directed.* BToM's posterior is over $(g, m)$ for an agent and has nowhere else to point; ours is over programs for a scene, and it is the _same_ search, under the same prior and the same success criterion, that solves `denoise` and `relocation`. Compression is undirected in the same way, which is why it is pooled rather than run per solution type (@compression): were it divided, the division would itself bias abstraction toward belief, since the belief families make up a larger share of the trajectory tasks than of the corpus as a whole. Pooled, the budget that could have paid for belief's structure is free to spend itself on the overlay, registration, obstacle and relocation families instead.

*The ordering is a prediction rather than an accommodation.* Because BToM's structure is complete from the start, nothing in the framework makes a goal attribution earlier or cheaper than a belief attribution, and the delay of @timeline has to be explained by something outside the model. Here the delay is the mechanism: a belief compound is a deep composition whose cost under the initial library puts it far outside any feasible budget window, and it becomes reachable only once earlier solutions have been compressed into tokens that shorten it. That is a claim that can fail --- the compounds might never come within budget, or they might come within budget in the wrong order --- and @criteria says what failure would look like.

== What would count as success or failure <criteria>

The measurements of Chapter 5 are tests rather than descriptions only if what they test is fixed in advance. This section fixes it.

*What counts as having discovered belief.* Not that the belief tasks were solved. A solved belief task with an extensional program is a failure of the corpus, not a success of the learner. The criterion is structural, and it is a condition on the invented term: an abstraction must enter the library whose body (i) opens a private channel, (ii) derives from it a world state that diverges from the actual one, (iii) evaluates an agent's policy against that divergent state, and (iv) commits only the resulting action --- not the divergent state --- back to the world, with the agent value shared between the policy and the commit. All four conjuncts are checkable on the term, and the fourth is the one that distinguishes attribution from mere counterfactual computation.

*It must be a compression win, not merely reachable.* Reachability is cheap: given enough budget, many things can be enumerated. The discovered compound has to be _selected_ by the joint objective of @compression over the non-mental rivals the same library can express, and the margin must be reported in nats against the shortest such rival --- including the rivals that survive the $k$-scene filter for structural reasons, such as the transient-wall spelling and the witness-agent construction. A margin at or below zero is a failure even if the mental program was found first.

*It must generalize behaviourally.* A learned belief term should predict, on a scene held out of the run entirely, that the agent goes where it _believes_ the goal to be rather than where the goal is. This is the false-belief test proper @wimmer_beliefsabout_nodate, and it is the criterion that separates a compressive coincidence from a term that means what we say it means. It also has to be run on a family where the two readings come apart behaviourally, which is why the goal-displacement family carries it: in a wall-belief scene the transient-wall rival reproduces the whole trajectory, so only description length separates the readings there, whereas a displaced goal sends the two readings to different cells.

*What would refute the hypothesis.* Any one of the following:

+ no belief-structured abstraction, in the four-conjunct sense above, enters the library within budget;
+ one enters, but loses on description length to an extensional rival expressible in the same library;
+ a rival that never opens a private channel reproduces all $k$ scenes of a mental family --- which would mean the family was misclassified and the corpus does not test what it claims to;
+ the discovered term fails the held-out behavioural probe, sending the agent to the true location;
+ the discovered term is not applied selectively --- if the library invents an agent constructor and then applies it to every mover in a scene, including one whose walk is fully explained by the bare world, then what was found is a habit of forking rather than an attribution.

Two things should be said now rather than left to the results chapter, because they bear on how the verdict should be read.

The first is that there is a depth past which the model's expressive claim outruns its search claim. Two agents holding contradictory false beliefs, each detouring around its own phantom wall, is a scene the language states without difficulty, but it names six latent literals across two nested forks, and no amount of budget we could give it brought a single such task within reach. The selective-attribution criterion above is therefore carried by the cheaper two-observer case --- two agents, one goal, one grid, where only one of them is attributed a belief --- and the contradictory-belief case is left as a limit, not a result.

The second is that the criteria are stated per endowment, and a verdict is a verdict about one of them. They are the criteria the library of combinators must meet. The control run of @sec-atomic-run is held to the same list and reported against it, but meeting them there does not discharge them, since the granted `fork`/`sync_to_world` pair is exactly what an unsympathetic reader would call `believe` in two halves.

#load-bib(read("refs.bib"))
