#import "@preview/illc-mol-thesis:0.2.0": *

#import "viz.typ": task-figure, scenes-figure, all-tasks, render-grid

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

Our model is a triple $lc cal(L)_0,D,"MDL" rc$: an initial language $cal(L)_0$ of primitive operations, a corpus $D$ of scenes, and the description-length objective $"MDL"$ (@sec-hbm). The model is a map from the triple to the final language $cal(L)^*$. Learning occurs in rounds, where each round involves solving tasks in $D$ (enumeration) and growing the library with new abstractions (compression). 

The learner is an observer who is shown scenes from $D$, visual episodes that unfold over discrete time steps on a small grid. In _enumeration_ the learner seeks to recover the underlying process that generated what it saw, by positing candidate hypotheses until it finds one that works. Hypotheses are written as compositions of primitives chosen from the library $cal(L)_i$. In _compression_, the learner finds syntactic structure common to its found hypotheses and abstracts them as new primitives added to the library, yielding $cal(L)_(i+1)$. So in the next enumeration, the learner's new hypotheses are formulated using higher-level terms, and the terms become iteratively more complex. 

We ask whether such a learner can come to learn structure of belief attribution as a new abstraction (@sec-false-belief), even without $cal(L)_0$ having terms for an agent, goal, or attitude, and without $D$ distinguishing the same (@no-btom). We claim that belief attribution is learnable, that it lies in the image of the map $lc cal(L)_0,D,"MDL" rc arrow cal(L)^*$. We fix $D$ in @corpus, $cal(L)_0$ in @space and @primitives, and MDL in @inference. An instance of our model gives a counterexample to the claim that belief attribution must be acquired through a native domain-specific module @leslie_pretense_nodate.

Take the world to be a state space $W$, so the data is the movement of entities in $W$. A hypothesis is a transition function $W arrow.r W$, a theorized dynamic by which the world advances to its next state. Our model adds the product $W times W$ with its canonical structure maps, so the learner has the capacity to hold two world states (@space). This is not the capacity for belief attribution, since nothing in a product says "whose" the second factor is, or that it "is" anybody's. 

Our model also adds a planner, which turns a utility into a dynamic that maximizes it (@rationality). We grant this substantive endowment on the grounds that efficiency-sensitive goal pursuit is in place some three years before false-belief attribution (@timeline). The third is the two-part MDL objective of @sec-hbm, which prices a language jointly with the corpus written in it.

Belief attribution is a particular way of putting these components together. To attribute a belief is to open the product, perturb the second factor, run the planner against that factor rather than the first, and collapse the pair by publishing a particular value (@signature). We ask whether an objective that only seeks to minimize description length finds this arrangement among the space of possible abstractions. So our question is about the map $lc cal(L)_0,D,"MDL" rc arrow cal(L)^*$ rather than about the expressive power of the language. 



// Nearly everything else is instantiation. The grids are $5 times 5$ because a scene has to be small enough to enumerate programs over; the cell values run $0$ to $9$ because a scene needs a handful of distinguishable entities; the budgets in seconds are facts about one machine; the abstractions the runs invent arrive with names like `fn_7` that carry no information at all. A reader wanting to know whether some detail belongs to the model or to the instantiation can apply one test: would changing it change which of the criteria of @criteria is met, and why? The size of the grid would not. The presence of the first-channel map beside the second-channel map would, since without the complement the learner's use of one channel rather than the other would not be a choice.

== The task corpus <corpus>

Our corpus consists of 316 tasks in 25 "families", groups of tasks that are all governed by a common underlying process. Of these 168 tasks in five families can only be solved by belief attribution, and 148 can be solved extensionally. The search doesn't know this distinction about tasks of course, it treats them all alike. 

Most tasks are _trajectory tasks_. A trajectory is a sequence of 2d grids over the time dimension, that depicts some dynamic at play (e.g. a body that falls, an agent that approaches something, etc.). A task consists of $k$ trajectories that all depict the same dynamic but vary in setup (where the entities start, how long the scene takes, what else is lying on the grid, etc). Given a task, the learner's goal is to find a transition function that, for any trajectory in the task, returns the next grid for any grid in the sequence. The transition function should be a general explanation of the underlying dynamic, invariant to initial conditions.

For example, the task below is an instance of the _constant-movement_ task family. It consists of four scenes that each depict the value 4 moving up. In the figure the grids of each timestep are collapsed into one, so each arrow indicates one timestep forward.

#figure(
scenes-figure("physics"),
kind: image,
//caption: [an instance of a _constant-movement_ task]
)

A solution to this constant-movement task would be a program that encodes "at each time step, value 4 moves to the adjacent cell above it". Such a program solves the task because for each scene, given that scene's starting positions, the program reproduces the rest of the trajectory. 

The following task is an instance of a _goal-directed movement_ task.

#figure(
scenes-figure("desire"),
kind: image,
//caption: [an instance of a _goal-directed movement_ task]
)

A solution to it would be a program that encodes "at each time step, the value 1 moves to the adjacent cell closest to value 2". The solution involves the optimizing planner (@sec-planner), but _does not_ require attributing a belief. It's a purely extensional description.

The following is an instance of an _obstacle_ task. The same solution program for the above goal-directed task reproduces the first three scenes. But notice that in scene 4, value 1 takes a suboptimal path to value 2 (it should just goes straight to the right but it goes up first). So we need a program that accounts for this, so it solves all four scenes. 

#figure(
  scenes-figure("obstacle", caption: none, posited: 3, posited-scenes: ()),
  kind: image,
) <fig-obstacle>

Say the origin (0,0) is the top left cell. A solution might encode "at each time step, value 1 moves to the adjacent cell closest to value 2, in a world where an impassable wall sits at cell (1,2)".

The wall is no part of what the learner is shown; it is posited by the hypothesis, and stamped onto the grid by the wall-stamping edit that hypothesis contains. Drawn in, hatched to mark it as the hypothesis's rather than the task's, the same four scenes look like this.

// the same task with the posited wall drawn in, hatched, in every scene
#figure(
  scenes-figure("obstacle", caption: none, posited: 3),
  kind: image,
) <fig-obstacle-wall>

Only scene 4 is evidence for that wall. It stands at the same cell in all four, but in scenes 1--3 a shortest route to the goal gets around it anyway, and in all three the agent takes one just what the wall-free predicts on its own. In scene 4 the agent and its goal share a row, so the wall blocks every shortest route, and the six steps the agent spends on a journey of Manhattan length four are something no amount of distance-minimising will reproduce.

These tasks, and most of the corpus, can be solved by programs that describe the scene extensionally via a purely behaviourist visual description. The task families we care about are those that can only be solved by attributing a belief to an agent.

#figure(
  scenes-figure("belief_goal"),
  kind: image,
  caption: [An instance of a goal-displacement task: the four scenes of one task, each collapsing its frames into a single panel, so that each arrow is one time step. The scenes vary in where the entities start and in how long they run, and share only the program that generated them],
) <fig-belief-goal>

The learner is shown those four scenes and nothing else: no labels, no segmentation of the grid into agent and goal, no indication that this task differs in kind from a falling body. What it must return is a single transition function which, started from each scene's own first frame, reproduces that scene's every frame. In all four, the value 1 takes an efficient path and then stops on an empty cell two rows above the value 2 --- on the cell, that is, that the 2 would occupy if it were two rows higher. The 2 itself never moves.

No description of where things are on the grid does that, because the cell the agent walks to is not where anything is. What does it is a description of where the agent _takes_ things to be. Written out in English, the program encodes

#quote(block: true)[at each time step, copy the grid and move the value 2 one cell up on the copy; in the copy, move the value 1 to the adjacent cell closest to 2; then in the world move 1 to wherever the copy left it, and change nothing else]

and that is how every program in this thesis will be given: by what it does to a grid at each time step, together with its type where the type matters --- here $G arrow.r G$, a transition function like any other, since nothing in the type of a program announces that it is about a mind. The symbols the search engine actually prints are an implementation matter, and are quoted only where a count of them is at issue.

Read that way it is not a program about grids but a proposition about a mind, and each of its parts answers to something Chapter 3 stipulated. The private copy is BToM's $m'$; the planner is its rationality assumption; and the commit, because it names the same value the planner moved, is what makes the copy _somebody's_ rather than merely counterfactual. The learner has to arrive at that program with none of the three given to it. @primitives says what it is given instead, @signature how the pieces assemble, and @criteria what would count as having assembled them for the right reasons.

Another task family that requires attribution to solve are the _false-wall_ tasks. Their trajectories have the shape of the obstacle task's: value 1 (agent) bends around a cell on its way to the goal (green), exactly as if a wall stood there. But this time no wall reaches the grid at all: where the obstacle task's program stamps a real one (@fig-obstacle), here no frame of any scene ever contains a 3, because the wall is private to the agent: agent 1 wrongly believes the cell is blocked. The cell it bends around is instead occupied, for the whole trajectory, by an inert bystander (the purple 5), and the bystander is not what turns the agent aside. Walls are the only impassable value in this world, occupied cells are freely entered (see the goal-seeking and obstacle tasks above, which end with the agent stepping onto the goal's own cell).

Why is the bystander there at all? A frame is only a snapshot, and a program is free to act between snapshots. So a devious program could reproduce these scenes without attributing anything: stamp a real wall onto that cell, exactly as the obstacle task's program does, let the agent take its step, then erase the wall again before the next frame is rendered. The bystander closes this loophole. Stamping a wall on its cell would wipe it out, and nothing in the language can put it back --- yet every frame shows it sitting there untouched. A program that reproduces these scenes must therefore keep the wall out of the world altogether, and the only place left for it is a model private to the agent (@app-generators makes this precise).

#figure(
  scenes-figure("belief_wall"),
  kind: image,
  caption: [An instance of a false-wall task],
) <fig-belief-wall>

Three further belief families vary the attribution rather than its content. In a _witness_ task, the false-wall scene gains a second agent that is not misled: it seeks a goal of its own and walks straight across the cell the first agent bends around, which no real wall could permit. In a _two-observer_ task, two agents seek the same goal and only one of them is wrong about where it is, so a solution must attribute a belief to one mover and withhold it from the other. And in a _false-obstacle_ task the agent is wrong about two things at once, detouring around a phantom wall on its way to a displaced goal, while a real wall stands elsewhere in the scene --- so a solution must hold apart, in one program, a wall that exists and a wall that is merely believed. Together with goal-displacement these are the five belief families of the corpus; @tab-families gives the program behind each, and @criteria says which family carries which criterion.

A second, smaller group are _template tasks_. Where a trajectory task's input is a single initial grid, a template task's input is a pair of grids: a _working_ grid and a _template_ grid, both $s times s$. The task is to produce the working grid modified according to the pattern given in the template. This format is loosely inspired by the ARC-AGI-1 corpus [CITE].

#figure(
  task-figure("registration", caption: none),
  kind: image,
  caption: [An instance of a registration task: the solution moves one named value in the working grid to the cell the template assigns it, and leaves everything else standing],
) <fig-registration>

The template families are in the corpus because they populate the pair interface from the non-mental side. A program that consumes a (working, template) pair is aligning two images, not holding a belief, and --- as @primitives will make concrete --- it is doing so with exactly the parts a belief attribution needs. This is a policy the whole corpus follows: whether a piece of machinery is mental will be settled by which families pay for it (Chapter 5), so each family should be readable as a claim on some corner of the language. That is worth a tour, since the results chapter refers to these families by name.

Among the trajectory tasks: _physics_ is the directional-movement family that opened this section, a body drifting one step per frame with nothing sought; _desire_ is one value seeking another, the planner's home family and the whole of the true-belief case; _flee_ is the same planner at the opposite utility polarity, running away rather than toward; _obstacle_ is desire plus a real wall (@fig-obstacle); and _relocation_, _deletion_ and _denoise_ are grid edits with no mover at all --- a wall taken down and re-stamped elsewhere, a cell cleared, a value scrubbed from the grid. Four further trajectory families exercise the pair machinery without any mental reading: _overlay_ and _underlay_ composite two layers of one scene, differing in which layer wins where they collide; _comet_ is a seeker that leaves a trail, its past positions composited behind it; and _wipe_ opens a blank second channel rather than a copied one. Among the template tasks: _registration_ (@fig-registration) aligns one named value to the template; _perception_ is the same alignment run in the opposite direction, the template updated with where the working grid actually has the value; _multi-registration_ aligns every value the two grids share, and _registration-except_ every value but one; _inpainting_ fills the working grid's empty cells from the template, and _readout_ discards the working grid for the template outright; _layer-compositing_ applies one edit to each grid before merging them; _drifting registration_ aligns against a working grid that is itself scrolling; and _map-update_ rewrites the template to match the working grid wholesale. None of these is filler: each was written to claim one operation on pairs, so that when Chapter 5 asks whether any non-mental family uses a given operation, every operation has a family that was given the chance to claim it (@tab-cube).

In numbers: the 168 belief tasks are 24 false-wall, 48 goal-displacement, 48 witness, 24 two-observer and 24 false-obstacle, and the remaining 148 are spread across the twenty non-mental families just toured. Every task consists of $k = 4$ scenes on $5 times 5$ grids, and the (goal, agent) values rotate through eight distinct pairs, so that no particular integer comes to mean "agent". The non-mental side is deliberately not uniform: obstacle is the densest family in the corpus at 48 tasks and relocation carries 16, because the wall-handling structure they share with belief must recur often enough for compression to abstract it --- a curriculum choice, defended when it is questioned (@sec-objections). @tab-families in the appendix lists every family beside the ground-truth program that generated it.

// The first is $k$ itself. The likelihood of @sec-bayes eliminates rather than ranks, so requiring $k$ scenes of one latent program conjoins $k$ eliminations: a rival that fit one scene by luck is removed rather than down-weighted. That is most of the work. Every task in the goal-displacement family admits a spelling in which the agent's bend is explained by a phantom wall rather than a displaced goal --- for _some single scene_ of the task, and for none of them across all four, since a single wall coordinate cannot follow a believed goal that sits somewhere different in each scene.

// The second and third are conditions on the scene, which $k$ does not deliver: the falsely believed wall must lie on the route the agent would otherwise take, so that the belief is doing work; and the scene must make a world-level stamp of that wall detectable, so that "believes a wall is there" is distinguishable from "a real wall stood there and was taken down before the frame was rendered". Both are enforced by the generators, and @app-generators states them precisely, together with the handful of rivals that have to be excluded by argument rather than by the scenes because they are extensional identities and no number of scenes separates them.

// Template tasks are evaluated against a second interpreter, which threads a working grid while pairing each frame with a constant external template and applying a commit policy. Here the second channel is a given input rather than a privately derived model, so a program that consumes the pair is aligning two images, not holding a belief. The same vocabulary therefore does mental work in one interpreter and non-mental work in the other, which is what lets the corpus price belief against genuinely non-mental competitors built from the very same parts.

== Program space: type-directed enumeration <space>

A hypothesis is a program, so the hypothesis space is the space of programs generated by type-directed enumeration. $G$ is the type of 2d grids of cell values, the frames of a scene. Write $G^*$ for stacks of grids, i.e. a full scene composed of a sequence of grids. So a transition function is then an element of $G arrow.r G$. The cell values are members of the ground set $V = {0, dots, 9}$, where each is an entity on the grid: value 3 is reserved as an impassable wall, while the other values are interpreted as agent or goal or any other entity. The grid coordinates are $C = {0, dots, s-1}$, a row or column index, from which a position in $C times C$ is built. The remaining sets are $D$, the four movement directions, and $U subset.eq (C times C arrow.r RR)$, the positional utilities that the planner scores (@primitives).

Everything else the language names is an arrow between these. The arrows relevant to our argument are the ones into and out of the product $G times G$, which serves as the (world, model) channel pair, and they arrive in the next section as the types of the product primitives. Whether a program opens a private second channel is a matter of which of those arrows its nodes inhabit, so a (world, model) representation is a structural property of the program. The fixed product does carry the commitment that a learner holding a private representation holds exactly one (generalizing this is left as further work, see @sec-arity).

A program is checked by running it through an interpreter (@app-interpreter). The solution to a trajectory task (@corpus) is a program of type $G arrow G$, a transition function that given the task's initial grid at step $i$ should return the next grid at step $i+1$. The interpreter iterates the program $G arrow G$ on the task's initial grid $G$, yielding the stack of grids $G^*$ which can be compared to the target scene. The interpreter's state at any step is just a single grid $G$, not a (world, model) pair $G times G$, and certainly not a registry of agents, goals, or entities. No pair is handed from one step to the next: a program that opens a second channel (a model channel that may diverge from the world) must build it, use it, and collapse it again inside that step. 

A template task starts from a pair in $G times G$, a working grid and a constant external template, so its solution is a _commit policy_ $G times G arrow.r G$, one of the collapses of @primitives. The commit answers what each frame should do with the template. Since a type-directed enumeration yields programs of one type only, trajectory tasks ($G arrow G$-rooted) and template tasks ($G times G arrow G$-rooted) are searched separately (@enumeration). Though compression pools the solutions of both (@compression).

== The primitive library <primitives>

The primitive library consists of grid primitives and pair primitives. The grid primitives are the interface that we hand-picked for our particular domain. Among them are a planner, a value mover, and the terminals a scene is made of. The pair primitives split in two, and only one of the halves was ours to pick.

One half is what the product $G times G$ determines by itself. Having a product just _is_ having the two projections, and the rest of this half follows from them by the universal property rather than by our choosing: the diagonal, the functorial actions, and the swap. The library holds a generating set for all of it --- the universal pairing $chevron.l f, f' chevron.r$, for instance, is the diagonal followed by the functorial action on both channels --- so nothing the product determines is missing from the language, and the composers close it up.

The other half is the operations that consume a pair: the two unions, the two scope collapses, and the reader and writer from which a single-value commit is assembled. The product does not determine these. They inhabit arrow types the product creates, but what each of them does with a pair is a fact about grids rather than about products, so each is a choice we made. @granularity gives the rule those choices follow, and Chapter 5 checks that every one of them is claimed by some non-mental family (@tab-cube).

Across both halves we impose one further discipline: every choice the pair layer offers comes with its opposite corner. The copy into both channels comes beside the pairing with a blank, the action on the second channel beside the action on the first, the projection that keeps the world beside the one that keeps the model, each union beside the union with the opposite precedence. This is a separate commitment from the completeness of the first half, and it is what the blank pairing is in the library for: the product supplies no distinguished empty grid, so that primitive earns its place as a mirror rather than as something the product forced. The discipline is what makes the learner's eventual route through the pair a choice rather than the only road, since no route is cheaper than its mirror image, and Chapter 5 checks that the mirrors are not merely available but taken, by other families in the same run (@tab-cube). The remaining decision is where the decomposition stops --- which compounds are handed over whole and which must be assembled from parts --- and @granularity states the rule that fixes it, together with the reason it leaves no room for tuning.

That is also why the library can be presented the way it is presented below: by saying what each operation does and giving its type, and not by giving the symbol the implementation spells it with. Nothing is lost, because there is nothing in these operations for a name to carry --- what an operation _is_, here, is exhausted by its type and its effect on a grid, and the plumbing in particular is fixed the moment one says "the language has the product $G times G$". The exhaustive catalogue, every symbol's name, signature and semantics under each endowment, is @app-primitives, for a reader who wants to count symbols rather than read descriptions.

Take the grid layer first. The _planner_, of type $U times V arrow.r (G arrow.r G)$, is the one from @sec-inverse: given a utility and a cell value it returns a transition function that at each time step moves that value one step toward the position maximising the utility. The two utilities it can be handed, both of type $V arrow.r U$, are minus and plus the distance to the nearest cell of a named value, so the same planner approaches a value under the one and flees it under the other. Beside it are the movements and edits, all of them transition functions once their arguments are supplied: a constant _step_, of type $V times D arrow.r (G arrow.r G)$, which moves a named value one cell in a named direction; _stamping a wall_ at a position and _clearing_ a single cell, both of type $C times C arrow.r (G arrow.r G)$; _erasing_, of type $V arrow.r (G arrow.r G)$, which scrubs every cell of a named value from the grid; and _composition_, of type $(G arrow.r G) times (G arrow.r G) arrow.r (G arrow.r G)$, which runs one transition function and then another. The terminals are the four directions, the ten cell values, and the five coordinates, a pair of which names a position.

Since what we are interested in is how transition functions $G arrow.r G$ interact with (world, model) pairs $G times G$, the rest of the library is the product: the maps it determines, and the operations we added to consume a pair once it is built. There are four jobs to be done with a pair, and an arrow for each: building a pair out of a grid, transforming a pair, collapsing a pair back to a grid, and reading and writing the positions a collapse goes off of. In what follows $w$ and $m$ are grids, $f$ and $f'$ transition functions, and $v$ a cell value.

Two operations _build_ a pair, both of type $G arrow.r G times G$. The diagonal $w mapsto (w,w)$ copies a grid into both channels. Its complement $w mapsto (w, bold(0))$ pairs the grid with an empty one instead, so that the second channel starts as scratch rather than as a copy of the world.

Four operations _transform_ a pair, all of type $G times G arrow.r G times G$. Three are the functorial actions --- the action of the product, as a bifunctor, on arrows --- which lift a transition function to act on the channels: $(w,m) mapsto (f(w),m)$ acts on the first factor, $(w,m) mapsto (w,f(m))$ on the second, and $(w,m) mapsto (f(w),f'(m))$ on both at once, with a separate function for each. The fourth is the _swap_ $(w,m) mapsto (m,w)$, the symmetry isomorphism of the product, which exchanges the channels, so that anything defined on one factor can be conjugated into a version acting on the other.

Several operations _collapse_ a pair to a single grid, of type $G times G arrow.r G$. The two projections, $(w,m) mapsto w$ and $(w,m) mapsto m$, return one channel and discard the other. Two more union the channels cell by cell, differing only in which of them wins where they disagree. The last two are _scope_ collapses. One moves every value the two channels share to the position it holds in the second; the other, of type $V arrow.r (G times G arrow.r G)$, does the same for every shared value but one named exception. Each adopts the second channel's account of where things stand, and they differ only in how much of it they adopt --- all of it, or all of it minus one named value. The second therefore publishes by blacklist: the value it names is the one it holds _back_, and everything unnamed goes through. The remaining point on that scale --- adopting the second channel's account of exactly one named value, a whitelist of one --- is deliberately not on the list, and assembling it is what the accessors below are for.

Finally, the accessors on positions. A _reader_, of type $V arrow.r (G arrow.r C times C)$, gives the position of a named value in a grid; a _writer_, of type $V arrow.r (G times (C times C) arrow.r G)$, gives the grid with that value moved to a given position. A third operation, of type $(G arrow.r C times C) times (G times (C times C) arrow.r G) arrow.r (G times G arrow.r G)$, joins a reader to a writer: it reads a position off the second channel and imposes it on the first. Handed the reader and the writer for one and the same value $v$, it yields the collapse that moves $v$ to wherever the second channel puts it and leaves everything else as the first channel had it. That collapse costs three symbols rather than one, since the learner must name the reader, the writer and the join --- it is assembled, not handed over.

Each of these arrows needs its own composer, since the composition of the grid layer joins two transition functions only. Three are given, one for each arrow that needs one. A builder followed by a transform, of type $(G arrow.r G times G) times (G times G arrow.r G times G) arrow.r (G arrow.r G times G)$, extends a builder. A builder followed by a collapse, of type $(G arrow.r G times G) times (G times G arrow.r G) arrow.r (G arrow.r G)$, is the composer that returns to $G arrow.r G$; since the interpreter only ever iterates a transition function, a program that opens a second channel must close it again within the same time step, and this is what closes it. A transform followed by a collapse, of type $(G times G arrow.r G times G) times (G times G arrow.r G) arrow.r (G times G arrow.r G)$, is the only one that acts on a pair the program did not itself build, which is what a template task's given pair calls for.

Nothing in any of these primitives distinguishes a world from a model. The pair is symmetric: both factors are grids, every map is present in both directions, and no primitive marks either channel as a particular agent's private representation. Which channel is the world and which is a belief is settled only by how a program assembles these pieces together, which is found by search. 

=== Composing belief attribution <signature>

In @sec-false-belief we presented belief attribution as a two-step computation. First, the observer must _derive_ a counterfactual grid which deviates from her own according to te agent's false belief $m' = "derive"(m)$. Then the observer _runs_ the agent's movement policy on the counterfactual grid $t' = "optimize"(g,m')$ to yield the agent's simulated trajectory which is optimal w.r.t the agent's world $m'$. 

Our model shows that this structure can be constructed by composing the $G times G$ primitives from @primitives. In our model only the _run_ step is done by a primitive, the planner, which builds the agent's movement policy. Nothing in the library opens a private grid, so the _derive_ step of creating and modifying a copy of the grid must be discovered by the learner.

Additionally, our model includes another step that isn't represented in @sec-false-belief's. In that model, the _attribution_ of a belief to an agent is stipulated. For example, if an observer is simulating agent $a$'s trajectory and given goal $g$ and model $m'$, she just calculates $t' = "optimize"(g,m')$. That $g$ and $m'$ are attributed to 

But in our model, we want to make the distinction between positing counterfactual a model in general vs. positing a counterfactual model and associating it with a particular agent, attributing it to that agent. 

But this architecture adds a third step with no counterpart in Chapter 3: _commit_ the result back to the world, publishing where the agent went and nothing else. The commit is owed to the interpreter rather than to the theory --- it renders one grid per step, so motion on a private grid doesn't affect the rendered grid unless an agent's motion is explicitly committed from it.




and nothing publishes a single value's position back out of one. Both have to be assembled from the product structure above. So _derive_ and _commit_ give us two target structures, and it is worth saying what each has to do before asking whether the learner can build it.


The first is the _derive-and-commit frame_, of type $(G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$. Handed a derive and a commit, it copies the world into a pair, applies the derive to the second channel to get the private grid, and gives the pair to the commit, which collapses it back to one grid. The private grid exists only for the duration of that call. Assembled out of @primitives it is the diagonal, then the derive lifted onto the second channel, then the commit, joined by two of the composers --- four symbols of plumbing around the two slots that do the work.

The second is the _single-value commit_, of type $V arrow.r (G times G arrow.r G)$. Given a value, it reads where that value sits in the private grid and moves the same value in the world to that position, so exactly one value's position is published and everything else stands as the world had it. This is the commit belief needs, and it is a reader and a writer for the same value, joined --- three symbols.

Belief is the composition of the two, with the derive doing the agent's error and the commit doing the attribution. Taking a phantom wall as the error, the belief program for an agent value $a$ that seeks a goal value $g$ and is wrong about a wall at row $r$, column $c$, encodes

#quote(block: true)[at each time step, copy the grid and stamp an impassable wall at cell $(r,c)$ on the copy; in the copy, move $a$ to the adjacent cell closest to $g$; then in the world move $a$ to wherever the copy left it, and change nothing else]

and it has type $G arrow.r G$, like every other transition function --- nothing in its type announces that it is a program about a mind. @fig-fork-anatomy traces the three steps through one transition of the false-wall task of @fig-belief-wall, one grid per intermediate state.

#let _fw-lane(lab, g, private: false) = stack(dir: ttb, spacing: 4pt,
  align(center, text(size: 8pt, fill: luma(110), lab)),
  box(
    inset: 4pt, radius: 2pt,
    stroke: if private { (paint: luma(120), thickness: 0.6pt, dash: "dashed") } else { none },
    render-grid(g, size: 10pt),
  ),
)
#let _fw-arr(lab) = stack(dir: ttb, spacing: 2pt,
  text(size: 8pt, lab),
  text(size: 11pt, sym.arrow.r.long),
)

#figure(
  grid(
    columns: 7,
    align: center + horizon,
    column-gutter: 8pt,
    _fw-lane([world $w$ --- frame $t$],
      ((0,0,0,0,0),(2,0,5,0,1),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0))),
    _fw-arr([derive: copy, wall at (1,2)]),
    _fw-lane([private copy, wall at (1,2)],
      ((0,0,0,0,0),(2,0,3,0,1),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0)),
      private: true),
    _fw-arr([run: 1 seeks 2]),
    _fw-lane([after the plan],
      ((0,0,0,0,1),(2,0,3,0,0),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0)),
      private: true),
    _fw-arr([commit: publish 1]),
    _fw-lane([world --- frame $t+1$],
      ((0,0,0,0,1),(2,0,5,0,0),(0,0,0,0,0),(0,0,0,0,0),(0,0,0,0,0))),
  ),
  kind: image,
  caption: [One transition of that program on the false-wall task of @fig-belief-wall, with agent 1 seeking goal 2 and wrong about a wall at row 1, column 2. _Derive_: the world is copied and a wall is stamped on the copy, overwriting the bystander 5 there --- the dashed grids are the private channel, which is never rendered and does not survive the step. _Run_: the agent's move is planned in the walled copy, and it steps _up_, a step no plan over the actual grid takes: the bystander's cell is freely enterable, so in the world the straight step left is strictly better. _Commit_: only the agent's resulting position is written back; the wall never reaches a rendered frame, and the bystander stands untouched.],
) <fig-fork-anatomy>

Note that the agent value $a$ occupies three slots in that program: the slot that says whose move is planned over the privately walled grid, the slot that says whose position is read off the copy, and the slot that says whose position is written back to the world. So to find this compound, the enumerator needs to independently fill the argument slots of the three primitives with the same value placeholder. We call that coincidence --- one value filling, at once, the slot that says whose move is planned and the slots that say what is published --- the _agency signature_. It is the structural mark that separates an attribution (a private grid that is _somebody's_) from a mere counterfactual computation, and both the success criteria of @criteria and the results of Chapter 5 are stated in terms of it.

=== Where the decomposition stops <granularity>

Even twelve-month-old infants encode an action as directed at a goal state, and expect an agent to use the most efficient means available to it @timeline. They employ the teleological stance, which relates an action, a goal state, and the constraints of the situation but without attributing any representation to agents @gergely_taking_1995 @gergely_teleological_2003. Under the teleological stance, the reality constraints that the observer consults are their own, not the agent's.  

We return to this distinction in @rationality, when we defend our choice to hand our model learner a primitive planner, which moves a value one step along the best available path toward improving some utility, even though the planner is algorithmically complex. From the developmental literature we know that efficient goal-directed interpretation is in place around a year of age, three years before anything recognizable as false-belief attribution (@timeline). And on Gergely and Csibra's analysis it isn't yet ToM at all, it just says how a body moves given a goal and a layout.

The library of @primitives cuts the language at conspicuously uneven depths. The commit that publishes one value must be assembled from a reader, a writer and a joiner, while the collapse that resolves an entire pair is a single token; one is entitled to ask why the knife fell where it did, since a granularity chosen freely is a granularity that could have been chosen to flatter the result. Two rules fix it, and neither leaves room for tuning.

The first rule is the completeness announced at the head of @primitives, and it covers the builders, the transforms and the projections. The two projections are what having a product _is_; the diagonal, the two functorial actions and their joint form, and the swap follow from them by the universal property, and the library holds a generating set for all of them, so the kit is complete rather than curated. The pairing with the empty grid is the one item here the product does not force, since it needs a distinguished blank grid, and it is present under the mirror discipline instead: each map comes with its complement --- the copy with the blank, the second channel with the first, the world with the model --- so no route through the pair is cheaper than its mirror image. At this layer there is nothing to decompose, and the only discretion we exercised was to include every mirror rather than a subset.

The second rule covers the collapses, where choices did have to be made, and it can be stated in one sentence: _the decomposition reaches exactly as far as the single-value vocabulary --- read where one named value sits, write one named value to a position --- can spell, and what that vocabulary cannot spell stays atomic._ The commit belief needs, publishing one named value's position out of the private channel, has a spelling in that vocabulary, so it is withheld as a token and the learner must build it out of parts that each do non-mental work elsewhere in the corpus. The scope collapses have no such spelling: both fold over _every_ value the two channels share, a set fixed by the scene rather than by the program, and writing them out would require granting iteration over $V$ --- a strictly more powerful construct than anything the language contains, and a far less innocent gift than the two collapses it would be used to spell. The two unions, which merge whole grids cell by cell, are atomic on the same grounds. So the rule is not "decompose everything"; it is that the compound under test must be assembled from parts, while its neighbours may be bought whole.

That policy has a consequence that should be stated here rather than discovered in the results: it prices the boundary belief draws asymmetrically. A commit can name what it publishes or name what it withholds --- publish only this one value, or publish everything except this one value --- and these draw the same kind of line from opposite sides, but the second costs one node where the first costs three. Nor are the two always distinguishable extensionally. On a scene where the agent is the only shared value the derive has moved, "publish everything" moves only the agent, and so denotes exactly the single-value commit; where the derive has also displaced one shared value, a believed-in goal, "publish everything but the goal" does. Wherever that coincidence holds, the cheap spelling and the assembled one compute the same collapse, and a description-length learner should be expected to buy the cheap one. The asymmetry does not bias the search toward belief --- the atomically priced scope collapses are the commits of non-mental families (_multi-registration_ and _registration-except_ in @tab-families), and within the mental families it taxes precisely the whitelist form of the agency signature --- but it does mean that the _spelling_ of a discovered commit is a prediction of the price list rather than a free observation. Chapter 5 reads it as such (@sec-selective), with the atomic control run  in which both spellings cost one token, separating what the price explains from what the denotation does. It is the one place in the thesis where saying what a program does and saying how many symbols it takes to say so come apart, and both readings are needed.

== Inference <inference>

Our procedure runs in alternating rounds of exploration and compression, the loop of @dechter_bootstrap_nodate as in DreamCoder @ellis_dreamcoder_2020. The exploration phase performs per-task inference, enumerating programs under the current library for the tasks not yet solved (@sec-search). The compression phase revises the library to compress the programs found (@sec-hbm). We omit DreamCoder's recognition model, for reasons explained in @no-amortization.

=== Search <enumeration>

The exploration phase finds a hypothesis that correctly explains a scene, by enumerating candidate programs in decreasing order of prior probability and checking if the program produces the scene. A hypothesis is a program (@sec-programs), represented as a typed expression tree over the current library (@space). Since enumeration in decreasing prior probability is enumeration in increasing description length, the search walks outward in expanding bands of cost (expanding $tilde e^ell$, see @sec-search). The first program it finds that reproduces the data is the shortest one that does, which is the maximum a posteriori hypothesis ($h^*$, see @sec-programs). The enumeration runs under a time budget, and a hypothesis whose cost band is not reached before the budget expires is simply not found (@sec-search). 

// Programs are represented as an expression tree over the library of primitives. Each node carries a type together with the types of the arguments it expects. A node with no arguments is a terminal whose value is itself, while a node with arguments is an operator applied to its filled-in arguments. The tree is directly executable, it's evaluated from the root by recursively evaluating its child nodes. 

The library fixes the prior. The prior $P(h) prop e^(-"DL"(h))$ (see @sec-programs) is realised as a distribution over the library's symbols, under which a program's cost is the summed $-log p$ of its nodes. Symbols are grouped by return type, and a symbol of type $tau$ is priced at $-ln$ of the number of symbols sharing $tau$, i.e. a type-uniform distribution#footnote[One exception to the uniform distribution is that a cell value which is visible in the task's first frame costs nothing, since naming what is already on display is not a hypothesis about anything. But a placed wall is full-price, since it's positing something that isn't there]. So the types give the branching factor at each argument slot, and the cost of each choice at that slot. Enumeration is type-directed: an argument slot draws only from symbols whose type matches, so no ill-typed tree is ever built.

A candidate program solves a task only if the interpreter, started from each scene's own first frame, reproduces that scene's every frame exactly, for all $k$ scenes (@corpus). So the indicator likelihood $bb(1)[h "solves" d]$ (@sec-bayes) is evaluated against all $k$ scenes. 

=== Library revision <compression>

Compression returns a set of abstractions together with the corpus rewritten to use them. Each abstraction is registered as a new symbol and added to the library of primitives. So what was a subtree found in round $i$ becomes a new atom in round $i+1$, so the next enumeration stage can reach deeper into program space (@sec-hbm) and the learner bootstraps more complex programs. 

Where exploration minimizes description length one hypothesis at a time against a library held fixed, compression minimises the total description length. It minimizes the length of the library itself plus the summed cost of every solved program written in it @ellis_dreamcoder_2020, the posterior over languages (@sec-hbm).

$
"DL"("library") + sum_"solutions" "DL"("program" | "library")
$

When a syntactic structure is common to a lot of programs, it gets added to the library as a new term. So the description length of the solutions shrinks, since rewriting the solutions with the new term replaces the common subtrees with that term (so $sum_"sols" "DL"("program" | "library")$ shrinks). But the library is now larger, and since a symbol's probability falls as the number of symbols competing with it rises, every program written in the enlarged library gets a little more expensive (so $"DL"("library")$ grows). This is the tradeoff that needs to be made when deciding whether to abstract some shared structure as a new primitive. The two terms ensure that an abstraction is kept only when the tokens it saves across the whole corpus outweigh the price of adding it. So we're less likely to abstract incidental structure, and more likely to abstract structure that's relevant to the corpus. 

DreamCoder looks for abstractions bottom-up, enumerating refactorings of each found program and intersecting them @ellis_dreamcoder_2020. Instead we delegate the search to the top-down Stitch @bowers_top-down_2023 library-learning system, which searches the space of abstractions directly and so never has to materialize every rewrite. Also, note that enumeration searches by root type (@space), so solutions to trajectory tasks (transition functions $G arrow G$) and template tasks (commit policies $G times G arrow G$) are searched for separately. This is just an implementation necessity which doesn't affect the overall outcome, since compression searches for common structure among the full pool of solutions together regardless of root type. 


== The model does not encode a Bayesian theory of mind <no-btom>

The goal of our model is to recover the belief attribution structure from Chapter 3 without stipulating it ourselves. A natural objection to our model's results would be we've merely re-encoded Chapter 3's stipulations in a different notation so the learner trivially "finds" them. To pre-empt this, we'll argue that no notion of belief attribution, agency, or world/model distinction is encoded in our setup (between the interpreter, the types, the library, the corpus, or the prior). Of course we have to give the learner some things, but connecting the given machinery to create a belief attribution is left to the learner. 

BToM hands agent, goal and belief to a planner as separately inferred arguments. Our model doesn't have a planner with argument slots to inherit that from, and no type denotes an agent, a goal or an attitude. There are just grids, cell values, coordinates and arrows between them. The thing we call the agent is whatever value occupies the policy slot and the commit slot at once (@signature). A function binding one hole to both the policy and the commit isn't given as a primitive. It's a compound that the searcher needs to assemble enough times so that compression abstracts it.

BToM evaluates the planner against a supplied world model $m'$ and permits $m eq.not m'$. What we give instead is a product type and general combinators that inhabit it. The pair is symmetric and content-free --- the diagonal produces $(w, w)$, and nothing in it says the second copy is a _model_ or that anyone _holds_ it. Belief is a matter of wiring: acting on the second channel rather than the first, committing directly rather than conjugating through the swap, publishing one named value rather than every value the channels share. Each of those choices has its opposite in the library, exercised by some non-mental family in @tab-families; the medium is neutral, and the asymmetry is made in program space by the search.

The interpreter does not supply it either. In BToM the divergent model is an input to a planner the observer runs _inside itself_; ours has one grid of state, no second channel to hand anybody, and no mode distinguishing evaluating a program from evaluating a program-as-an-agent-would. A private grid exists only because a program built it, and the interpreter cannot tell it from any other. Nor is either target compound of @signature given as an atom --- that is the atomic control ru-- since the searcher here is handed the diagonal, the channel maps, the composers and the position accessors, product-category combinators no one would describe as mental.

A stronger form of the worry puts the concession one level down: the ability to build a copy of the world that differs from the world just _is_ the representational capacity theory of mind consists in. It is not. Building a world state that differs from the actual one is counterfactual reasoning --- the capacity to consider what would be the case if things were otherwise --- and it is domain-general, with a developmental record of its own well before false-belief attribution. Theory of mind is an abstraction that _uses_ that capacity, by attributing the divergent world to a particular agent, so that it is _that agent's_ world. That attribution is the conjunct the shared hole encodes, and it is exactly what the pair does not supply: the derive-and-commit frame will as happily derive a private grid, compute over it, and publish an unrelated value, and the language will let you write that. What it will not do is tell you not to.

The regress does not end, and we do not claim otherwise. The diagonal could itself be decomposed into reads and writes over a flat memory, so that holding a second world model becomes a discovered pattern of buffer use; but "allocate a buffer" is then the primitive, and no more mental than "form a pair", because intensionality just _is_ the coexistence of a world and a model of it, so any substrate expressive enough to state the structure can hold two things at once. The pair is to theory of mind roughly what the integers are to arithmetic: not a disguised form of the thing to be learned, but the medium without which it cannot be stated. There is no substrate in which theory of mind is discovered from nothing, only substrates whose primitives are, or are not, individually non-mental and general. Call this pair of replies --- attribution is a further conjunct on top of counterfactual reasoning, and decomposing the medium only relocates the regress --- the _no-separation point_; it is what remains of the objection once measurement runs out, and Chapter 5's objections section leans on it by that name (@sec-objections).

=== The space of admissible divergences <no-range>

BToM must also fix which divergences are admissible, and the scenarios that produce them. Here there is no range to fix: the derive slot accepts _any_ transition function $G arrow.r G$, so the belief families differ in what the belief is _about_ rather than in some parameter of a common schema --- a phantom wall is a wall stamped at a coordinate, a displaced goal is a one-cell shove of the goal value, and being wrong about both at once is the two composed. The goal-displacement family varies its content across four shove shapes, so an abstraction specialised to one of them would not recur and compression must keep the content subtree as a hole: the productivity and systematicity @explanandum demanded.

The concession is that what the learner finds is a divergent world model, not a belief that is _formed_ and could be _revised_. Nothing represents how the agent came to be wrong, and nothing updates when it looks again. In BToM that lack is a stipulation about admissible divergences; here it is a limit on what the corpus depicts and what a single-frame transition function can express, and the most important respect in which the discovered structure is thinner than the thing it is a model of (@sec-persistent).'

The second concession is about what is being asked for. BToM's $m'$ is a belief in the full sense: a state the agent came to be in, that could have been otherwise, and that would be revised if the agent looked again. What the learner here is asked to find is thinner --- a divergent world model, attributed to an agent and acted on by that agent's policy, but with no account of how the agent came to hold it and no machinery for updating it. Nothing in a single-frame transition function represents perceptual access, or a record of what was witnessed, or an update on being shown the truth. That is a real gap between the discovered structure and the thing it models, and it is the most important one; @sec-persistent takes it up as further work.

=== The space of goals <no-goal-space>

BToM's goal space, and the prior on it, are given; Baker et al. compare several goal priors precisely because the choice matters and is the modeller's to make @baker_action_2009. Here a goal is whichever cell value happens to fill the argument of the attracting utility, drawn from the same ten integer terminals that supply agent identifiers and step magnitudes. There is no goal type and no goal prior --- there is inference over programs, and a program seeks something because a symbol landed in that slot. Even the polarity is a search choice: the repelling utility sits alongside the attracting one, so maximising utility flees a value as readily as it approaches one, and the _flee_ family exists to keep that corner inhabited. The only prior over any of it is the type-uniform distribution, which prices a goal value by the same rule that prices a direction or a composition.

=== Rationality, and the one thing we do grant <rationality>

Rationality is the one stipulation of BToM our learner also receives, in the form of the planner primitive, on the grounds Chapter 3 gave: it is the only item on the list with independent developmental support, in place well before the first birthday and some three years before anything recognizable as false-belief attribution (@sec-planner, @timeline) @gergely_taking_1995, and on the best analysis of it not mentalistic, since the teleological stance relates an action, a goal state and the constraints of a situation without attributing any representation to the agent @gergely_teleological_2003.

Its signature makes the point formally. The planner has type $U times V arrow.r (G arrow.r G)$: it receives one grid, has no parameter identifying whose grid it is and no access to any other channel, and treats whatever grid it is handed as the layout. It cannot represent a divergence, because it cannot see two things to diverge, and it cannot attribute anything, because it has no argument for an attributee. Belief arises only when something _else_ hands it a grid that is not the world and then publishes only part of the result --- and that something else is the composition, which is what the learner has to find. Nor is the planner reserved for the mental families: _desire_ uses it to walk to a goal, _flee_ to run from a hazard, _comet_ to bend a trail, and _obstacle_ to detour around a real wall, on scenes whose rendered trajectories closely resemble the belief ones. Availability is not use.

The residual cost is real and we do not pretend otherwise: the planner packs a full breadth-first search into one token, and a more elementary treatment would decompose it into memory, conditionals and interaction with the environment (@sec-optimize). What we claim is only that granting a planner is not granting a theory of mind.

=== Two confounds that are ours rather than BToM's <confounds>

Two further items answer to nobody on Chapter 3's list; they are properties of how the experiment is run, and the two knobs a sceptical reader would reach for first.

*The success criterion admits no partial credit.* A candidate program either reproduces all $k$ scenes of a task cell for cell or it does not solve the task (@corpus, @sec-bayes). A graded likelihood is exactly where a thumb could rest on the scale: under partial credit a mental program that is nearly right could be preferred to an extensional rival that is exactly right, and the comparison of @criteria --- whether the mental program wins on description length _among programs that all reproduce the data_ --- would no longer be the one being made.

*Nothing in the loop is conditioned on what a scene looks like.* We omit DreamCoder's recognition model, for the reason @no-amortization gives at length: a proposal distribution trained on solved tasks has its prior over the library fit to the data, and belief is precisely the structure the corpus is meant to leave unfitted. Enumeration runs under the type-uniform prior over the current library and nothing else, so the only thing that can make a belief compound reachable is that earlier compression made it short --- the mechanism the ordering claim of @criteria depends on. The choice was settled by measurement rather than by design, and @sec-limits reports what the loop gave when it was run both ways.

=== Two structural differences worth naming <positive-differences>

*Inference here is not agent-directed.* BToM's posterior is over $(g, m)$ for an agent and has nowhere else to point; ours is over programs for a scene, and it is the _same_ search, under the same prior and the same success criterion, that solves _denoise_ and _relocation_. Compression is undirected in the same way, which is why it is pooled rather than run per solution type (@compression): were it divided, the division would itself bias abstraction toward belief, since the belief families make up a larger share of the trajectory tasks than of the corpus as a whole. Pooled, the budget that could have paid for belief's structure is free to spend itself on the overlay, registration, obstacle and relocation families instead.

*The ordering is a prediction rather than an accommodation.* Because BToM's structure is complete from the start, nothing in the framework makes a goal attribution earlier or cheaper than a belief attribution, and the delay of @timeline has to be explained by something outside the model. Here the delay is the mechanism: a belief compound is a deep composition whose cost under the initial library puts it far outside any feasible budget window, and it becomes reachable only once earlier solutions have been compressed into tokens that shorten it. That is a claim that can fail --- the compounds might never come within budget, or they might come within budget in the wrong order --- and @criteria says what failure would look like.

== What would count as success or failure <criteria>

The measurements of Chapter 5 are tests rather than descriptions only if what they test is fixed in advance. This section fixes it.

*What counts as having discovered belief.* Not that the belief tasks were solved. A solved belief task with an extensional program is a failure of the corpus, not a success of the learner. The criterion is structural, and it is a condition on the invented term: an abstraction must enter the library whose body (i) opens a private channel, (ii) derives from it a world state that diverges from the actual one, (iii) evaluates an agent's policy against that divergent state, and (iv) commits only the resulting action --- not the divergent state --- back to the world, with the agent value shared between the policy and the commit. All four conjuncts are checkable on the term, and the fourth is the one that distinguishes attribution from mere counterfactual computation.

*It must be a compression win, not merely reachable.* Reachability is cheap: given enough budget, many things can be enumerated. The discovered compound has to be _selected_ by the joint objective of @compression over the non-mental rivals the same library can express, and the margin must be reported in nats against the shortest such rival --- including the rivals that survive the $k$-scene filter for structural reasons, such as the transient-wall spelling and the witness-agent construction. A margin at or below zero is a failure even if the mental program was found first.

*It must generalize behaviourally.* A learned belief term should predict, on a scene held out of the run entirely, that the agent goes where it _believes_ the goal to be rather than where the goal is. This is the false-belief test proper @wimmer_beliefsabout_nodate, and it is the criterion that separates a compressive coincidence from a term that means what we say it means. It also has to be run on a family where the two readings come apart behaviourally, which is why the goal-displacement family carries it: in a false-wall scene the transient-wall rival reproduces the whole trajectory, so only description length separates the readings there, whereas a displaced goal sends the two readings to different cells.

*What would refute the hypothesis.* Any one of the following:

+ no belief-structured abstraction, in the four-conjunct sense above, enters the library within budget;
+ one enters, but loses on description length to an extensional rival expressible in the same library;
+ a rival that never opens a private channel reproduces all $k$ scenes of a mental family --- which would mean the family was misclassified and the corpus does not test what it claims to;
+ the discovered term fails the held-out behavioural probe, sending the agent to the true location;
+ the discovered term is not applied selectively --- if the library invents an agent constructor and then applies it to every mover in a scene, including one whose walk is fully explained by the bare world, then what was found is a habit of forking rather than an attribution.

Two things should be said now rather than left to the results chapter, because they bear on how the verdict should be read.

The first is that there is a depth past which the model's expressive claim outruns its search claim. Two agents holding contradictory false beliefs, each detouring around its own phantom wall, is a scene the language states without difficulty, but it names six latent literals across two nested forks, and no amount of budget we could give it brought a single such task within reach. The selective-attribution criterion above is therefore carried by the cheaper two-observer case --- two agents, one goal, one grid, where only one of them is attributed a belief --- and the contradictory-belief case is left as a limit, not a result.

#load-bib(read("refs.bib"))
