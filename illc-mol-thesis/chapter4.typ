#import "lib.typ": *

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

#mol-chapter("The Model", lbl: <ch-model>)

== The learning problem <problem>

In our model the learner is an observer shown tasks from a corpus $D$: sets of visual scenes that unfold over discrete time steps on a small grid (@corpus). For each task its goal is to recover the process that generated what it saw, by positing candidate hypotheses until one reproduces the scenes. A hypothesis is a program, a composition of symbols drawn from the current library, and the learner starts from an initial library $cal(L)_0$ of primitives (@primitives). Learning alternates two phases (@inference). In _search_, the learner finds for each task the shortest program under the current library that reproduces its scenes. In _compression_, it finds syntactic structure shared across the found programs. Commonly shared structure is named as a new abstraction and is added to the library, yielding $cal(L)_(i+1)$. Each abstraction turns what was a deep subtree into a single symbol, so compositions that were unreachable in one round come within the search budget of the next, and the learner's library grows iteratively more abstract, useful, and domain-specific.

We ask whether belief attribution (@sec-false-belief) is among the abstractions such a learner invents, given a domain-general initial library $cal(L)_0$ and task corpus $D$. In this context, to attribute a belief is to open a private copy of the world, modify it, run the planner against the copy rather than the world, and modify the real world according to the result computed in the copy (@signature). If the learner adds an abstraction with this structure to the library, then belief attribution is learnable by the same domain-general induction the learner applies to everything else. So the demonstration would be a counterexample to the claim that belief attribution must be acquired through a native domain-specific module @leslie_pretense_1987.

== The task corpus <corpus>

Our corpus consists of 316 tasks in 25 "families", groups of tasks that are all governed by a common underlying process. Of these 168 tasks in five families can only be solved by belief attribution, and 148 can be solved extensionally. The search doesn't know this distinction about tasks of course, it treats them all alike. 

Most tasks are _trajectory tasks_. A trajectory is a sequence of 2d grids over the time dimension, that depicts some dynamic at play (e.g. a body that falls, an agent that approaches something, etc.). A task consists of $k$ trajectories that all depict the same dynamic but vary in setup (where the entities start, how long the scene takes, what else is lying on the grid, etc). Given a task, the learner's goal is to find a transition function that, for any trajectory in the task, returns the next grid for any grid in the sequence. The transition function should be a general explanation of the underlying dynamic, invariant to initial conditions.

For example, the task below is an instance of the _constant-movement_ task family. It consists of four scenes that each depict the value 4 moving up. In the figure the grids of each timestep are collapsed into one, so each arrow indicates one timestep forward.

#figure(
scenes-figure("physics"),
kind: image,
//caption: [an instance of a _constant-movement_ task]
)

A solution to this constant-movement task would be a program that encodes _"at each time step, value 4 moves to the adjacent cell above it"_. Such a program solves the task because for each scene, given that scene's starting positions, the program reproduces the rest of the trajectory. 

The following task is an instance of a _goal-directed movement_ task. Value 1 is an agent moving towards its goal 2. Throughout this chapter, if we say "agent" and "goal" it's just for didactic reasons, since that's the role these values are playing. But they're not encoded as "agent" and "goal", to the observer they're just the values 1 and 2 on the grid. 

#figure(
scenes-figure("desire"),
kind: image,
//caption: [an instance of a _goal-directed movement_ task]
)

A solution to it would be a program that encodes _"at each time step, the value 1 moves to the adjacent cell closest to value 2"_. The solution involves the optimizing planner (@sec-planner), but _does not_ require attributing a belief. It's a purely teleological description.

The following is an instance of an _obstacle_ task. The same solution program for the above goal-directed task reproduces the first three scenes. But notice that in scene 4, value 1 takes a suboptimal path to value 2 (it should just goes straight to the right but it goes up first). So we need a program that accounts for this, so it solves all four scenes. 

#figure(
  scenes-figure("obstacle", caption: none, posited: 3, posited-scenes: ()),
  kind: image,
) <fig-obstacle>

The origin (0,0) is the top left cell. A solution might encode _"at each time step, value 1 moves to the adjacent cell closest to value 2, in a world where an impassable wall sits at cell (2,1)"_. The wall isn't part of the world that the observer sees. The observer hypothesizes that there's a wall there, to make sense of the agent's suboptimal movement. 

// the same task with the posited wall drawn in, hatched, in every scene
#figure(
  scenes-figure("obstacle", caption: none, posited: 3),
  kind: image,
) <fig-obstacle-wall>

The four scenes of a task are different "observations" of the same "world", governed by the same underlying process, so a solution to the task is a single program that solves all four. So even though only scene 4 needs the wall for its trajectory to make sense, the program posits a wall in the other scenes as well. The scene 4 trajectory could just as well be explained by a wall at (1,1) or (3,1), but those would interfere with the trajectories of the other scenes which pass through those points. So given the constraints, a wall at (2,1) is the solution. 

The solution to the above task may seem like ToM or belief attribution, but it is not. The observer does indeed postulate a world model that differs from her own, but she doesn't _attribute_ that divergent model to a particular agent. The solution _"at each time step, value 1 moves to the adjacent cell closest to value 2, in a world where an impassable wall sits at cell (2,1)"_ is a hypothesis about an agent's behavior in a counterfactual model. This is distinct from a hypothesis about an agent's behavior in _that agent's_ counterfactual model. It's not that the agent takes a wall to be at (2,1), it's the _observer_ who takes it to be there. So there's no belief attributed to the agent. 

These task families, among others, are "non-mental": they don't require belief attribution to be solved. The task families we care about are the mental ones, those that can only be solved by attributing a belief to an agent; since the mental state they all require attributing is a belief, we call them the _belief families_ from here on. The following task is one of them, an instance of the _goal-displacement_ family. This is our version of the Sally--Anne task. The light blue cell is value 1's final location at the end of the scene. So to the observer, value 1 moves towards an empty space which is consistently two cells above value 2. 

#figure(
  scenes-figure("belief_goal"),
  kind: image,
  //caption: [An instance of a goal-displacement task],
) <fig-belief-goal>

The task would be solved by a program that encodes _"at each time step, value 1 moves to the adjacent cell closest to the cell two cells above value 2"_. But this program isn't expressible in our language. The planner is an expression of the capacity for taking the teleological stance @gergely_teleological_2003, a goal-directed efficiency-sensitive measure. So it can only aim at an object on the grid, not at a location.#footnote[Coordinates do exist as terminals, but they are arguments to the two grid edits (stamping a wall, clearing a cell) and nothing carries them to a utility, nor is there any arithmetic on them with which to form "two above" in the first place. And the agent can't be given something to aim at, since the only primitive that writes at a named coordinate writes a wall, walls are impassable, and a wall would show up in the frames.] 

Instead, a candidate program that is expressible by our language is _"at each time step, move value 2 two cells up, then value 1 moves to the adjacent cell closest to value 2"_. But that doesn't reproduce the scenes correctly. Below is a frame-by-frame of the result of running the program on the first frame of scene 1. At $t_1$ value 2's jumps two cells up, at $t_2$ it's off the top edge and gone from the grid, so value 1 stays put since no adjacent cell movement decreases distance to value 2. The problem is that the program needs to relocate 2 but only for the purposes of serving as a reference for 1's movement. The program shouldn't _actually_ be moving 2 in the real world.

#figure(
  task-figure("belief_goal_rival", mode: "frames",
              data: json("rival_samples.json")),
  kind: image,
  //caption: [],
) <fig-belief-goal-rival>

What we need is a program that encodes _"at each time step, value 1 moves to the adjacent cell closest to 2, given that value 1 is navigating according to its own (false) world model in which value 2 is two cells above where it actually is."_ This is the belief attribution of BToM: a hypothesis that posits that an agent is navigating according to their own private model of the world, which may differ from the real world. The program posits the agent is navigating by its private model, but doesn't render the model as the real world. The private model affects the agent, but only the agent: the goal's position is unaffected. 

Another belief family are the _witness_ tasks. In the instance below, value 1 bends around cell (3,1) to navigate to value 2, while value 4 navigates to value 5 without avoiding (3,1).

#figure(
  scenes-figure("belief_witness"),
  kind: image,
  //caption: [An instance of a witness task. The believer (1) bends around the cell it wrongly takes to be walled --- here (1,3), and the same cell in all four scenes --- on its way to its goal (2), spending more steps than the direct route would have cost it. The witness (4), which seeks a goal of its own (5) and is wrong about nothing, walks through that cell. No frame of any scene contains a wall],
) <fig-belief-witness>

A hypothesis that posits a wall at (3,1) won't do, since value 4 wouldn't be able to pass through that cell if there were a wall there. So to explain these scenes, a program needs to differentiate between the model by which value 1 navigates and the model by which value 4 navigates. It needs to attribute a counterfactual model to value 1, in which there's a wall at (3,1). But to value 4 the observer doesn't need to attribute any belief, it can just explain its behavior with respect to the real world that the observer sees. So a solution program would have to encode _"at each time step, value 4 moves towards value 5. and value 1 moves towards value 2 according to a model that has a wall at (3,1)."_

A third belief family, the _two-observer_ family, combines the goal displacement with a second agent. In the instance below both value 1 and value 6 move towards value 2, but value 6 walks to value 2 itself, while value 1 walks to the empty cell one above value 2 and stops there.

#figure(
  scenes-figure("belief_observers"),
  kind: image,
  //caption: [An instance of a two-observer task],
) <fig-belief-observers>

As in the goal-displacement family, value 1's endpoint is explained by a private model in which value 2 sits one cell above where it actually is. Value 6's path needs no such model, it's explained by the world as the observer sees it. So a solution program has to encode _"at each time step, value 6 moves towards value 2, and value 1 moves towards value 2 according to its own model in which value 2 is one cell above its actual position."_ Both agents pursue the same value, and the false belief about it must be attributed to exactly one of them --- the same selectivity the witness family demands, but now with a single shared goal.

All of the task families so far, belief or not, are _trajectory tasks_ where for each scene the input is a starting grid and the goal is to find a transition function. A second, smaller group of tasks are _template tasks_. A template task's input is a pair of grids (canvas, template), and the goal is to find a program mapping the canvas and template pair to an output grid. The format is loosely inspired by the ARC-AGI-1 corpus [CITE].

For example, below is an instance of a _registration_ task. A program that solves this task encodes _"change value 1's position on the canvas to its position specified by the template"_.

#figure(
  task-figure("registration", caption: none),
  kind: image,
  //caption: [An instance of a registration task: the solution moves one named value in the canvas to the cell the template assigns it, and leaves everything else standing],
) <fig-registration>

Registration comes with a handful of variant families that turn the same alignment job around. In _perception_ tasks it's the template that is updated to record where the value stands on the canvas. In _drifting registration_ the alignment is carried out inside a moving image. We refer to them collectively as the _registration families_. The template families are in the corpus because they populate the pair interface from the non-mental side. To entertain two models of the world at once is not specifically a mental capacity. It's the domain general capacity to entertain a counterfactual, to "what-if". A program that consumes a canvas--template pair is aligning two images it was given, not holding or attributing a belief.

The template families are not the only non-mental claim on two-grid structure. Three _picture-processing_ trajectory families (_overlay_, _underlay_ and _comet_) depict a value that smears across the grid rather than moving through it. The cells it has passed through are never cleared, so each frame is the previous frame with the move unioned onto it. Reproducing that takes the same copy-and-collapse machinery as everything else in this section, but in those tasks it's used to paint motion blur, just graphics, with nothing mental in it.

We created the families in the corpus to claim all the pair combinators (@primitives), to show that the primitives of our library are domain-general. See the full corpus of task families in @app-corpus. 

== Program space: type-directed enumeration <space>

A hypothesis is a program, so the hypothesis space is the space of programs generated by type-directed enumeration. $G$ is the type of 2d grids of cell values, the frames of a scene. Write $G^*$ for stacks of grids, i.e. a scene (among the four in a task). A transition function alters a grid, so it's of type $G arrow.r G$. The cell values on the grid are members of the ground set $V = {0, dots, 9}$. Value 3 is reserved as an impassable wall, while the other values are interpreted as agent or goal or any other entity. The grid coordinates are $C = {0, dots, 4}$, a row or column index, from which a position in $C times C$ is built. The remaining sets are $D$, the four movement directions, and $U subset.eq (C times C arrow.r RR)$, the positional utilities that the planner scores (@primitives).

Everything else the language names is a function between these. The ones relevant to our argument are the functions into and out of the pair type $G times G$, which serves as the (world, model) channel pair. Whether a program opens a private second channel is a matter of which of those types its nodes inhabit, so a (world, model) representation is a structural property of the program. The fixed pair type does carry the commitment that a learner holding a private representation holds exactly one; generalizing the arity is left as further work.

A program is checked by running it through an interpreter (@app-interpreter). The solution to a trajectory task (@corpus) is a program of type $G arrow G$, a transition function that given the task's initial grid at step $i$ should return the next grid at step $i+1$. The interpreter iterates the program $G arrow G$ on the task's initial grid $G$, yielding the stack of grids $G^*$ which can be compared to the target scene. The interpreter's state at any step is just a single grid $G$, not a (world, model) pair $G times G$, and certainly not a registry of agents, goals, or entities. No pair is handed from one step to the next: a program that opens a second channel (one that may diverge from the world) must build it, use it, and collapse it again inside that step.

A template task starts from a pair in $G times G$, a canvas and a constant external template, so its solution is a _commit_ $G times G arrow.r G$, one of the collapses of @primitives. The commit answers what each frame should do with the template; the grid it returns is the next frame's canvas, while the template is re-paired with it unchanged. Since a type-directed enumeration yields programs of one type only, trajectory tasks ($G arrow G$-rooted) and template tasks ($G times G arrow G$-rooted) are searched separately (@enumeration). Though compression pools the solutions of both (@compression).

== The primitive library <primitives>

Our learner is initialized with an initial library $cal(L)_0$ of primitives. We designed our library so it would be domain-general, with the primitives low-level enough that they are each used in a variety of task families, combined in different ways with various other primitives. None of our primitives encode the structure of ToM, a belief attribution, or any other mental operation. For every primitive we added to the library we also included another that plays its complementary role. This way we have a point of comparison to show that the learner chooses certain primitives for a program but not others. The full list of primitives is in @app-primitives. 

To interface with elements on a grid, our library includes primitives for changing the values on grids. The _planner_, of type $U times V arrow.r (G arrow.r G)$, is the one from @sec-inverse: given a utility and a cell value it returns a transition function that at each time step moves that value one step toward the position maximising the utility. The two utilities it can be handed, both of type $V arrow.r U$, are minus and plus the distance to the nearest cell of a named value, so the same planner approaches a value under the one and flees it under the other. Beside it are the movements and edits, all of them transition functions once their arguments are supplied: a constant _step_, of type $V times D arrow.r (G arrow.r G)$, which moves a named value one cell in a named direction; _stamping a wall_ at a position and _clearing_ a single cell, both of type $C times C arrow.r (G arrow.r G)$; _erasing_, of type $V arrow.r (G arrow.r G)$, which scrubs every cell of a named value from the grid; and _composition_, of type $(G arrow.r G) times (G arrow.r G) arrow.r (G arrow.r G)$, which runs one transition function and then another. The terminals are the four directions, the ten cell values, and the five coordinates, a pair of which names a position.

Since our programs involve pairs of grids $G times G$, we need primitives for accessing values in them. A _reader_, of type $V arrow.r (G arrow.r C times C)$, gives the position of a named value in a grid. A _writer_, of type $V arrow.r (G times (C times C) arrow.r G)$, gives the grid with that value moved to a given position. A third primitive, of type $(G arrow.r C times C) times (G times (C times C) arrow.r G) arrow.r (G times G arrow.r G)$, joins a reader to a writer. It reads a position off the second channel and imposes it on the first.

The rest of the primitives are the _pair combinators_: standard, content-blind operations on a pair of grids that never look inside either of them. There are four jobs to be done with a pair, and a group of combinators for each: building a pair out of a grid, transforming a pair, collapsing a pair back to a grid, and reading and writing the positions a collapse goes off of. In what follows $w$ and $m$ are grids, $f$ and $f'$ transition functions, and $v$ a cell value.

Two combinators _build_ a pair, both of type $G arrow.r G times G$. The _copy_ $w mapsto (w,w)$ puts a grid in both channels. Its complement $w mapsto (w, bold(0))$ pairs the grid with an empty one instead, so that the second channel starts as scratch rather than as a copy of the world.

Four combinators _transform_ a pair, all of type $G times G arrow.r G times G$. Three of them _map_ a transition function over the channels, leaving the pair a pair: $(w,m) mapsto (f(w),m)$ applies it to the first channel, $(w,m) mapsto (w,f(m))$ to the second, and $(w,m) mapsto (f(w),f'(m))$ applies one to each. Each is a separate primitive. The fourth is the _swap_ $(w,m) mapsto (m,w)$, which exchanges the channels, so that anything defined on one of them can be turned into a version acting on the other by swapping before and after.

Several combinators _collapse_ a pair to a single grid, of type $G times G arrow.r G$. The two projections, $(w,m) mapsto w$ and $(w,m) mapsto m$, return one channel and discard the other. Two more union the channels cell by cell, differing only in which of them wins where they disagree. The last two are _scope_ collapses. One moves every value the two channels share to the position it holds in the second. The other, of type $V arrow.r (G times G arrow.r G)$, does the same for every shared value but one named exception. So they both adopt the second channel's value but they differ in how much of it to adopt. 

Each of these arrows needs its own composer, since the composition of the grid layer joins two transition functions only. Three are given, one for each arrow that needs one. A builder followed by a transform, of type $(G arrow.r G times G) times (G times G arrow.r G times G) arrow.r (G arrow.r G times G)$, extends a builder. A builder followed by a collapse, of type $(G arrow.r G times G) times (G times G arrow.r G) arrow.r (G arrow.r G)$, is the composer that returns to $G arrow.r G$; since the interpreter only ever iterates a transition function, a program that opens a second channel must close it again within the same time step, and this is what closes it. A transform followed by a collapse, of type $(G times G arrow.r G times G) times (G times G arrow.r G) arrow.r (G times G arrow.r G)$, is the only one that acts on a pair the program did not itself build, which is what a template task's given pair calls for.

Nothing in any of these primitives distinguishes a world from a model. The pair is symmetric: both factors are grids, every map is present in both directions, and no primitive marks either channel as a particular agent's private representation. Which channel is the world and which is a belief is settled only by how a program assembles these pieces together, which is found by search. 

== Composing belief attribution <signature>

In @sec-false-belief we presented belief attribution as a two-step computation. First, the observer must _derive_ a counterfactual model which deviates from her own according to the agent's false belief $m' = "derive"(m)$. Then the observer _runs_ the agent's movement policy on the counterfactual model $t' = "optimize"(g,m')$ to yield the agent's simulated trajectory which is optimal w.r.t the agent's world $m'$. 

This structure can be constructed by composing the $G times G$ primitives from @primitives. In our setup only the _run_ step is done by a primitive, the planner, which builds the agent's movement policy. Nothing in the library opens a private copy, so the _derive_ step of creating and modifying a copy of the grid must be discovered by the learner.

Our setup also has a third step, _commit_, which writes the result of the private computation back to the world. This step doesn't have a counterpart in @sec-false-belief. BToM's observer runs the planner inside herself and gets the trajectory $t'$ back as a value, which she then compares against the observed $t$. What happened in the private model never has to re-enter the world, because the observer is the one holding both. Our programs have no such vantage point. A hypothesis is a transition function $G arrow.r G$ and the interpreter renders one grid per step (@space), so a move made on a private copy is invisible unless something explicitly writes it into the grid that gets rendered. Without a commit, nothing an agent does on its own model ever reaches the predicted trajectory outputted by the program.

Another reason why our setup has a _commit_ step is that the commit is where attribution lives. In BToM, that $g$ and $m'$ are agent $a$'s is settled by the modeler before the computation starts: they are handed to the planner as $a$'s arguments, and the trajectory that comes back is $a$'s by declaration. Our learner has no place to make that declaration, since no type says whose a grid is and the interpreter cannot tell a private copy from any other grid (@no-btom). So if a counterfactual grid is to be _somebody's_ rather than merely counterfactual, the program has to specify that. The commit is where the program can encode which unique entity's movement is determined by the counterfactual model, i.e. the entity who has that model as their private model. The commit separates between positing a counterfactual world and positing it as the world a particular agent is acting on. 

So in our setup there are two target structures needed for recreating the belief-attribution structure. We call them the _single-value commit_ and the _derive-and-commit frame_. The _single-value commit_ is of type $V arrow.r (G times G arrow.r G)$. Given a value, it reads where that value sits in the private copy and moves the same value in the world to that position, so exactly one value's position is published and everything else stands as the world had it.

The derive is the transition function $(G arrow.r G)$ that is applied to the believing agent's private copy. The _derive-and-commit frame_ is of type $(G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$. Handed a derive and a commit, it copies the world into a pair, applies the derive to the second channel to get the private copy, and gives the pair to the commit, which collapses it back to one grid. The private copy exists only for the duration of that call.

Belief is the composition of the two, with the derive applying the agent's error and the commit doing the attribution. The derive-and-commit frame $(G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$ applied to the derive $G arrow G$ and commit $G times G arrow G$ yields a transition function $G arrow G$. 

We'll walk through the process of belief attribution through derive-run-commit on a _false-wall_ task. This is a simpler variant of the witness task (@corpus): rather than a witness whose path crosses the location of the agent's falsely believed wall, the bystander remains static at that location the entire run. In our experimental setup, 3 is an impassable wall, and any value other than 3 is a generic value through which another value can pass.

Take scene 1 below. An agent without a false belief that there's a wall would just take the direct path to 2. So value 1 must believe there's a wall somewhere between it and 2. The only available cell for a wall there is at (2,1), since value 1's trajectories pass through (1,1) and (3,1) in scenes 2 and 3 respectively. 

#figure(
  scenes-figure("belief_wall")
)

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

We'll now show frame-by-frame the derive-run-commit process, for scene 1. Observer starts with its initial world model. It then _derives_ a copy of the grid and applies a transformation (placing a wall) to it. So the private copy now has wall 3 where the bystander 5 was before. The observer then _runs_ the planner on the private copy, which yields agent 1's next position according to that copy. Now that the observer has agent 1's position, it _commits_ that value to the world grid, and only that value. So by the second timestep, nothing in the world is changed except for agent 1's position, which was changed according to its private model. So the wall never reaches the shared world, and the bystander 5 remains untouched. 

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
  //caption: [],
) <fig-fork-anatomy>

Note that the agent value $a$ occupies three slots in that program: the slot that says whose move is planned over the privately walled grid, the slot that says whose position is read off the copy, and the slot that says whose position is written back to the world. So to find this compound --- a composition of several symbols that is not itself in the library --- the search needs to independently fill the argument slots of the three primitives with the same value placeholder. We call that coincidence the _agency signature_, the structural mark that separates an attribution (a private grid that is _somebody's_) from a mere counterfactual world.

The Sally-Anne false belief task is easily solved by an observer whose library carries belief attribution as a single symbol. As stated in @corpus, the task below is solved by a program that encodes _"at each time step, value 1 moves to the adjacent cell closest to 2, given that value 1 is navigating according to its own (false) world model in which value 2 is two cells above where it actually is."_

#figure(
  scenes-figure("belief_goal"),
  kind: image,
  //caption: [An instance of a goal-displacement task],
) <fig-belief-goal>

We can formulate that program purely in terms of grid operations, as:

#quote(block: true)[at each time step, _derive_ a model of the grid by moving value 2 two steps up in a copy of the grid. Then in the model, _run_ the planner to find the adjacent cell closest to value 2 and move value 1 to that cell. Then _commit_ value 1's location from the model to the world, i.e. in the world move value 1's position based on its position in the model, and don't move anything else.]



== Inference <inference>

Our procedure runs in alternating rounds of exploration and compression, the loop of @dechter_bootstrap_2013 as in DreamCoder @ellis_dreamcoder_2020. The exploration phase performs per-task inference, enumerating programs under the current library for the tasks not yet solved (@sec-search). The compression phase revises the library to compress the programs found (@sec-hbm).

DreamCoder's third phase trains a neural recognition model that reads a task and emits a task-conditional prior $Q(dot|x)$, so that enumeration proceeds under a $Q$ biased toward the primitives that task is likely to need. We don't implement it, because of the difference in scale between our domain and the DreamCoder experiments.

The amortized inference that the recognition model offers is essential when the branching factor is large and the loop is long. DreamCoder works with libraries of hundreds of symbols over dozens of iterations. Ours is a small library run for a few rounds with fewer than 50 symbols. There is no long tail of iterations across which a learned proposal could amortize anything.

=== Search <enumeration>

The exploration phase finds a hypothesis that correctly explains a scene, by enumerating candidate programs in decreasing order of prior probability and checking if the program produces the scene. A hypothesis is a program (@sec-programs), represented as a typed expression tree over the current library (@space). Since enumeration in decreasing prior probability is enumeration in increasing description length, the search walks outward in expanding bands of cost (expanding $tilde e^ell$, see @sec-search). The first program it finds that reproduces the data is the shortest one that does, which is the maximum a posteriori hypothesis ($h^*$, see @sec-programs). The enumeration runs under a time budget, and a hypothesis whose cost band is not reached before the budget expires is simply not found (@sec-search). 

// Programs are represented as an expression tree over the library of primitives. Each node carries a type together with the types of the arguments it expects. A node with no arguments is a terminal whose value is itself, while a node with arguments is an operator applied to its filled-in arguments. The tree is directly executable, it's evaluated from the root by recursively evaluating its child nodes. 

The library fixes the prior. The prior $P(h) prop e^(-"DL"(h))$ (see @sec-programs) is realised as a distribution over the library's symbols, under which a program's cost is the summed $-log p$ of its tokens, the occurrences of library symbols at its nodes. Symbols are grouped by return type, and a symbol of type $tau$ is priced at $-ln$ of the number of symbols sharing $tau$, i.e. a type-uniform distribution. So the types give the branching factor at each argument slot, and the cost of each choice at that slot. Enumeration is type-directed: an argument slot draws only from symbols whose type matches, so no ill-typed tree is ever built. (Note that one exception to the type-uniform distribution is that a cell value which is visible in the task's first frame costs nothing, since naming what is already on display is not a hypothesis about anything. But a placed wall is full-price, since it's positing something that isn't there). 

A candidate program solves a task only if the interpreter, started from each scene's own first frame, reproduces that scene's every frame exactly, for all $k$ scenes (@corpus). So the indicator likelihood $bb(1)[h "solves" d]$ (@sec-bayes) is evaluated against all $k$ scenes. 

=== Library revision <compression>

Compression returns a set of abstractions together with the corpus rewritten to use them. Each abstraction is registered as a new symbol in the library. So what was a subtree found in round $i$ becomes a single symbol in round $i+1$, so the next enumeration stage can reach deeper into program space (@sec-hbm) and the learner bootstraps more complex programs. 

Where exploration minimizes description length one hypothesis at a time against a library held fixed, compression minimises the total description length. It minimizes the length of the library itself plus the summed cost of every solved program written in it @ellis_dreamcoder_2020, the posterior over languages (@sec-hbm).

$
"DL"("library") + sum_"solutions" "DL"("program" | "library")
$

When a syntactic structure is common to a lot of programs, it gets added to the library as a new abstraction. So the description length of the solutions shrinks, since rewriting the solutions replaces the common subtrees with the abstraction's symbol (so $sum_"sols" "DL"("program" | "library")$ shrinks). But the library is now larger, and since a symbol's probability falls as the number of symbols competing with it rises, every program written in the enlarged library gets a little more expensive (so $"DL"("library")$ grows). This is the tradeoff that needs to be made when deciding whether to abstract some shared structure. The two summands ensure that an abstraction is kept only when the tokens it saves across the whole corpus outweigh the price of adding it. So we're less likely to abstract incidental structure, and more likely to abstract structure that's relevant to the corpus. 

DreamCoder looks for abstractions bottom-up, enumerating refactorings of each found program and intersecting them @ellis_dreamcoder_2020. Instead we delegate the search to the top-down Stitch @bowers_top-down_2023 library-learning system, which searches the space of abstractions directly and so never has to materialize every rewrite. Also, note that enumeration searches by root type (@space), so solutions to trajectory tasks (transition functions $G arrow G$) and template tasks (commits $G times G arrow G$) are searched for separately. This is just an implementation necessity which doesn't affect the overall outcome, since compression searches for common structure among the full pool of solutions together regardless of root type. 

== The setup does not encode BToM <no-btom>

The goal of our setup is to recover the belief attribution structure from Chapter 3 without stipulating it ourselves. A natural objection to our results would be we've merely re-encoded Chapter 3's stipulations in a different notation so the learner trivially "finds" them. To pre-empt this, we'll argue that no notion of belief attribution, agency, or world/model distinction is encoded in our setup.

In our primitive library we give the pair type $G times G$ and the pair combinators over it. The pair is symmetric and content-free: the copy produces $(w, w)$, and nothing in it says the second copy is a _model_ or that anyone _holds_ it. Belief is assembled by composition: acting on the second channel rather than the first, committing directly rather than conjugating through the swap, publishing one named value rather than every value the channels share. Each of those choices has its opposite in the library, and each opposite is exercised by some non-mental family in the same run. So the medium is neutral, and any choice of particular abstractions is made because they are useful to the corpus. 

The interpreter does not supply ToM either. In BToM the divergent model is an input to a planner that the observer runs inside itself, so the observer already distinguishes between its own model and any counterfactual model. Our interpreter has one grid of state, no second channel to hand anybody, and no mode distinguishing evaluating a program from evaluating a program-as-an-agent-would. A private copy exists only because a program built it.

One might then argue more strongly that the ability to build a copy of the world that differs from the world just _is_ the representational capacity of ToM. But it is not. Building a world state that differs from the actual one is just counterfactual reasoning, the capacity to "what-if", to consider what would be the case if things were otherwise. It's a domain-general ability, and infants have this ability way before false-belief attribution.

ToM is an abstraction that _uses_ the counterfactual capacity, by _attributing_ the divergent world to a particular agent, so that it is _that agent's_ world. That attribution is the conjunct the agency signature encodes (@signature). The primitives don't supply that signature: the derive-and-commit frame can just as well derive a private copy, compute over it, and publish an unrelated value rather than the agent's movement. 

In BToM the modeler must also fix the ways in which a private model may diverge from the observer's, and the scenarios that produce the divergence. In our setup there's no range to fix, since the derive slot accepts any transition function $G arrow.r G$. So the different task families that require false belief attribution differ in what the belief is about. 

BToM's goal space, and the prior on it, are given by the modeler @baker_action_2009. In our setup, a goal is whichever cell value happens to fill the argument of the attracting utility, drawn from the same ten integer terminals that supply agent identifiers and step magnitudes. There is no goal type and no goal prior. A value seeks something because a symbol landed in that slot. Even the polarity is a search choice, since the repelling utility sits alongside the attracting one, so maximising utility flees a value as readily as it approaches one. The only prior over any of it is the type-uniform distribution, which prices a goal value by the same rule that prices a direction or a composition. The same applies to "agents", which have no special type or status. We just call a value an "agent" when it's used in a program in a way that it plays the role of an agent, like having a belief attributed to it. 

Rationality is the one stipulation of BToM that our learner also receives, in the form of a planning primitive. We grant it because it's not a mental operation: the teleological stance relates an action, a goal state and the constraints of a situation without attributing any representation to the agent @gergely_teleological_2003. Planning is in place well before the first birthday and some three years before anything recognizable as false-belief attribution (@sec-planner, @timeline) @gergely_taking_1995. 

The transition function given by the planner $U times V arrow.r (G arrow.r G)$ just takes a single grid as input. It has no parameter identifying whose grid it is and no access to any other channel, it just modifies a grid like any other grid-edit primitive. It cannot represent a divergence of models because it cannot see two things to diverge, and it cannot attribute anything because it has no argument for an attributee.

Belief arises only when the surrounding program hands the planner a grid that is not the world, and then publishes only part of the result (@signature). And furthermore the planner is used in a variety of task families that don't involve belief attribution. The planner is certainly a hefty primitive, packing a full breadth-first search into one token. But this algorithmic complexity isn't evidence that the planner is doing mental work, that it grants a theory of mind in and of itself.

In BToM the posterior ranges over goals and models $(g,m)$. So nothing in the framework makes a goal attribution earlier or cheaper than a belief attribution, the inference proceeds the same over both. This isn't a problem with BToM, since it models mature ToM capacity where we can reasonably assume that the difference in processing power between goal attribution and model attribution is negligible. But for our purposes we're interested in the developmental timeline, in explaining the fact that goal attribution arrives at least two years before belief attribution. So our setup accounts for that delay: a belief compound (@signature) is a deep composition whose cost under the initial library puts it far outside any feasible budget window, and it becomes reachable only once earlier solutions have been compressed into abstractions that shorten it.

== Success criteria <criteria>

In our setup, successfully acquiring the ability to attribute belief means the learner adds an abstraction to its library whose body (i) opens a private channel, (ii) derives from it a world state that diverges from the actual one, (iii) evaluates an agent's policy against that divergent state, and (iv) commits the agent's resulting move, and nothing else of the divergence, back to the world.

Condition (iv) is what distinguishes attributing a belief from positing a counterfactual. A private copy is _somebody's_ belief only if it's used in a way that makes sense with respect to the world and agent. Take  @fig-fork-anatomy, where the private model shows agent 1 navigates according to a wall, which is agent 1's false belief. The relevant value that should be committed from that model is the agent, not the wall. The agent's movement is caused by something false, but that movement is real. 

Our language allows for several ways of demarcating which value or values get committed from the model to the real world. In the @fig-fork-anatomy example, the commit is "publish value 1", naming the one value that goes through from the model to the world. The same commit can be formulated as the complement, "publish everything except the believed content (the wall)", holding back the false value while everything else moves to where the model has it. And the commit can be left to a default, to just publish the union of values, the values that the grids share.

For our purposes these spellings are interchangable (mostly because our tasks are so small, in bigger tasks the difference would be meaningful). For example, the commit from @fig-fork-anatomy could be rewritten as "commit everything except the wall", since the wall is the only value in the model that is false or whose position deviates from its position in the real world. 

Regardless of whether the commit is formulated as inclusion or exclusion, the program would only count towards the criterion if the value the commit takes is bound correctly. The value whose move reaches the world must be the value the policy of (iii) moves, and no false belief postulated by the derive of (ii) may reach the real world.

So which values need to be bound depends on which commit spelling the program uses. For example, if the program uses the inclusion commit "publish value 1", then value 1 must be the value that moves according to the false belief. Similarly, if the program uses the exclusion commit "publish everything but value 3", then value 3 must be the value that is false in the private model. A program whose commit publishes an unrelated value (or renders the private copy in its entirety including a posited wall) would just be a counterfactual world with no owner. So it would fail the condition on account of a binding mismatch (@signature). 

Other than the structure of the found abstractions, we are also interested in the process by which the abstractions are selected for the library and this is the most relevant factor for our original point of inquiry about the acquisition of ToM. 
So our success criteria also include that the abstraction must be selected by the joint compression objective of @compression. An abstraction should be selected only if the description length it saves across the solved corpus outweighs the cost of registering it.

The components that an abstraction is assembled from must be demonstrably domain-general. They should occur across the corpus in different combinations, solving a variety of families of tasks. The belief-attribution abstractions must arrive by the bootstrapping detailed in @inference. They should be out of reach under $cal(L)_0$, but within reach after earlier solutions are compressed. 

In Chapter 5 we report the run against these criteria, including the library it converged to and the bodies of its abstractions (@sec-found), and the order in which those abstractions were assembled (@sec-order).

#load-bib(read("refs2.bib"))
