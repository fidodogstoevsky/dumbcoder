#import "@preview/illc-mol-thesis:0.2.0": *

#import "world_tape.typ": world-tape, grid-view, arc-colors

#import "viz.typ": task-figure, all-tasks

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

#mol-chapter("The Model")

== The Learning Problem <problem>

In our model the learner is an observer, in a setup analogous to a child in a developmental psychology experiment. In each task the learner is shown a set of _scenes_, visual episodes unfolding oer discrete time steps. For each set of scenes, the learner's task is to recover the underlying process that explains the scenes, by positing canididate hypotheses until it finds the correct one. We model hypotheses as programs, so a hypothesis is correct iff executing the program reproduces the scenes exactly (every frame, not just the final one).

The learner's 

Scenes come in families that 

each emphasize a particular concept which explains the underlying process. In particular we study mental families, i.e. scenes that cannot easily be explained by extensional programs that just describe the observable surface of the scene. For such scenes the shortest available explanation must instead posit a representation held by an agent that diverging from the world, must attribute an intensional representation to an agent. 

In our model, theorizing about the world is a program synthesis problem while concept learning is a library learning problem. 

We ask whether a learner equipped with nothing mental to begin with will converge on a mental explanation because it is the most compressive one. 

To learn, in our setting, is to grow one's conceptual vocabulary by enriching it with useful conceptual abstractions. So our claim is that a reusable conceptual abstraction corresponding to belief attribution appears in a library that began without one.

@audit through @primitives fix what the learner is given, what it is shown, what it can express, and where it starts; @inference gives the inference procedure; @criteria states what would count as the claim succeeding or failing.

== What is given, and what must be learned <audit>



The argument of @problem is only as strong as the audit of the learner's starting point. If any part of the apparatus — the interpreter, the type system, the primitive library — already encodes the world/model distinction, or already carries a notion of agency, then a discovered belief abstraction shows nothing, because belief was present from the start in a thin disguise. This section makes the audit explicit.

#block(inset: (left: 1em))[
  *TODO:* the given/not-given table. Left column, _given_: the minimal interpreter (@interpreter); the domain-general core primitives (@primitives); the monomorphic type system; the MDL objective; the exact-match success criterion. Right column, _not given_: any primitive or type denoting an agent, a goal, a belief, or a mental state; the world/model channel distinction; any commitment in interpreter state to more than one representation of the world; any task-specific bias toward the belief families. Each row should point at the section where the corresponding claim is discharged.
]

=== The interpreter <interpreter>

A hypothesis is a transition function, not a scene. A trajectory scene is generated from its first frame by iterated application, so for a candidate transition function $f$ and first frame $x_0$ the generated scene is fixed by the recurrence $x_(i+1) = f(x_i)$. That recurrence lives in the interpreter rather than in program space, and the choice matters for two reasons.

The first is compositionality. "Iteratively apply $f$, starting from $x_0$, for $n$ timesteps" is structure common to every solution in the corpus. Were it part of the program, it would be baked into every abstraction the learner discovers: a corpus of ten falling-object scenes would yield not the reusable $lambda$`0`. `(step down #0)` but an overspecified _apply `(step down #0)` from $x_0$ for `#1` steps_, whose output type is an entire rendered scene. Such an abstraction cannot be combined with another, so it could never serve as a component of the deeper compounds that belief requires, and library learning would be defeated at the first round. Keeping the recurrence out of program space keeps abstractions `fn`-typed and therefore composable.

The second is that the recurrence carries no information the learner could be credited with discovering. Since it is shared by every solution, what search is actually looking for is only $f$; the rest is what is needed to render a scene from $f$ in order to check it against the observation. So we move the rendering machinery out of program space and into the evaluation framework (the interpreter is given in full in [app-interpreter]).

The interpreter is minimal by design, and its minimality is the load-bearing part of the audit. Its state at any step is a _single grid_. It does not encode a (world, model) grid pair, and it holds no registry of agents, goals, or entities: there is nothing in it that distinguishes a cell playing the role of agent from a cell playing the role of obstacle. There is likewise nothing representational about it, since it only ever builds up one sequence of grids. The ability to hold multiple representations of the world, to hold a representation that deviates from the world and to attribute that representation to a particular agent, is nowhere in the interpreter. So assuming the interpreter as innate to the system does not smuggle in a theory of mind. Whether a program opens a second, private channel is a _structural_ property of the synthesized program — a matter of which types its nodes carry (@types) — not a commitment built into interpreter state.

Template tasks (@corpus) are evaluated against a second interpreter, which threads a working grid while pairing each frame with a _constant external_ template and applying a commit policy of type `fn_p_g`. The distinction from the trajectory case is not incidental bookkeeping but a control on what the pair types can be taken to mean: here the second channel is a given input rather than a privately derived model, so a program that consumes the pair with `sync_to_world` is performing registration, not holding a belief. The same primitive vocabulary therefore does mental work in one interpreter and non-mental work in the other, which is what lets the corpus price belief against genuinely non-mental competitors that use the very same combinators.

== Scenes and the task corpus <corpus>

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

=== Tasks as sets of scenes

#block(inset: (left: 1em))[
  *TODO:* the $k$-scene task format. A task is not a single scene but $k$ scenes (currently $k=4$) sharing one latent program, and a candidate solves the task only if it reproduces all $k$. State why this is a model-level commitment rather than an implementation choice: a single scene underdetermines its generating program, so a program can reproduce one belief scene by coincidence — fitting the particular geometry of that scene rather than the dynamic it depicts. Requiring one program across $k$ independently sampled scenes of the same family is what makes the identifiability condition of @identifiability bite, and it is what the rival measurements of @criteria are measured against. Give the measured effect: rival programs that pass on one scene of the goal-displacement family fail on all four.
]

=== The families

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

=== Identifiability of the mental families <identifiability>

The corpus only tests what it claims to test if the belief families are not solvable extensionally. A scene in which an agent detours around a wall that isn't there admits a mental explanation, but it may equally admit a shorter non-mental one, and if it does then a learner that solves it has demonstrated nothing about mentalizing. So membership in a mental family is a condition on the scene, not a label attached to it: the scene must be one that no program describing only its observable surface can reproduce.

The condition can fail in ways that are easy to miss. A wall false belief scene might place the falsely-believed wall at a location nowhere near the agent's path to its goal, in which case the agent navigates to the goal exactly as it would have without the false belief, and positing the belief is neither necessary nor helpful in explaining what is seen. Such a scene is underdetermined and is excluded from the corpus. The generators enforce the condition by rejection, and the checks they apply are given in [app-generators].

== The hypothesis space

=== Programs as typed expression trees

A hypothesis is an expression tree over the library. Each node carries a type, together with the types of the arguments it expects; a node with no arguments is a terminal whose value is itself, like `up :: dir` or the literal `4 :: cellvalue`, and a node with arguments is an operator, like `step :: cellvalue, dir -> fn`, which is applied to its filled-in arguments. The tree is directly executable, and evaluating it bottom-up yields the object its root type names — so `(step 4 up)` evaluates its two leaves to the cell value `4` and the direction `(-1,0)` and applies `step` to obtain a transition function `fn :: grid -> grid`.

Enumeration is type-directed: an argument slot draws only from symbols whose type matches, so no ill-typed tree is ever built. This is why the type system is a substantive part of the model rather than bookkeeping — it is the type discipline that determines which compounds are even expressible, and therefore what the learner could in principle discover.

An invented abstraction is a node like any other, but it names a body: a tree over lower-level primitives with numbered holes standing in for its parameters, which is substituted in when the node is called. The point of naming is that the name is what gets charged. A program's description length is its node count, one for a leaf and one plus its arguments for an operator, and an abstraction used as a token adds just one, because the subtree it abbreviates lives inside the body and does not appear in the program that uses it. Description length is thus the quantity in which library learning trades: an abstraction converts many charged nodes into one. The representation that implements this is given in [app-delta].

=== The library

The library is the learner's language of thought, and it has two parts: the `core` primitives a run begins with, and the `invented` abstractions that compression adds to it as they are discovered. Search always draws from their concatenation, so an abstraction discovered in round $N$ is available as an atom in round $N+1$ on exactly the same footing as a primitive that was given.

The library also fixes the prior. Symbols are grouped by return type, and a symbol of type $tau$ is priced at $-ln$ of the number of symbols sharing $tau$ — the type-uniform distribution. So the grouping does double duty: it gives the branching factor at each argument slot and the cost of each choice at that slot (@enumeration). It also means growing the library is not free, which is what makes the compression tradeoff of @compression non-trivial.

=== A monomorphic type system <types>

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

== The primitive library <primitives>

The library a run begins with is the model's substantive claim about the learner's initial endowment, and the three phases below weaken that endowment in turn. Each phase hands the learner a strictly more elementary starting vocabulary, so that a belief abstraction discovered in a later phase is discovered from less.

=== Core primitives

every run's library starts with core primitives used to construct the transformation functions. `step :: cellvalue, dir -> fn` moves the value one step in a specified direction, chosen among `left :: dir`, `up :: dir`, etc. `optimize :: util, cellvalue -> fn` moves the value one greedy (BFS-optimal) step towards improving its utility, chosen among `distance :: cellvalue -> util` or `neg_dist :: cellvalue -> util`.

#show table.cell.where(y: 0): set text(style: "normal", weight: "bold")
#set table(stroke: (_, y) => if y == 1 { (top: 0.9pt) } else if y > 1 { (top: 0.2pt) })

#block(inset: (left: 1em))[
  *TODO:* the full core primitive table (symbol, type, gloss), or a pointer to it in the appendix.
]

=== Atomic (phase 1)

In our first trial (phase 1) we show that, given the domain-general capacity to represent counterfactual world states, a system can learn a functional abstraction that encodes a notion of belief attribution. The learned function attributes a counterfactual world state to a particular agent, determines what the agent's movement would be based on its own private (non-world) representation, and then renders only the agent's movement as a real-world artifact (so it doesn't render the agent's believed world as the real world). So this use of the capacity for counterfactual representation is discovered in program space.

The capacity is given as two primitives:

- `fork :: fn -> (grid, grid) -> fn`
- `sync_to_world :: cellvalue -> (grid, grid) -> grid`

=== Decomposed (phase 2) <decomposed>

In the second trial (phase 2), we show that the system can arrive at an abstracted notion of belief starting from an even more basic language of 2-ary combinators. We decompose the counterfactual representation machinery, initializing the system with a library of product-category combinators that are unambiguously domain-general. We show that enumeration recovers the basic structure of belief attribution even from these atomized primitive operations.

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

=== Arity (phase 3) <arity>

#block(inset: (left: 1em))[
  *TODO:* the arity-generalization phase. The `pair_gg` type of phases 1 and 2 stipulates that a learner holding a private representation holds _exactly one_, which is a commitment the model should not be making on the learner's behalf — a one-model frame is precisely what a theory of mind is claimed to have. Replacing the pair with the recursive grid-stack `gstack` (`cons`/`nil`, with the lift/endomorphism/render arrows `fn_g_s`, `fn_s_s`, `fn_s_g` of @types) makes channel arity a free parameter, so the number of private channels becomes something search selects rather than something the type system fixes. State the verdict: MDL never selects more than one private channel, so belief's single-model frame is discovered rather than stipulated.
]

== Inference <inference>

ECD is a wake-sleep library-learning loop inspired by the DreamCoder @ellis_dreamcoder_2020. Each round consists of an enumeration phase, a compression phase, and a dreaming phase, and the three correspond to the three moves of the previous chapter: search is Bayesian inference over programs under a prior; compression is revision of the language in which hypotheses are written; dreaming is amortization of the first by a learned recognition model.

=== Search <enumeration>

In principle the enumerator doesn't itself set out to synthesize a solution program for each particular task. Rather it generates candidate programs arbitrarily from the grammar in order from most to least probable (from lowest to highest cost), checking candidate programs against the task corpus.

Say the task to solve depicts the value 3 rising in each successive grid, so the target solution is the program `(step 4 up)`. Its head is `step :: cellvalue, direction -> fn` and its leaves are `4 :: cellvalue` and `up :: direction`. A program's cost is given by the sum of the costs of its constituent primitives. Say $Q$ is the initial flat prior, the type-uniform distribution, so each primitive's cost is the negative natural log of the number of symbols in the grammar that share its type. For example, there are $N=4$ symbols of type `direction` so each gets $p = 1/N = 1/4$. By Shannon, we have cost(`up`) $= -ln p = -ln 1/4 = ln 4 approx 1.386$ nats. By the same process, `step`'s cost is $ln 8 approx 2.1$ nats and `4`'s cost is $ln 10 approx 2.3$. By logarithmic additivity, to find the program's total cost we simply sum the costs of its constituent primitives which yields $approx 5.8$ nats for `(step 4 up)`.

Search proceeds cheapest-first, so the solution it returns for a task is the shortest program that solves it under the current library — the maximum a posteriori hypothesis, given that the prior prices a program by its description length. A candidate counts as solving a task only if the interpreter, run on the task's own first frame, reproduces the task's every frame exactly. Each still-unsolved task is enumerated as its own cost-ordered stream, which is what allows the prior $Q$ to be conditioned on the individual scene rather than shared across the corpus — a fact we rely on in @dreaming. The mechanics of cost-bounded search are given in [app-enumeration].

=== Library revision <compression>

While Enumeration minimizes the cost of each program under a fixed grammar, Compression lets the system revise the grammar in light of the solutions found by enumeration. It identifies syntactic structure common across found solutions, and abstracts it as a new primitive to add to the library. The goal of compression is to minimize the joint cost of the library plus the programs written in it, given by:

$
min_"library" ("DL"("library") + sum_"tasks" "DL"("program" | "library"))
$

The choice of whether to abstract a given shared structure is non-trivial. Consider a program tree fragment that recurs across many solutions. Say it gets added to the library as a new primitive `fn_k`, so now each subsequent usage of the fragment collapses from a multi-node subtree to a single leaf, so it costs one token rather than the several that `fn_k` abbreviates. But the library is now larger, and since a symbol's cost is $-ln$ of the number of symbols sharing its type, adding one more `fn_k`-typed symbol raises the cost of every primitive of that type. The abstraction is worthwile only when the fragment is large enough, and recurs often enough, that the tokens saved across the corpus outweigh the price of carrying one more symbol. So accidentally-shared structure is never abstracted, while shared structural motifs do.

Where DreamCoder @ellis_dreamcoder_2020 searches for abstractions bottom-up, by enumerating refactorings of each found program and intersecting them, we delegate the search to the top-down Stitch @bowers_top-down_2023 library-learning system, which searches the space of abstractions directly and so never has to materialize every rewrite. The search procedure and its utility bound are given in [app-stitch].

enumeration is by root types so `fn` and `fn_p_g` are searched separately. but compression is joint over root types, their solutions are pooled into a single stitch call. so an abstraction for belief has to fight over the space of all solved programs, including those from the other root type. so it really is that much more efficient. This preempts the attack that by separating the search to different root types we're biasing abstraction creation towards the belief tasks since they make up a higher percentage of `fn`-rooted tasks rather than `fn_p_g`-rooted. the same objective that could have paid for belief's structure is free to spend its budget on the overlay, registration, and obstacle families instead, but it ends up going to the belief families because those abstractions compress so much structure.

What compression returns is a set of abstractions together with the corpus rewritten to use them. Each abstraction is registered as a new symbol in the library, and the rewritten corpus becomes both the training data for the dreaming phase and the starting point for the next round — in which programs a level deeper, belief compounds too expensive to enumerate from bare primitives, have become reachable because their shared core now costs a single token.

=== Amortization <dreaming>

Enumeration searches under a prior $Q$ that is fixed across the whole corpus: every task is enumerated against the same type-uniform distribution. But the tasks are not interchangeable. A scene in which the agent takes a straight path to its goal and a scene in which it detours around a wall call for very different transition functions, and a prior that cannot see the scene has no way to tell them apart — it must spend the same budget windows on both.

In the dreaming phase we train a recognition model @ellis_dreamcoder_2020, a neural network that reads a task $x$ and emits a task-conditional prior $Q(dot | x)$, biasing the grammar toward the primitives that the scene is likely to need. It is trained by the wake–sleep procedure of a Helmholtz machine on two sources of program–scene pairs: _replays_, the solutions enumeration has actually found, run back through the interpreter on their own task grids; and _fantasies_, programs sampled from the current library prior and executed on freshly generated grids, which supply new pairs at no labelling cost where replays are scarce. The architecture, its training regime, and the rescaling that puts its output on the enumerator's cost scale are given in [app-recognition].

Two of its commitments are part of the audit of @audit rather than matters of engineering. The first is that entity _roles are read off from motion, not from cell values_: the corpus deliberately uses diverse agent and goal identifiers, so the encoder cannot key on "value 1 is the agent", and instead labels a cell occupied in the first frame but vacated by the last a _mover_ and a non-background cell occupied at both ends a stationary _goal_. Roles in fantasies are likewise decided by the sampled program's motion, so a fantasy only ever _lets_ a program drive two movers; it never manufactures a belief scene by hand.

The second is that the dreamed prior is gated to families with at least one solved instance. A family with no solves is invisible to the recognition model and could only be mispriced by it: it would flood the early budget windows with the non-belief primitives the model _has_ seen and thereby delay the primitives that family actually needs. Such families stay on the uniform baseline and earn the dreamed prior only once they are solved and can contribute replays of their own. So dreaming accelerates the families the curriculum has already reached, and never the frontier it has not — which matters here, because belief is the frontier: the first belief solve cannot be an artifact of a prior trained to expect belief.

== What would count as success or failure <criteria>

#block(inset: (left: 1em))[
  *TODO:* the discovery criteria and the rival criteria — the section that makes the next chapter's measurements tests rather than descriptions. It should fix, in advance:

  - *What counts as having discovered belief.* Not "the belief tasks were solved" — a solved belief task with an extensional program is a failure of the corpus, not a success of the learner. The criterion is structural: an abstraction appears in the library whose body opens a private channel, derives a world state that diverges from the actual one, evaluates the agent's action against that divergent state, and commits only the action back to the world. State it as a condition on the term, so it is checkable.
  - *That it must be a compression win, not merely reachable.* The discovered compound has to be selected by the joint objective of @compression against the non-mental rivals available in the same library — this is what the MDL margin measures, and the margin must be reported against the shortest rival the library can express, including the witness agent.
  - *That it must generalize behaviourally.* A learned belief term should predict, on a held-out scene, that the agent goes where it believes the goal to be rather than where the goal is. This is the false-belief test proper, and it is the criterion that distinguishes a compressive coincidence from a term that means what we say it means.
  - *What would refute the hypothesis.* Any of: no belief-structured abstraction enters the library within budget; one enters but loses on description length to an extensional rival; a rival that never opens a private channel reproduces all $k$ scenes of a mental family; the belief term fails the held-out behavioural probe. Say which of these the runs in fact rule out, and which remain open — the unsolved dual-belief family should be named here rather than left to the results chapter.
]

#load-bib(read("chapter1.bib") + read("chapter2.bib"))
