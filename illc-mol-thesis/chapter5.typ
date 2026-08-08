#import "lib.typ": *
#import "bootstrap_diagram.typ": bootstrap-diagram

#show raw.where(lang: "lisp"): it => block(
  fill: rgb("#f6f8fa"),
  stroke: 0.5pt + rgb("#d0d7de"),
  inset: 8pt,
  radius: 4pt,
  width: 100%,
)[
  #set text(font: "DejaVu Sans Mono", size: 8.5pt)
  #it
]

#show table.cell.where(y: 0): set text(style: "normal", weight: "bold")
#show table: set par(justify: false)
#set table(stroke: (_, y) => if y == 1 { (top: 0.9pt) } else if y > 1 { (top: 0.2pt) })

#mol-chapter("Results", lbl: <ch-results>)

We ran the learner of Chapter 4 on the corpus (@corpus), initialized with the primitive library (@primitives). We ran it until convergence, until it no longer found new solutions so there was nothing new to abstract from. It converged in four rounds, solving 280 of the 316 tasks (132 of the 168 belief tasks), and its final library contains six learned abstractions. This chapter presents those abstractions.

For each abstraction we present its type, its body, and what the transition function it returns does at each time step. Bodies are written with letters assigned by role#footnote[To be clear, these "roles" are just the names we give them based on how they're used in the given example]: $a$ is an agent in $V$, $g$ a goal in $V$, $b$ a believed-about value in $V$, $n$ a second actor in $V$, $r, c$ a wall's coordinates in $C$, $d$ a direction in $D$, $v$ a value in $V$, $f$ a transition function in $G arrow G$, $k$ a commit in $G times G arrow G$. The run's own symbols, argument orders and verbatim bodies are in the appendix (@tab-lib-p2-raw).

== The library <sec-found>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, center),
    table.header([Abstraction], [Symbol], [Type], [Solves \ (belief / non-mental)]),
    [phantom-wall constructor], [`fn_9`], [$C times C times V times V arrow.r (G arrow.r G)$], [75 / 0],
    [two-actor constructor], [`fn_7`], [$V times D times V times V arrow.r (G arrow.r G)$], [26 / 0],
    [displaced-goal constructor], [`fn_10`], [$V times D times V arrow.r (G arrow.r G)$], [22 / 0],
    [open constructor], [`fn_6`], [$V times V times (G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$], [12 / 0],
    [seek block], [`fn_8`], [$V times V times (G arrow.r G) arrow.r (G arrow.r G)$], [46 / 47],
    [agent commit], [`fn_11`], [$V arrow.r (G times G arrow.r G)$], [9 / 12],
  ),
  caption: [The library the run converged to],
) <tab-lib-p2>

For a given abstraction, the solve counts indicate how many solution programs contain the abstraction as a subtree. For example, the phantom-wall constructor appears in 75 solutions to belief tasks, but in no solutions to non-mental tasks. The full abstractions are at @tab-lib-p2-raw.

We gave each of the final abstractions a name for ease of reference. We'll go through each of these final abstractions, showing its type and its body, both in terms of its constituent abstractions and also fully inlined. Note that we show the abstractions written as they are outputted by the run itself. We haven't introduced these implementation-level names for the primitives and they're not necessary because we also give the types and we'll explain their semantics. 

\

*Open constructor* --- `fn_6` : $V times V times (G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$

```lisp
(pipe_gpg (compose_gp dup (mapsnd (compose f (optimize (neg_dist g) a)))) k)
```

Applied to an agent $a$, a goal $g$, a derive $f$ and a commit $k$, it returns the transition function: at each time step, copy the grid; on the copy, make the alteration $f$, then move $a$ one step toward $g$; then hand the (world, copy) pair to $k$, which collapses it back to one grid. This is the derive-and-commit frame of @signature with the planner already inside it: what the copy is wrong about ($f$) and how the copy is published back to the world ($k$) are both left as holes. Twelve belief solutions use the open constructor directly, with the search filling both holes.

The next three abstractions are the open constructor's instantiations. The compression step emits each of their bodies as a call to `fn_6` with the two holes filled. So we give each body in that call form first, with its expansion beneath it.

*Phantom-wall constructor* --- `fn_9` : $C times C times V times V arrow.r (G arrow.r G)$

```lisp
(fn_6 a g (wall_at r c) sync_all)
= (pipe_gpg (compose_gp dup (mapsnd (compose (wall_at r c)
      (optimize (neg_dist g) a)))) sync_all)
```

At each time step: copy the grid; on the copy, stamp an impassable wall at $(r, c)$, then move $a$ one step toward $g$; then move every value the two grids share to the position the copy gives it.

The wall exists only in the copy, so it is not a shared value, and $a$ is the only shared value the derive moves. So moving all the values moves just $a$ and nothing else, and the wall never reaches the rendered world. This is the false-wall attribution: $a$ detours around a wall that is not there. This abstraction solves 75 belief solutions, the most of any abstraction.

*Displaced-goal constructor* --- `fn_10` : $V times D times V arrow.r (G arrow.r G)$

```lisp
(fn_6 a b (step b d) (sync_except b))
= (pipe_gpg (compose_gp dup (mapsnd (compose (step b d)
      (optimize (neg_dist b) a)))) (sync_except b))
```

At each time step: copy the grid; on the copy, displace $b$ one cell in direction $d$, then move $a$ one step toward $b$; then move every shared value _except $b$_ to where the copy has it.

So $a$ walks toward where $b$ is not, $a$'s move is published to the the world, and the displacement (being $b$'s alone) is never published to the world. This is the Sally--Anne attribution: $a$ acts on a false belief about where $b$ is. The value $b$ fills three slots at once: the value displaced on the copy, the value $a$'s plan pursues, and the one value the commit withholds. This abstraction appears in twenty-two belief solutions.

*Two-actor constructor* --- `fn_7` : $V times D times V times V arrow.r (G arrow.r G)$

```lisp
(compose (optimize (neg_dist b) n)
  (fn_6 a b (step b d) (sync_except b)))
= (compose (optimize (neg_dist b) n)
    (pipe_gpg (compose_gp dup (mapsnd (compose (step b d)
        (optimize (neg_dist b) a)))) (sync_except b)))
```

At each time step, $n$ moves one step toward $b$ in the world itself, and then the displaced-goal constructor runs for $a$.

This is the displaced-goal constructor with a second actor composed in front. The `fn_6` call here is the displaced-goal constructor's whole body. Both actors pursue the same value $b$; $n$ plans against the world as it is and $a$ against a private copy in which $b$ has been displaced, so the two end up at different cells. Here $b$ fills four independently searchable slots: it's displaced on the copy, sought by $n$, sought by $a$, and withheld by the commit. The abstraction binds one value to all four arguments.

Given the value 2, the direction up, the value 1 and the value 6, the abstraction encodes _"agent 6 navigates towards goal 2. agent 1 also does, but believes 2 is one cell up"_. Twenty-six belief solutions use this abstraction, 24 of them the two-observer family's tasks, each solved by this one symbol applied to four values.

*Seek block* --- `fn_8` : $V times V times (G arrow.r G) arrow.r (G arrow.r G)$

```lisp
(compose f (optimize (neg_dist g) a))
```

Make the edit $f$ to the grid, then move $a$ one step toward $g$ --- all of it in the world, with no copy and no commit.

This is also, symbol for symbol, what the open constructor runs on the copy (the library carries that occurrence inlined inside `fn_6` rather than as a call). This is the goal-directed movement family's planning policy with room for one edit in front of it. This is the abstraction found from the non-mental corpus in round 1. 47 obstacle solutions run it with a real wall as the edit, and 46 belief solutions (nearly all from the witness family) run it with a phantom-wall constructor in the edit slot: the believer's whole attribution step as the "edit", followed by the witness's seek on the world it leaves behind.

*Agent commit* --- `fn_11` : $V arrow.r (G times G arrow.r G)$

```lisp
(register (locate v) (place v))
```

Applied to a value $v$, it returns a commit: read where $v$ sits in the second grid, move $v$ in the first grid to that position, and leave everything else as the first grid had it.

This is the single-value commit of @signature, reassembled from the registration parts, with $v$ bound twice: once in the reader and once in the writer. Twenty-one solutions use it: 9 belief solutions, where it publishes exactly the believer's move, and 12 from the registration families, where the same abstraction does image alignment against a template.

The three constructors spell their publication three different ways, committing three different things. The agent commit commits the _agent_: publish this one value and nothing else. The phantom-wall constructor commits the _everything_: move everything the grids share, which, on a scene where the copy differs from the world only at an unshared wall and at the agent, is the same one-value transfer. The displaced-goal and two-actor constructors commit the _false-belief content_: publish everything except the value the agent is wrong about. In every case the value whose behaviour reaches the world is the value whose reasoning was simulated, and the falsehood stays private. The same agency signature of @signature manifests in three different ways according to which commit policy is being used. 

== The order of arrival <sec-order>

One of the main themes of the developmental literature is the order in which ToM capacities come into fruition @timeline. Rational constructivism engages with the timeline evidence most directly, so in our experimental setup we are also interested in the order in which abstractions are acquired.

The figure below denotes each abstraction with a box, in the columb of the round during which it was invented. The counts inside the box denote the number of tasks that shared the common structure from which the abstraction was made, and the counts are split by whether it was a solution to a belief task (blue) or non-mental task (violet). The column header has the numbers of the full pool of solutions.

Solid arrows denote that the right-hand abstraction's body instantiates the left one, that the body of the right-hand abstraction is composed in part of the left-hand abstraction. Solid arrows denote that the single-value commit fills the open constructor's commit hole; the early wall-and-seek policies are re-cut into the seek block. For the full per-round results, see @tab-lib-rounds-raw.

#figure(
  bootstrap-diagram(),
  //caption: [],
) <fig-bootstrap>

Round 1 solves 148 tasks, none of which are belief tasks. When the library only has base primitives, every belief family is out of enumeration's reach. Compression over those 148 non-mental solutions returns six abstractions. Among them are exactly the two target structures of @signature: the _derive-and-commit frame_, of type $(G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$, is abstracted from the picture-processing families (overlay, underlay, comet), and the _single-value commit_ $V arrow (G times G arrow G)$, the "agent commit" above (under an earlier symbol) is abstracted from image registration. Neither of them were found from solutions to belief tasks, since no belief tasks were solved yet. 

With those two abstractions as single symbols, in the enumeration phase of round 2 the abstractions are composed into more complex belief-attribution programs, which are used to solve 75 belief tasks. The compression phase then invents the phantom-wall, displaced-goal and open constructors. Round 3 solves 32 more and adds the two-actor constructor and the seek block, re-cutting the early wall-and-seek policies into the latter.

From round 3 the library is fixed. Round 4 only grows uses it more, and the run converges at 132 of 168 belief tasks. An abstraction discovered in round $N$ is a single symbol in round $N+1$, and the round-1 abstractions are precisely the two capacities (hold a second grid, transfer one named value between grids) that belief needs and that non-mental tasks independently use for. This order just falls out of what is cheap at each stage.

@fig-corpus-dl shows how the description length of the corpus decreases as more complex abstractions are introduced. To plot this, we hold the corpus fixed at its base-primitive solutions and re-express every program in each round's library. The belief families' total falls from 3,857 to 1,828 nats, and the steepest single drop (1,108 nats) comes at round 2, the round the private-copy constructors are invented. The description length of the non-mental tasks barely moves. Its slight rise after round 1 is the joint compression re-cutting early non-mental spellings toward the structure the belief solutions share.

#figure(
  image("corpus_dl_p2.pdf", width: 92%),
  caption: [Total description length of the corpus under each round's library, split into the belief families and the pooled non-mental families.],
) <fig-corpus-dl>

We also run an "atomic" control run, in the derive-and-commit frame and the single-value commit are included in the initial library as primitives. The atomic control run shows similar results but does everything one round earlier. It solves 55 belief tasks in round 1, and converges in two rounds at 155 of 168. Its final library also has six abstractions, five of which open a private copy and bind the agent value in both the planner and the publication. So it's the same shape as with the main run, but with coarser granularity, with the agent value shared two ways where the combinator run's registration spelling shares it three. Its constructors and their bodies are @tab-lib-p1-read and @tab-lib-p1 in the appendix.
