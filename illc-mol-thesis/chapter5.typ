#import "@preview/illc-mol-thesis:0.2.0": *

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

#show table.cell.where(y: 0): set text(style: "normal", weight: "bold")
#show table: set par(justify: false)
#set table(stroke: (_, y) => if y == 1 { (top: 0.9pt) } else if y > 1 { (top: 0.2pt) })

#mol-chapter("Results")

Chapter 3 set out the structure a Bayesian theory of mind stipulates: a planner evaluated against a model of the environment supplied as an input, where that model may diverge from the state of affairs the observer takes to obtain, and where the planner runs on a particular agent's behalf. Chapter 4 built a learner with none of that structure --- a single-grid interpreter, a monomorphic type discipline over grids and cell values, and a stock of domain-general primitives. The claim of this chapter is not that the belief tasks were solved; a belief task solved by an extensional program would be a failure of the corpus. It is that the joint description-length objective of @compression --- the same objective, with no term keyed to the mental families --- selects into the library a term whose body has the shape Chapter 3 stipulates, and does so from a starting point where that shape was neither a primitive nor a type.

@sec-found reports the libraries the runs converged to and @sec-btom-map lays the discovered term beside the BToM structure it is supposed to resemble. @sec-compression through @sec-probe discharge the criteria of @criteria in turn --- compression win, reachability and ordering, generality and selectivity, and the false-belief test on held-out scenes --- and @sec-scaffold recovers, from the runs themselves, which shallower tasks the term was assembled out of. @sec-verdict tallies the result including what failed, @sec-objections takes the objections, and @sec-limits says what the results do not show.

== What the loop finds <sec-found>

=== The combinator library: a two-stage bootstrap <sec-combinator-run>

The learner is handed the product combinators of @tab-combinators and must assemble the pair interface before it can use one. What follows is not a slower version of the control run of @sec-atomic-run but a two-stage bootstrap, and the first stage is the strongest evidence in the chapter that the machinery belief uses is not belief's machinery.

Round 1 solves 144 tasks and *none* of them are belief tasks. Every non-mental family falls but one --- physics, desire, the obstacle and relocation families, the overlay and underlay and comet families, the registration family and its variants; the exception is the layer-compositing family, a search-budget miss @sec-verdict accounts for --- while the belief families are, at 1200 s, out of reach entirely. The compression step then runs over those 144 purely non-mental solutions and returns, among its six abstractions,

#terminal("fn_2: (pipe_gpg (compose_gp dup (mapsnd #0)))                [fn, fn_p_g] -> fn
fn_1: (register (locate #0) (place #0))                    [cellvalue] -> fn_p_g")

which are, by the identities verified before the run began (@signature), exactly `fork` and `sync_to_world`. The learner has reassembled the pair interface out of the categorical plumbing, on evidence in which no agent holds a belief about anything: `fn_2` is induced by the overlay, underlay and comet families, `fn_1` by image registration.

Round 2 therefore starts where the control run starts, and 75 belief tasks fall. Its compression step returns the agency signature proper --- the term the verdict machinery certifies as the agent constructor, worth reading line by line:

#terminal("fn_7: (compose
  (optimize (neg_dist $0) $3)          ; a second actor $3 seeks $0 where it really is
  (pipe_gpg
    (compose_gp dup                    ; split the grid into (world, model)
      (mapsnd                          ; ... and in the model channel only:
        (compose (step $0 $1)          ;   displace $0 one cell in direction $1
          (optimize (neg_dist $0) $2)) ;   plan $2's move against the displaced $0
      ))
    (sync_except $0)))                 ; commit: publish every value EXCEPT $0")

Each application moves the naive actor `$3` toward `$0`'s true position; then duplicates the grid, displaces `$0` in the private copy only, plans the believer `$2`'s move against that copy, and commits by writing the model back into the world *for every value but `$0`* --- so both agents' moves reach the world, and the displacement never does. The hole `$0` --- the *content* of the belief --- appears *four* times: as the value that is displaced, as the value both plans pursue, and as the one value the commit withholds. A whole belief task is this token with four arguments: `(fn_7 2 up 1 6)` reads "agent 1 believes 2 is one cell up; agent 6 knows better". Rounds 3 and 4 take the remaining 57, and the run converges at 132 of 168 belief tasks. The final library, expanded to base primitives, is @tab-lib-p2.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, center),
    table.header([Symbol], [Body], [Belief solves]),
    [`fn_6`], [`(pipe_gpg (compose_gp dup (mapsnd (compose $2 (optimize (neg_dist $1) $0)))) $3)`], [12],
    [`fn_7`], [`(compose (optimize (neg_dist $0) $3) (pipe_gpg (compose_gp dup (mapsnd (compose (step $0 $1) (optimize (neg_dist $0) $2)))) (sync_except $0)))`], [26],
    [`fn_8`], [`(compose $2 (optimize (neg_dist $1) $0))`], [46],
    [`fn_9`], [`(pipe_gpg (compose_gp dup (mapsnd (compose (wall_at $1 $0) (optimize (neg_dist $2) $3)))) sync_all)`], [75],
    [`fn_10`], [`(pipe_gpg (compose_gp dup (mapsnd (compose (step $0 $1) (optimize (neg_dist $0) $2)))) (sync_except $0))`], [22],
    [`fn_11`], [`(register (locate $0) (place $0))`], [9],
  ),
  caption: [The final library (`p2-nodream`), bodies expanded to base primitives. `fn_7` is the agent constructor proper --- one hole bound four ways; `fn_10` is its fork block alone, and `fn_9` the phantom-wall variant committing through `sync_all` (@sec-selective); `fn_6` is the constructor with derive and commit both left as holes. The two terms with no fork in them are belief's donated parts: `fn_8` is the bare seek policy belief shares with the obstacle family, through which 47 obstacle solves are also rewritten, and `fn_11` is `sync_to_world` reassembled, shared with the registration, perception and drifting-registration families.],
) <tab-lib-p2>

Four of the six carry the fork --- three complete agent constructors differing in their derive and their commit spelling, and `fn_6`, the constructor with both slots still open. The two terms with no fork in them are exactly the parts the non-mental families donate: `fn_8` the seek policy, the token belief and the obstacle family use in common, and `fn_11` the reassembled commit, the token belief and the registration families use in common --- the sharing @sec-scaffold takes up. The commits themselves come in two spellings. `fn_11` names the boundary by the *agent*: publish this one value. `sync_except` and `sync_all` name the same boundary by the *content*: publish everything but this. On scenes where model and world differ only at the displaced content and the movers, the two spellings denote the same commit, which @sec-selective verifies by substitution and re-rendering --- and the choice between them is not a retreat from the single-value corner but a price fact @sec-selective returns to.

This is the bootstrapping structure of @sec-hbm in its clearest form. An abstraction discovered in round $N$ is an atom in round $N+1$, and the round-1 atoms are precisely the two capacities --- hold a second grid, transfer one value between two grids --- that belief needs and that non-mental tasks independently pay for. Nothing schedules this order; it falls out of what is cheap at each stage.

=== The atomic control <sec-atomic-run>

The second endowment is a *control*, in which `fork` and `sync_to_world` are granted outright as single nodes rather than assembled from the product structure, so that belief is the two-token composition `(fork (compose (wall_at r c) (optimize (neg_dist gv) av)) (sync_to_world av))` with the agent value appearing twice instead of three times. It is not a second result and the argument does not rest on it (@no-separation); its use is as an instrument. Because the two vocabularies are expressively identical --- which is what the identities of @signature establish --- a task the control solves and the combinator run does not is a task lost to the search budget rather than to the language, and that is a distinction no single run can draw. It also prices the same non-mental rivals at a second granularity, which is what shows the compression result is not an artefact of where the primitives were cut. It solves 291 of its 304 tasks and 155 of 168 belief tasks against the combinator cell's 276 of 316 and 132, and on the tasks the two corpora share its solve set strictly contains the combinator cell's.

Round 1 begins with no library, a uniform prior over 37 primitives, and a 1200 s budget per task, and solves every non-mental family and 55 of the 168 belief tasks by brute cheapest-first enumeration. The obstacle families fall almost immediately, at position 859 in the cost-ordered walk --- `(compose (wall_at c1 c1) (optimize (neg_dist 5) 4))` --- and the belief tasks built on the same wall content fall two and a half orders of magnitude later, at position 297342: `(fork (compose (wall_at c1 c2) (optimize (neg_dist 2) 1)) (sync_to_world 1))`. That ratio is the concrete form of the point @sec-search made abstractly: a belief compound is not a slightly less probable hypothesis than an obstacle compound but a hypothesis some three hundred times deeper into an exponentially growing space. Nor is the comparison rigged by the choice of content --- the cheapest belief task of any family in round 1, whose private derive is a bare `optimize` rather than a wall, still lands at position 21528.

Of the six abstractions the round-1 compression step returns, two matter: the wall-and-navigate policy that the obstacle family shares with the belief tasks' derive, and, built on top of the seek fragment, the thing this thesis is about,

#terminal("fn_0: (fork (compose #2 (optimize (neg_dist #1) #0)) (sync_to_world #0))  [cellvalue, cellvalue, fn] -> fn")

`fn_0` takes a goal value `#1`, an agent value `#0`, and an arbitrary `fn` in the hole `#2`. It runs `#2` on a private copy of the grid, plans `#0`'s move on the result, and publishes `#0`'s move --- and only `#0`'s move --- back to the world. The content of the false belief is the hole; the agent is the parameter that appears twice.

In round 2 the budget is the same but the library is not, and the remaining 100 belief tasks fall. What had been a search hundreds of thousands of steps deep is now a fill-in-the-blanks: a witness task --- an agent misled about where its goal went alongside a second agent who is not --- is caught at position 8668 by composing a plain `optimize` for the witness with `fn_0` carrying a `step` in its hole. Round 3 adds nothing, at which point the loop has converged on the library of @tab-lib-p1, in the appendix.

Five of its six invented terms are agent constructors: a fork whose derive ends in a policy for some value and whose commit publishes that same value. They differ in what the private derive does --- an arbitrary hole in `fn_6`, a shove of the goal in `fn_8`, a phantom wall at a free coordinate in `fn_11` and at a fixed column in `fn_10` --- and the most-used is `fn_11`, through which 52 of the 155 belief solutions are rewritten. `fn_9` is worth a remark: its derive contains another fork, so it is an agent whose private model is itself derived by forking, and fourteen solutions are rewritten through it. That is not second-order mentalizing in the sense of @timeline, since the inner fork is part of the outer agent's model rather than a model of another mind, but it does establish that the discovered shape embeds, which is one of the properties @explanandum asked for.

=== The agency signature

Under either endowment the invented term has the same shape, and the shape is a coincidence in the grammar that the objective declined to treat as one. Written in the abbreviations of @signature, the constructor is

```lisp
(fork (compose <derive> (optimize (neg_dist gv) av))
      (sync_to_world av))
```

`optimize` takes a `cellvalue` saying whose move to compute; the commit takes a `cellvalue` saying what to publish, or what to withhold. These are different argument slots on different primitives, filled independently by the type-directed enumerator, with ten cell-value terminals available for each. Nothing in the grammar, the type system or the interpreter requires them to agree. In the invented abstractions they are one hole, and how many ways the one hole is bound depends on how the commit is spelled. The atomic constructor binds it twice, planner and published value. When the commit is assembled from the registration parts, as in the 9 solutions that spell it `register (locate v) (place v)`, the same cell value lands in the `locate` and the `place` as well, and the coincidence is threefold. And in the combinator cell's constructor `fn_7` it is fourfold: the value that is displaced in the private channel, the value both plans pursue, and the value the `sync_except` withholds from publication are one hole. The two runs pin the same boundary from opposite sides --- the control's constructor names it by the agent (publish this value), the combinator cell's by the content (publish everything but this) --- and either way the boundary must be *drawn*, by a coincidence of independently-fillable slots that the objective had no obligation to keep.

That collapse is what makes the term an attribution rather than a computation. A program that derives a counterfactual grid, plans on it, and publishes the result of planning for *some other* value is expressible, is no more expensive to describe, and does not occur. What compression kept is the version in which the value whose behaviour is published is the value whose reasoning was simulated --- and, in the content-side spelling, the version in which the value withheld from the world is exactly the value misrepresented in the model. Which is to say: the version in which the private grid is *somebody's*, wrong about *something in particular*.

Nothing in this identifies the term as *belief* in any thicker sense, and on the account of intentionality this thesis has been working with, nothing needs to: what makes a state a belief is the role it plays in the predictive scheme of the intentional stance, not an intrinsic mark it carries @dennett_intentional_2002. The invented term is a state that is derived privately, is planned against, is attributed to one value rather than to the scene, and predicts where that value will go. That is the role, and asking whether the thing filling it is *really* a belief is asking a question the framework declines.

== The discovered term against BToM <sec-btom-map>

@stipulated listed what BToM assumes in order to model mature performance; the list is now a list of things to look for in the invented term. @tab-btom-map in the appendix sets the correspondence out item by item, and it runs as follows.

BToM's planner maps a utility and an environment model to an action. The learner's is `optimize (neg_dist gv) av`, applied to whatever grid it is handed, and this is the one item we grant outright, as a primitive, for the reason @rationality gave. The environment model $m'$ that BToM supplies to that planner as an input is here the second channel of the pair, and it is the only channel the derive writes: the invented abstraction contains `mapsnd` and not `mapfst`, so the world channel passes through untouched. Which of the two channels is the model is therefore something the search settled rather than something the architecture declared. The same goes for the divergence $m eq.not m'$ that makes false belief expressible at all: in BToM a permission written into the architecture, here an arbitrary `fn` applied to the private copy only --- and the opposite choice is not merely available but exercised, since @sec-selectivity reports the families that transform the world channel instead. The goal fares differently. BToM ranges $g$ over a given space under a given prior; in the discovered term the goal is whichever `cellvalue` happens to land in `neg_dist`'s argument slot, priced like every other symbol. It is not a space at all --- there is no goal type and no goal prior (@no-btom) --- but an argument position with ten terminals that could fill it. And the attribution is the item the previous section turned on: BToM stipulates that $m'$ is #emph[this agent's] model, and the invented term has the shared hole instead --- the atomic constructor binding the planner's actor to the commit's published value, the combinator one binding the derive's edit, the plans and the commit's exception to one content value --- either way, one value pinned to both sides of the model/world boundary.

Three further pieces of BToM's machinery have counterparts of a different sort. Its nested evaluation --- the observer running a planner inside itself --- is here structural, in program space: `fork`'s private grid lives for the duration of one call and is discarded, never rendered, and the interpreter has no such mode of its own (@app-interpreter). Its inverse planning, which infers $(g, m)$ from a trajectory by Bayes, is here cheapest-first enumeration under the library prior against an indicator likelihood: the inference of @sec-bayes, unchanged, and not agent-directed, since the same procedure solves denoising. And the last piece has no counterpart. Belief in BToM is a state that is #emph[formed] and could be #emph[revised]; the model discovered here exists for one transition and has no history. That is the honest gap, and @sec-limits returns to it.

Set out that way the invented abstraction is a Bayesian theory of mind's agent, written in a language that had no word for one. The difference from Chapter 3 is where each item came from: BToM's are architecture, fixed before any scene is seen, and its inference locates a point inside them, whereas three of them here --- which channel diverges, whose model it is, and what the divergence consists in --- are decisions made in program space by a search that also solves image registration and denoising, and kept by an objective that had the option of spending its budget on the overlay and relocation families instead.

Two grants will be pressed. The planner we grant for the reason Chapter 3 gave: on Gergely and Csibra's analysis the teleological stance relates action, goal and situational constraint without attributing any representation to anybody, and it is in place by twelve months, three years ahead of false-belief attribution @gergely_teleological_2003 @gergely_taking_1995. The results bear that out --- `optimize` appears in the desire family (16 solves), the flee family, the comet family and all 48 obstacle solves, scenes nearly indistinguishable at the level of the rendered trajectory from the wall-belief scenes and containing nothing mental at all. What separates the mental families is not that they use the planner but what is wired around it. The pair we also grant, but `dup` produces $(w, w)$ and says nothing about which copy is a model or who holds it: the asymmetry is entirely in the composition, and @sec-selectivity shows the symmetric alternatives are not merely available but *used*, by other families in the same runs.

== The objective selected it <sec-compression>

The guiding thought behind the whole project is that an intensional theory --- one that attributes a belief to an agent --- is the most compressed available description of that agent's behaviour, and that this is no coincidence: the agent navigates by its own representation of the world, so a theory that posits that representation is tracking the process that actually generated the data. One can describe the same trajectory without attributing anything, but the description has to account for each step of the walk separately, and it will not transfer to the next scene. The prediction is that the description-length objective should prefer the intensional reading, and prefer it more the more experience the learner has had. @criteria demanded that the compound be a compression win rather than merely reachable, and there are three measurements: what the library does to the description length of the whole corpus, what it does to belief programs specifically, and how the belief program compares to the non-mental rivals the same library can express.

=== The corpus description length

The quantity the compression step minimises (@compression) is the joint description length of the library together with the corpus written in it. Fixing the corpus and repricing it under each round's library gives, for round $r$,

$
  "DL"(X|r) = sum_(x in X) "DL"(p_x|L_r) + "DL"(L_r)
$

where $X$ is the corpus, $p_x$ the program the run found for task $x$, and $L_r$ the library after round $r$'s compression. @fig-corpus-dl plots this, split by family.

#figure(
  image("corpus_dl_p2.png", width: 100%),
  caption: [Corpus description length by round (`p2-nodream`). Left: the stacked total, with the library's own cost as the black line. Right: the per-family contribution. The belief band collapses at round 2, the round after the one that rebuilt `fork` and `sync_to_world` from non-mental evidence (@sec-combinator-run).],
) <fig-corpus-dl>

Under the base primitives the 276 solved programs cost 5329 nats; after four rounds they cost 3315, including the 47 nats the library charges for itself. Of the 2014 nats saved, 2029 come from the belief family alone --- 3857 nats down to 1828 --- and the largest single step is the 1108 nats that go at round 2, the round whose compression step invents the constructor. The non-mental families barely move: `world` (the nineteen non-mental families pooled) goes from 1333 to 1290, nearly all of it the registration-flavoured families buying the reassembled `register (locate v) (place v)` token, desire from 116 to 125, physics from 23 to 25 --- the last two slightly *worse*, which is the library tax of @compression showing up exactly where it should, since a family that gains no abstraction still pays for the enlarged symbol set.

That asymmetry is the argument. If the initial primitives had encoded anything mental, belief would have been cheap from the start and the belief band would look like the desire band. It is because belief is expensive under the base library and cheap under the learned one that we can say the mental content is in what was learned rather than in what was given. The atomic control gives the same picture a round earlier and from a higher starting point: 5765 nats down to 3516, belief carrying 4404 to 2090, with its largest drop --- 1841 nats --- at round 1, since there the constructor can be invented as soon as anything has been solved.

=== What the library buys each family

Per task, the same result in nats. Pricing each family's found program under the base primitives and again under the final library (@tab-dl-census, in the appendix) turns up one substantial mover and a handful of small ones: belief falls from a median 34.51 nats to 17.40, a saving of 17.11 per task and 412 over the priced solves; obstacle from 15.97 to 12.79, through `fn_8`, the seek fragment the two share; and registration, perception and drifting registration each save exactly 2.30 nats --- the price of one token, the reassembled `register (locate v) (place v)`. Every other family saves 0.00. So belief is the only family the library substantially compresses, and the small savers are instructive rather than embarrassing: the same objective that invents the agent constructor also invents a purely non-mental abstraction where purely non-mental reuse exists, and the two shared tokens are exactly the two donor relations @sec-scaffold recovers independently. The atomic control gives the same shape with a smaller belief saving, 33.71 down to 13.38 nats, since less of the compound is left to abstract once the pair interface is given.

One caveat about the belief row: this census prices only solutions whose library-rewritten form re-parses under the final library, here 25 of the 132 belief solves, all of them from the witness family, so the 17.11 is a median over a fifth of the family rather than over all of it. @fig-corpus-dl is the unrestricted version of the same measurement --- it prices every solved program in every round, and gives the belief band's 3857 to 1828 nats directly.

=== No non-mental rival reproduces the scenes <sec-rivals>

The third measurement is the one @criteria cared most about: whether a non-mental program is available that is as short and as accurate. Each belief task carries a battery of hand-written rival spellings --- the transient-wall schedule that stamps a wall, lets the agent walk, and erases it before the frame is rendered; the version that shoves the goal in the world rather than in a model; pure desire; a physics reading with no wall at all; and, for the witness family, the four orderings of stamp, act, witness and erase. Each rival is rendered over every scene of every belief task and compared frame for frame.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header([Rival spelling], [Tasks], [Reproduces\ all 4 scenes], [Reproduces\ the agents], [Median margin\ (library, nats)]),
    [transient wall (stamp / act / erase)], [24], [0], [24], [$+10.67$],
    [transient wall (erase-3 variant)],     [24], [0], [24], [$+9.76$],
    [transient shove (shove / act / restore)], [51], [0], [51], [$+18.19$],
    [witness then stamp / act / erase],     [48], [0], [0],  [$+13.85$],
    [stamp / act / erase then witness],     [48], [0], [0],  [$+10.67$],
    [stamp, witness, act, erase],           [48], [0], [0],  [$+10.67$],
    [stamp, act, witness, erase],           [48], [0], [0],  [$+10.67$],
    [transient phantom $+$ shoved goal],    [9],  [0], [0],  [$+12.35$],
    [shove goal in the world],              [60], [0], [0],  [$+4.79$],
    [both agents seek the goal],            [24], [0], [0],  [$+4.09$],
    [transient, witness ignored],           [48], [0], [0],  [$+3.58$],
    [pure desire (no belief)],              [27], [0], [0],  [$-0.69$],
    [no wall (physics)],                    [72], [0], [0],  [$-2.53$],
    [no belief (seek past the real wall)],  [9],  [0], [0],  [$-9.51$],
  ),
  caption: [The non-mental rivals (`p2-nodream`), over the 132 solved belief tasks. "Margin" is $"DL"("rival") - "DL"("found")$ under the final library; positive means the mental reading is shorter. Not one rival reproduces all four scenes of any task.],
) <tab-rivals>

The headline of @tab-rivals is the third column. Across 132 belief tasks and fourteen rival spellings, *no non-mental rival reproduces all four scenes of any task*. This is a change in kind from earlier versions of the corpus, where the transient-wall rival reproduced the wall-belief scenes exactly and had to be beaten on description length, so that the honest form of the argument was a margin. Two corpus changes closed it. An inert bystander object now sits on the phantom-wall cell for the whole trajectory, so stamping a real wall there overwrites something the frame shows and nothing in the language can put it back; and the $k$-scene format (@corpus) requires a rival to survive four independent placements of the same latent, which a single wall coordinate chasing a believed goal cannot do. The fourth column records how close the rivals get: the transient-wall spelling still reproduces the *agents'* trajectories on all 24 wall-belief tasks, and the transient shove on 51 goal tasks. They get the walk right and the world wrong.

This is exclusion by reproduction rather than by price, and it is the stronger form of the argument --- a rival that cannot render the data is not a rival at all, whatever it costs. It also keeps the argument honest in the other direction: 120 of the 540 rival pricings come out *shorter* than the mental program, almost all of them the bottom three rows, and every one of them is a program that simply does not depict the scene --- an agent walking to a goal it is not wrong about. Cheapness without accuracy is worth nothing, and it is the likelihood of @sec-bayes rather than the prior that disposes of them. But the price is still informative, and the last column is where the compression story reappears. Take the closest rival, the transient wall. Under the base primitives it costs $0.08$ nats *less* than the mental program --- a tie leaning the rival's way, as one would expect, since neither reading has any structure to reuse and the mental one names one value more. Under the final library the mental program is $10.67$ nats *shorter*, a swing of four orders of magnitude in prior probability. Nothing about the rival changed; what changed is that the mental reading collapsed into a token the library had a reason to invent and a bespoke stamp-act-erase schedule, recurring nowhere else in the corpus, did not. The atomic control gives the same result from a tie leaning the other way, $+1.28$ nats widening to $+10.91$, and that the effect survives both carvings of the primitive set is the point of running the control.

== Reachability, and the order in which things arrive <sec-order>

@sec-search argued that description length in this setting is simultaneously a measure of plausibility and a budget: a hypothesis that is long under the current library is not merely improbable but unreachable. @criteria turned that into a prediction about ordering. @fig-solve-dynamics is the prediction met.

#figure(
  image("solve_dynamics_p2.png", width: 100%),
  caption: [Solve dynamics (`p2-nodream`). Left: cumulative solves by family, by round. Right: for each belief task solved after at least one miss, its enumeration time in the round it missed against the round it was solved.],
) <fig-solve-dynamics>

The non-mental families are flat lines at their ceiling from round 1: they need no abstraction and get none. Belief is an S-curve --- 0, then 75, 107, 132 --- whose step is the round after the constructor enters the library. And the first point of that curve is a zero rather than a small number: with the pair interface given only in parts, not one belief task is solved until the compression step has reassembled `fork` and `sync_to_world` out of non-mental evidence. Belief here is not merely slow in round 1; it is out of reach.

The right-hand panel is the same fact per task, and it is more equivocal. All 132 belief solves were missed at least once, and the worst case goes from the full 3600 s timeout to 22.2 s, a factor of 162 --- but the *median* moves only from a 1200 s round-1 timeout to a 1146 s solve. What the library buys in this cell is therefore reachability rather than speed: it brings the compound inside the budget window at all, and most tasks then spend most of that window. The atomic control moves the median further --- of 100 tasks missed at least once, from a 1200 s timeout to a 708 s solve, with the steepest collapse from 1200 s to 2.0 s --- which is what one expects of a cell that starts a rung higher; but there too the typical task spends more than half its budget. In neither case did the tasks or the budget change; what changed is the description length of their solutions, and with it the position of those solutions in the cost-ordered walk.

This does not show that a child's four-year delay has this cause. What it shows is that in a learner of this kind the delay does not have to be put in. @stipulated's complaint against BToM was that its structure is complete from the start, so every parameter is available for inference on day one and nothing in the framework makes goal attribution earlier or cheaper than belief attribution; the ordering has to be imported from outside as maturation or as a performance limitation. Here the ordering is the mechanism. Desire is solved in round 1 because `(optimize (neg_dist 2) 1)` is a three-symbol program; belief is not solved until round 2 because its compound is out of budget until earlier learning has shortened it. That is the cumulative, staged picture @explanandum demanded, produced rather than accommodated: the bootstrapping of @sec-hbm, measured rather than assumed.

== Generality and selectivity <sec-selectivity>

An abstraction that is a compression win might still be a bad theory in two ways: it might have baked in the particular values it was trained on, and it might be applied to everything that moves --- the second being @criteria's fifth refutation condition, a habit of forking rather than an attribution. Three measurements answer them.

=== The constructor spans the corpus

The corpus rotates through eight distinct (goal value, agent value) pairs so that no integer is privileged (@no-btom). @fig-agent-tiling counts, for each combination and each invented token, how many solved belief programs are rewritten through it.

#figure(
  image("agent_tiling_p2.png", width: 92%),
  caption: [Constructor coverage (`p2-nodream`). Rows are the eight (goal, agent) value combinations; columns are the six invented abstractions. Four of the six columns are occupied in every row; the two that are not are the two rarest spellings.],
) <fig-agent-tiling>

Every fork-bearing spelling spans the matrix. The most-used token, `fn_9` --- the phantom-wall constructor --- carries 75 of the 132 belief solves and appears in every row; the agent constructor `fn_7` carries 26 and appears in every row; the bare seek `fn_8` (46) and the commitless fork block `fn_10` (22) likewise. Only the two rarest tokens miss anything: `fn_6` (12 solves) and the reassembled commit `fn_11` (9) are each absent from two rows. So the constructor is not a memorised (goal, agent) pair with a wrapper --- its cell-value slots are genuine holes, exercised across every pair the corpus rotates through. The instructive comparison is the obstacle family, whose policy did *not* generalise into a single token in this run: because obstacle scenes vary their wall coordinate more than their (goal, agent) pair, no single wall-bearing policy recurred often enough to be abstracted with holes, so belief's derive is not a cheap token even after compression --- the belief compound is not shallow even with the constructor in hand. The atomic control gives the same picture slightly less evenly: its constructor `fn_11` spans all eight combinations, by 52 of 155 solves, but the bare seek policy `fn_7` is missing from two rows.

=== The symmetric complements go elsewhere

@tab-combinators lists, beside each primitive belief uses, the opposite corner of an independent symmetry axis: channel direction, commit scope, z-order, projection, pairing, the bifunctor axis, and utility polarity. The point of including them was that an unused complement proves nothing. @tab-cube reports which family claimed which corner.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header([Corner], [Family that selected it]),
    [`register (locate v) (place v)` (read model, write world, one value)], [*belief* (81 of 132, of which 9 spell it literally --- @sec-selective), registration, perception, drifting registration],
    [`sync_all` (every shared value)], [multi-registration, map-update],
    [`sync_except` (all but one)], [registration-except],
    [`dup` / `mapsnd` (diagonal; act on the second factor)], [*belief* (81), overlay, underlay, comet],
    [`mapfst` (act on the world factor)], [drifting registration],
    [`swap` (the twist)], [perception, map-update],
    [`pair_blank` (pair with a blank)], [wipe],
    [`overlay` / `underlay` (z-order)], [overlay, comet / underlay, inpainting],
    [`snd_gg` (project the model channel)], [readout, wipe],
    [`neg_dist` / `distance` (utility polarity)], [belief, desire, obstacle / flee],
    [`wall_at` / `clear_at` / `erase` (grid edits)], [belief, obstacle, relocation / deletion, relocation / denoise],
    [`bimap` (act on both factors at once)], [no family --- see below],
  ),
  caption: [The cube census (`p2-nodream`). Belief selects the one asymmetric corner; every complement any family claims is claimed by a non-mental one. The atomic control gives the same census with `sync_to_world` in place of the register triple.],
) <tab-cube>

Every complement the census tracks --- the channel direction, the two scope corners, the world-side functor, the twist, the blank pairing, the reversed z-order, the model-side projection, the repulsive utility, the removal edits --- is claimed by some non-mental family, and belief claims none of them. Availability is not use: the search had all of them in hand at the same prior cost, and it took the corner that corresponds to acting on the world through a private model of it. The one row without a claimant is `bimap`, and its emptiness is measured rather than mysterious: the layer-compositing family written to claim it --- two channels, one map each, a task a paint program performs --- went 0 of 4, and the miss is a template-budget artefact, its intended program sitting just past where the shared per-round template sweep ran out of time (@sec-verdict). A corner the search could have taken and did not is what makes taking the belief corner a choice, and eleven of the twelve corners are now also independently *exercised*; the twelfth awaits a run with the budget to reach it.

=== Attribution is selective <sec-selective>

The two-observer family puts two agents in one scene with one goal and one grid, where only one of them is wrong about anything and the other's walk is plain desire. If what the learner acquired were a habit of forking, it would fork for both. In `p2-nodream`, 24 of 24 solved scenes commit for the believer and not for the bystander; in `p1-nodream`, 24 of 24. There is no over-attribution anywhere in the 48 solved two-observer scenes.

This is also the place to explain a number that looks alarming in the combinator census and is not. Of the 132 belief solutions in that cell, 9 commit through the literal `register (locate av) (place av)` and 123 through a *scope complement* --- `sync_all`, or `sync_except gv`. Read as a token, that looks like belief giving up its corner. Read as a computation, it is not: on a scene whose model diverges from the world only at the misrepresented content and the movers, "move everything to its model position" and "move everything but the goal" both denote "move the agent", and the verdict machinery verifies this by swapping the commit for the single-value one and re-rendering all four scenes of every task. Every one of the 123 passes, and the falsifying class --- a scope commit that adopts a world-edit no modelled agency explains --- is empty in both cells.

Why the searcher prefers the scope spelling is not a mystery either; it is minimum description length doing exactly what this thesis says it does, one level down. Under the combinator vocabulary the whitelist commit costs three symbols --- `register`, `locate`, `place` --- while the blacklist commit costs one, because the scope complements fold over an unbounded value set and were left atomic on principle (@tab-combinators): the decomposition reaches exactly as far as the single-value `locate`/`place` vocabulary does. Same denotation, cheaper intension, so compression buys the blacklist 123 times out of 132. In the atomic control, where both spellings cost one token, it buys the whitelist 155 times out of 155. The commit *form* tracks the price list while the commit *denotation* stays fixed --- and note that this decomposition boundary handicaps the agency spelling, not its rivals, so the census asymmetry runs against the result, not for it. The internal control is the false-obstacle family, constructed so that no scope complement reproduces its scenes: a real wall and a stationary goal sit alongside the phantom one, so a wholesale commit would move them. Every solved false-obstacle task in either run --- 16 of 24 in `p1-nodream`, 9 of 24 in `p2-nodream` --- commits through the literal single-value spelling. The spelling switches exactly at the semantic boundary, where the blacklist would be extensionally wrong --- which is not the behaviour of a classifier reading tea leaves off strings. Over both cells the census reports *zero* non-mental commits: every belief solution in either run publishes exactly what some model sought.

The coverage all of this is computed over is the weakest number in the chapter. Three families are complete --- wall-belief 24 of 24, witness 48 of 48, two-observer 24 of 24 --- but goal-displacement is 27 of 48 and the false-obstacle family, the deepest compound in the corpus, naming a real wall, a phantom wall, a displaced goal and an agent, is 9 of 24. The atomic control reaches 155 of 168 on the same corpus: complete on the first three, 43 of 48 on goal-displacement and 16 of 24 on false-obstacle. Since its vocabulary expresses nothing this one does not, the 23 tasks in the gap are tasks this run ran out of budget on rather than tasks it could not state --- @sec-limits is careful about how much that argument is worth.

== Scaffolding <sec-scaffold>

@timeline records that theory of mind arrives in a fixed order, and @constructivism's account of why is that later concepts are built out of earlier ones. Nothing in the corpus asserts such an order: a task carries only its own kind, and no family is marked as scaffolding for another. Whether the order is there is therefore a question to be put to the runs, and it can be put in a way that does not presuppose the answer. Take each solved belief program, abstract its concrete values and coordinates to typed holes, and ask which *other* families' solved programs occur inside it as subterms.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    table.header([Donor family], [Belief tasks it feeds], [The fragment, and the role it plays]),
    [desire], [132], [`(optimize (neg_dist #v) #v)` --- the action policy],
    [obstacle], [84], [`(compose (wall_at #c #c) (optimize (neg_dist #v) #v))` --- the derive],
    [registration-except], [51], [`(sync_except #v)` --- the commit, in its scope spelling],
    [wall belief], [51], [a phantom-wall constructor, feeding the witness family],
    [physics], [48], [`(step #v up)` --- the displacement inside a displaced-goal derive],
    [goal belief], [26], [a displaced-goal constructor],
    [registration], [9], [`(register (locate #v) (place #v))` --- the agency commit],
    [witness belief], [3], [a composed pair of agent blocks],
    [flee], [2], [`(optimize (distance #v) #v)`],
    [two-observer belief], [2], [a seek composed with an agent block],
  ),
  caption: [Structural scaffolding recovered from `p2-nodream`: which shallower solved programs occur as subterms of belief solutions. Nothing in the corpus labels any of these a scaffold; the relation is read off the programs.],
) <tab-scaffold>

Every one of the 132 belief solutions contains a desire solution. That is the strongest and least surprising row: belief's policy *is* the desire family's program, and that family has nothing mental about it --- it is one object approaching another. The commit is the same story split across two spellings: 9 solutions contain the registration family's `register (locate v) (place v)` outright, and 51 the registration-except family's scope commit, which on these scenes denotes it (@sec-selective). The obstacle family supplies the derive for 84. And the belief families feed each other in the order the corpus's difficulty predicts: a wall-belief constructor sits inside 51 witness-belief solutions, a witness-belief being a wall-belief with a second, unforked agent walking across the phantom cell. The tool prints the same fact as an assembly --- here a witness-belief solution, annotated with the family that certifies each maximal fragment:

#terminal("(compose
  (optimize (neg_dist 4) 5)                            <- desire  [action policy]
  (pipe_gpg (compose_gp dup
              (mapsnd (compose (wall_at c3 c1)
                               (optimize (neg_dist 2) 1))))
            sync_all))                                 <- wall belief  [agent block]")

The second mechanism is compression rather than containment, and it is the one the objective actually paid for: when a belief program and a non-belief program are rewritten through the *same* invented token, that token is shared reuse. Here there are exactly two such tokens, and they are exactly the two donated parts: `fn_8`, the seek fragment shared between belief and the obstacle family, and `fn_11`, the reassembled `register (locate v) (place v)` shared between belief and the registration, perception and drifting-registration families --- alongside the two round-1 tokens that first rebuilt `fork` and `sync_to_world` out of the overlay and registration families. The atomic control has only the seek fragment, since there is no pair interface left for it to rebuild. So the sharing is real but narrow, and it is narrow in the right place: what the non-mental families donate is the *parts* --- the policy and the commit --- and every fork-bearing token is belief's alone. What is belief-specific is the *wiring*. An abstraction shared wholesale with a non-mental family would not be an agent constructor.

Two caveats. This is scaffolding in the sense of compositional dependence recovered from a converged run, not a longitudinal claim about a learner acquiring one rung and then the next; the ordering evidence for that is @sec-order. And the curriculum is ours: the obstacle and relocation families are deliberately over-weighted precisely so that the wall-and-navigate policy would recur often enough to be abstracted. That is a modelling choice about what the learner sees, not a thumb on the scale of what it concludes, but it is a choice, and @sec-limits returns to it.

== The behavioural probe <sec-probe>

Everything so far is description length. @criteria's third condition was behavioural: on a scene held out of the run entirely, does the learned term predict that the agent goes where it *believes* the goal to be?

The probe uses the goal-displacement family (@fig-belief-goal), because it is the one where the two readings come apart in behaviour. In a wall-belief scene the transient-wall rival walks the agent along the same path, so only description length separates the readings there --- that was @sec-rivals. In a goal-displacement scene the agent walks to where it believes the goal is, one cell from the true goal, which never moves, so a program that does not represent the belief cannot land on the right cell.

The scenes are drawn from a seed that never entered the run and are checked to be absent from the training corpus, and the library is rebuilt by re-stitching the run's own found programs, so the terms priced are the terms the run converged to. We then instantiate the discovered belief compound on the novel scene and, as its foil, the shortest program the same library can express without it.

#figure(
  image("behavioral_probe_p2.png", width: 100%),
  caption: [The false-belief test on a held-out Sally-Anne scene (`p2-nodream`). The learned belief compound sends the agent to the believed cell; the shortest non-mental program under the same library sends it to the true goal.],
) <fig-behavioral-probe>

Through the library, the belief compound is written `(fn_6 1 2 (step 2 up) (fn_11 1))`, which expands to

```lisp
(pipe_gpg (compose_gp dup (mapsnd (compose (step 2 up)
                                           (optimize (neg_dist 2) 1))))
          (register (locate 1) (place 1)))
```

--- the discovered constructor applied to a derive that privately displaces the goal, committing only the agent's own move, and committing it here in the whitelist spelling, `fn_11` naming the agent outright. Its final frame puts the agent on the believed cell, one step short of the true goal. The learner restricted to the non-mental fragment writes the shortest thing that fragment can say, plain desire `(optimize (neg_dist 2) 1)`, and walks straight to where the goal really is. Across all 24 held-out scenes the belief compound lands on the believed cell every time; the non-mental program lands on the true goal cell in 18 of 24 and, in the remaining six, one step short of it on the true-goal side, walking toward the real goal and running out of frames. It never once lands on the believed cell. The same probe against the atomic control's library gives the same 24 of 24, as it must: the two endowments differ in how the compound is *spelled*, not in what it computes.

The nats are the point of the exercise. Here the non-mental program is *shorter* --- 7.8 against 17.8, and 7.9 against 13.4 under the atomic control --- which is the opposite of @sec-rivals, and that is why both measurements are needed. Where a non-mental rival reproduces the scene, it does so only at greater length, so description length rules it out; where a non-mental rival is cheaper, it is cheaper only because it predicts the wrong behaviour, so the data rule it out. There is no program in the non-mental fragment that is both as short as the belief compound and as accurate. A learner that has compressed its experience into that compound passes the false-belief task; a learner confined to the fragment fails it, and fails it in the specific way the developmental literature documents --- by answering with the true location @wimmer_beliefsabout_nodate.

== The verdict <sec-verdict>

@criteria fixed in advance what would count as success and what would refute the hypothesis. @tab-verdict is the tally.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([Criterion (@criteria)], [Combinator], [Atomic]),
    [An abstraction enters the library that opens a private channel, derives a divergent state, evaluates a policy against it, and commits only the action, with one value shared between derive, policy and commit],
    [Met, round 2 (`fn_7`, one shared hole used four times)],
    [Met, round 1 (`fn_11`, one shared hole used twice)],

    [Selected by the joint objective over the non-mental rivals the same library expresses],
    [Met: $+10.67$ nats over the nearest rival under the library, from $-0.08$ under base primitives],
    [Met: $+10.91$ nats, from $+1.28$],

    [No rival that never opens a private channel reproduces all $k$ scenes of a mental family],
    [Met: 0 of 14 rival spellings over 132 tasks],
    [Met: 0 of 14 over 155 tasks],

    [The term generalises behaviourally on a held-out scene],
    [Met: 24/24 to the believed cell],
    [Met: 24/24],

    [The term is applied selectively, not to every mover],
    [Met: 24/24],
    [Met: 24/24],

    [Belief coverage],
    [132/168 (79%)],
    [155/168 (92%)],
  ),
  caption: [The criteria of @criteria against the runs.],
) <tab-verdict>

Four things need saying here rather than in a footnote, the first of them for the first time in this corpus's history to the good.

*The combinator cell's machine verdict passes as computed.* Every automated criterion --- parts general, agency unique, zero non-mental commits, constructor invented and belief-specific, the cube census --- comes out true in the run's own verdict artifact, with the scope-commit verification of @sec-selective performed live rather than by later reclassification. Earlier revisions of this corpus could not say that, and the difference is not a relaxed check but a repaired corpus: the bystander and $k$-scene changes of @sec-rivals are what removed the leaks the old verdicts caught.

*The atomic cell fails one check,* `obstacle_belief_misrepresents_obstacle`, which demands that *every* solved task in the two families whose agent is wrong about an obstacle --- plain wall-belief and false-obstacle --- have a private model in which the obstacle is misplaced. Plain wall-belief passes 24 of 24, so the failures are in the false-obstacle family, where a handful of tasks were solved by a program representing the agent as wrong about the *goal* rather than about the wall: a nested-fork displaced-goal model, which reproduces the scenes and is a belief attribution, but not the one the family was built to force. This is a mis-specified check rather than a non-mental rival --- the criterion asks about the wall because that is what the family names, but the family's scenes admit a second mental explanation the check does not recognise.

*One non-mental family went unsolved.* The layer-compositing family --- the four tasks written to claim `bimap`, the one corner of @tab-cube without a claimant --- was missed in every round. The cause is measured, not conjectured: its intended program sits at roughly the 40,000th position in the template sweep's enumeration order, and the shared 60 s per-round budget reached about the 30,000th on the run's hardware; the same sweep on faster hardware solves all four in under 40 s, with the intended `bimap` program. It is a budget artefact of exactly the kind @sec-order says description length creates, and the fix --- a larger template budget --- is a rerun, not a redesign.

*Coverage is incomplete, and unevenly so.* The combinator cell misses 21 of 48 goal-displacement tasks and 15 of 24 false-obstacle tasks. The case that these are budget misses rather than expressive ones is the atomic control: its solve set strictly *contains* the combinator cell's --- 23 belief tasks solved there and not here, none the other way --- in a vocabulary that expresses nothing this one cannot, and the combinator solve-time distribution (@fig-solve-dynamics) has a large mass pressed against the 3600 s ceiling. That is an argument from a shorter spelling of the same programs, not from a longer run, and a longer run is what would settle it; this thesis does not have one. The `belief_dual` family, removed from the corpus for the reasons @criteria gives, is the version of this that could not be papered over at all.

Against that, what did come out clean is the load-bearing part. In both cells the compression step invented a term with the shared-hole signature; in both the belief families used it and no non-mental family did; in both the cube census found belief on the one asymmetric corner, with every complement any family reached claimed by a non-mental one; and in both, not one belief solution committed a value that no model had sought.

== Objections <sec-objections>

=== "`fork` and `sync_to_world` are just `believe` cut in half"

The sharpest objection is that the atomic primitive set of the control run already contains the answer. By the definition this thesis took from the literature, mentalizing is attributing to an agent an intensional state --- a representation of the world that may diverge from it. `fork` builds a divergent representation and `sync_to_world` reads an agent's position off it. Putting them together is not discovery; it is reassembly of something that was handed over in two pieces.

Three replies, each of them a measurement already reported. The parts have non-mental uses and take them: `fork` is what the overlay, underlay and comet families' *ground-truth* programs are made of, `sync_to_world` what image registration's are, and a primitive that only ever appeared inside belief compounds would be a `believe` in disguise where one image registration uses is a general-purpose tool. The field is symmetric and the complements are live: @tab-cube finds all of them but one inhabited by other families in the same run at the same prior cost, with belief taking none --- and it is one thing to observe that a learner did not go the other way, another to observe that other tasks in the same run *did*. And the objection is in any case to a variant the thesis does not rest on: remove the two primitives, as the combinator library does, and @sec-combinator-run is the structure coming back, reassembled in round 1 out of non-mental evidence with no belief task solved. If `fork` were `believe` in disguise, that run would have had to discover `believe` from scratch, and it did.

Two residual forms of the objection turn on nothing the runs measured, and @no-separation answers them: that the pair is itself the representational capacity, when building a divergent world state is counterfactual reasoning and attributing it to an agent is the conjunct the shared hole adds; and that `dup` could in turn be decomposed into allocation, reads and writes, which relocates the regress rather than ending it.

=== "The corpus was built to produce this result"

The last objection is about the curriculum rather than the language, and it has the most substance. The obstacle family is deliberately the densest corner of the corpus, at six tasks per (goal, agent) combination, because the wall-and-navigate policy has to recur often enough for compression to abstract it; belief's derive is short only because that abstraction exists. Change the mix and the result moves.

But this is a claim about what the learner is *shown*, not about what it concludes, and it is exactly the claim @constructivism makes on the theory theory's behalf: a concept becomes available when earlier learning has made it cheap, so the composition of earlier experience is a determinant of what is learnable and when. A model in which the curriculum did not matter would be one in which the ordering result of @sec-order was vacuous. Nor is the corpus stratified in the way the objection needs: it is one undifferentiated bag with no mental / non-mental label anywhere in the search or the compression step, and the joint stitch of @compression pools both root types (@positive-differences), so the budget that paid for belief's structure was free to be spent on overlay, registration, obstacle or relocation instead. What is missing is the dose-response experiment that would quantify the mix --- vary the non-mental fraction, measure whether the constructor is still invented. It is the single most informative experiment left undone, and @sec-limits says so.

== What the results do not show <sec-limits>

Seven limits, the costliest first.

*The belief has no history.* @no-range states the concession and the results do not soften it: nothing in the discovered term represents how the agent came to be wrong, and nothing updates when it looks again. What should be added is that the comparison to BToM runs the wrong way. BToM's belief is at least a state that could in principle be revised, and this one is not, so the gap is not the symmetrical one that section's parallel might suggest. @sec-persistent takes it up.

*The curriculum mix is untested.* As @sec-objections concedes, the corpus-mix dose-response has not been run. Nor have the two others that would sharpen the result: a silo-versus-joint stitch, which would show whether the constructor survives being compressed against belief alone, and a seed-robustness sweep reporting every verdict metric as $k$ out of five corpus seeds. Each of the two cells reported here is a single seed.

*Omitting the recognition model was decided on the evidence, not in advance.* @no-amortization gives a good reason for leaving DreamCoder's third phase out, but it is not the order in which the work happened: the loop was run both ways first, and the numbers settled it. The measurement is from the July runs, on the 300-task corpus that predates the pair-plumbing families of @corpus; the ablation was not repeated on the final corpus. Under the atomic control the dreamed proposal was a small win --- 291 tasks and 159 belief tasks against 287 and 155 --- concentrated in the two hardest families; under the combinator library it was a substantial loss, 240 against 264, with the damage in the witness family, which fell from 48 of 48 to 26 of 48. The mechanism is not the obvious one. The dreamed prior was gated to families with at least one solved instance, so it cannot be what delayed the first belief solve, and both combinator cells made theirs in round 2. What differed was the *library*: the two cells solved slightly different sets of tasks in round 1, the joint stitch therefore saw a different corpus, and the dreamed cell converged on a library through which the witness compound is longer. The deficit was transmitted by the compression step rather than by the proposal distribution --- the two interact, since a proposal that changes *which* tasks are solved changes what the next library can be. Characterising that interaction is left undone. What can be said is that the constructor was invented in the same round, with the same shape, in all four cells, so the structural result does not turn on the choice either way.

*Coverage is not complete, and the misses are not proven to be budget misses.* @sec-verdict gives the numbers. The claim that a longer run would close them is supported by the comparison between the two endowments and by the shape of @fig-solve-dynamics, and it is not supported by a run. The experiment that would settle it is a budget dose-response --- the same corpus at 600, 1200 and 2400 s per task, with the solve rate per belief family plotted against the budget --- which would separate a family that is out of *reach* from one that is merely out of *time*.

*The one-model frame is fixed by the type system, not discovered.* Both endowments give the learner the product $G times G$, which says that a learner holding a private representation holds exactly one --- and a one-model frame is precisely the kind of thing a theory of mind is supposed to have. Nothing reported here shows that compression would *select* a single private channel if more were expressible, because in these runs no more are. The experiment that would settle it replaces the fixed pair with a recursive grid-stack, so that channel arity becomes a free parameter, and measures which arity each family selects; @sec-arity sets it out as further work.

*The planner is a single token.* @rationality defends granting a planner but not granting it in this form; `optimize` packs a full breadth-first search into one symbol, and a more elementary treatment would decompose it into memory, conditionals and interaction (@sec-optimize).

*None of this is a claim about children.* The model is a sufficiency result at the computational level: it shows that a learner with domain-general representational resources and a preference for short programs *can* arrive at belief-structured abstractions, which blocks the inference from "no mechanism has been named" to "no mechanism exists". Whether human learners use this mechanism, or another one that solves the same problem, is not settled by anything in this chapter.

#load-bib(read("refs.bib"))
