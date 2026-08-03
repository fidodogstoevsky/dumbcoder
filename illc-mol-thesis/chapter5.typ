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

#show table.cell.where(y: 0): set text(style: "normal", weight: "bold")
#show table: set par(justify: false)
#set table(stroke: (_, y) => if y == 1 { (top: 0.9pt) } else if y > 1 { (top: 0.2pt) })

#mol-chapter("Results")

Chapter 3 wrote down the structure that a Bayesian theory of mind stipulates: a planner evaluated against a model of the environment supplied as an input, that input permitted to diverge from the state of affairs the observer takes to obtain, and the whole thing run on a particular agent's behalf. Chapter 4 built a learner that has none of it --- a single-grid interpreter, a monomorphic type discipline over grids and cell values, and a stock of domain-general primitives every one of which is exercised by some non-mental family --- and stated in advance (@criteria) what would count as that learner constructing the structure anyway. This chapter reports what the learner did.

The claim the results are meant to support is a structural one, and it is worth stating precisely before any numbers. It is not that the belief tasks were solved; a solved belief task with an extensional program would be a failure of the corpus. It is that the joint description-length objective of @compression --- the same objective, with no term keyed to the mental families --- selects into the library a term whose body has the shape Chapter 3 stipulates, and that it does so from a starting point where that shape was neither a primitive nor a type. @sec-runs fixes what was run. @sec-found reports the libraries the runs converged to and @sec-btom-map lays the discovered term beside the BToM structure it is supposed to resemble. @sec-compression through @sec-probe then discharge the four criteria of @criteria in turn: that the term is a compression win over the non-mental rivals, that its reachability explains the developmental ordering, that it generalises and is applied selectively, and that it passes the false-belief test on held-out scenes. @sec-scaffold recovers, from the runs themselves, which shallower tasks it was assembled out of. @sec-verdict then tallies the result against the criteria, including what failed, and @sec-objections takes the objections.

== The runs <sec-runs>

Everything reported here comes from two full runs on the Snellius cluster, one for each endowment of @primitives: the assembled combinator library of @tab-combinators, and the control that grants `fork` and `sync_to_world` outright (@sec-atomic-run). Each run enumerates a corpus of 300 tasks --- 168 belief tasks across five families and 132 non-mental ones across sixteen --- under a per-task budget of 3600 s (1200 s in round 1, which is a cap on the round that has no library yet), for up to six ECD rounds, with a joint Stitch call of six iterations at the end of each. A task is four scenes sharing one latent program (@corpus), so the 300 tasks are 1200 trajectories, and a candidate solves a task only if it reproduces every frame of all four.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, center, center, center),
    table.header([Cell], [Primitives], [Solved], [Belief solved], [Constructor round]),
    [`p2-nodream`], [combinator (@tab-combinators)], [264/300], [132/168], [2],
    [`p1-nodream`], [atomic control],                [287/300], [155/168], [1],
  ),
  caption: [The two runs; the cell names are the run identifiers, `p2` for the combinator endowment and `p1` for the atomic one, with `nodream` recording that neither uses a recognition model (@no-amortization). "Constructor round" is the ECD round at whose compression step an abstraction with the agency signature of @sec-found first entered the library. The atomic cell was run on 2026-07-29 at commit `a6965cd`; the combinator cell on 2026-07-28 at commit `decb58d`, with its commit-form census re-derived under the later classifier (see @sec-selectivity).],
) <tab-runs>

Two facts about @tab-runs should be recorded now rather than discovered later. The first is that the combinator cell reported here is a day older than the atomic one. A combinator rerun at the later commit was launched and did not finish: the deeper search needs roughly fourteen hours of wall-clock at these budgets and the job was given eight, so it died in the middle of its third round without writing a verdict. What changed between the two commits is a classifier and a check, not the search, and the combinator solutions have accordingly been re-classified under the later code without being re-searched. Where that re-classification changes a number I say so.

The second is that the figures and tables in this chapter are computed from `p2-nodream` --- this is the run the argument of @primitives rests on. Where the atomic control is informative the corresponding number from `p1-nodream` is given alongside, and every claim in @sec-verdict is checked against both cells.

Each run writes three artifacts: the programs it found (with their library-rewritten forms), a per-round trajectory with timings, and a verdict object recording every criterion of @criteria as a machine-checked boolean. The figures and tables below are computed from those artifacts by separate analysis scripts, so that nothing reported here is produced by the code that produced the result.

== What the loop finds <sec-found>

=== The combinator library: a two-stage bootstrap <sec-combinator-run>

The learner is handed the product combinators of @tab-combinators and must assemble the pair interface before it can use one. The result is not a slower version of the control run of @sec-atomic-run; it is a two-stage bootstrap, and the two stages are worth separating because the first is the strongest evidence in the chapter that the machinery belief uses is not belief's machinery.

Round 1 solves 132 tasks and *none* of them are belief tasks. Every non-mental family falls --- physics, desire, the obstacle and relocation families, the overlay and underlay and comet families, all four registration variants --- and the belief families are, at 1200 s, out of reach entirely. The compression step then runs over those 132 purely non-mental solutions and returns, among its six abstractions,

#terminal("fn_1: (pipe_gpg (compose_gp dup (mapsnd #0)))                [fn, fn_p_g] -> fn
fn_5: (register (locate #0) (place #0))                    [cellvalue] -> fn_p_g")

which are, by the identities verified before the run began (@signature), exactly `fork` and `sync_to_world`. The learner has reassembled the pair interface out of the categorical plumbing, and it has done so from evidence in which no agent holds a belief about anything: `fn_1` is induced by the overlay, underlay and comet families, `fn_5` by image registration.

Round 2 therefore starts where the control run of @sec-atomic-run starts, and 75 belief tasks fall. Its compression step returns the agency signature proper,

#terminal("fn_7: (pipe_gpg (compose_gp dup (mapsnd (compose (wall_at $2 $1) (optimize (neg_dist $3) $0))))
           (register (locate $0) (place $0)))")

in which `$0` --- the agent --- appears *three* times: once as the value `optimize` plans for, once in the `locate` that reads its position off the private model, and once in the `place` that writes that position into the world. Rounds 3 and 4 take the remaining 57, and the run converges at 132 of 168 belief tasks. The final library, expanded to base primitives, is @tab-lib-p2.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, center),
    table.header([Symbol], [Body], [Belief solves]),
    [`fn_6`], [`(pipe_gpg (compose_gp dup (mapsnd (compose $2 (optimize (neg_dist $1) $0)))) $3)`], [18],
    [`fn_7`], [`(pipe_gpg (compose_gp dup (mapsnd (compose (wall_at $2 $1) (optimize (neg_dist $3) $0)))) (register (locate $0) (place $0)))`], [37],
    [`fn_8`], [`(compose (optimize (neg_dist $0) $3) (pipe_gpg (compose_gp dup (mapsnd (compose (step $0 $1) (optimize (neg_dist $0) $2)))) (sync_except $0)))`], [26],
    [`fn_9`], [`(compose $2 (optimize (neg_dist $1) $0))`], [46],
    [`fn_10`], [`(pipe_gpg (compose_gp dup (mapsnd (compose (wall_at $0 c2) (optimize (neg_dist $1) $2)))) sync_all)`], [32],
    [`fn_11`], [`(pipe_gpg (compose_gp dup (mapsnd (compose (step $0 $1) (optimize (neg_dist $0) $2)))) (sync_except $0))`], [22],
  ),
  caption: [The final library (`p2-nodream`), bodies expanded to base primitives. `fn_7` is the agent constructor proper --- one hole bound three ways. `fn_8`, `fn_10` and `fn_11` are the same shape spelled through a scope commit (@sec-selective); `fn_9` is the bare seek policy belief shares with the obstacle family, through which 47 obstacle solves are also rewritten; `fn_6` is the constructor with its commit left as a hole.],
) <tab-lib-p2>

Four of the six carry the agency signature and a fifth, `fn_6`, is the constructor with its commit slot still open. The one term with no fork in it, `fn_9`, is the seek policy --- and it is the only invented token that belief and a non-mental family use in common, which is the sharing @sec-scaffold takes up. The three terms that commit through `sync_all` or `sync_except` are not a retreat from the single-value corner; on scenes whose only mover in the model is the agent those commits *denote* the single-value one, which @sec-selective verifies by substitution and re-rendering.

This is the bootstrapping structure of @sec-hbm in its clearest form. An abstraction discovered in round $N$ is an atom in round $N+1$, and here the round-1 atoms are precisely the two capacities --- hold a second grid, transfer one value between two grids --- that belief needs and that non-mental tasks independently pay for. Nothing schedules this order. It falls out of what is cheap at each stage.

=== The atomic control <sec-atomic-run>

The second endowment is a *control*, in which `fork` and `sync_to_world` are granted outright as single nodes rather than assembled from the product structure, so that belief is the two-token composition `(fork (compose (wall_at r c) (optimize (neg_dist gv) av)) (sync_to_world av))` with the agent value appearing twice instead of three times. This is not a second result, and it is not a weaker version of the one claimed above; the argument does not rest on it (@no-separation). Its use is as an instrument. Because the two vocabularies are expressively identical --- which is what the identities of @signature establish --- a task the control solves and the combinator run does not is a task lost to the search budget rather than to the language, and that is a distinction no single run can draw. It also prices the same non-mental rivals at a second granularity, which is what shows the compression result is not an artefact of where the primitives were cut.

The control hands over at the start what the combinator run spent its first round building, so the whole of the search goes on the wiring rather than on the parts. It solves more --- 287 of 300 tasks and 155 of 168 belief tasks against 264 and 132 --- and its solve set strictly contains the combinator cell's, which is the fact @sec-limits leans on: 23 of the 36 belief tasks the combinator run missed are solved here, in a vocabulary that expresses nothing the other does not.

Round 1 begins with no library, a uniform prior over 37 primitives, and a 1200 s budget per task. It solves every non-mental family and 55 of the 168 belief tasks by brute cheapest-first enumeration. The obstacle families fall almost immediately,

#terminal("[   859] caught (compose (wall_at c1 c1) (optimize (neg_dist 5) 4))")

and the belief tasks built on the same wall content fall two and a half orders of magnitude later in the cost walk:

#terminal("[297342] caught (fork (compose (wall_at c1 c2) (optimize (neg_dist 2) 1)) (sync_to_world 1))")

The bracketed number is the position in the enumeration, and the ratio between the two is the concrete form of the point @sec-search made abstractly: a belief compound is not a slightly less probable hypothesis than an obstacle compound but a hypothesis some three hundred times deeper into an exponentially growing space. The comparison is not rigged by the choice of content: the cheapest belief task of any family in round 1, one whose private derive is a bare `optimize` rather than a wall, still lands at position 21528.

The compression step at the end of round 1 returns six abstractions. Two matter. The first is the wall-and-navigate policy that the obstacle family shares with the belief tasks' derive:

#terminal("fn_1: (compose (wall_at #3 #2) (optimize (neg_dist #1) #0))  [cellvalue, cellvalue, coord, coord] -> fn")

and the second is built on top of the seek fragment, and is the thing this thesis is about:

#terminal("fn_0: (fork (compose #2 (optimize (neg_dist #1) #0)) (sync_to_world #0))  [cellvalue, cellvalue, fn] -> fn")

`fn_0` takes a goal value `#1`, an agent value `#0`, and an arbitrary `fn` in the hole `#2`. It runs `#2` on a private copy of the grid, plans `#0`'s move on the result, and publishes `#0`'s move --- and only `#0`'s move --- back to the world. The content of the false belief is the hole; the agent is the parameter that appears twice.

In round 2 the budget is the same but the library is not, and the remaining 100 belief tasks fall. What had been a search hundreds of thousands of steps deep is now a fill-in-the-blanks --- here a witness task, an agent misled about where its goal went alongside a second agent who is not, caught by composing a plain `optimize` for the witness with `fn_2`, which is `fn_0` with a `step` in its hole:

#terminal("[  8668] caught (compose (optimize (neg_dist 2) 6) (fn_2 up 2 1))")

Round 3 adds nothing, at which point the loop has converged. The final library, expanded to base primitives, is @tab-lib-p1.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, center),
    table.header([Symbol], [Body], [Belief solves]),
    [`fn_6`], [`(fork (compose $2 (optimize (neg_dist $1) $0)) (sync_to_world $0))`], [37],
    [`fn_7`], [`(compose $2 (optimize (neg_dist $1) $0))`], [18],
    [`fn_8`], [`(compose (optimize (neg_dist $1) $3) (fork (compose (step $1 $0) (optimize (neg_dist $1) $2)) (sync_to_world $2)))`], [24],
    [`fn_9`], [`(fork (compose (fork (compose (optimize (neg_dist $1) $0) (optimize (neg_dist 0) $0)) (sync_to_world $0)) (optimize (neg_dist $0) $1)) (sync_to_world $1))`], [14],
    [`fn_10`], [`(compose (fork (compose (wall_at c2 $0) (optimize (neg_dist $1) $2)) (sync_to_world $2)) (optimize (neg_dist $3) $4))`], [29],
    [`fn_11`], [`(fork (compose (wall_at $1 $0) (optimize (neg_dist $2) $3)) (sync_to_world $3))`], [52],
  ),
  caption: [The final library under the atomic control (`p1-nodream`), bodies expanded to base primitives. Five of the six carry the agency signature; `fn_7` is the bare seek policy, which belief shares with the obstacle family (47 obstacle solves are rewritten through it).],
) <tab-lib-p1>

Five of the six invented terms are agent constructors: a fork whose derive ends in a policy for some value and whose commit publishes that same value. They differ in what the private derive does --- an arbitrary hole in `fn_6`, a shove of the goal in `fn_8`, a phantom wall at a free coordinate in `fn_11` and at a fixed column in `fn_10` --- and in what else is composed around them. The most-used of them is `fn_11`, the phantom-wall constructor, through which 52 of the 155 belief solutions are rewritten. `fn_9` is worth a remark: its derive contains another fork, so it is an agent whose private model is itself derived by forking. Fourteen belief solutions are rewritten through it. That is not second-order mentalizing in the sense of @timeline, since the inner fork is part of the outer agent's model rather than a model of another mind, but it does establish that the discovered shape embeds, which is one of the properties @explanandum asked for.

=== The agency signature

Under either endowment the invented term has the same shape, and the shape is a coincidence in the grammar that the objective declined to treat as one. Written in the abbreviations of @signature, the constructor is

```lisp
(fork (compose <derive> (optimize (neg_dist gv) av))
      (sync_to_world av))
```

`optimize` takes a `cellvalue` saying whose move to compute. The commit takes a `cellvalue` saying whose position to publish. These are different argument slots on different primitives, filled independently by the type-directed enumerator, and there are ten cell-value terminals to fill each of them with. Nothing in the grammar, the type system or the interpreter requires them to agree. In the invented abstraction they are one hole --- and when the commit is itself assembled, as it is in `fn_7`, the same cell value has to land in the `locate` and the `place` as well, so the coincidence is threefold rather than twofold.

That collapse is what makes the term an attribution rather than a computation. A program that derives a counterfactual grid, plans on it, and publishes the result of planning for *some other* value is expressible, is one token cheaper to describe in no respect, and does not occur. What compression kept is the version in which the value whose behaviour is published is the value whose reasoning was simulated --- which is to say, the version in which the private grid is *somebody's*.

It should be said that nothing in this identifies the term as *belief* in any thicker sense, and on the account of intentionality this thesis has been working with, nothing needs to. To adopt the intentional stance towards a system is to treat it as having the beliefs it ought to have and to predict its behaviour accordingly @dennett_intentional_2002; what makes a state a belief is the role it plays in that predictive scheme, not an intrinsic mark it carries. The invented term is a state that is derived privately, is planned against, is attributed to one value rather than to the scene, and predicts where that value will go. That is the role. Asking whether the thing filling it is *really* a belief is asking a question the framework declines.

== The discovered term against BToM <sec-btom-map>

@stipulated listed what BToM assumes in order to model mature performance. The list is now a list of things to look for in the invented term, and @tab-btom-map is the correspondence.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([BToM (Chapter 3)], [In the discovered term], [Status here]),
    [A planner mapping (utility, environment model) to action],
    [`optimize (neg_dist gv) av`, applied to whatever grid it is handed],
    [Granted as a primitive (@rationality)],

    [The environment model $m'$ supplied as an #emph[input] to the planner],
    [The second channel of the pair, the only one the derive writes],
    [Discovered: `mapsnd`, not `mapfst`; the world channel is untouched],

    [$m eq.not m'$ permitted --- the divergence that makes false belief expressible],
    [The derive is an arbitrary `fn` applied to the private copy only],
    [Discovered: which channel is transformed is a search choice with an exercised opposite],

    [The goal $g$, ranging over a given space under a given prior],
    [Whichever `cellvalue` lands in `neg_dist`'s slot, priced like every other symbol],
    [Not a space: no goal type, no goal prior (@no-btom)],

    [The attribution: $m'$ is #emph[this agent's] model, on whose behalf the planner runs],
    [The shared hole --- `optimize`'s actor #emph[is] `sync_to_world`'s committed value],
    [Discovered: the conjunct the compression step kept],

    [The nested evaluation: the observer runs the planner inside itself],
    [`fork`'s private grid lives for one call and is discarded, never rendered],
    [Structural, in program space; the interpreter has no such mode (interpreter)],

    [Inverse planning: infer $(g, m)$ from the trajectory by Bayes],
    [Cheapest-first enumeration under the library prior, indicator likelihood],
    [The inference of @sec-bayes, unchanged and not agent-directed],

    [Belief as a state that is #emph[formed] and could be #emph[revised]],
    [Nothing: the model exists for one transition and has no history],
    [Absent --- the honest gap (@sec-limits)],
  ),
  caption: [What BToM stipulates, and where it turns up in the term the compression step invented.],
) <tab-btom-map>

Read down the middle column and the invented abstraction is a Bayesian theory of mind's agent, written in a language that had no word for one. Read down the right and the difference from Chapter 3 is where each row came from. BToM's rows are architecture, fixed before any scene is seen, and its inference locates a point inside them. Here, three of the rows --- which channel diverges, whose model it is, and what the divergence consists in --- are decisions made in program space by a search that also solves image registration and denoising, and kept by an objective that also had the option of spending its budget on the overlay and relocation families.

Two rows are worth dwelling on because they are the ones an unsympathetic reader will press.

The first is the planner. We grant it, and Chapter 3 gave the reason: on Gergely and Csibra's analysis the teleological stance relates action, goal and situational constraint without attributing any representation to anybody, and it is in place by twelve months, three years ahead of false-belief attribution @gergely_teleological_2003 @gergely_taking_1995. The results bear out the formal version of that argument. `optimize` appears in the desire family (16 solves), the flee family, the comet family and all 48 obstacle solves --- scenes which are, at the level of the rendered trajectory, nearly indistinguishable from the wall-belief scenes and contain nothing mental at all. What separates the mental families is not that they use the planner but what is wired around it.

The second is the pair. `dup` produces $(w, w)$ and says nothing about which copy is a model or who holds it; the asymmetry is entirely in the composition. @sec-selectivity shows that the symmetric alternatives are not merely available but *used* --- by other families, in the same runs --- which is what turns "the learner could have gone the other way" from an assertion into a measurement.

== The objective selected it <sec-compression>

The guiding thought behind the whole project is that an intensional theory --- one that attributes a belief to an agent --- is the most compressed available description of that agent's behaviour, and that this is not a coincidence. The agent navigates by its own representation of the world, so a theory that posits that representation is tracking the process that actually generated the data. One can of course describe the same trajectory without attributing anything, but the description has to account for each step of the walk separately, and it will not transfer to the next scene. The prediction, then, is that the description-length objective should prefer the intensional reading, and prefer it more the more experience the learner has had.

@criteria demanded that the compound be a compression win rather than merely reachable. There are three measurements here: what the library does to the description length of the whole corpus, what it does to belief programs specifically, and how the belief program compares to the non-mental rivals that the same library can express.

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

Under the base primitives the 264 solved programs cost 5328 nats. After four rounds they cost 3195, including the 54 nats the library charges for itself. Of the 2133 nats saved, 2187 come from the belief family alone --- 3986 nats down to 1799 --- and the largest single step is the 1216 nats that go at round 2, the round whose compression step invents the constructor. The non-mental families do not move. `world` (the sixteen non-mental families pooled) goes from 1203 to 1191, desire from 116 to 126, physics from 23 to 25: the last two get slightly *worse*, which is the library tax of @compression showing up exactly where it should --- a family that gains no abstraction still pays for the enlarged symbol set.

That asymmetry is the argument. If the initial primitives had encoded anything mental, belief would have been cheap from the start and the belief band would look like the desire band. It is because belief is expensive under the base library and cheap under the learned one that we can say the mental content is in what was learned rather than in what was given.

The atomic control gives the same picture a round earlier and from a higher starting point: 6017 nats down to 3687, belief carrying 4723 to 2345, with its largest drop --- 2025 nats --- at round 1, since there the constructor can be invented as soon as anything has been solved.

=== What the library buys each family

Per task, the same result in nats. @tab-dl-census prices each family's found program under the base primitives and under the final library.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header([Family], [Solves priced], [Base (nats)], [Final library], [Saved]),
    [belief],       [25], [34.87], [15.95], [*18.92*],
    [obstacle],     [48], [16.21], [12.95], [3.26],
    [desire],       [16], [7.86],  [7.86],  [0.00],
    [comet],        [4],  [16.09], [16.09], [0.00],
    [relocate],     [16], [14.13], [14.13], [0.00],
    [registration], [4],  [6.68],  [6.68],  [0.00],
    [perception],   [4],  [4.16],  [4.16],  [0.00],
    [flee],         [4],  [7.86],  [7.86],  [0.00],
    [deletion],     [4],  [5.78],  [5.78],  [0.00],
    [denoise],      [4],  [4.87],  [4.87],  [0.00],
    [multi_reg / reg_except / inpaint / readout], [4 each], [2.08--4.38], [unchanged], [0.00],
  ),
  caption: [Median description length per task (`p2-nodream`), before and after the learned library. Belief saves 18.92 nats per task and 472 nats over the priced solves; every family but obstacle saves nothing.],
) <tab-dl-census>

Belief is the only family the library substantially compresses, and obstacle is the only other family it touches at all --- through `fn_9`, the seek fragment the two share, which is exactly the donor relation @sec-scaffold recovers independently. The atomic control gives the same shape with a smaller belief saving, 33.71 down to 13.38 nats, since less of the compound is left to abstract once the pair interface is given.

One caveat about the belief row. This census prices only solutions whose library-rewritten form re-parses under the final library, which here is 25 of the 132 belief solves --- all of them from the witness family --- so the 18.92 is a median over a fifth of the family rather than over all of it. @fig-corpus-dl is the unrestricted version of the same measurement: it prices every solved program in every round, and gives the belief band's 3986 to 1799 nats directly.

=== No non-mental rival reproduces the scenes <sec-rivals>

The third measurement is the one @criteria cared most about: whether a non-mental program is available that is as short and as accurate. Each belief task carries a battery of hand-written rival spellings --- the transient-wall schedule that stamps a wall, lets the agent walk, and erases it before the frame is rendered; the version that shoves the goal in the world rather than in a model; pure desire; a physics reading with no wall at all; and, for the witness family, the four orderings of stamp, act, witness and erase. Each rival is rendered over every scene of every belief task and compared frame for frame.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header([Rival spelling], [Tasks], [Reproduces\ all 4 scenes], [Reproduces\ the agents], [Median margin\ (library, nats)]),
    [transient wall (stamp / act / erase)], [24], [0], [24], [$+6.27$],
    [transient wall (erase-3 variant)],     [24], [0], [24], [$+5.35$],
    [transient shove (shove / act / restore)], [51], [0], [51], [$+18.59$],
    [witness then stamp / act / erase],     [48], [0], [0],  [$+14.17$],
    [stamp / act / erase then witness],     [48], [0], [0],  [$+10.91$],
    [stamp, witness, act, erase],           [48], [0], [0],  [$+10.91$],
    [stamp, act, witness, erase],           [48], [0], [0],  [$+10.91$],
    [transient phantom $+$ shoved goal],    [9],  [0], [0],  [$+19.73$],
    [shove goal in the world],              [60], [0], [0],  [$+4.87$],
    [both agents seek the goal],            [24], [0], [0],  [$+4.17$],
    [transient, witness ignored],           [48], [0], [0],  [$+3.74$],
    [pure desire (no belief)],              [27], [0], [0],  [$-0.69$],
    [no wall (physics)],                    [72], [0], [0],  [$-2.53$],
    [no belief (seek past the real wall)],  [9],  [0], [0],  [$-2.53$],
  ),
  caption: [The non-mental rivals (`p2-nodream`), over the 132 solved belief tasks. "Margin" is $"DL"("rival") - "DL"("found")$ under the final library; positive means the mental reading is shorter. Not one rival reproduces all four scenes of any task.],
) <tab-rivals>

The headline of @tab-rivals is the third column. Across 132 belief tasks and fourteen rival spellings, *no non-mental rival reproduces all four scenes of any task*. This is a change from earlier versions of the corpus and it is a change in kind. It used to be that the transient-wall rival reproduced the wall-belief scenes exactly and had to be beaten on description length; the honest form of the argument was then a margin. Two corpus changes closed it. An inert bystander object now sits on the phantom-wall cell for the whole trajectory, so stamping a real wall there overwrites something the frame shows and nothing in the language can put it back; and the $k$-scene format (@corpus) requires a rival to survive four independent placements of the same latent, which a single wall coordinate chasing a believed goal cannot do. The fourth column records how close the rivals get: the transient-wall spelling still reproduces the *agents'* trajectories on all 24 wall-belief tasks, and the transient shove on 51 goal tasks. They get the walk right and the world wrong.

This is exclusion by reproduction rather than by price, and it is the stronger form of the argument --- a rival that cannot render the data is not a rival at all, whatever it costs. It is also the form that keeps the argument honest in the other direction: 111 of the 540 rival pricings in @tab-rivals come out *shorter* than the mental program, almost all of them the bottom three rows, and every one of them is a program that simply does not depict the scene --- an agent walking to a goal it is not wrong about. Cheapness without accuracy is worth nothing, and it is the likelihood of @sec-bayes rather than the prior that disposes of them. But the price is still informative, and the last column is where the compression story reappears. Take the closest rival, the transient wall. Under the base primitives it costs $0.12$ nats more than the mental program --- to three decimal places a tie, which is what one would expect, since neither reading has any structure to reuse. Under the final library it costs $6.27$ nats more --- a factor of about $e^6 approx 500$ in prior probability. Nothing about the rival changed. What changed is that the mental reading collapsed into a token the library had a reason to invent and the transient-wall schedule did not, because a bespoke stamp-act-erase sequence recurs nowhere else in the corpus.

The atomic control gives the same result from a slightly looser starting tie: $+1.28$ nats under base primitives, widening to $+10.91$ under its final library. That the effect survives both carvings of the primitive set is the point of running the control --- the compression win is not an artefact of how coarsely the parts were cut.

== Reachability, and the order in which things arrive <sec-order>

@sec-search argued that description length in this setting is simultaneously a measure of plausibility and a budget: a hypothesis that is long under the current library is not merely improbable but unreachable. @criteria turned that into a prediction about ordering. @fig-solve-dynamics is the prediction met.

#figure(
  image("solve_dynamics_p2.png", width: 100%),
  caption: [Solve dynamics (`p2-nodream`). Left: cumulative solves by family, by round. Right: for each belief task solved after at least one miss, its enumeration time in the round it missed against the round it was solved.],
) <fig-solve-dynamics>

The non-mental families are flat lines at their ceiling from round 1: they need no abstraction and get none. Belief is an S-curve --- 0, then 75, 107, 132 --- whose step is the round after the constructor enters the library. And the first point of that curve is a zero rather than a small number: with the pair interface given only in parts, not one belief task is solved until the compression step has reassembled `fork` and `sync_to_world` out of non-mental evidence. Belief here is not merely slow in round 1; it is out of reach.

The right-hand panel is the same fact per task, and it is more equivocal. All 132 belief solves were missed at least once, and the worst case goes from the full 3600 s timeout to 26.4 s, a factor of 136 --- but the *median* moves only from a 1200 s round-1 timeout to a 1164 s solve. What the library buys in this cell is therefore reachability rather than speed: it brings the compound inside the budget window at all, and most tasks then spend most of that window. The atomic control moves the median further --- of 100 tasks missed at least once, it goes from a 1200 s round-1 timeout to a 691 s solve, and the steepest collapse is from 1200 s to 2.1 s, a factor of 562 --- which is what one expects of a cell that starts a rung higher; but there too the typical task still spends more than half its budget. In neither case did the tasks or the budget change; what changed is the description length of their solutions, and with it the position of those solutions in the cost-ordered walk.

It is worth being clear about what this does and does not establish. It does not show that a child's four-year delay has this cause. What it shows is that in a learner of this kind the delay is not something that has to be put in. @stipulated's complaint against BToM was that its structure is complete from the start, so every parameter is available for inference on day one and nothing in the framework makes goal attribution earlier or cheaper than belief attribution; the ordering has to be imported from outside as maturation or as a performance limitation. Here the ordering is the mechanism. Desire is solved in round 1 because `(optimize (neg_dist 2) 1)` is a three-symbol program; belief is not solved until round 2 because its compound is out of budget until earlier learning has shortened it. That is the cumulative, staged picture @explanandum demanded, produced rather than accommodated: the bootstrapping of @sec-hbm, measured rather than assumed.

== Generality and selectivity <sec-selectivity>

An abstraction that is a compression win might still be a bad theory in two ways: it might have baked in the particular values it was trained on, and it might be applied to everything that moves. @criteria's fifth refutation condition was precisely the second of these --- a habit of forking rather than an attribution. Three measurements answer them.

=== The constructor spans the corpus

The corpus rotates through eight distinct (goal value, agent value) pairs so that no integer is privileged (@no-btom). @fig-agent-tiling counts, for each combination and each invented token, how many solved belief programs are rewritten through it.

#figure(
  image("agent_tiling_p2.png", width: 92%),
  caption: [Constructor coverage (`p2-nodream`). Rows are the eight (goal, agent) value combinations; columns are the six invented abstractions. Every column is occupied in every row.],
) <fig-agent-tiling>

The matrix is fully occupied: each of the six invented tokens is used in all eight combinations. The agent constructor `fn_7` carries 37 of the 132 belief solves and appears in every row; the most-used token is `fn_9`, the bare seek policy, at 46. So the constructor is not a memorised (goal, agent) pair with a wrapper --- its cell-value slots are genuine holes, exercised across every pair the corpus rotates through. The instructive comparison is the obstacle family, whose policy did *not* generalise into a single token in this run: because obstacle scenes vary their wall coordinate more than their (goal, agent) pair, no single wall-bearing policy recurred often enough to be abstracted with holes. Belief's derive is therefore not a single cheap token even after compression, which is to say that the belief compound is not shallow even with the constructor in hand. The atomic control gives the same picture slightly less evenly: its constructor `fn_11` spans all eight combinations, by 52 of 155 solves, but one of its six tokens --- the bare seek policy `fn_7` --- is missing from two rows.

=== The symmetric complements go elsewhere

@tab-combinators lists, beside each primitive belief uses, the opposite corner of an independent symmetry axis: channel direction, commit scope, z-order, projection, pairing, the bifunctor axis, and utility polarity. The point of including them was that an unused complement proves nothing. @tab-cube reports which family claimed which corner.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header([Corner], [Family that selected it]),
    [`register (locate v) (place v)` (read model, write world, one value)], [*belief* (37 solves), registration],
    [`via_swap` (the same commit through the swapped pair)], [perception],
    [`sync_all` (every shared value)], [multi-registration, perception],
    [`sync_except` (all but one)], [registration-except],
    [`dup` / `mapsnd` (diagonal; act on the second factor)], [*belief* (81), overlay, underlay, comet],
    [`overlay` / `underlay` (z-order)], [overlay, comet / underlay, inpainting],
    [`snd_gg` (project the model channel)], [readout],
    [`neg_dist` / `distance` (utility polarity)], [belief, desire, obstacle / flee],
    [`wall_at` / `clear_at` / `erase` (grid edits)], [belief, obstacle, relocation / deletion, relocation / denoise],
    [`mapfst`, `bimap`, `pair_blank`, `swap`], [no family, belief included],
  ),
  caption: [The cube census (`p2-nodream`). Belief selects the one asymmetric corner; every complement that any family claims is claimed by a non-mental one. The atomic control gives the same census with `sync_to_world` in place of the register triple and `sync_to_model` in place of `via_swap`.],
) <tab-cube>

Every complement the run's census tracks --- the channel direction, the two scope corners, the reversed z-order, the model-side projection, the repulsive utility, the removal edits --- is claimed by some family, and belief claims none of them. Availability is not use: the search had all of them in hand at the same prior cost, and it took the corner that corresponds to acting on the world through a private model of it.

The last row is the honest exception. Four of the combinators offered --- acting on the first factor, acting on both, pairing with a blank, and the bare twist --- are used by *no* family in this run. Their being available still does the work the design asked of them, since a corner the search could have taken and did not is what makes taking the belief corner a choice; but they are not, in these runs, independently exercised the way the other complements are, and a corpus with a family that needed them would make the point more cleanly.

=== Attribution is selective <sec-selective>

The two-observer family puts two agents in one scene with one goal and one grid, where only one of them is wrong about anything and the other's walk is plain desire. If what the learner acquired were a habit of forking, it would fork for both. In `p2-nodream`, 24 of 24 solved scenes commit for the believer and not for the bystander; in `p1-nodream`, 24 of 24. There is no over-attribution anywhere in the 48 solved two-observer scenes.

This is also the place to explain a number that looks alarming in the combinator census and is not. Of the 132 belief solutions in that cell, 37 commit through the literal `register (locate av) (place av)` and 95 commit through a *scope complement* --- `sync_all`, or `sync_except gv`. Read as a token, that looks like belief giving up its corner. Read as a computation, it is not: on a scene whose only moving value in the model is the agent, "move everything to its model position" and "move everything but the goal" both denote "move the agent", and the classifier verifies this by swapping the commit for the single-value one and re-rendering all four scenes. Every one of the 95 passes. What differs is the spelling --- the agency hole expressed on the goal rather than on the actor --- and it is cheaper by a token, which is why compression prefers it wherever the scene permits.

The family that does not permit it is the false-obstacle family, which is constructed so that no scope complement reproduces its scenes: a real wall and a stationary goal are present alongside the phantom one, so a wholesale commit would move them. Every solved false-obstacle task in either run --- 16 of 24 in `p1-nodream`, 9 of 24 in `p2-nodream` --- commits through the literal single-value `sync_to_world`. Where the degeneracy is excluded by construction, the agency commit is forced, and it is found.

Under the classifier at the later commit, which reads the commit form off the world-level commit position rather than off the solution string, the census over both cells reports *zero* non-mental commits: every belief solution in either run publishes exactly what some model sought.

#figure(
  image("belief_solved_p2.png", width: 92%),
  caption: [Belief coverage by family (`p2-nodream`). The faint bar is the corpus total; the coloured bar is what search recovered.],
) <fig-belief-solved>

@fig-belief-solved gives the coverage that all of this is computed over, and it is the weakest number in the chapter. Three families are complete --- wall-belief 24 of 24, witness 48 of 48, two-observer 24 of 24 --- but goal-displacement is 27 of 48 and the false-obstacle family, the deepest compound in the corpus, naming a real wall, a phantom wall, a displaced goal and an agent, is 9 of 24. The atomic control reaches 155 of 168 on the same corpus: complete on the first three, 43 of 48 on goal-displacement and 16 of 24 on false-obstacle. Since its vocabulary expresses nothing this one does not, the 23 tasks in the gap are tasks this run ran out of budget on rather than tasks it could not state --- @sec-limits is careful about how much that argument is worth.

== Scaffolding <sec-scaffold>

@timeline records that theory of mind arrives in a fixed order, and @constructivism's account of why is that later concepts are built out of earlier ones. Nothing in the corpus asserts such an order: a task carries only its own kind, and no family is marked as scaffolding for another . Whether the order is there is therefore a question to be put to the runs, and it can be put in a way that does not presuppose the answer. Take each solved belief program, abstract its concrete values and coordinates to typed holes, and ask which *other* families' solved programs occur inside it as subterms.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    table.header([Donor family], [Belief tasks it feeds], [The fragment, and the role it plays]),
    [desire], [132], [`(optimize (neg_dist #v) #v)` --- the action policy],
    [obstacle], [84], [`(compose (wall_at #c #c) (optimize (neg_dist #v) #v))` --- the derive],
    [registration-except], [51], [`(sync_except #v)` --- the commit, in its scope spelling],
    [physics], [48], [`(step #v up)` --- the displacement inside a displaced-goal derive],
    [registration], [37], [`(register (locate #v) (place #v))` --- the agency commit],
    [false-obstacle belief], [28], [the whole constructor, feeding witness and wall],
    [goal belief], [26], [a displaced-goal constructor],
    [wall belief], [23], [a phantom-wall constructor, feeding the witness family],
    [witness belief], [3], [a composed pair of agent blocks],
    [flee], [2], [`(optimize (distance #v) #v)`],
    [two-observer belief], [2], [a seek composed with an agent block],
  ),
  caption: [Structural scaffolding recovered from `p2-nodream`: which shallower solved programs occur as subterms of belief solutions. Nothing in the corpus labels any of these a scaffold; the relation is read off the programs.],
) <tab-scaffold>

Every one of the 132 belief solutions contains a desire solution. That is the strongest and least surprising row: belief's policy *is* the desire family's program, and that family has nothing mental about it --- it is one object approaching another. The commit is the same story split across two spellings: 37 solutions contain the registration family's `register (locate v) (place v)` outright, and 51 contain the registration-except family's scope commit, which on these scenes denotes it (@sec-selective). The obstacle family supplies the derive for 84. And the belief families feed each other in the order the corpus's difficulty predicts: a wall-belief constructor sits inside witness-belief, which is a wall-belief with a second, unforked agent walking across the phantom cell.

The tool prints the same fact as an assembly. A witness-belief solution, annotated with the family that certifies each maximal fragment:

#terminal("(compose
  (optimize (neg_dist 4) 5)                            <- desire  [action policy]
  (pipe_gpg (compose_gp dup
              (mapsnd (compose (wall_at c3 c1)
                               (optimize (neg_dist 2) 1))))
            (register (locate 1) (place 1))))          <- belief  [agent constructor]")

The second mechanism is compression rather than containment, and it is the one the objective actually paid for: when a belief program and a non-belief program are rewritten through the *same* invented token, that token is shared reuse. Here there is exactly one such token, `fn_9`, the seek fragment shared between belief and the obstacle family, alongside the two round-1 tokens that reassembled `fork` and `sync_to_world` out of the overlay and registration families. The atomic control has only the seek fragment, since there is no pair interface left for it to rebuild. So the sharing is real but narrow, and that is the correct result: what the non-mental families donate is the *parts*, and what is belief-specific is the *wiring*. An abstraction that were shared wholesale with a non-mental family would not be an agent constructor.

Two caveats. This is scaffolding in the sense of compositional dependence recovered from a converged run, not a longitudinal claim about a learner acquiring one rung and then the next; the ordering evidence for that is @sec-order. And the curriculum is ours: the obstacle and relocation families are deliberately over-weighted  precisely so that the wall-and-navigate policy would recur often enough to be abstracted. That is a modelling choice about what the learner sees, not a thumb on the scale of what it concludes, but it is a choice, and @sec-limits returns to it.

== The behavioural probe <sec-probe>

Everything so far is description length. @criteria's third condition was behavioural: on a scene held out of the run entirely, does the learned term predict that the agent goes where it *believes* the goal to be?

The probe uses the goal-displacement family, because it is the one where the two readings come apart in behaviour. In a wall-belief scene the transient-wall rival walks the agent along the same path, so only description length separates the readings there --- that was @sec-rivals. In a goal-displacement scene the agent walks to where it believes the goal is, one cell from the true goal, which never moves, so a program that does not represent the belief cannot land on the right cell.

#task-figure("belief_goal")

The scenes are drawn from a seed that never entered the run and are checked to be absent from the training corpus. The library is rebuilt by re-stitching the run's own found programs, so the terms priced are the terms the run converged to. We then instantiate the discovered belief compound on the novel scene and, as its foil, the shortest program the same library can express without it.

#figure(
  image("behavioral_probe_p2.png", width: 100%),
  caption: [The false-belief test on a held-out Sally-Anne scene (`p2-nodream`). The learned belief compound sends the agent to the believed cell; the shortest non-mental program under the same library sends it to the true goal.],
) <fig-behavioral-probe>

Through the library, the belief compound is written

```lisp
(fn_0 1 2 (step 2 up) (register (locate 1) (place 1)))
```

which expands to

```lisp
(pipe_gpg (compose_gp dup (mapsnd (compose (step 2 up)
                                           (optimize (neg_dist 2) 1))))
          (register (locate 1) (place 1)))
```

--- the discovered constructor applied to a derive that privately displaces the goal, committing only the agent's own move. Its final frame puts the agent on the believed cell, one step short of the true goal. The learner restricted to the non-mental fragment writes the shortest thing that fragment can say, plain desire `(optimize (neg_dist 2) 1)`, and walks straight to where the goal really is.

Across all 24 held-out scenes the belief compound lands on the believed cell every time. The non-mental program lands on the true goal cell in 18 of 24 and, in the remaining six, one step short of it on the true-goal side --- it is walking toward the real goal and runs out of frames. It never once lands on the believed cell. The same probe run against the atomic control's library gives the same 24 of 24, as it must: the two endowments differ in how the compound is *spelled*, not in what it computes.

The nats are the point of the exercise. Here the non-mental program is *shorter* --- 7.9 against 20.1 here, and 7.9 against 13.4 under the atomic control --- which is the opposite of @sec-rivals, and that is exactly why both measurements are needed. Where a non-mental rival reproduces the scene, it does so only at greater length, so description length rules it out. Where a non-mental rival is cheaper, it is cheaper only because it predicts the wrong behaviour, so the data rule it out. There is no program in the non-mental fragment that is both as short as the belief compound and as accurate. A learner that has compressed its experience into that compound passes the false-belief task; a learner confined to the fragment fails it, and fails it in the specific way the developmental literature documents --- by answering with the true location @wimmer_beliefsabout_nodate.

== The verdict <sec-verdict>

@criteria fixed in advance what would count as success and what would refute the hypothesis. @tab-verdict is the tally.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([Criterion (@criteria)], [Combinator], [Atomic]),
    [An abstraction enters the library that opens a private channel, derives a divergent state, evaluates a policy against it, and commits only the action, with the agent value shared between policy and commit],
    [Met, round 2 (`fn_7`, one shared hole used three times)],
    [Met, round 1 (`fn_6`, one shared hole used twice)],

    [Selected by the joint objective over the non-mental rivals the same library expresses],
    [Met: $+6.27$ nats over the nearest rival under the library, from $+0.12$ under base primitives],
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

Two things did not come out clean, and they belong here rather than in a footnote.

*Both runs print `ok: false`,* and for two different reasons, neither of which is a row of @tab-verdict. I report them because a run that prints `ok: false` should not be quoted as if it printed `ok: true`.

The combinator cell fails `no_belief_complement`. That check was written before the commit-form classifier learned to tell a scope commit that *denotes* the single-value one from a genuinely non-mental commit, so it fires on all 95 of the `sync_all` and `sync_except` spellings discussed in @sec-selective. Under the later classifier those 95 are verified by substitution and re-rendering, the census splits into 37 literal and 95 degenerate, and the count of non-mental commits is zero. The failing boolean is the old check, not a finding.

The atomic cell fails `obstacle_belief_misrepresents_obstacle`, which demands that *every* solved task in the two families whose agent is wrong about an obstacle --- plain wall-belief and false-obstacle --- have a private model in which the obstacle is misplaced. Plain wall-belief passes 24 of 24, so the failures are in the false-obstacle family, and the diagnosis recorded for that run is that one or two of its tasks were solved by a program representing the agent as wrong about the *goal* rather than about the wall --- a nested-fork displaced-goal model, which reproduces the scenes and is a belief attribution, but not the one the family was built to force. On that diagnosis this too is a mis-specified check rather than a non-mental rival: the criterion asks about the wall because that is what the family names, but the family's scenes admit a second mental explanation the check does not recognise.

*Coverage is incomplete, and unevenly so.* The combinator cell misses 21 of 48 goal-displacement tasks and 15 of 24 false-obstacle tasks. The case that these are budget misses rather than expressive ones is the atomic control: its solve set strictly *contains* the combinator cell's --- 23 belief tasks solved there and not here, none the other way --- in a vocabulary that expresses nothing this one cannot, and the combinator solve-time distribution (@fig-solve-dynamics) has a large mass pressed against the 3600 s ceiling. That is an argument from a shorter spelling of the same programs, not from a longer run, and a longer run is what would settle it; this thesis does not have one. The `belief_dual` family, removed from the corpus for the reasons @criteria gives, is the version of this that could not be papered over at all.

Against that, what did come out clean is the load-bearing part. In both cells the compression step invented a term with the shared-hole signature; in both the belief families used it and no non-mental family did; in both the cube census found belief on the one asymmetric corner with every complement claimed by somebody else; and in both, under the later classifier, not one belief solution committed a value that no model had sought.

== Objections <sec-objections>

=== "`fork` and `sync_to_world` are just `believe` cut in half"

The sharpest objection is that the atomic primitive set of the control run already contains the answer. By the definition this thesis took from the literature, mentalizing is attributing to an agent an intensional state --- a representation of the world that may diverge from it. `fork` builds a divergent representation and `sync_to_world` reads an agent's position off it. Putting them together is not discovery; it is reassembly of something that was handed over in two pieces.

Four things answer this, in increasing order of force.

First, the parts have non-mental uses and take them. `fork` is used by the overlay, underlay and comet families; `sync_to_world` by image registration. These are not token gestures: they are what those families' *ground-truth* programs are made of, and the census in @sec-selectivity shows the searcher reaching for them there. A primitive that only ever appeared inside belief compounds would be a `believe` in disguise; one that image registration uses is a general-purpose tool.

Second, the field is symmetric and the complements are live. @tab-cube is the measurement: the opposite channel direction, the two wider commit scopes, the reversed z-order, the other projection and the reversed utility polarity are all inhabited by other families in the same run, at the same prior cost, and belief takes none of them. It is one thing to observe that a learner did not go the other way; it is another to observe that other tasks in the same run *did*.

Third, the composition is where the content is, and the composition is what was searched for. The pair is symmetric and content-free --- `dup` makes $(w, w)$, and nothing in it says which copy is a model or who holds it. Three independent decisions make one copy a belief: transform the second and not the first, commit from model to world and not the reverse, and publish exactly the value whose policy was run. Each has an available opposite, each opposite is exercised elsewhere, and all three are made in program space.

Fourth, and most directly, the objection is to a variant the thesis does not rest on. Remove the two primitives, as the combinator library of combinators does, and the structure comes back. The searcher is handed `dup`, `mapsnd`, `compose_gp`, `pipe_gpg`, `locate`, `place` and `register` --- product-category combinators nobody would call mental --- along with the wrong-channel complements `swap`, `mapfst`, `bimap` and `pair_blank`. What happens is @sec-combinator-run: the non-mental families reassemble `fork` and `sync_to_world` in round 1 with no belief task solved, and the agency signature is built on top of them in round 2 with the agent value now shared three ways instead of two. If `fork` were `believe` in disguise, that run would have had to discover `believe` from scratch, and it did.

=== "But `fork` still smuggles in a representational capacity"

A weaker version of the objection concedes all of that and locates the smuggling one level down: the ability to make a copy of the world that differs from the world just *is* the representational capacity that theory of mind consists in.

The reply is that it is not, and the distinction matters. Building a world-state that differs from the actual one is counterfactual reasoning --- the capacity to consider what would be the case if things were otherwise --- and it is thoroughly domain-general, with a developmental record of its own well before false-belief attribution. Theory of mind is not that capacity; it is an abstraction that *uses* it, by attributing the counterfactual world to a particular agent, such that the divergent world is *that agent's* world. That attribution is precisely the conjunct that the shared hole encodes, and it is precisely the conjunct that nothing in `fork` supplies: `fork` will happily derive a private grid, compute over it, and publish an unrelated value, and the language will let you write that. What it will not do is tell you not to.

=== "The regress does not end"

Nor do we claim it does. One can always ask for `dup` to be decomposed in turn --- into allocation, reads and writes over a flat memory, so that "hold a second world model" becomes a discovered pattern of buffer use. That relocates the regress rather than ending it. "Allocate a buffer" is then the primitive, and it is neither more nor less mental than "form a pair", because the capacity for plurality is presupposed by intensionality at every level. Intensionality just *is* the coexistence of a world and a model of it, so any substrate expressive enough to state "act on the world via a transformed model of it" must be able to hold two things at once. The pair is to theory of mind roughly what the integers are to arithmetic: not a disguised form of the thing to be learned, but the medium without which it cannot be stated. There is no substrate in which theory of mind is discovered from nothing. There are only substrates whose primitives are, or are not, individually non-mental and general, and the argument of this section is that these ones are.

=== "The corpus was built to produce this result"

The last objection is about the curriculum rather than the language, and it has the most substance. The obstacle family is deliberately the densest corner of the corpus, at six tasks per (goal, agent) combination, because the wall-and-navigate policy has to recur often enough for compression to abstract it; belief's derive is short only because that abstraction exists. Change the mix and the result moves.

Three things are worth saying. The first is that this is a claim about what the learner is *shown*, not about what it concludes, and it is exactly the claim @constructivism makes on the theory theory's behalf: a concept becomes available when earlier learning has made it cheap, so the composition of earlier experience is a determinant of what is learnable and when. A model in which the curriculum did not matter would be a model in which the ordering result of @sec-order was vacuous. The second is that the corpus is not stratified: it is one undifferentiated bag with no mental / non-mental label anywhere in the search or the compression step , and the joint stitch of @compression pools both root types, so the budget that paid for belief's structure was free to be spent on overlay, registration, obstacle or relocation instead. The third is that the dose-response experiment that would quantify this --- vary the non-mental fraction, measure whether the constructor is still invented --- is not in this thesis. It is the single most informative experiment left undone, and @sec-limits says so.

== What the results do not show <sec-limits>

Seven limits, in roughly decreasing order of how much they cost the argument.

*The belief has no history.* What the learner finds is a divergent world model, not a belief that is formed and could be revised. Nothing in the discovered term represents how the agent came to be wrong: no perceptual access, no record of what was witnessed, no update when the agent looks again. This is a limit of the corpus and of a single-frame transition function, and it is the largest respect in which the discovered structure is thinner than the thing it is a model of. Chapter 3's BToM has an analogous gap --- the space of admissible divergences is stipulated rather than derived --- but BToM's belief is at least a state that could in principle be revised, and this one is not. @sec-persistent takes it up.

*The curriculum mix is untested.* As @sec-objections concedes, the corpus-mix dose-response has not been run. Nor have the two others that would sharpen the result: a silo-versus-joint stitch, which would show whether the constructor survives being compressed against belief alone, and a seed-robustness sweep reporting every verdict metric as $k$ out of five corpus seeds. Each of the two cells reported here is a single seed.

*Omitting the recognition model was decided on the evidence, not in advance.* @no-amortization gives the reason for leaving DreamCoder's third phase out, and the reason is a good one, but it is not the order in which the work happened: the loop was run both ways first, and the numbers are what settled it. Under the atomic control the dreamed proposal was a small win --- 291 tasks and 159 belief tasks against 287 and 155 --- concentrated in the two hardest families. Under the combinator library it was a substantial loss: 240 against 264, with the damage concentrated in the witness family, which fell from 48 of 48 to 26 of 48. The mechanism is worth recording because it is not the obvious one. The dreamed prior was gated to families with at least one solved instance, so it cannot be what delayed the first belief solve, and indeed both combinator cells made theirs in round 2. What differed was the *library*: the two cells solved slightly different sets of tasks in round 1, the joint stitch therefore saw a different corpus, and the abstractions it returned differed accordingly --- the dreamed cell converging on a library through which the witness compound is longer. The deficit was transmitted by the compression step rather than by the proposal distribution, which is to say that the two interact: a proposal that changes *which* tasks are solved changes what the next library can be. That interaction is a real property of the architecture, and characterising it is left undone here. What can be said is that the constructor was invented in the same round, with the same shape, in all four cells, so the structural result does not turn on the choice either way.

*Coverage is not complete, and the misses are not proven to be budget misses.* @sec-verdict gives the numbers. The claim that a longer run would close them is supported by the comparison between the two endowments and by the shape of @fig-solve-dynamics, and it is not supported by a run. The experiment that would settle it is a budget dose-response --- the same corpus at 600, 1200 and 2400 s per task, with the solve rate per belief family plotted against the budget --- which would separate a family that is out of *reach* from one that is merely out of *time*.

*The one-model frame is fixed by the type system, not discovered.* Both endowments give the learner the product $G times G$, which says that a learner holding a private representation holds exactly one --- and a one-model frame is precisely the kind of thing a theory of mind is supposed to have. Nothing reported here shows that compression would *select* a single private channel if more were expressible, because in these runs no more are. The experiment that would settle it replaces the fixed pair with a recursive grid-stack, so that channel arity becomes a free parameter, and measures which arity each family selects; it is not in this thesis, and @sec-arity sets it out as further work.

*The planner is a single token.* `optimize` packs a full breadth-first search into one symbol. @rationality defends granting a planner, on grounds the developmental record supports, but it does not defend granting it in this form, and a more elementary treatment would decompose it into memory, conditionals and interaction (@sec-optimize).

*None of this is a claim about children.* The model is a sufficiency result at the computational level: it shows that a learner with domain-general representational resources and a preference for short programs *can* arrive at belief-structured abstractions, which blocks the inference from "no mechanism has been named" to "no mechanism exists". Whether human learners use this mechanism, or another one that solves the same problem, is not settled by anything in this chapter.

#load-bib(read("refs.bib"))
