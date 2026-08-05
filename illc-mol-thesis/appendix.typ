#import "@preview/illc-mol-thesis:0.2.0": *

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

#show table.cell.where(y: 0): set text(style: "normal", weight: "bold")
#show table: set par(justify: false)
#set table(stroke: (_, y) => if y == 1 { (top: 0.9pt) } else if y > 1 { (top: 0.2pt) })

#set heading(numbering: "1.")
#mol-chapter("Appendix")

This appendix collects the implementation of the model outlined in Chapter 3.

- full list of tasks
- full list of primitives

== Full Corpus

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([Family], [Ground-truth program], [What it probes]),
    [`physics`], [`step v d`], [the bare interpreter, no private model],
    [`desire`], [`optimize (neg_dist gv) av`], [goal-directed motion without a model],
    [`obstacle`], [`compose (wall_at r c) (optimize …)`], [a detour round a #emph[real] wall],
    [`flee`], [`optimize (distance hv) av`], [the repulsion corner of the utility axis],
    [`deletion` / `denoise`], [`clear_at r c` / `erase v`], [the grid-edit removal corners],
    [`relocation`], [`compose (clear_at …) (wall_at …)`], [the grid-edit "move" corner],
    [`overlay` / `underlay`], [`fork (step v d) overlay`], [`fork` with a #emph[non-mental] commit; z-order],
    [`comet`], [`fork (optimize …) overlay`], [`fork`'s derive slot taking a seek policy],
    [`wipe`], [`pipe_gpg pair_blank snd`], [the pairing complement: a fresh blank channel, not a copy],
    [`belief_wall` (false-wall)], [`fork (compose (wall_at r c) (optimize …)) (sync_to_world av)`], [detour round an #emph[invisible] wall],
    [`belief_witness` (witness)], [the above, composed with a second bare seeker], [a witness who crosses the phantom wall],
    [`belief_goal` (goal-displacement)], [`fork (seq (step gv d) (optimize …)) (sync_to_world av)`], [Sally-Anne: belief whose #emph[content] varies],
    [`belief_observers` (two-observer)], [a bare seeker composed with a forked one], [selective attribution: one agent forks, one does not],
    [`belief_false_obstacle` (false-obstacle)], [`fork (seq (wall_at Wb) (step gv d) (optimize …)) (sync_to_world av)`], [wrong about obstacle #emph[and] goal, with a real wall present],
    [`registration` / `perception`], [`sync_to_world v` / `sync_to_model v`], [the sync commits doing #emph[non-mental] alignment],
    [`multi_registration` / `registration_except`], [`sync_all` / `sync_except v`], [the scope complements],
    [`inpainting` / `readout`], [`underlay` / `snd`], [the z-order and projection complements],
    [`composite` (layer-compositing)], [`compose_pg (bimap (step u d) (erase w)) overlay`], [the full bifunctor: two layers, one map each],
    [`drift_reg` (drifting registration)], [`compose_pg (mapfst (step u d)) (register …)`], [the world-channel map: register while the image scrolls],
    [`map_update`], [`compose_pg swap sync_all`], [the twist: the model adopts the world wholesale],
  ),
  caption: [The task families, by the program behind them. The first block are trajectory tasks, solved by transition functions; the last are template tasks, solved by commits (roots). Programs are written throughout in the abbreviated spelling of @signature: `fork` and `sync_to_world` are not primitives of the learner's library but compounds it must assemble . The last three families exist only under the combinator library: nothing in the atomic control maps both channels, maps the world channel of a #emph[given] pair, or commits wholesale into the model, so they have no program there at any length. `wipe` has one in both, at three nodes against eight.],
) <tab-families>

== The BToM correspondence <app-btom-map>

@sec-btom-map reads the invented term against the stipulations of @stipulated. The same correspondence, item by item.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([BToM (Chapter 3)], [In the discovered term], [Status here]),
    [A planner mapping (utility, environment model) to action],
    [$pi_a^g$ (that is, `optimize (neg_dist g) a`), applied to whatever grid it is handed],
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
    [The shared hole --- planner, derive and commit boundary bound to one value: the agent in the atomic constructor, the believed content in the combinator one],
    [Discovered: the conjunct the compression step kept, from either side of the boundary],

    [The nested evaluation: the observer runs the planner inside itself],
    [`fork`'s private grid lives for one call and is discarded, never rendered],
    [Structural, in program space; the interpreter has no such mode (@app-interpreter)],

    [Inverse planning: infer $(g, m)$ from the trajectory by Bayes],
    [Cheapest-first enumeration under the library prior, indicator likelihood],
    [The inference of @sec-bayes, unchanged and not agent-directed],

    [Belief as a state that is #emph[formed] and could be #emph[revised]],
    [Nothing: the model exists for one transition and has no history],
    [Absent --- the honest gap (@sec-limits)],
  ),
  caption: [What BToM stipulates, and where it turns up in the term the compression step invented.],
) <tab-btom-map>

== The libraries, verbatim <app-lib-p1>

@tab-lib-p2 gives the combinator run's final library as arrows, under names of our own. This is the same library as the compression step emitted it: the run's names, the types, the argument orders, and the bodies expanded to base primitives, for a reader who wants to count symbols rather than read denotations --- the two views differ in exactly the way @sec-selective turns on. The arguments a term takes are written `$0`, `$1`, … in the body, and are supplied in that order, which is the searcher's and carries no meaning; @tab-lib-p2's letters are assigned by role instead.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([Symbol], [Type, in argument order], [Body]),
    [`fn_6`], [$V times V times (G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$], [`(pipe_gpg (compose_gp dup (mapsnd (compose $2 (optimize (neg_dist $1) $0)))) $3)`],
    [`fn_7`], [$V times D times V times V arrow.r (G arrow.r G)$], [`(compose (optimize (neg_dist $0) $3) (pipe_gpg (compose_gp dup (mapsnd (compose (step $0 $1) (optimize (neg_dist $0) $2)))) (sync_except $0)))`],
    [`fn_8`], [$V times V times (G arrow.r G) arrow.r (G arrow.r G)$], [`(compose $2 (optimize (neg_dist $1) $0))`],
    [`fn_9`], [$C times C times V times V arrow.r (G arrow.r G)$], [`(pipe_gpg (compose_gp dup (mapsnd (compose (wall_at $1 $0) (optimize (neg_dist $2) $3)))) sync_all)`],
    [`fn_10`], [$V times D times V arrow.r (G arrow.r G)$], [`(pipe_gpg (compose_gp dup (mapsnd (compose (step $0 $1) (optimize (neg_dist $0) $2)))) (sync_except $0))`],
    [`fn_11`], [$V arrow.r (G times G arrow.r G)$], [`(register (locate $0) (place $0))`],
  ),
  caption: [The final library under the combinator endowment (run `p2-nodream`), verbatim. Read as arrows in @tab-lib-p2.],
) <tab-lib-p2-raw>

@sec-found also summarizes the control run; this is the library it converged to. The five constructors differ in what the private derive does. In the notation of @signature they read

$
"fn"_(6)(a,g,delta) &= "fork"(pi_a^g compose delta, space "sync"_a) \
"fn"_(7)(a,g,delta) &= pi_a^g compose delta \
"fn"_(8)(d,b,a,n) &= "fork"(pi_a^b compose "step"_(b,d), space "sync"_a) compose pi_n^b \
"fn"_(9)(u,v) &= "fork"(pi_v^u compose "fork"(pi_u^0 compose pi_u^v, space "sync"_u), space "sync"_v) \
"fn"_(10)(c,g,a,g',n) &= pi_n^(g') compose "fork"(pi_a^g compose "wall"_(c_2,c), space "sync"_a) \
"fn"_(11)(c,r,g,a) &= "fork"(pi_a^g compose "wall"_(r,c), space "sync"_a) \
$

so the derive is an arbitrary hole $delta$ in `fn_6`, a shove of the goal in `fn_8`, a phantom wall at a free coordinate in `fn_11` --- the most-used, carrying 52 of the 155 belief solutions --- and at a fixed column in `fn_10`. `fn_9` is worth a remark: its derive contains another fork, so it is an agent whose private model is itself derived by forking, and fourteen solutions are rewritten through it. That is not second-order mentalizing in the sense of @timeline, since the inner fork is part of the outer agent's model rather than a model of another mind, but it does establish that the discovered shape embeds, one of the properties @explanandum asked for.

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
  caption: [The final library under the atomic control (run `p1-nodream`), bodies expanded to base primitives; the readings are above. Five of the six carry the agency signature; `fn_7` is the bare seek policy $pi_a^g$ with a derive prefix, which belief shares with the obstacle family (47 obstacle solves are rewritten through it).],
) <tab-lib-p1>

== The per-family description-length census <app-dl-census>

@sec-compression reports what the learned library buys each family. The full census, from which those figures are taken.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header([Family], [Solves priced], [Base (nats)], [Final library], [Saved]),
    [belief],       [25], [34.51], [17.40], [*17.11*],
    [obstacle],     [48], [15.97], [12.79], [3.18],
    [registration], [4],  [6.80],  [4.50],  [2.30],
    [perception],   [4],  [10.39], [8.08],  [2.30],
    [drift_reg],    [4],  [16.56], [14.26], [2.30],
    [desire],       [16], [7.78],  [7.78],  [0.00],
    [comet],        [4],  [16.05], [16.05], [0.00],
    [relocate],     [16], [13.89], [13.89], [0.00],
    [flee],         [4],  [7.78],  [7.78],  [0.00],
    [deletion],     [4],  [5.70],  [5.70],  [0.00],
    [denoise],      [4],  [4.79],  [4.79],  [0.00],
    [wipe / map_update], [4 each], [5.78], [5.78], [0.00],
    [multi_reg / reg_except / inpaint / readout], [4 each], [2.20--4.50], [unchanged], [0.00],
  ),
  caption: [Median description length per task (`p2-nodream`), before and after the learned library. Belief saves 17.11 nats per task and 412 nats over the priced solves; obstacle saves through the shared seek `fn_8`, and the three registration-flavoured families each save exactly 2.30 nats --- the price of the reassembled $"sync"_v$ token.],
) <tab-dl-census>

== The interpreter <app-interpreter>


The first is compositionality. "Iteratively apply $f$, starting from $x_0$, for $n$ time steps" is structure common to every solution in the corpus. Were it part of the program, it would be baked into every abstraction the learner discovers: a corpus of ten falling-object scenes would yield not the reusable $lambda$`0. (step down #0)` but an overspecified _apply `(step down #0)` from $x_0$ for `#1` steps_, whose output type is an entire rendered scene. Such an abstraction cannot be combined with another, so it could never serve as a component of the deeper compounds that belief requires, and library learning would be defeated at the first round. Keeping the recurrence out of program space keeps every abstraction an arrow $G arrow.r G$ and therefore composable.

The second is that the recurrence carries no information the learner could be credited with discovering. Since it is shared by every solution, what search is actually looking for is only $f$; the rest is what is needed to render a scene from $f$ in order to check it against the observation. So we move the rendering machinery out of program space and into the evaluation framework. The interpreter is given in full in @app-interpreter.


The interpreter is discussed at the model level in @app-interpreter. What follows is the design argument in the form it was originally worked out, followed by the two interpreters themselves.

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

`unfold` is the anamorphism that at each timestep applies $f$ to the previous grid to generate the new grid, and appends the new grid to the list of grids. So `unfold`'s state is just a single grid: it doesn't encode a world/model grid pair, nor a registry of agents and goals. A transition function $f$ is a `grid` endomorphism `fn :: grid -> grid`.

For some of the tasks, we use an interpreter for programs of root `fn_p_g :: (grid, grid) -> grid` since those tasks don't involve just one input grid but a pair. That uses `unfold_with_template(g, template, T, c)`

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

== The primitive catalogue <app-primitives>

The primitive sets handed to the searcher are summarised at the model level in @primitives. The exhaustive catalogue — every symbol's repr, type signature and semantics, under each endowment — belongs here.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([Symbol], [Type], [Semantics]),
    [`compose`], [$(G arrow.r G)^2 arrow.r (G arrow.r G)$], [run one transition then the other],
    [`step`], [$V times D arrow.r (G arrow.r G)$], [move every cell of the given value one step in the direction],
    [`optimize`], [$U times V arrow.r (G arrow.r G)$], [move the given value one greedy (BFS-optimal) step to improve the utility],
    [`neg_dist`], [$V arrow.r U$], [minus the distance to the nearest cell of the target value (attract)],
    [`distance`], [$V arrow.r U$], [plus that distance, so maximising it flees the target],
    [`wall_at`], [$C times C arrow.r (G arrow.r G)$], [stamp a wall at a position],
    [`clear_at`], [$C times C arrow.r (G arrow.r G)$], [clear one cell],
    [`erase`], [$V arrow.r (G arrow.r G)$], [remove every cell of a value],
    [`right left up down`], [$D$], [the four movement directions],
    [`c0 … c4`], [$C$], [coordinate terminals, a pair naming a position (the invisible latent pool)],
    [`0 … 9`], [$V$], [cell-value terminals (content-priced when visible)],
  ),
  caption: [The grid core: the primitives that move cell values about a single grid, shared by both endowments.],
) <tab-core>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([Symbol], [Type], [Semantics]),
    [`dup`], [$G arrow.r G times G$], [$Delta$: pair a grid with a copy of itself],
    [`mapsnd`], [$(G arrow.r G) arrow.r (G times G arrow.r G times G)$], [$"id" times f$: act on the second channel],
    [`compose_gp`], [$(G arrow.r G times G) times (G times G arrow.r G times G) arrow.r (G arrow.r G times G)$], [compose a pair producer with a pair endomorphism],
    [`pipe_gpg`], [$(G arrow.r G times G) times (G times G arrow.r G) arrow.r (G arrow.r G)$], [produce a pair, then consume it],
    [`locate`], [$V arrow.r (G arrow.r C times C)$], [$"loc"_v$: where $v$ is in a grid],
    [`place`], [$V arrow.r (G times (C times C) arrow.r G)$], [$"put"_v$: move $v$ to a position],
    [`register`], [$(G arrow.r C times C) times (G times (C times C) arrow.r G) arrow.r (G times G arrow.r G)$], [$(w, m) arrow.r.bar "plc"(w, "loc"(m))$: read a position off one channel, impose it on the other],
    [`overlay`], [$G times G arrow.r G$], [union the channels, model wins ties --- a #emph[non-mental] commit],
    [`underlay`], [$G times G arrow.r G$], [union the channels, world wins ties],
    [`sync_all`], [$G times G arrow.r G$], [move #emph[every] shared value to its model position],
    [`sync_except`], [$V arrow.r (G times G arrow.r G)$], [every shared value but one],
    [`fst` / `snd`], [$G times G arrow.r G$], [project the world channel / the model channel],
    [`swap`], [$G times G arrow.r G times G$], [the twist: exchange the channels],
    [`mapfst`], [$(G arrow.r G) arrow.r (G times G arrow.r G times G)$], [$f times "id"$: act on the #emph[first] channel],
    [`bimap`], [$(G arrow.r G)^2 arrow.r (G times G arrow.r G times G)$], [$f times g$: act on both],
    [`pair_blank`], [$G arrow.r G times G$], [pair a grid with a fresh blank rather than a copy],
    [`compose_pg`], [$(G times G arrow.r G times G) times (G times G arrow.r G) arrow.r (G times G arrow.r G)$], [transform a pair, then consume it --- the third composition, and the only one that acts on a pair the program did not build],
  ),
  caption: [The product combinators: the plumbing, the commits that consume a pair, and the symmetric complements. The lower rows are the complements discussed in @primitives.],
) <tab-combinators>

#block(inset: (left: 1em))[
  *TODO:* the atomic variant's pair interface, transcribed from `prims.py`.
]

== Program representation <app-delta>

The hypothesis space is described at the model level in !types. This is the data structure that represents it.

Every program, primitive, and invented abstraction is a `Delta`, i.e. an object that can serve as a node in an expression tree. A `Delta` carries a `head` (the Python value or callable it denotes), a `type` (the type its subtree returns), a list of `tailtypes` (the types of the arguments it expects), and a list of `tails` (the actual argument subtrees, once they are filled in). So a `Delta` with no `tailtypes` is a _terminal_, a leaf whose value is just its head, for example `up :: dir` or the literal `4 :: cellvalue`. A `Delta` with `tailtypes` is an _operator_: an interior node like `step :: cellvalue, dir -> fn`, whose head is applied to its filled-in tails. Then a node's `arrow` is the signature that the enumerator matches against when it fills argument slots. For a terminal, its arrow is just its `type`. For an operator, it's the pair `(tailtypes, type)`, the arrow from its input types to its output types.

The expression tree is directly executable. Calling a `Delta` evaluates the program it represents, so a terminal returns its head while an operator recursively evaluates each of its tails and applies its head to the results. So the program `(step 4 up)` is a `step`-headed node with tails `4` and `up`; calling it evaluates the two leaves to the cell value `4` and the direction `(-1,0)`, then applies `step` to obtain a transition function `fn :: grid -> grid`.

An invented abstraction is represented as a `Delta` as well, but it's distinguished by a `hiddentail` which is the body of the abstraction. The `hiddentail` is itself a `Delta` tree over lower-level primitives, with numbered argument holes `$0, $1, ...` standing in for the abstraction's parameters. When such a node is called, its body is copied and each hole `$i` is replaced by the corresponding tail before the body is evaluated. This is what makes an abstraction cost a single token. The whole subtree it abbreviates is hidden inside the `hiddentail` and does not appear in the program that uses it.

The description length of a program is its node count, computed by `length`. a leaf counts one, and an operator counts one plus the lengths of its tails. So an invented abstraction used as a token adds just one, since its body lives in the un-charged `hiddentail`. This node count is the minimum-description-length quantity the enumerator uses to tiebreak between programs that solve a task within the same budget window (see !enumeration).

=== The library object

The library of primitives is a `Deltas` object. It is instantiated with a list of `core` primitives and grows a parallel list of `invented` abstractions as compression discovers them. When the enumerator draws primitives, it draws from the concatenation of `core` and `invented` which is the full library. Whenever a primitive is added, `Deltas.infer` rebuilds the indices the search depends on — most importantly `bytype`, which groups the alphabet's symbols by their return type. This grouping is what the type-uniform prior is computed from: a symbol of type $tau$ is priced at $-ln$ of the number of symbols sharing type $tau$, so `bytype` gives both the branching factor at each argument slot and the cost of each choice (!enumeration).

== Task generators and necessity filters <app-generators>

The identifiability condition on the mental families is stated in identifiability beyond the $k$ scenes, two conditions on the scene itself and a set of exemptions that no scene can settle. This section states them and says how the generators enforce them. Both conditions are sharpest in the false-wall family , where the agent detours around a wall that is not there.

*The falsely believed wall must matter.* Placed nowhere near the agent's route, it leaves the agent navigating to its goal exactly as it would have without any false belief; positing the belief is then neither necessary nor helpful, and the scene shows nothing. So the phantom wall lies on the shortest path the agent would take were its belief true, and costs it extra steps over that path. A merely monotone "detour" is just another shortest path, which plain desire reproduces under different tie-breaking. Scenes failing this are rejected by the generator.

*A world-level stamp must be detectable.* This one is structural. On a bare grid nothing distinguishes a wall the agent _believes_ in from a real wall stamped into the world and taken back down before the frame is rendered: a rival that stamps, acts, then erases reproduces every scene honestly, at every $k$. The scenes must therefore make the stamp visible, and they do so by leaving an inert object sitting on the phantom-wall cell for the whole trajectory --- the cell the agent bends around in  Stamping a wall there overwrites that object, and nothing in the language can put it back, so a world-level stamp shows up in the rendered frame. The witness family makes the same argument with a moving object in place of a static one: an agent who walks across the phantom cell would be clobbered by any real wall stamped there.

*Some rivals are identities, and are exempted by argument.* A last few rivals cannot be excluded by the scenes at all, because what they are is an _identity_ rather than a coincidence, and no number of scenes separates two programs that denote the same function on the scenes in question. On a two-value scene, "move every shared value except the goal to its model position" simply _is_ "move the agent to its model position". These are exempted from the rival census of @sec-rivals by argument, and the full set is:

== Cost-bounded enumeration <app-enumeration>

The search is described at the model level in !enumeration. These are its mechanics.

While a single cost-ordered stream can in principle be checked against every task in the corpus at once, in the experiments each still-unsolved task is instead enumerated as its own cost-ordered stream, parallelized across tasks. The prior $Q$ each stream runs under is the same one: the type-uniform prior over the current library, with the content-literal boost of !enumeration. Nothing in the loop conditions $Q$ on the individual scene (!no-amortization).

Program search proceeds by budget window to ensure that cheaper programs are enumerated first. For `(step 4 up)`, the first two windows are skipped by pruning. We start enumerating in window $[0,2)$, but since `step`'s cost is 2.1 nats it exceeds the budget, so it cannot be chosen as the root node of the program tree and we proceed to budget window $[2,4)$. After paying 2.1 nats for `step`, only 1.9 nats remain to fill the two holes in the partial program tree. `step`'s tail types (arguments) are `cellvalue` and `step`. But any primitive of type `cellvalue` costs 2.3 nats, so `step`'s first argument can't be placed. So trees with a `step` node are pruned, since they would never complete within the second budget window. Search is cheapest-first, since more complex programs only become affordable once the window is wide enough.

At budget window $[4,6)$ the ceiling is finally sufficiently wide to admit all three symbols. The tree is assembled incrementally: each enumerated subtree is appended to a copy of its parent node, and a callback fills the remaining argument slots by the same recursion (so programs of any arity or nesting depth are handled by the same procedure). Since enumeration is type-directed, each argument slot draws candidates only from symbols of the matching type, so no ill-typed partial tree is ever constructed. `(step 4 up)` completes at its total cost of $approx 5.8$ nats.

When `(step 4 up)` is found the tree rooted by `step` has no more available holes, so the callback fires to evaluate the found program $rho$. For each task $x$ in the corpus, the interpreter generates the output matrix given the input grid of the task by calling `unfold` on $rho$, initial grid $x_0$, and $x$'s third dimension $n$. Then the resulting output matrix $x'$ is hashed and compared against task $x$'s hash. So for a program to solve a task, it must produce a grid that exactly matches that task's entire trajectory, not just the endpoint of the trajectory. This ensures that programs capture the full behavior rather than just the end result.

If `(step 4 up)` is the first solution found for the task, it's stored as the task's solution. Because the windows are tried in increasing order of cost, the first solution the search returns is in principle the shortest program that solves the task. But enumeration is parallelized so search unfolds simultaneously over program trees, so we need to tiebreak when multiple programs solve a task within the same budget window. To tiebreak we first run `Delta.simplify` on the programs, a semantics-preserving rewrite that collapses spurious nesting which sometimes occurs during enumeration. We then run `Delta.length` to get the node count of the expression trees we need to compare. Every operator and leaf adds to the count, and an invented abstraction used as a token adds one to the count too. The body of the abstraction isn't charged, since the components now count for just application. The program with the lower node count wins since it has the lower minimum description length (since by symbol count it's shorter).

Also, the search space contains lots of junk programs that crash or return nothing even though they type-check. So the enumerator callback wraps both the interpreter and the evaluator in `try/except` blocks and skips any degenerate or non-callable program.

== Compression with Stitch <app-stitch>

The compression objective and the tradeoff it makes are given in !compression. This is the search that optimizes it.

In DreamCoder @ellis_dreamcoder_2020 the compression step is bottom-up: for each found program it enumerates the exponential set of program refactorings (stored as version spaces) and intersects them to grow abstractions up from reoccurring subexpressions. Rather than this bottom-up approach, we delegate the search for abstractions to the top-down Stitch @bowers_top-down_2023 library-learning system. Instead of enumerating refactorings of every program, stitch searches the space of abstractions directly. So stitch doesn't have to materialize every rewrite.

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

=== Normalization before compression

Before passing the found solutions to stitch for abstraction, we first need to fully expand any existing primitives in those solutions. For this we implement `normalize`, the complementary operation to abstraction. It fully inlines every invented `fn_k` back to the bare primitive alphabet, recursively substituting each abstraction's argument holes with its actual arguments until nothing but core primitives remain.

We need to normalize because stitch's top-down compression can only abstract over structure it can see, and an opaque `fn_k` token has no visible interior. So any abstraction that would need to reach inside `fn_k`, or span its boundary and surrounding context, is simply invisble to the search.

the compression step must be handed programs in a common vocabulary — feeding it already-compressed programs, in which a prior round's fn_0 appears as an opaque token, would let those tokens collide with the fresh abstractions the compressor invents under the same names. Normalizing before compression guarantees the compressor sees only primitives and the structure it discovers is genuinely its own. Renaming it to `inv_0` avoids the lexical clash but leaves it just as opaque; stitch still can't see the `(compose (wall_at …) (optimize …))` sitting inside it. Inlining flattens the token back to that primitive structure, and only then can the compressor find abstractions that cut across the old boundaries.

The best syntactic fragment to abstract from round $N$'s solution may carve the primitives differently than round $N+1$'s. For example,

So we can refactor abstractions that were previously ideal but are no longer ideal

=== No recognition model <no-amortization>

DreamCoder's third phase trains a neural recognition model that reads a task and emits a task-conditional prior $Q(dot|x)$, so that enumeration proceeds under a $Q$ biased toward the primitives. We don't implement it. 

The reason is a difference of scale between DreamCoder's setting and this one. Amortization earns its place when the branching factor is large and the loop is long: DreamCoder works with libraries of hundreds of symbols over dozens of iterations, and there the cost of one forward pass per task is repaid many times over by the cost bands it lets the enumerator skip. Ours is a small language run for a few rounds. The learner starts with 37 primitives under the atomic control and 45 under the combinator library, and ends with 43 and 51 once compression has added its six; and the abstraction the whole argument turns on enters at the first compression step under the one endowment and the second under the other, with the loop converging a round or two later (@sec-order). There is no long tail of iterations across which a learned proposal could amortize anything. What such a proposal would buy is reaching sooner what is already reachable, and reachability, not speed, is the quantity the argument of @sec-search depends on.

Omitting the phase also removes it as a place where the result could be smuggled in. A recognition model trained on solved tasks is a component whose prior over the library is fit to data, and belief is precisely the structure the corpus is meant to leave unfitted. With no such component, no distribution anywhere in the loop is conditioned on what a scene looks like, and the audit of @no-btom has one fewer item to defend.

This is a departure from DreamCoder, and it is one we arrived at by measurement rather than by design: the loop was run both ways before it was settled, and @sec-limits reports what each way gave.

The second is to abandon enumeration in prior order for enumeration under a learned proposal distribution, trained to guess which primitives a task of this appearance is likely to need. This is amortized inference. The system this thesis builds uses only the first, for reasons given in @no-amortization; it is the first that does the work the argument needs, since it is the one that changes what is _reachable_ rather than merely what is reached sooner --- an intervention on the problem rather than on the procedure for solving it. 