#import "lib.typ": *
#import "viz.typ": all-tasks, legend

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
#mol-chapter("Appendix", lbl: <ch-appendix>)

Reference material for Chapters 4 and 5: the corpus, the primitives handed to the learner, the libraries it converged to, and the per-family cost census.

== The task corpus <app-corpus>

The 168 belief tasks are 24 false-wall, 48 goal-displacement, 48 witness, 24 two-observer and 24 false-obstacle; the remaining 148 are spread across the twenty non-mental families. Every task consists of $k = 4$ scenes on $5 times 5$ grids, and the (goal, agent) values rotate through eight distinct pairs, so that no particular integer comes to mean "agent". The non-mental side is deliberately not uniform: obstacle is the densest family at 48 tasks and relocation carries 16, because the wall-handling structure they share with belief must recur often enough for compression to abstract it --- a deliberate curriculum choice.

One task of each family, all four of its scenes drawn side by side, each as the trajectory collapsed onto its opening grid: one numbered arrow per transition, cells reached only later ghosted in and deletions marked. The scenes of a task share one latent program, and a solution counts only if it reproduces every one of them.

#align(center, legend())
#v(6pt)

#all-tasks(mode: "path", show-steps: true, scenes: auto, spacing: 11pt)

== The primitive catalogue <app-primitives>

The primitive sets handed to the learner, summarised in @primitives: every symbol's repr, type signature and semantics, under each endowment.

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
  caption: [The grid-edit primitives that move cell values about a single grid],
) <tab-core>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([Symbol], [Type], [Semantics]),
    [`dup`], [$G arrow.r G times G$], [pair a grid with a copy of itself],
    [`mapsnd`], [$(G arrow.r G) arrow.r (G times G arrow.r G times G)$], [$(w,m) arrow.r.bar (w, f(m))$: act on the second channel, leave the first],
    [`compose_gp`], [$(G arrow.r G times G) times (G times G arrow.r G times G) arrow.r (G arrow.r G times G)$], [build a pair, then transform it],
    [`pipe_gpg`], [$(G arrow.r G times G) times (G times G arrow.r G) arrow.r (G arrow.r G)$], [produce a pair, then consume it],
    [`locate`], [$V arrow.r (G arrow.r C times C)$], [where a named value is in a grid],
    [`place`], [$V arrow.r (G times (C times C) arrow.r G)$], [move a named value to a position],
    [`register`], [$(G arrow.r C times C) times (G times (C times C) arrow.r G) arrow.r (G times G arrow.r G)$], [$(w, m) arrow.r.bar "plc"(w, "loc"(m))$: read a position off one channel, impose it on the other],
    [`overlay`], [$G times G arrow.r G$], [union the channels, the second wins ties --- a #emph[non-mental] commit],
    [`underlay`], [$G times G arrow.r G$], [union the channels, the first wins ties],
    [`sync_all`], [$G times G arrow.r G$], [move #emph[every] shared value to its position in the second channel],
    [`sync_except`], [$V arrow.r (G times G arrow.r G)$], [every shared value but one],
    [`fst` / `snd`], [$G times G arrow.r G$], [project the first channel / the second],
    [`swap`], [$G times G arrow.r G times G$], [exchange the channels],
    [`mapfst`], [$(G arrow.r G) arrow.r (G times G arrow.r G times G)$], [$(w,m) arrow.r.bar (f(w), m)$: act on the #emph[first] channel],
    [`bimap`], [$(G arrow.r G)^2 arrow.r (G times G arrow.r G times G)$], [$(w,m) arrow.r.bar (f(w), g(m))$: act on both, one function each],
    [`pair_blank`], [$G arrow.r G times G$], [pair a grid with a fresh blank rather than a copy],
    [`compose_pg`], [$(G times G arrow.r G times G) times (G times G arrow.r G) arrow.r (G times G arrow.r G)$], [transform a pair, then consume it --- the third composition, and the only one that acts on a pair the program did not build],
  ),
  caption: [The pair combinators],
) <tab-combinators>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([Symbol], [Type], [Semantics]),
    [`fork`], [$(G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$], [the derive-and-commit frame of @signature, bought whole: copy the grid, run the derive on the copy, hand the (world, copy) pair to the commit],
    [`sync_to_world`], [$V arrow.r (G times G arrow.r G)$], [the single-value commit bought whole: move $v$ in the world to its position in the model, return the world],
    [`sync_to_model`], [$V arrow.r (G times G arrow.r G)$], [the direction complement, also atomic here: move $v$ in the model to its position in the world, return the model --- perception rather than action],
    [`then_sync`], [$(G times G arrow.r G) times V arrow.r (G times G arrow.r G)$], [run the given commit, then `sync_to_world` $v$ on its result. Absent from the combinator endowment, where it would smuggle an atomic `sync_to_world` back in --- with it, registration solves as `(then_sync fst_gg v)` and never reaches for the register/locate/place spelling],
    [`sync_all` / `sync_except`], [$G times G arrow.r G$ #h(0.4em)/#h(0.4em) $V arrow.r (G times G arrow.r G)$], [the scope collapses, atomic in both endowments (@tab-combinators)],
    [`overlay` / `underlay`], [$G times G arrow.r G$], [the z-order unions, atomic in both endowments],
    [`fst_gg` / `snd_gg`], [$G times G arrow.r G$], [the projections --- the `fst`/`snd` of @primitives, under the run's reprs --- atomic in both endowments],
  ),
  caption: [The atomic control's pair interface, which hands the learner pre-composed higher-level primitives],
) <tab-atomic>

== The final libraries <app-lib-p1>

The final library of the combinator run. The run's names, the types, the argument orders, and the bodies are expanded to base primitives. The arguments an abstraction takes are written `$0`, `$1`, … in the body and are supplied in that order, which is the run's own and carries no meaning.

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
  caption: [The final library],
) <tab-lib-p2-raw>

The library shown in @tab-lib-p2-raw lists the abstractions found in round 3 on, since no new abstractions are found in round 4; the library is just reused to solve new tasks.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, left, left, left),
    table.header([Round], [Symbol], [Type, in argument order], [Body]),
    [1], [`fn_0`], [$V times V times C times C arrow.r (G arrow.r G)$], [`(compose (wall_at $3 $2) (optimize (neg_dist $1) $0))`],
    [1], [`fn_1`], [$V arrow.r (G times G arrow.r G)$], [`(register (locate $0) (place $0))`],
    [1], [`fn_2`], [$(G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$], [`(pipe_gpg (compose_gp dup (mapsnd $0)) $1)`],
    [1], [`fn_3`], [$C times C times C times C arrow.r (G arrow.r G)$], [`(compose (wall_at $3 $2) (clear_at $1 $0))`],
    [1], [`fn_4`], [$V times V arrow.r (G arrow.r G)$], [`(optimize (neg_dist $0) $1)`],
    [1], [`fn_5`], [$V times V times C arrow.r (G arrow.r G)$], [`(compose (wall_at $2 c2) (optimize (neg_dist $0) $1))`],
    [2], [`fn_6`], [$V times V times (G arrow.r G) times (G times G arrow.r G) arrow.r (G arrow.r G)$], [`(pipe_gpg (compose_gp dup (mapsnd (compose $2 (optimize (neg_dist $1) $0)))) $3)`],
    [2], [`fn_7`], [$V times V times C times C arrow.r (G arrow.r G)$], [`(compose (wall_at $3 $2) (optimize (neg_dist $1) $0))`],
    [2], [`fn_8`], [$V times V times C times V times V arrow.r (G arrow.r G)$], [`(compose (pipe_gpg (compose_gp dup (mapsnd (compose (wall_at $2 c2) (optimize (neg_dist $3) $4)))) sync_all) (optimize (neg_dist $1) $0))`],
    [2], [`fn_9`], [$V times D times V arrow.r (G arrow.r G)$], [`(pipe_gpg (compose_gp dup (mapsnd (compose (step $0 $1) (optimize (neg_dist $0) $2)))) (sync_except $0))`],
    [2], [`fn_10`], [$V arrow.r (G times G arrow.r G)$], [`(register (locate $0) (place $0))`],
    [2], [`fn_11`], [$C times C times V times V arrow.r (G arrow.r G)$], [`(pipe_gpg (compose_gp dup (mapsnd (compose (wall_at $1 $0) (optimize (neg_dist $2) $3)))) sync_all)`],
  ),
  caption: [Rounds 1 and 2, with the bodies expanded to base primitives.],
) <tab-lib-rounds-raw>

@tab-lib-rounds-raw shows the abstractions found at the end of round 1 and 2. Round 1's `fn_2` is the derive-and-commit frame and its `fn_1` the single-value commit, both discovered before any belief task is solved; round 2's `fn_6`, `fn_8`, `fn_9` and `fn_11` are the first four abstractions that open a private copy, and `fn_8` is the two-mover abstraction round 3 discards. Note the literal `c2` where the final library's two-actor constructor carries a hole. The compression step re-derives the library from scratch each round, so symbols are not stable across rounds; `fn_7` below and `fn_7` above are different abstractions.

@sec-found also summarizes the control run. Its five constructors differ in what is done to the private copy. Here is what each does at a time step, with its arguments as free letters, and the bodies they are read from below.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    table.header([Symbol], [What it does at each time step]),
    [`fn_6`$(a,g,delta)$], [on a private copy of the grid altered in some unspecified way $delta$, $a$ moves to the adjacent cell closest to $g$; then $a$'s new position, and nothing else, is written back to the world],
    [`fn_7`$(a,g,delta)$], [the alteration $delta$ is made to the grid itself, and then $a$ moves to the adjacent cell closest to $g$ --- no copy is opened],
    [`fn_8`$(d,b,a,n)$], [$n$ moves to the adjacent cell closest to $b$; then, on a private copy in which $b$ has been displaced one cell in direction $d$, $a$ moves to the adjacent cell closest to $b$; then $a$'s position alone is written back],
    [`fn_9`$(u,v)$], [on a private copy: $u$ moves to the adjacent cell closest to $v$ and then to the adjacent cell closest to the nearest 0, and $u$'s position is written into that copy; still in the copy, $v$ then moves to the adjacent cell closest to $u$; finally $v$'s position alone is written back to the world],
    [`fn_10`$(c,g,a,g',n)$], [on a private copy with an impassable wall stamped in row 2 at column $c$, $a$ moves to the adjacent cell closest to $g$, and $a$'s position alone is written back; then, in the world, $n$ moves to the adjacent cell closest to $g'$],
    [`fn_11`$(c,r,g,a)$], [on a private copy with an impassable wall stamped at row $r$, column $c$, $a$ moves to the adjacent cell closest to $g$; then $a$'s position alone is written back],
  ),
  caption: [The final abstractions from the atomic control, read as prose.],
) <tab-lib-p1-read>

One remark: what `fn_9` does to its private copy itself opens a second private copy, so the discovered shape embeds --- though the inner copy is part of the outer agent's model rather than a model of another mind, so this is not second-order mentalizing in the sense of @timeline.

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
  caption: [The final library under the atomic control, bodies expanded to base primitives],
) <tab-lib-p1>

Five of the six carry the agency signature; `fn_7` is the bare seek policy (move $a$ toward $g$) with room for one edit in front of it, which belief shares with the obstacle family (47 obstacle solves are rewritten through it).

== The per-family description-length census <app-dl-census>

The census below prices every family's solutions before and after the library was learned.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header([Family], [Solves priced], [Base (nats)], [Final library], [Saved]),
    [belief],       [132], [34.51], [10.78], [*17.11*],
    [obstacle],     [48], [15.97], [12.79], [3.18],
    [registration], [4],  [6.80],  [4.50],  [2.30],
    [perception],   [4],  [10.39], [8.08],  [2.30],
    [drift_reg],    [4],  [16.56], [14.26], [2.30],
    [goal-directed movement], [16], [7.78], [7.78], [0.00],
    [constant movement], [4], [6.17], [6.17], [0.00],
    [comet],        [4],  [16.05], [16.05], [0.00],
    [overlay / underlay], [4 each], [14.44], [14.44], [0.00],
    [composite],    [4],  [16.74], [16.74], [0.00],
    [relocate],     [16], [13.89], [13.89], [0.00],
    [flee],         [4],  [7.78],  [7.78],  [0.00],
    [deletion],     [4],  [5.70],  [5.70],  [0.00],
    [denoise],      [4],  [4.79],  [4.79],  [0.00],
    [wipe / map_update], [4 each], [5.78], [5.78], [0.00],
    [multi_reg / reg_except / inpaint / readout], [4 each], [2.20--4.50], [unchanged], [0.00],
  ),
  caption: [Median description length per task, before and after the learned library, over all 280 solved tasks.],
) <tab-dl-census>

The census is priced under the type-uniform prior alone, that is, with @enumeration's visible-value exception removed. Repricing the same programs with the exception restored at half and at full strength (the prior enumeration actually searches under) leaves belief's median saving at 17.11 nats in all three cases, while the three registration-flavoured savings fall from 2.30 to 1.15 to exactly 0.00: their whole economy is naming a visible value once instead of twice, and once the discount makes visible values free there is nothing left to save. Under the full discount, belief and obstacle are the only families the library shortens at all. Within belief, the wall and witness variants are invariant to the last digit; the goal-displacement, two-observer and false-obstacle variants each give back 2.30 nats --- the uniform cost of one cell-value terminal --- per visible value their abstraction absorbs (two, three and one respectively), and still save between 9.25 and 19.59 nats with the discount at full strength.

== The interpreters <app-interpreter>

Iterating a transition function from an initial frame is structure common to every solution, so it is kept out of program space and put in the evaluation framework instead (@space). Were it in program space it would be baked into every abstraction the learner discovers --- a corpus of falling-object scenes would yield not the reusable `(step down $0)` but an overspecified "apply `(step down $0)` from $x_0$ for `$1` steps", whose output type is an entire rendered scene and which therefore composes with nothing. Keeping the recurrence out of program space keeps every abstraction an arrow $G arrow.r G$.

A trajectory task is checked with the standard unfold: its state is a single grid, not a (world, model) pair, and not a registry of agents or goals.

```python
def unfold(g: grid, T: int, f: fn) -> mat:
    frames = [g.copy()]
    for _ in range(T - 1):
        g = f(g)
        frames.append(g.copy())
    return np.stack(frames)
```

A template task starts from a pair, so it is checked with a second interpreter, which threads the canvas and re-pairs each frame with the same constant external template.

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