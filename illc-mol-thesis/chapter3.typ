#import "@preview/illc-mol-thesis:0.2.0": *

#import "world_tape.typ": world-tape, grid-view, arc-colors

#import "viz.typ": task-figure, all-tasks

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

#mol-chapter("Implementation")

== Overview

Modularity-nativists argue for the existence of a Theory of Mind Module (ToMM), a natively-endowed domain-specific system for understanding the behavior of intentional agents. In Alan Leslie's characterization the ToMM introduces attitude concepts like _Believes_ and uses a "proprietary representational system" for ascribing attitudes to agents @leslie_pretending_1994. Our goal is to demonstrate that the ability to theorize in mental terms can in principle be _learned_, without the need for a native ToMM. 

In our first trial we show that, given a non-mental representational substrate, a system can learn a functional abstraction that encodes a notion of belief attribution. The learned function `fork(derive, commit)`


We initialize the system with a library of domain-general primitives, none of which posit an agent's intentional state, nor encodes attitude concepts like _believes_. Lacking bespoke ToM capacities, such a system would on Leslie's view not be able to reason about intentional agents, to attribute a belief to an agent, to explain an agent's movement as navigation according to a private intensional representation. We show that the system learns _useful abstractions_ of domain-general functions, abstractions that encode attitude concepts like _Believes_ and attribute an intensional state to an agent.

In the second iteration, we show that the implementation can arrive at an abstracted notion of belief starting from an even more basic language of 2-ary combinators. 

== ECD architecture 

ECD is a wake-sleep library-learning loop inspired by the DreamCoder @ellis_dreamcoder_2020. Each iteration of the loop consists of an enumeration phase, a compression phase, and a dreaming phase. In enumeration, the system generates candidate programs to solve tasks

=== Program representation

Every program, primitive, and invented abstraction is an instance of the `Delta` class, i.e. it is an object that can be a node in an expression tree. A `Delta` is initialized with a `head` 

The library of primitives is represented by an instance of the `Deltas` class. It's instantiated with `core` primitives (the initial primitives) and is enriched by `invented` abstractions 

=== Enumeration

Enumeration produces candidate programs cheapest-first under a cost model $Q$. A program's cost is the summed $-log p$ of its nodes. $Q$ is either the type-conditioned uniform prior, the content-aware prior, or the dreamed recognition model. For example, a program like `(step 4 right)` vs. `(fork (compose (wall_at c2 c3) (optimize(neg_dist 2) 1)) (sync_to_world 1))`

best-first enumeration by description length: `cenumerate` walks the search in expanding $-log p$ bands $("LOGPGAP"*"idx", "LOGPGAP"*("idx"+1))$




== boneyard


theory of mind involves attributing a belief/desire to an agent. a belief is a representation of the world that is separate from it. to posit a mind to explain false belief is to hypothesize: the agent has its own private representation of the world, which differs from the world in some way, and the agent is acting according to that believed world

signature of belief: derive a counterfactually modified world (`wall_at`), run a policy on it (`optimize`), commit just one entity's position to the real world (`sync av`)

goal is to abstract `fn_agent` which takes an agent value, goal value, and wall coordinates (propositional content of belief) and associates them. 

components of intensionality:
1. a representation of the world that's separate from it
2. reasoning over the representation
3. acting according to the results



#terminal("INVENTED ABSTRACTIONS:
fn_0  [<class 'int'>, <class 'int'>, <class 'int'>, <class 'int'>] -> fn
      body: (fork (compose (wall_at $3 $2) (optimize (neg_dist $1) $0)) (sync_to_world $0))")

== Type system

Bare-string, monomorphic, first-order, exact-match. A type is either a ground string (`grid`, `int`, `dir`, or `coord`) or an arrow name (`fn`, `pair_gg`, `fn_p_g`, `fn_g_p`, `fn_p_p`, `fn_g_c`, `fn_gc_g`). Delta.arrow = (tuple(tailtypes), type) (dsl.py:716), and enumeration resolves a hole by possibles = D.bytype[tailtype] — a dict keyed on string identity (ecd.py:132). Invented abstractions get types by reading concrete hole types straight off the body via typize (ecd.py:506). There is no type variable and no unification anywhere.

 

== Primitives

`sync_to_world(w)` bakes in three choices
1. read coordinate from model channel
2. find one particular value, transfer that
3. transfer it to the world, discard the model

DEFENDING the `optimize` primitive:

"9- to 12-month-old infants have principled expectations about the way agents approach their goals that are consistent with the assumption that agents choose rational means to approach their goals (Gergely, Nadasdy, Csibra, & Bird, 1995)." Sodian

- compose
- step
- optimize
- wall_at
- neg_distance
- direction terminals right/left/up/down
- coord terminals c0..c{SIZE-1}
- cellvalue terminals 0..9.

== Tasks

the curriculum scaffold:
- overlay → the fork skeleton (fork `$f` overlay) (fork + a commit)
- registration → sync_to_world as a commit
- obstacle → policy = (compose (wall_at) (seek))
- desire → seek

A scene requires belief attribution iff the world and the agent's model must hold contradictory contents at the same frame, and something in the world is sensitive to the true content at that cell/time. unfold runs one fixed per-frame fn, so any "stamp a real wall / move the real goal / erase" rival corrupts that witness. 


WITNESS false belief tasks, to avoid extensional explanations

=== false belief

Goal-displacement false belief (canonical Sally-Anne)
- goal object visibly relocates in-world mid-episode (from location A to B) but the agent still heads to A. - solved by a program that posits that, in the agent's model, the goal is still at A. And then the derive runs on that model, so the agent navigates optimally wrt the goal being at A rather than B. 
- it's different than the phantom-wall false belief: false belief about an object's location vs. about an obstacle
- so fork generalizes about what the model is wrong about

two agents, contradictory beliefs
- two agents detour around different phantom walls
- solved by `compose(fork(...), fork(...))`
- 



Physical: rising

#figure(task-figure("physics", caption: none))

True belief: agent 1 has goal 2, navigates optimally towards it.

#figure(task-figure("desire", caption: none))

False belief: agent 1 has goal 2, but believes there's an impermeable wall at the second row and third column from the top left. 

#figure(task-figure("belief", caption: none))

The above $t_0$ is the actual initial grid from the task. The below $t_0$ is the initial grid according to which the agent navigates, according to which its navigation is optimal

== Interpreter

Say we have a candidate program $p$. The enumerator needs to check (for each task in the corpus) whether $p$ solves task `x`.  i.e. whether running $p$ produces a 

```python
def unfold(g, T, f):
    "grid, int, fn -> mat: iterate f:grid->grid from g, rendering each frame."
    frames = [g.copy()]
    for _ in range(T - 1):
        g = f(g)
        frames.append(g.copy())
    return np.stack(frames)
```

The interpreter just threads a single grid, so the intensionality of an additional grid representation isn't baked into it, rather it needs to be discovered. So only tasks that require an intensional representation, that require a theory of mind, will make use of it. It's not enforced by the interpreter. 

So every program is `fn :: grid -> grid`. Say `f` is a candidate program, `x` is a task with `g = x[0]` that consists of `T` 2d grids. Running `unfold(g, T, f)` yields a `mat` (3d matrix, sequence of 2d grids). If `unfold(g, T, f) == x`, then `f` is a solution to `x`. 

goal is to have an interpreter that's as minimal as possible

for some of the tasks, we use an interpreter for programs of root `fn_p_g :: (grid, grid) -> grid` since those tasks don't involve just one input grid but a pair. 

== `fork` and `sync`

Goal: show that "mentalizing utilizes domain-general cognitive equipment"

At this point the goal is just to initialize the system with a bunch of primitives that operate on one or two grids. then whether 

goal is to show that `fn_agent` is synthesized from a base of non-mental primitives

child scientist/theory theory: "mentalizing utilizes domain-general cognitive equipment" 

modularity nativist: there's an innate Theory of Mind Module (ToMM) 

"ToMM constructs agent-centered descriptions of situations or “metarepresentations “. Agent-centered descriptions place agents in relation to information. By relating behavior to the attitudes agents take to the truth of propositions, ToMM makes possible a commonsense causal interpretation of agents’ behavior as the result of circumstances that are imaginary rather than physical." @leslie_pretending_1994

Leslie argues that what the ToMM provides is a structure for "relating behavior to the attitudes agents take to the truth of propositions"

I show that this structure which relates agents, behaviors, attitudes, and propositions can be learned from a domain-general cognitive equipment. 

Leslie asks: "how is the preschool child able to learn about mental states when these are unobservable, theoretical constructs?" and "how is the young brain able to attend to mental states when they can be neither seen, heard, nor felt?"

solving these tasks requires computing an internal representation, a metarepresentation (Leslie)

Leslie assumption: "native to our mental architecture is a domain-specific processing stream adapted for understanding the behavior of agents. A major component of this system is a mechanism which computes the M-representation."

Leslie claims
- ToMM uses _proprietary_ representational system
- ToMM introduces basic _attitude_ concepts

Leslie's informational relations

"My assumption is that there is a small set of primitive informational relations available early on, among them BELIEVE and PRETEND. These notions are primitive in the sense that they cannot be analyzed into more basic components such that the original notion is eliminated. "

"decoupling introduces extra structure"

`derive` builds a private grid from a copy of the world (e.g. stamp a phantom
wall and run the policy on it); `commit` reconciles the (world, derived) pair back to a single grid (e.g. move the agent to the position it reached in the derived grid).  The second grid lives only for the duration of this call.

The second grid is introduced locally in program space by the `fork` combinator, which applies `derive` to `w` and then commits particular values to the original `w`. 

`fork` applies a derived transform to a copy and 

The abstracted agent function is `fork(policy-on-modified-copy, sync_to_world av)`

The goal of this first pass is to show that, 

The pair is constructed inside `fork`

```python
def fork(derive, commit):
    def _f(w):
        return commit((w.copy(), derive(w.copy())))
    return _f
```

The *S* combinator knows nothing of walls, policies, or agents

Charge: `fork` and `sync` are just a `believe` primitive split in two gears that only ever re-mesh into `believe`. Answered by:
- overlay tasks `(fork (step v d) overlay)` is fork without sync
- registration tasks `(sync_to_world v)` over `(working, template)` is sync without fork
so you can't fuse `fork` and `sync` without losing the ability to solve those other tasks, so they're genuinely useful on their own

belief is the triad:
1. derive a counterfactually-modified world (`wall_at` stamps an obstacle that isn't there)
2. run a policy on it (`optimize`)
3. commit only one entity's resulting position (`sync_to_world av`, with `av` shared between 2 and 3)

the intensionality of `belief` is in the `fork`-closure, not `sync`


content of belief is what you do to the private model before the policy runs. in `fork(compose(wall_at(r,c), optimize(neg_dist(gv), av)),  sync_to_world(av))` we have 

derive: `compose(wall_at(r,c), optimize(neg_dist(gv), av))` which builds the private model and runs the policy

commit: `sync_to_world(av)`

content is an extensional world fact `wall_at(r,c)`

goal is to extract the co-occurrence of `av` across the action policy `optimize(neg_dist(gv), av))` and the sync policy `sync_to_world(av)`.

goal is to abstract `fn_agent($av, $gv, $r, $c)`

DSL:

- 

Result: solves false belief tasks (example)

#terminal("[ 10077] caught (fork (compose (wall_at 3 2) (optimize (neg_dist 2) 4)) (sync_to_world 4))")

Abstracted primitive:

#terminal("fn_0  [<class 'int'>, <class 'int'>, <class 'int'>, <class 'int'>] -> fn
      body: (fork (compose (wall_at $3 $2) (optimize (neg_dist $1) $0)) (sync_to_world $0))
")

Base form

```lisp
(fork
  (compose
    (wall_at $3 $2)
    (optimize (neg_dist $1) $0))
  (sync_to_world $0)
)```

With agent value `$0`, goal value `$1`, and percieved wall coordinates `($3, $2)`

`$0` appears twice, in `optimize` (the actor) and `sync_to_world` (the commiter)

the private model is born from `dup` in `fork` and collapsed by `sync`

=== primitives where `fork` and `sync` are ATOMIC

- `fork`
- direction of information flow:
  - `sync_to_world` (read from model, then write to world): takes action based on model
  - `sync_to_model` (read from world, then write to model): percieves/records based on world 
- scope of information flow:
  - `sync_to_world` (read one value `v`)
  - `sync_all` (read every value, copy entire grid)
  - `sync_except` (multi-object registration)
- which channel's information gets precedence if they disagree when one is copied onto the other:
  - `overlay` (model wins ties)
  - `underlay` (world wins ties)
- utility:
  - `neg_dist` (attract)
  - `distance` (repel)
- grid editing:
  - `wall_at`
  - `clear_at`
  - `erase`

This is a general DSL of operations. If minimum description length still picks exactly the `read-model`, `write-world`, `single-av`, `attract`, `add-wall` sequence while the other tasks use other primitives, then the agency signature is genuinely found as a composition of primitives, it's not just forced by hardcoding which primitives are available

=== library where `fork` and `sync` are DECOMPOSED



- `fork` decomposed as `commit∘mapsnd(derive)∘dup`
- `sync` decomposed as `register(locate, place)`

- which channel to render:
  - `fst_gg` (render the world)
  - `snd_gg` (render the model)
- symmetry witness
  - `swap`
  - `via_swap`

the searcher exploited mapsnd's functoriality, finding (compose_gp (compose_gp dup (mapsnd (wall_at ..))) (mapsnd (optimize ..))) — i.e. mapsnd(g∘f) = mapsnd(g)∘mapsnd(f) — rather than mapsnd(compose(wall_at, optimize)). Equivalent, same node count, shared-av intact.

Why specifically four

There are two objects in this little category, `grid` and `pair`, so there are four arrows between them: `grid` $arrow$ `grid` (`fn`), `grid` $arrow$ `pair` (`fn_g_p`), `pair` $arrow$ `grid` (`fn_p_g`), and `pair` $arrow$ `pair` (`fn_p_p`). We use all four because we want `dup`, `mapsnd`, and the two composers to be separate, discoverable nodes.

In a monomorphic DSL, the number of types is dual to how finely you decompose. Every type you delete = one combinator you must pre-fuse = one node you wanted discoverable that you've now re-hidden.


Principled, but a bigger change — parametric polymorphism. All four function types are instances of one polymorphic arrow a → b, and compose/compose_gp/pipe_gpg are literally the same function compose : (a→b)→(b→c)→(a→c). The proliferation is purely an artifact of having no type variables. With Hindley–Milner-style types + unification (what DreamCoder actually does) you'd keep one →, one compose, and just dup : grid→pair, mapsnd : (grid→grid)→(pair→pair), sync : int→(pair→grid). The four strings collapse into the unifier's job. Cost: the enumerator is built on exact string-keyed bytype buckets; you'd have to replace those with unification at each step. That's the honest "right" simplification, but it's an enumerator rewrite, not a DSL tweak.

why use types grid, pair_gg, fn_g_p, and fn_p_g when we could use grid, (grid, grid), grid -> (grid grid), and (grid, grid) -> grid?

Because in this enumerator a type is nothing but an opaque hash key. The only thing the searcher ever does with a type is D.bytype[tailtype] — bucket primitives by exact equality of the type value (see infer, makepaths, groom, and isequal's n1.type == n2.type). Nothing ever parses or decomposes a type string.

So the direct answer: 'fn_g_p' and the literal string 'grid -> (grid, grid)' are behaviorally identical. You could globally swap in the structural strings today and the run would be byte-for-byte the same — they're just longer, equally-opaque dict keys. The short names are an ergonomic, honest label for an atom; the structural notation would be cosmetic at best and misleading at worst, because it advertises structure the machinery doesn't honor (a reader would expect the checker to know (grid,grid)->grid is a function returning a grid, and it doesn't — it's one indivisible token).

Three things make this precise:

1. There already is structure, exactly one level deep — at the primitive signature. Each Delta carries arrow = (tuple(tailtypes), type). So sync_to_world really is typed (int,) -> fn_p_g structurally. What's atomic is the slot types — the return type and each argument type. fn_g_p/fn_p_g/fn_p_p are precisely those slot atoms.

2. They're atoms because the arguments are higher-order. fork takes commit :: fn_p_g — the argument is itself a function value. For the flat bytype bucket to enumerate "things that produce a pair→grid function," that function type has to be a single token. fn_p_g is the reified name of the arrow (grid,grid)->grid so a first-class function can be a bucket key. Writing it as the structured (grid,grid)->grid only buys you something if argument slots are matched structurally rather than by equality — i.e., if the enumerator unifies type terms instead of hashing them.

3. Structural notation is load-bearing only with unification. This is the same fork in the road as last time: ground (variable-free) structured types matched by structural equality are identical in power to atomic strings — you gain nothing until you add type variables. compose : (a->b) -> (b->c) -> (a->c) is the only thing that needs to look inside -> (to unify b), and that's what would collapse your four arrow-atoms + three composers into one -> and one compose. Without polymorphism, grid -> (grid, grid) is just a verbose alias for fn_g_p.

So the choice isn't "named atoms vs. structural types" — it's "atoms-as-keys (cheap dict lookup, current) vs. structural-types-with-unification (parses ->/(,), enables one polymorphic compose)." The structural spelling without the unifying machinery is pure notation. Given you're at two objects and four arrows, the atoms are the right call; the structural representation earns its keep only once you add a HM-style checker, which is worth it only if the object/arrow count starts exploding.

=== tasks

the variety of tasks serves two purposes
1. tasks that show that all the DSL primitives are useful
2. tasks that show that the primitives used to solve belief tasks are also meaningfully used to solve other tasks in other combinations


tasks with `fn`-root, aka `grid -> grid`
- `flee`: flee the nearest hazard, uses `optimize` and `pos_dist`
- `deletion`: punch one hole in a solid object, uses `clear_at`
- `denoise`: drop a particular value from the entire grid, uses `erase`
- `movement`: move a value at constant rate in some direction, uses `step`
- `desire`: true belief tasks, uses `optimize` and `neg_dist`
- `belief`: 
- `witness_belief`
- `overlay`

tasks with `fn_p_g`-root, aka `(grid,grid) -> grid`
- `registration`


in `witness-belief` tasks, the private-copy `fork` is the unique explanation. This isn't a problem with the minimal DSL, but once we expose the full symmetric cube DSL then there's an easier, non-mental explanation `compose (compose (wall_at r c) optimize... ) (clear_at r c)`. Stamp a real wall, optimize around it to the goal, then remove that wall. There's no belief here, no use of the private channel, no representation. This isn't belief because the propositional content `wall_at r c` isn't attributed to a particular agent, it's just an extensional fact. This is actually the same setup as my earlier DSL from February which had `mask(unfold(wall_at optimize))`, and the problem with that was similarly that the masking wasn't associated with the agent that was navigating, it was just an extensional fact. So as long as the DSL contains the `clear_at` ability to remove values, the extensional explanation will be cheaper than the intensional explanation. So we could just remove `clear_at`, but then we're limiting the expressivity and biasing the project towards discovering the theory of mind abstraction.

So instead we need to create niche tasks that cannot be explained with these simpler extensional explanations. So in the witness tasks, a second agent (navigating to a different goal) crosses the "wall" without a problem. So the task can't just be solved by placing a real wall there, running navigation, and then removing it, since then both agents would be affected. So the only explanation requires using the private channel to account for the navigating agent's behavior, that it has a belief that there's a wall there to avoid, since we `fork` it so the wall lives in the detouring agent's private model, not in the shared world. 

To be clear, I'm not trying to make a Piaget claim that this learning occurs by first combining lower-level physical theory successes into knowledge units that are used to build at the higher level. the first enumeration cycle here is enough, theories that solve the false belief tasks are constructed from even fairly simple primitives.

I include these extraneous tasks and primitives just to show that even in a broad program space it's still possible. And that these aren't just hand picked primitives for ToM put together. and that solutions for false belief live deeper in program space. 



#pagebreak()

== decomposing `fork` and `sync`

```lisp
(pipe_gpg
  (compose_gp
    (compose_gp
      dup
      (mapsnd (wall_at $3 $2))
    )
    (mapsnd
    (optimize (neg_dist $1) $0))) (sync_to_world $0))
```

Standard form of $Phi$ combinator: `P f g h x = f (g x) (h x)`. A binary $f$ applied to the results of two arbitrary unary branches $g$ and $h$, both fed the same $x$. 

`fork` is `fork(derive, commit) w = commit(w, derive(w))`

Mapping it onto $Phi$, it's a special case. Rather than general $lc g,h rc$ it's $lc id, "derive" rc$. And rather than the general $f$, `fork` applies the pair constructor `(,)` to get the 2-tuple. And then `fork` post-composes with `commit`, the product eliminator `(pair -> grid)`, which isn't part of $Phi$. So `fork` is the product pairing with identity in the first slot, post-composed with the eliminator. `fork(derive, commit)` is $"commit" compose lc id, "derive" rc$

So the full `fork` is composing *B* and $Phi$, where *B* is function composition `f g x = f (g x)`

Since the first branch is pinned to `ID`




. It consists of a `derive` component and a `commit` component, as 

`fork(derive, commit)(w) = commit((w, derive(w)))`

applied to `w` is a grid state, `x[0]`, the initial grid of a task

It duplicates an input, runs parallel computations, then merges them. 

It's the $Phi$ combinator $lambda a b c d . a(b d)(c d)$ of type $(b -> c -> d) -> (a -> b) -> (a -> c) -> a -> d$

It applies a binary function to the results of two unary functions that share one argument

`P f g h x = f(g x)(h x)`

Input $x$ is duplicated, $g$ and $h$ are applied to it at each branch, and the outputs are merged by $f$. 

It's a generalization of the *S* (substitution) combinator (Haskell's applicative "`<*>`"), which takes three arguments, applies the first (`f`) to the third (`x`), and then applies that to the result of the second (`h`) to the third (`x`)

`S f h x = f x (h x)`

The *S* combinator is just the $Phi$ combinator with `g` as the identity function, so `f(g x)` is just `f x`. 

The *B* combinator is `B f g x = f (g x)`, just function composition: Haskell's "`.`"

So `fork(derive, commit)(w) = commit((w, derive(w)))` is $Phi$ with identity on one of the grids. Compare to `P f g h x = f(g x)(h x)`, where `commit` is `f`, `id` is `g`, `derive` is `h`. What `fork` does is: 

#theorem[The $Phi$ combinator decomposes to *SKI*]

#proof[
  We start with `P f g h x = f(g x)(h x)`, which is equivalent to `(B f g)x (h x)` by composition. This is equivalent to `S (B f g) h x` by substitution, so we have `P f g h x = S (B f g) h x`. By $eta$-reduction we have `P f g = S (B f g)`,
]

`fork` is a generic combinator that, given `w`, applies a transformation to it and then reconciles the `w'` with the original `w` based on the `commit` instructions. It locally introduces another grid and then collapses it into the original grid. 

`fn, fn_p_g -> fn: w |-> commit((w, derive(w))).`



`int -> fn_p_g: move value v in world (first) to its position in derived (second)`




- `fork(derive, commit)(w)`
- `commit((w, derive(w)))`
- `(commit ∘ mapsnd(derive) ∘ dup)(w)`

where `dup` takes a `w` and returns a `(w,w)`. Then `mapsnd(derive)` transforms just the second `w`, adding the wall and executing the policy. then `commit` (which is `sync`) collapses the pair back into one grid, writing the agent result to the real world. 

*monomorphism*

The decomposition is monomorphic, but that's just because of how we've implemented it. 

- a _type_ is a label for the kind of thing a value is, like `int`, `grid`, etc
- a _type variable_ is a placeholder `a`, `b`, `c` for any type you plug in. So a function `a -> a` is a function that takes something of any type and returns something of that same type, like `+ :: int -> int`
- a _polymorphic_ function works for many types, like `map :: (a -> b) -> [a] -> [b]` takes a function between two types `(a -> b)` and a list of the first type `[a]` and returns a list of the second type `[b]`. For example take the following where type variable `a` stands for `int` and `b` stands for `bool`

```lisp
(map
  (lambda x: x > 2) ; a function from ints to bools, type (int -> bool)
  '(1 2 3 4)        ; a list of ints, type [int]
)
```

The expression checks each digit in the list whether it's greater than 2. So it returns `[False, False, True, True]`, a list of bools, so it's type `[bool]`, which is the placeholder `[b]`.

*the argument*

The concept is polymorphic, but the implementation happens to be monomorphic because there's only one type at play (`grid`). 


== PLAN

phase 1: assume two-channel capability, `fork` and `sync`. world channel and model channel, something. then given this generic capacity, show that the system can learn a theory of mind `Believes` primitive, the "informational relation" that Leslie assumes, which relates an informational state (a grid representation) to an agent, attributing belief to that agent. The representational system isn't proprietary, and the concepts are learned from combinations of lower-level concepts. Show that those primitives can be used in other settings and can be combined in other ways, so it's not just `Believes` decomposed. show that solutions to false belief tasks live deeper in program space. 

phase 2: more of the same but with `fork` and `sync` broken down further. 

phase 3: learning the channel arity. 

- enumerate typed program trees by probability 
- enumeration weighed by description length
- goal is MDL
- wake/sleep library learning by compression
- I'm doing inductive synthesis on I/O pairs, not deductive

types
- I've got a simply-typed and monomorphic DSL
- the types are only used to prune enumeration
- so they just constrain which wirings are legal
- they don't constrain what the program must prove
- no generic composer
- type-pruned, example-driven, compression-based synthesis

goal:
- show that "belief"/ToM can be a discoverable compound under MDL rather than a built-in primitive
- 

#pagebreak()

== arity-generalization: discovering the number of grids needed, discovering that you even need a private channel for belief



#pagebreak()

#all-tasks()

#load-bib(read("references.bib"))