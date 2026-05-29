= Fixing Stitch problems

Stitch represents abstractions in lambda calculus, where a hole can appear in function position like ```(#0 1 5)``` which means "apply function $lambda$0 to arguments 1 and 5"

The DSL uses fixed-head trees where the head is always a known primitive, so can't represent hole-as-head

Stitch then rewrites programs using `fn_3` e.g. `(place_wall blank 1 5)` → `(fn_3 place_wall)`, which your parser can't resolve.

Currying would make this representable but explodes tree depth — every n-ary primitive becomes n levels, pushing solutions into deeper enumeration windows.

The actual fix: inline-expand skipped abstractions

  When parsing a rewritten program that references a skipped abstraction, substitute the body:

```python
  # fn_3 was skipped but its body is known: "(#0 1 5)"
  # rewritten: "(fn_3 place_wall)"
  # → substitute #0=place_wall: "(place_wall 1 5)"
  # → now parseable
```

  Concretely, before giving up on a rewritten program, try expanding each reference to a skipped
  abstraction by substituting its arguments into the saved body string:

```python
  # track skipped abstraction bodies keyed by name
  skipped_bodies = {}  # e.g. {'fn_3': '(#0 1 5)', 'fn_4': '(#0 0 3)'}


  # when parsing fails due to fn_3, expand it:
  def expand_skipped(prog_str, skipped_bodies):
      import re
      changed = True
      while changed:
          changed = False
          for name, body in skipped_bodies.items():
              # match (fn_3 arg0 arg1 ...)
              pattern = rf'\({name}((?:\s+\S+)*)\)'
              def replacer(m):
                  args = m.group(1).split()
                  result = body
                  for i, arg in enumerate(args):
                      result = result.replace(f'#{i}', arg)
                  return result
              new = re.sub(pattern, replacer, prog_str)
              if new != prog_str:
                  prog_str, changed = new, True
      return prog_str
```

  Then in saturate_stitch, save skipped bodies and try expansion before failing:

  \# in the abstractions loop, when skipping:
  skipped_bodies[name] = body_str   \# save it
  print(f"skipping abstraction '{name}' ...")

  \# in the rewritten programs loop:
  for prog_str in result.rewritten:
      expanded = expand_skipped(prog_str, skipped_bodies)
      try:
          tree = tr(D, expanded)
          ...
      except Exception as e:
          print(f"could not parse rewritten program '{prog_str}': {e}")

  Why this happens / how to debug

  Print the raw stitch output before any filtering to understand what abstractions are being created:

  for abs_result in result.abstractions:
      print(f"  {abs_result.name}: {abs_result.body}  (arity={abs_result.arity})")

  If you see many abstractions with \#0 in head position, it means stitch is finding patterns where the
  same arguments recur but with different primitives applied to them — e.g. (place_wall blank 1 5),
  (place_ag blank 1 5 ...) both have blank 1 5 repeated. Stitch factors out the recurring arguments and
   puts the varying primitive in hole position. This is a sign your programs share a lot of int
  constants (1, 5) across multiple primitives, which is common when grid coordinates repeat across
  tasks.

  The inline-expansion approach recovers these rewritten programs without needing currying.

  - skipped_bodies dict: collects the stitch body string (with \#i holes) for every abstraction that
  gets skipped — both parse failures and 0-arity/partial-application cases.
  - expand_skipped(prog_str): iterates re.sub until stable. The regex `\(fn_3(\s+\S+)*\)` captures the
  arguments, then substitutes \#0, \#1, ... into the saved body. So (fn_3 place_wall) with body (\#0 1 5)
  → (place_wall 1 5). The loop handles nesting: if a skipped abstraction references another skipped
  abstraction, multiple passes resolve it.
  - Applied before parsing each rewritten program: expanded = expand_skipped(prog_str), then tr(D,
  expanded). If expansion produces a parseable program, it joins the training corpus. The error message
   also shows the expanded form to make failures easier to debug.



*actually....*

   Got rid of the regex, replaced with s expression parser


 fn_1: (\#0 1 0) has the hole \#0 in head/function position — (\$0 1 0) means "call \$0 as a function with
   args 1, 0". Your DSL has fixed-head trees where the head is always a known primitive, so there's no
  type to assign to a hole-as-head. typize walks the tree looking for \$i placeholders and infers their
  type from the parent's tailtypes — but a hole in head position has no parent tailtype entry, so it
  collects zero typed holes.

  This was always going to be skipped. What changed with the fix is the consequence: previously,
  rewritten programs referencing fn_1 would fail to parse and be dropped. Now they get inline-expanded
  ((fn_1 place_wall) → (place_wall 1 0)) and parse fine. The skip message is correct and expected —
  fn_1 is genuinely unrepresentable as a DSL primitive. It just no longer causes downstream failures.

= New DSL

Keep:
- blank, gset, place_wall, place_agent_goal (grid construction)
- unfold (generic iteration)
- approach (generic move A toward B one step)
- navigate (convenience terminal, just approach(1,2))
- exists (generic grid predicate)
- if_goal/if_else (generic conditionals)

remove:
- nav_unfold (pre-packages intention)
- hide_walls (encodes observer perspective) replace with mask(mat,val)
- at, optimize, goal type (pre-packages desire)
- full_obs, assume, optimize_k, know type (belief)

= New Test Suite

=== round 1

6 total tasks
- 3 nav (0 wall)
- 3 false-belief

bootstrap the 3 nav tasks, see if ECD can find the 3 false-belief tasks

hypothesis: expected discovery is:
  mask(unfold(believed_grid, T, navigate), 3) where believed_grid has phantom walls

test so that the hypothesis is the only clean explanation
of the false belief tasks

Q trains on unfold(place_ag(...), T, navigate),
must discover that mask(unfold(place_wall(place_ag(...), ...), T, navigate), 3) explains the detour trajectories

=== round 2

increased to 12 nav seeds. But stitch didn't discover abstractions. all 12 programs have the same top-level structure `(unfold (place_ag blank ...) T navigate)` but stitch finds no useful abstraction because `T` varies and the int coordinates vary. Stitch's compression metric is based on size reduction; abstracting `(unfold #0 #1 navigate)` would save nothing since the holes just push the varying parts up. The shared parts (`unfold, blank, navigate, place_ag`) are all single-token terminals — there's no repeated subtree to compress.

= May 8

Belief: a primitive that captures which grid drives the agent's trajectory

== files

== primitive encoding of grids, remove bootstrapping

rather than needing to find the encoding of the starting grid (as `(place_ag blank 3 3 2 0)` for example) by enumerating coordinates, the starting grid is encoded as a task-specific primitive

- each task `Xs[i]` gets a terminal `Delta(x[0], grid, repr='ig_i')` 
- stitch sees distinct tokens per task, so it creates holes for them `fn_nav($grid, $T) = (unfold $grid $T navigate)`
- so ECD only searches over T and step functions
- for false belief tasks, it searches over phantom wall position (since that's invisible in `x[0]`)

so no bootstrapping, ECD discovers everything

== removing `T`

rather than needing to enumerate ints to find the timespan `T` over which to unfold (how many grids in the sequence), when evaluating a candidate program just run it for as many timesteps as are in the target task

- added `unfold_auto(grid, fn)` to `dsl.py` which reads a module-level `_unfold_steps`
- in exploration, before enumerating each task, set `_unfold_steps` to `x.shape[0]` which is the `T` dimension of the task matrix x. So the found program should unfold for exactly as many steps as are in the target matrix. 
- in dreaming, set `_unfold_steps` to a sampled `T` when evaluating solution trees (dream just needs a valid input, it doesn't need a precise T)
- so instead of `Delta(unfold, mat, [grid, int, fn])` we have `Delta(unfold_auto, mat, [grid, fn])` (but for simplicty we rename `unfold_auto` to `unfold`)

=== file 1

mvp

_Tasks :_ 10 navs on $4 times 4$ grids with a fixed 2-cell vertical barrier at $(1,2),(2,2)$.

`
0 0 0 0
0 3 3 0
0 0 0 0
0 0 0 0
`

Each task gets a task-specific terminal `ig_i = x[0]` (agent + goal + walls).

All tasks solved instantly in enumeration, after just two trees. Solutions:

- `(unfold ig_0 navigate)`
- `(unfold ig_1 navigate)`
- etc

So stitch abstracts
`fn_0: (unfold #0 navigate)  [mat]`

and programs are rewritten

- `(fn_0 ig_0)`
- `(fn_0 ig_1)`

it's just a nav primitive paramterised over the initial grid. 

=== file 2

dataset:
- 8 nav tasks $4 times 4$, one wall
- 20 false-belief tasks $4 times 4$, one phantom wall

DSL:
- 20 core primitives
- 28 task-specific terminals

i.e. the initial grid $x[0]$ that the system sees as a terminal primitive contains just the agent and goal, walls aren't visible to the system. This is the "true" grid, the grid that the system sees with its own "eyes". 

_Explore 0_

just as before, the system immediately finds `(unfold ig_0 navigate)` etc for each of the 8 simple nav tasks.

then to solve the false belief tasks, it enumerates about 1.2 million trees. it doesn't find anything so it moves to compression.

_Compress 0_

stitch abstracts `fn_0: (unfold #0 navigate)  [mat]` as it did in file 1. 

_Explore 1_

then in the new ECD iteration, the system finds solutions to the false belief tasks within $1000$ to $2000$ trees.

- `(mask (fn_0 (place_wall ig_8 2 1)) 3)`
- `(mask (fn_0 (place_wall ig_9 2 1)) 3)`
- `(mask (fn_0 (place_wall ig_10 1 1)) 3)`
- etc

_Compress 1_

Stitch finds

`fn_0  [<class 'int'>, <class 'int'>, grid]
  body: (mask (unfold (place_wall $2 $1 $0) navigate) 3)`

and 

`fn_2  [grid]
  body: (unfold $0 navigate)`

along with some other useless coincidental abstractions (baked-in column values)

if run with capped iterations `saturate_stitch(D, sols, iterations=2)`, iteration 1 finds the general belief primitive `fn_0($grid, $pwr, $pwc)` and nav `fn_1($grid)`

So the system successfully finds the expected discovery, abstracting `fn0: (mask (unfold (place_wall $2 $1 $0) navigate) 3)`. It's a function that takes a grid (the initial grid, the task-specific terminal primitive) and a coordinate pair, and returns the sequence of grids produced by an agent navigating a version of the task-grid augmented with a wall placed at the coordinate. 

To discover `(unfold #0 navigate)` is to discover the concept of 

So to discover `f0` is to discover


TODO:
- implement other step functions, right now `navigate` is the only one. There should be other possible explanations for a movement, and there should be a cost to choosing more complex explanations. `navigate` already assumes so much, it's a complicated primitive. 
- unpack `navigate`, so it's discovered from lower-level primitives?
- unpack `mask` and discover from lower-level, structurally. or, figure out a reason for `mask` to exist outside of just this one use case
- basically, it all feels too tailored. make it more general so that it feels like it's actually putting stuff together and discovering something. right now I'm just waiting for it to apply `mask` to `unfold` and calling that learning theory of mind. but there's already so much encoded there
- so keep removing and making lower level, and show the point at which it breaks down

= May 11: new step functions

`approach(agent, goal)` is expensive, it runs BFS internally and takes 2 int args

so `step(agent, dir)` is a shorter explanation since it's simpler internally (doesn't run bfs) and also dir only has four options

Semantic evaluation:
- for belief: does mask wrap a grid with extra content?
- for desire: does a hole appear in both a world-placement position and a step-function position?

= May 17: decomposing "approach"

```python
core_prims = [
    Delta(unfold_auto,  mat,  [grid, fn],             repr='unfold'),
    Delta(gset,         grid, [grid, int, int, int],  repr='gset'),
    Delta(optimize,     fn,   [util, int],            repr='optimize'),
    Delta(neg_distance, util, [int],                  repr='neg_dist'),
    Delta(distance,     util, [int],                  repr='distance'),
    Delta(neg_util,     util, [util],                 repr='neg_util'),
    Delta(add_util,     util, [util, util],           repr='add_util'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
]
```

*desire tasks*: navigation towards goal, but paramterized by goal value (could be 2,4,5)

stitch finds `(unfold (gset $3 $2 $1 $0) (optimize (neg_dist $0) 1))`

e.g.`
goal_val=4  agent=(3, 1)  goal=(3, 3)
    found:     (unfold (gset ig_15 3 3 4) (optimize (neg_dist 4) 1))
    rewritten: (fn_0 4 3 3 ig_15)`

this all happens in the first enumeration

= May 19: file 5

*Sequential desire tasks* agent(1) approaches goal(gv1) then goal(gv2)

added primitive `if_fn(exists(gv1), optimize(neg_dist(gv1),1), optimize(neg_dist(gv2),1))`

Hope:

1. `neg_dist($gv)` desire as utility
2. `fn_want($gv)=optimize(neg_dist($gv), 1)` desire compiled to action
3. `fn_cond_desire($gv1,$gv2)=if_fn(exists($gv1), fn_want($gv1), fn_want($gv2))` goal ordering combinator
4. `fn_seq_desire(...)` full task

*RESULTS:* ran for 8 iterations. only found simple desire, and accidentally some sequential ones that matched.

So added `fn_want` which wraps `approach(1, goal_val)`. So `fn_want =  fn_want = optimize(neg_dist($0), 1)`. 

```python
core_prims = [
    Delta(unfold_auto,  mat,     [grid, fn],           repr='unfold'),
    Delta(gset,         grid,    [grid, int, int, int], repr='gset'),
    Delta(fn_want,      fn,      [int],                 repr='fn_want'),
    Delta(optimize,     fn,      [util, int],           repr='optimize'),
    Delta(neg_distance, util,    [int],                 repr='neg_dist'),
    Delta(distance,     util,    [int],                 repr='distance'),
    Delta(neg_util,     util,    [util],                repr='neg_util'),
    Delta(add_util,     util,    [util, util],          repr='add_util'),
    Delta(if_fn,        fn,      [fn_pred, fn, fn],     repr='if_fn'),
    Delta(exists,       fn_pred, [int],                 repr='exists'),
    Delta(0, int, repr='0'), Delta(1, int, repr='1'),
    Delta(2, int, repr='2'), Delta(3, int, repr='3'),
    Delta(4, int, repr='4'), Delta(5, int, repr='5'),
]
```


hope:

1. `fn_desire($ig,$gr,$gc,$gv)` from simple desire `(fn_want ×1, gv ×2)`
2. `fn_cond_desire($gv1,$gv2)=if_fn(exists($gv1), fn_want($gv1), fn_want($gv2))` from sequential desire
3. `fn_seq_desire($ig,$r1,$c1,$gv1,$r2,$c2,$gv2)` from sequential desire (full task)

= May 22

Ran file5 on a mix of simple nav tasks and multi step nav tasks, in the hope that it would abstract `(optimize (neg distance $gv))` from all the solutions `(unfold grid (optimize (neg distance goal)))`. But that didn't happen because the solutions were all of the same form, they all have `unfold` as their root, so it abstracts the full `(unfold (gset #3 #2 #1 #0) (optimize (neg_dist #0) 1))` which is too specific.

So I changed it so that now the enumerator is trying to find an `fn` and the system implicitly unfolds the `fn` when evaluating it against the target matrix. 

- solve_enumeration: enumerates fn-type programs; callback evaluates each as `unfold(x[0], T, fn_val)`
per task, keyed by ig_map. `_unfold_steps` global is no longer touched here.
- dream: takes training_Xs=None; generates random fn trees (depth 5 instead of 10); evaluates by
picking a random training frame as the initial grid and calling unfold(ig, T, fn_val) directly.
- ECD: passes training_Xs=Xs to dream.
- file5.py: core_prims drops unfold_auto and gset; ig terminal generation removed; D = Deltas(core_prims) only. Unused imports (mat, grid, unfold_auto, gset, task_terminals) cleaned up.

And indeed that does fix the problem. It first finds `(optimize (neg_dist 2) 1)` for a simple desire task, then finds `(if_fn (exists 2) (optimize (neg_dist 2) 1) (optimize (neg_dist 4) 1))` for multi-goal tasks. In fact it finds it quite quickly, solutions for both types of tasks within the first enumeration. So I can make things more complicated. 

So yeah it successfully abstracts `(if_fn (exists $1) (optimize (neg_dist $1) 1) (optimize (neg_dist $0) 1))` so it can do

`
(2→4)  agent=(1, 2)  goal1=(0, 3) goal2=(3, 2)  T=7
    found:     (if_fn (exists 2) (optimize (neg_dist 2) 1) (optimize (neg_dist 4) 1))
    rewritten: (fn_0 4 2)
`

= May 24

The problem with making `unfold` implicit and searching for an `fn` rather than a `grid` is that now we have to take the initial grid as a given and we can't search over possible different ways for the starting grid to be - we can't `mask`, we can't posit other possible states of the world

initial fix: two modes of ECD, the classic one where the resulting program is a `grid`, and the new one (just for file 5) where the resulting program is an `fn`. So then for file 5 it'll find `optimize(neg_dist($gv), $av)` or something, but for all the rest it'll find `unfold(grid, fn)` like it did previously

but of course that's not the point. the goal is to enumerate over a combination of initial grid states and transition functions, to find the pair that yields the best explanation. 

a `belief` function would look something like `believes(agent, grid)` associating an agent with the grid that they believe represents the real world, the grid according to which they navigate. For example, if agent 1 navigates according to a grid that's just like the initial grid but has a wall placed at (1,2) it'd be `(believes 1 (place_wall ig 1 2))`.

= May 25

making `unfold` explicit opened up a new variable to play with: not just choosing the DSL and choosing the data, but choosing the root type. i.e. choosing the type of the final programs, the programs to enumerate over. previously it was just generating `mat` type, but in the version of file5 without unfold it's generating `fn` type. 

Delete *file7*, it is stupid and useless. It has root type `grid` and evaluator `unfold_belief_steps(actual_g, believed_g, T, approach(agent_val, goal_val))`. Its DSL is just `place_wall`. So it just searches over possible wall placements to find the matching initial grid. For two-agent tasks, it's hard coded to separate them out into separate tasks and then runs ECD on each and then abstracts `place_wall($ig, $r, $c)`. 

Delete *file8* it's something like that but for desire, finding the goal for each agent, it's also stupid and useless. 

Then I tried with *file6* originally it was finding `fn_belief`, which is a function `int -> (grid -> grid)`, given an agent returns a grid to grid transformation function `fn`, i.e. a function for augmenting the initial grid (adding a wall somewhere) to make the navigation optimal, i.e. a function that given the initial grid returns the grid according to which the agent is navigating optimally. 

The actual grid is passed at evaluation time. So there are no task-specific grid primitives, it's passed in. So the primitives are

Delta(assign_belief, fn_belief, [int, fn, fn_belief], repr='assign_belief'),
Delta(no_belief_fn,  fn_belief,                       repr='no_belief_fn'),
Delta(set_at,        fn,        [int, int, int],      repr='set_at'),

the solution to simple nav tasks is `assign_belief($av, set_at($r, $c, 3), no_belief_fn)`

so stitch should abstract `assign_belief($av, set_at($r, $c, 3), no_belief_fn)` as a function that assigns a singular belief to one agent (`no_belief_fn` is the zero case, i.e. for all other agents if any they just navigate on the standard grid)

then for two agents the solution would be `assign_belief(1, set_at(r1, c1, 3), fn_one_belief(4, r2, c2))`, it's nested. Agent 1 navigates according to the grid given a wall at (r1, c1), and agent 4 navigates according to the grid given a wall at (r2, c2), and if there are any other agents they navigate according to the default grid. 

= May 26

The above is garbage. file6 produces programs of type fn_belief. But my goal is for ECD to discover that structure by itself. I want it to discover the very notion of assigning an fn (or grid) to an int, assigning a proposition to an agent (i.e. a belief). I want it to discover that structure, not just to discover for particular tasks what the correct assignment is.  

so ECD just discovers the correct _values_ (which agent, which coordinate) not the _structure_(that beliefs are `int -> grid -> grid`) assignments

`assign_belief` is what I want to abstract. I don't want it as a DSL primitive. then what's the point. 

`assign_belief` can be decomposed with lambda abstraction `lam a. if a == av then f else id_fn`

Ideally it'd abstract something like `unfold(ig_i, pair(av1, f1), pair(av2, f2))`

The problem is that stitch finds repeated subtrees. So for stitch to discover `pair(av, f)` (the belief function, pairing an agent with a transformation function), then `(av, f)` would need to appear as a subtree, not just as arguments in `unfold`. With flat `unfold(ig_i, av1, f1, av2, f2)`, `av` and `f` are siblings. So stitch wouldn't abstract their connection. 

So I could just accept `pair` as a primitive and treat it like `cons` in a list or something, and then argue that `pair(av, f)` is belief. But that's not satisfying. 

or I could use flat `unfold(ig_i, av1, f1, av2, f2)` and when stitch finds a partial application abstraction say that it's belief. but then it's not general, it's just about those specific agents

*SOOO* obviously what I should do is lambda abstraction. then `pair` is found as "the function that, given an agent, returns this transformation", i.e. `λa. if a==av then f else id_fn`

problem is that `Delta.__call__` strictly evaluates arguments. So by the time `lam(body)` runs, `body`'s already been evaluated. So there's nowhere to substitute `a`. 

So I need to add an explicit binder in the tree. 

<<< lambda implementation >>>

problem is that `sim` is still top level primitive always, since root type is `mat`. So once again the discovered abstractions are too specific because they include `sim`, we need a component of it. So maybe ditch stitch or change it or something, so that it abstracts just components and not the whole thing? i.e. make the abstraction shittier. Or vary the data more. but any mat type will have sim at the root. like, say fn0 is the abstraction. take every subtree of fn0 and make that an abstraction too. 

building data structures using lambda calculus?

= May 27

nested conditional if statements. lambda DSL. everything is an `fn` except int terminals and variable sentinel.

*enumeration*

root type: `fn`, so enumerates over functions

but it can't just evaluate the program tree directly because it might have `var` in it

so it passes the unevaluated tree to `_lam_impl`, which returns a closure (the expression with the original value with which to evaluate it, according to when it was created)

then it replaces each `var` with the agent value at the time of simulation

so the beta-reduced expression `wm` is of type `int -> fn`. It maps an agent to its belief. That gets passed to `unfold_multiagent_fn_belief_steps`, which calls `wm` for each agent as `wm(1)` and `wm(4)`. That gets each agent's world transformation.

Then it simulates both agents navigating under their respective worlds, and checks if that produces the grid. 

*DSL*

`Delta(_if_int_eq,    fn,  [int, int, fn, fn], repr='cond'),`

conditional. takes two ints and two fns. if the ints are equal, return the first function. otherwise return the second function. the ints are for comparing which agent the given agent is. The fns are for choosing the corresponding grid transformation, whether to transform the grid for that agent or not (or to check the other agents). 

`Delta(set_at,        fn,  [int, int, int],    repr='set'),`

grid transformation function. takes the coordinates at which to place a wall, and the value to place there. since the value is irrelevant anyway I might as well just hardcode this as 3, do that later

`Delta(_var_sentinel, int,                     repr='var'),`

the sentinel value in which to place the agent value when we do the beta reduction at evaluation

`Delta(id_fn,         fn,                      repr='id'),`

just the id function. in principle the id function and conditional can be found from pure lambda calculus but they're given as primitives to make search easier

*solution to 1-agent task:*

`(cond var 1 (set 1 2 3) id)`

if the agent (`var`) is agent 1, apply `(set 1 2 3)` to the default grid (i.e place a wall at 1,2) otherwise, apply `id` to the grid (so any other agent, in this case none, operates based on the default grid)

finds solutions like this within $~2000$ enumerations. also finds some solutions like `(set_at 4 var 3)` but those are degenerate

abstracts

`fn_0: (if_int_eq var 1 id_fn (set_at #2 #1 #0))  [fn]`

*solution to 2-agent task:*

`
a1=(4, 2) pw1=(3, 2)  a2=(1, 2) pw2=(1, 3)  T=5
    body:      (if_int_eq var 1 (set_at 3 2 3) (if_int_eq var 1 id_fn (set_at 1 3 3)))
    rewritten: (fn_2 3 2)
`

abstracts

`(if_int_eq var 1 (set_at $0 $1 $0) (if_int_eq var 1 id_fn (set_at 1 3 3)))`

*summary of the implementation so far*

over the last few days I've discovered that a recursive/nested list data structure makes sense for associating grid states with agents, to form beliefs. 

a candidate program is a `fn`, it's a chain of conditionals where each conditional checks the agent being evaluated and if it's that corresponding agent returns the transformation function `fn` that will transform the grid into the grid according to which the agent is navigating optimally. So if the agent is seemingly navigating suboptimally because it takes a detour at (3,3), the corresponding transformation function is one that takes the intial grid and places a wall at that coordinate. 

The whole program is a lambda expression that takes an `int` agent value, since it's evaluated for each agent

To evaluate/simulate the program, replace `var` with each possible agent (or other) value. 

So for example, to evaluate the program

`(cond var 1 (set 1 2 3) (cond var 4 (set 2 2 3) id)))`

first replace all instances of `var` with `1` for agent 1 (beta reduction). So the first condition is met, since `var` is now `1` so `var==1`. So the expression evaluates to `(set 1 2 3)`, which is the grid transformation function for agent 1. So agent 1 navigates on a grid just like the initial grid but with a wall `3` at coordinate `1,2`. 

Now replace all instances of `var` with 4 for agent 4. The first condition isn't met since `var!=1`, so move to the fallback value which itself is another conditional. This time the predicate is true `var==4`, so we return grid transformation function `(set 2 2 3)` for agent 4

now say there's also agent 6. we replace all var instances, and it doesn't meet any of the predicates. So we return the fallback value of the second conditional, `id`. So agent 6 simply navigates on the unchanged initial grid. 

so agent 1 and agent 4 have false beliefs, they operate based on altered grids. But agent 6 has a true belief, operating based on the actual grid. 

(it's hard to say what's a false and what's a true belief. it's relative, to the other agents. it's more like just all the agents have different beleifs you need to account for.)

So now that we have a grid function for each agent (i.e. each agent's belief), we need to check whether these functions are correct. to check this, we need to use the functions to generate the data, the grids. 

simulation: we assume that the agents move optimally toward their goal, we take movement as a given. then to evaluate the functions, we hand them to the simulator and see whether ideal navigation on such transformed grids would yield the target grid. The simulator unfolds based on optimal navigation on the corresponding grid for each agent.  

= May 28: after the meeting

in the above implementation we take movement as a given, we just search over grid transformations. but what I really want is to also search over movement. I want there to also be the option of physical explanations, non-mental. And I want it all to be on one grid at the same time. Three cases for example:

- You see agent $1$ optimally navigating towards goal $2$. to understand this you just need to posit `optimize(neg_dist(2), 1)`, that at each timestep $1$ becomes slightly closer to $2$. this isn't mental
- You see agent $1$ seemingly suboptimally navigating towards goal $2$. to understand this you can't just posit that it's an agent navigating to a goal (as above), since if it were an agent it'd be navigating optimally. so you also need to posit that it's navigating according to `fn(grid)`, a different grid than the actual grid, transformed by grid transformation function `fn`. 
- You see object `4` moving left. To understand this you can just posit `step(left, 4)`

So for an agent with a false belief you need to posit that it has a belief

my new goal is to synthesize both the believed grid (transformation function), and the movement (transition function). It'll be in the same format as above, nested conditionals. 

importantly, I want it to only assign a believed grid to agents, not to inanimate objects

so take a scene where $1$ is navigating to goal $2$ by true belief (empty grid), $4$ is navigating to goal $5$ by false belief (wall at 2,2), $3$ is the wall, and $6$ is a non agent that just moves down (affected by gravity or somethign). So I'd want the conditional chain to look something like

so now, the grid transformation function takes the movement transition function as an argument. 

`(cond var 1 
// condition for object 1

    (id (opt (negd 2))) 
    // object 1 navigates to goal 2 based on the default grid

    (cond var 4 
    // condition for object 4

        ((set 2 2 3) (opt (negd 5))) 
        // object 2 navigates to goal 5 based on a grid given by adding a wall to
        // the default grid at (2,2)

        (cond var 6
        // condition for object 6
            
            (step left)
            // object 6 moves left
            
            id
            // else condition. for any object other than 1,4,6, at each new timestep it's the same as the previous timestep. So it's just static. 
        )
    )
)`

- In the case of object 1, its transition function is `(id (opt (negd 2)))` meaning that it moves towards $2$ on an unmodified grid (`id`). 
- For object 4, it moves towards 2 on a grid modified with a wall at 2,2
- for object 6, it moves left
- for all other objects, they don't move at all

my hope is that for agents we'll always see the movement function wrapped in a grid modification function, even when that function is `id`, to account for the agents having a representation of the world (even when the representation is correct). whereas for non-agents, I'd hope that we wouldn't see wrapping like `(id (step left))` since there wouldn't be solutions that would require a moving non-agent to navigate over something that isn't the actual world. 

the challenge with this (enumerating over both possible grids and possible movement functions) is that currently in simulation it assumes optimal navigation when unfolding. we run `opt(neg_dist(goal))` (with goal and agent values handed in freely as an argument), so we run an optimal agent on the transformed grid (and then mask the walls), and then if that path matches the path in the target task then we know that the posited grid was the one that it was actually navigating on (since we assume it's navigating optimally)

so the change is that now in the simulation we need to unfold based on the particular movement function itself that we found, not just on the default optimize function

so then to actually generate the grid from a program, we need to loop through each timestep, for each timestep initializing an empty grid with unknown values. then within the timestep loop, loop through each object on the grid (each int value). then we augment the previous timestep's grid with the transformed grid, transformed by `fn` which is the nested transition function (which is at least `id` and at most `(grid_transformation movement_transition)`). We build up the new timestep object by object, only adding the values that are relevant to that specific object so we avoid collisions. Then we add it to the history, and then we move on to the next timestep. 