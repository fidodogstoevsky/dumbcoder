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

== phases

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

=== phase 1

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

=== phase 2

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

stitch abstracts `fn_0: (unfold #0 navigate)  [mat]` as it did in phase 1. 

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

= May 19: phase 5

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

Ran phase5 on a mix of simple nav tasks and multi step nav tasks, in the hope that it would abstract `(optimize (neg distance $gv))` from all the solutions `(unfold grid (optimize (neg distance goal)))`. But that didn't happen because the solutions were all of the same form, they all have `unfold` as their root, so it abstracts the full `(unfold (gset #3 #2 #1 #0) (optimize (neg_dist #0) 1))` which is too specific.

So I changed it so that now the enumerator is trying to find an `fn` and the system implicitly unfolds the `fn` when evaluating it against the target matrix. 

- solve_enumeration: enumerates fn-type programs; callback evaluates each as `unfold(x[0], T, fn_val)`
per task, keyed by ig_map. `_unfold_steps` global is no longer touched here.
- dream: takes training_Xs=None; generates random fn trees (depth 5 instead of 10); evaluates by
picking a random training frame as the initial grid and calling unfold(ig, T, fn_val) directly.
- ECD: passes training_Xs=Xs to dream.
- phase5.py: core_prims drops unfold_auto and gset; ig terminal generation removed; D = 
Deltas(core_prims) only. Unused imports (mat, grid, unfold_auto, gset, task_terminals) cleaned up.