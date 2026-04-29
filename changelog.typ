== Wednesday, April 29th

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



   === Actually

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