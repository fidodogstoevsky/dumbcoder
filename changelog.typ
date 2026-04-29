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