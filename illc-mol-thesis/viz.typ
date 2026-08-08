// Task-family visualisations — a small library + a standalone preview.
//
// USE AS A LIBRARY (selective figures in your paper):
//   #import "viz.typ": task-figure, all-tasks
//   #figure(task-figure("physics", "belief_wall"), caption: [...])  // early demo
//   #all-tasks()                                                // appendix: everything
//
// TWO MODES (`mode:` on task-figure / all-tasks / render-sample):
//   "frames" (default) — the trajectory as a row of grids t0, t1, … tn
//   "path"             — ONE grid (the t0 state) with an arrow per transition, so
//                        the whole trajectory reads as a path; cells reached only
//                        later are ghosted in, deletions get an ✕.  Add
//                        `show-steps: true` to number each arrow with its frame.
//   #task-figure("belief_wall", mode: "path")
//
// HOW MANY SCENES (`scenes:` on the same three, plus `scenes-figure`):
//   A task is a SET of k scenes sharing one latent program (scenes.py) — a program
//   solves it only by reproducing ALL of them, so the scene set, not one
//   trajectory, is what the searcher is up against.  `scenes: 1` (the default)
//   shows a representative scene; `scenes: auto` shows all k, and `scenes: n` the
//   first n.  In "path" mode the scenes sit side by side (one arrowed grid each),
//   in "frames" mode they stack, since a row of frames is already wide.
//   #scenes-figure("belief_wall")            // the k scenes, path view, in a row
//   #task-figure("desire", mode: "path", scenes: auto)   // the same thing, longhand
//
// OBJECTS THE PROGRAM POSITS (`posited:` / `posited-scenes:`):
//   Some values are not scenery the scene arrives with — the obstacle family's
//   wall is stamped onto the grid by `wall_at`, the very term the learner has to
//   find.  `posited: 3` draws that value hollow and hatched rather than as a solid
//   numbered cell, so a reader does not take it for a given, and `posited-scenes`
//   (1-based, auto = all) restricts which scenes draw it at all — elsewhere it is
//   erased.  Chapter 4 uses the pair to show the obstacle task twice: first as the
//   learner gets it, with nothing on the grid to explain the detour, then with the
//   wall the solution posits drawn in.  A figure that suppresses it is no longer
//   the raw scene set, so its caption has to say so.
//   #scenes-figure("obstacle", posited: 3, posited-scenes: ())   // no wall drawn
//   #scenes-figure("obstacle", posited: 3)                       // hatched, all k
//
// PLACEMENT AND CAPTIONS:
//   task-figure / scenes-figure draw the grids and nothing else, centred on the
//   page (`caption: none` and `alignment: center` are the defaults; `alignment:
//   none` puts them flush left).  Say what a figure IS with Typst's own figure —
//   that is where the numbered caption below, and the label to @-reference, come
//   from:
//     #figure(scenes-figure("belief_wall"),
//             caption: [The four scenes of one false-wall task.]) <fig-belief-wall>
//   (a figure centres its body anyway, so the two never fight).
//
//   `caption: auto` brings back viz's own HEADER above the grids — the family
//   title and a plain-English account of the mechanism, named to the values in the
//   grids underneath — which is what the `all-tasks` catalogue uses, since nothing
//   else there says which family is which.
//
// COMPILE STANDALONE (preview every family):
//   typst compile viz.typ
//
// Importing only pulls the #let bindings; the preview at the bottom of this file
// is discarded in the importing document, so it is safe to `#import`.
//
// DATA: task_samples.json holds one ACTUAL generated example task per family, written
// by experiment.export_task_samples (the `--samples` / `--samples-only` path of a
// phase run).  The single kind='belief' is split into its five display variants
// (wall / witness / goal / false-obstacle / observers).  Each sample is
// {kind, tag, k, scenes:[{T, panels:[{label, grid}]}]}, with scene 0's `T`/`panels`
// also lifted to the top level; fn families list successive `unfold` frames, fn_p_g
// families canvas | template | result (each scene carrying its own template).

#let task-data = json("task_samples.json")

// ── cell / grid rendering ─────────────────────────────────────────────────────
// Values are object ids reused across tasks; their role (agent / goal / hazard /
// noise / signal) is given by each task's tag.  0 = empty, 3 = wall.
#let palette = (
  "0": rgb("#f4f4f6"),   // empty
  "1": rgb("#4e79a7"),   // blue
  "2": rgb("#59a14f"),   // green
  "3": rgb("#54585c"),   // wall (dark)
  "4": rgb("#e1812c"),   // orange
  "5": rgb("#b07aa1"),   // purple
  "6": rgb("#e15759"),   // red
  "7": rgb("#76b7b2"),   // teal
  "8": rgb("#b8a02e"),   // olive
  "9": rgb("#9c755f"),   // brown
)

#let grid-gutter = 1pt

// A POSITED cell: hollow, hatched, dashed, and carrying no digit.  Some objects
// are not scenery given with the scene but are stamped onto the grid by the very
// program the learner is looking for (the obstacle family's wall, `wall_at`).
// Drawing those as ordinary solid cells reads as "the task came with a wall in
// it", which is exactly wrong, so they get a notation of their own.
#let posited-cell(size: 11pt, ink: rgb("#54585c")) = box(
  width: size, height: size,
  radius: 1.2pt, inset: 0pt, clip: true,
  stroke: (paint: ink, thickness: 0.8pt, dash: "dashed"),
  {
    // hatching: the anti-diagonals x + y = d, clipped to the square
    let s = (paint: ink.lighten(35%), thickness: 0.6pt)
    for d in (0.35, 0.7, 1.05, 1.4, 1.75) {
      let (p, q) = if d <= 1 { ((0pt, size * d), (size * d, 0pt)) }
                   else { ((size * (d - 1), size), (size, size * (d - 1))) }
      place(top + left, line(start: p, end: q, stroke: s))
    }
  },
)

// `ghost: true` draws the cell as a washed-out outline — used by the path view to
// show where an object ENDS UP without pretending it is there at t0.
#let cell(v, size: 11pt, ghost: false, posited: false) = if posited {
  posited-cell(size: size)
} else { box(
  width: size, height: size,
  radius: 1.2pt, inset: 0pt,
  fill: if ghost { palette.at(str(v), default: rgb("#cccccc")).lighten(78%) }
        else { palette.at(str(v), default: rgb("#cccccc")) },
  stroke: if ghost { (paint: palette.at(str(v), default: rgb("#cccccc")).lighten(30%),
                      thickness: 0.5pt, dash: "dotted") }
          else { 0.3pt + rgb("#c2c2c8") },
  align(center + horizon,
    if v != 0 {
      text(size: size * 0.55,
           fill: if ghost { palette.at(str(v), default: rgb("#cccccc")).darken(15%) } else { white },
           weight: "bold", str(v))
    }),
) }

// `posited` (a cell value, or none) names the value to draw in the posited
// notation wherever it occurs — see `posited-cell`.
#let render-grid(g, size: 11pt, posited: none) = grid(
  columns: (size,) * g.at(0).len(),
  rows: (size,) * g.len(),
  gutter: grid-gutter,
  ..g.flatten().map(v => cell(v, size: size, posited: v == posited)),
)

// a labelled grid (one frame / channel)
#let panel(p, size: 11pt, posited: none) = stack(dir: ttb, spacing: 3pt,
  text(size: 8pt, fill: luma(110), p.label),
  render-grid(p.grid, size: size, posited: posited),
)

// ── collapsed path view ───────────────────────────────────────────────────────
// Instead of laying the frames t0 … tn out side by side, draw ONE grid (the t0
// state) and overlay an arrow per transition, so a trajectory reads as a path.
//
// Object identity is the cell value, as everywhere else in this file.  Between
// consecutive frames, each cell a value GAINS is matched to the nearest cell it
// LOST (greedy, Manhattan) — that pairing is the move.  A gain with nothing lost
// is a growth step (motion trails: `overlay` paints a new cell without clearing
// the old one) and is drawn from the nearest cell the value already occupied, but
// only if it is adjacent; otherwise the object appeared from nowhere (`wall_at`)
// and gets no arrow.  A loss with nothing gained is a deletion and gets an ✕.

#let _cells-of(g, v) = {
  let out = ()
  for (r, row) in g.enumerate() {
    for (c, x) in row.enumerate() { if x == v { out.push((r, c)) } }
  }
  out
}

#let _manhattan(a, b) = calc.abs(a.at(0) - b.at(0)) + calc.abs(a.at(1) - b.at(1))

#let _nearest(cands, p) = cands.fold(none,
  (best, q) => if best == none or _manhattan(q, p) < _manhattan(best, p) { q } else { best })

// every non-empty value occurring anywhere in the trajectory, low ids first
#let _values-in(frames) = {
  let vals = ()
  for f in frames { for row in f { for v in row {
    if v != 0 and not vals.contains(v) { vals.push(v) }
  } } }
  vals.sorted()
}

// [(from, to, v, t)] for the whole trajectory, plus the cells that vanish
#let _transitions(frames) = {
  let moves = ()
  let gone = ()
  for t in range(frames.len() - 1) {
    let (a, b) = (frames.at(t), frames.at(t + 1))
    for v in _values-in((a, b)) {
      let (pa, pb) = (_cells-of(a, v), _cells-of(b, v))
      let added = pb.filter(p => not pa.contains(p))
      let lost = pa.filter(p => not pb.contains(p))
      for p in added {
        // prefer a cell the value just vacated; else an adjacent cell it kept
        let src = if lost.len() > 0 { lost } else { pa.filter(q => _manhattan(q, p) <= 1) }
        let from = _nearest(src, p)
        if from != none {
          moves.push((from: from, to: p, v: v, t: t))
          lost = lost.filter(q => q != from)
        }
      }
      for q in lost { gone.push((cell: q, v: v, t: t)) }
    }
  }
  (moves: moves, gone: gone)
}

// One grid carrying the whole trajectory.  `show-steps` numbers each arrow with
// the frame index it leaves, for trajectories where the order is not obvious.
#let path-grid(frames, size: 20pt, show-steps: false, posited: none) = {
  let base = frames.first()
  let final = frames.last()
  let (rows, cols) = (base.len(), base.at(0).len())
  let pitch = size + grid-gutter
  let (w, h) = (cols * pitch - grid-gutter, rows * pitch - grid-gutter)

  // centre of cell (r, c) in the box's coordinates
  let cx(c) = c * pitch + size / 2
  let cy(r) = r * pitch + size / 2

  let tr = _transitions(frames)
  // a value that disappears because something MOVED ONTO it (an agent reaching its
  // goal) was occluded, not deleted — the incoming arrowhead already says so
  let dead = tr.gone.filter(g => not tr.moves.any(m => m.t == g.t and m.to == g.cell))

  // movers get a perpendicular offset each, so crossing paths stay readable
  let movers = _values-in(frames).filter(v => tr.moves.any(m => m.v == v))
  let lane(v) = {
    let i = movers.position(x => x == v)
    if movers.len() < 2 { 0pt } else { (i - (movers.len() - 1) / 2) * 1.7pt }
  }
  let ink(v) = palette.at(str(v), default: rgb("#cccccc")).darken(22%)

  let arrow(m) = {
    let (x1, y1) = (cx(m.from.at(1)), cy(m.from.at(0)))
    let (x2, y2) = (cx(m.to.at(1)), cy(m.to.at(0)))
    let (dx, dy) = (x2 - x1, y2 - y1)
    let len = calc.sqrt(calc.pow(dx / 1pt, 2) + calc.pow(dy / 1pt, 2)) * 1pt
    if len == 0pt { return }
    let (ux, uy) = (dx / len, dy / len)          // unit vector along the move
    let (px, py) = (-uy * lane(m.v), ux * lane(m.v))  // perpendicular lane offset
    let head = calc.min(size * 0.30, 6pt)
    let tail = size * 0.28                        // keep the shaft out of the digits
    let tip = (x2 - ux * tail + px, y2 - uy * tail + py)
    let foot = (x1 + ux * tail + px, y1 + uy * tail + py)
    let shaft-end = (tip.at(0) - ux * head, tip.at(1) - uy * head)
    let ang = calc.atan2(dx / 1pt, dy / 1pt)

    place(top + left, line(start: foot, end: shaft-end,
                           stroke: (paint: ink(m.v), thickness: 1.1pt, cap: "round")))
    place(top + left, dx: tip.at(0) - ux * head / 2, dy: tip.at(1) - uy * head / 2,
      box(width: 0pt, height: 0pt, place(center + horizon, rotate(ang,
        polygon(fill: ink(m.v), stroke: none,
          (0pt, 0pt), (0pt, head * 0.84), (head, head * 0.42))))))
    if show-steps {
      place(top + left,
        dx: (foot.at(0) + shaft-end.at(0)) / 2 - uy * 4pt,
        dy: (foot.at(1) + shaft-end.at(1)) / 2 + ux * 4pt,
        box(width: 0pt, height: 0pt, place(center + horizon,
          text(size: size * 0.26, fill: ink(m.v), str(m.t)))))
    }
  }

  // a struck-out cell: the value was there and is not any more
  let strike(g) = {
    let (x, y) = (cx(g.cell.at(1)), cy(g.cell.at(0)))
    let d = size * 0.24
    let s = (paint: ink(g.v), thickness: 1pt, cap: "round")
    place(top + left, line(start: (x - d, y - d), end: (x + d, y + d), stroke: s))
    place(top + left, line(start: (x + d, y - d), end: (x - d, y + d), stroke: s))
  }

  // base = t0; cells occupied only at the end are ghosted in, so a path that walks
  // into empty space still shows where it lands, and deleted cells are washed out
  // under their ✕ rather than sitting there at full strength.  A `posited` value
  // takes its own notation whether it is in the base or arrives later — it is not
  // a thing the scene shows up with, however long it then stands.
  let layer = ()
  for r in range(rows) {
    let row = ()
    for c in range(cols) {
      let (v, f) = (base.at(r).at(c), final.at(r).at(c))
      row.push(if v == posited or (v == 0 and f == posited) { posited-cell(size: size) }
               else if v != 0 { cell(v, size: size, ghost: dead.any(g => g.cell == (r, c))) }
               else if f != 0 { cell(f, size: size, ghost: true) }
               else { cell(0, size: size) })
    }
    layer.push(row)
  }

  box(width: w, height: h, {
    place(top + left, grid(
      columns: (size,) * cols, rows: (size,) * rows, gutter: grid-gutter,
      ..layer.flatten()))
    for m in tr.moves { arrow(m) }
    for g in dead { strike(g) }
  })
}

// the path counterpart of `panel`: one grid, labelled with the span it covers.
// `label` overrides that text — the multi-scene views say which scene it is
// instead, which matters more there than the frame span.
#let path-panel(frames, labels, size: 20pt, show-steps: false, label: auto,
                posited: none) = stack(
  dir: ttb, spacing: 3pt,
  text(size: 8pt, fill: luma(110),
       if label == auto {
         labels.first() + sym.arrow.r + labels.last() + " (" + str(frames.len()) + " frames)"
       } else { label }),
  path-grid(frames, size: size, show-steps: show-steps, posited: posited),
)

// Which panels of a sample form a trajectory?  fn families list t0 … tn; fn_p_g
// families list canvas | template | t1, where canvas → t1 is the (one-step)
// trajectory and the template is a separate channel shown alongside.  For the
// fn_p_g families that REPLACE the grid rather than move things in it (readout,
// perception) that one step is a diff, not motion — render those with
// mode: "frames" if the arrows read as more than they are.
// ("world" is the pre-canvas spelling of the same panel; kept so older artifacts
// and rival_samples.json keep rendering.)
#let _is-frame(p) = p.label == "canvas" or p.label == "world" or (
  p.label.starts-with("t") and p.label.slice(1).match(regex("^[0-9]+$")) != none)

// takes anything carrying `panels` — a whole sample or one of its scenes
#let _split-panels(s) = (
  s.panels.filter(_is-frame),
  s.panels.filter(p => not _is-frame(p)),
)

// The scenes of a sample.  Pre-multi-scene artifacts have no `scenes` key; their
// top-level T/panels ARE the one scene, so they keep rendering unchanged.
#let _scenes-of(s) = if "scenes" in s { s.scenes } else { ((T: s.T, panels: s.panels),) }

// ── per-kind captions: human title + what the task's mechanism is ─────────────
// A sample's `tag` records the latent the generator sampled — "av=1, gv=2,
// pw=(1, 2), bv=5".  Parsed into a dict, it lets a caption name the values and
// cells the reader is actually looking at instead of talking about "the agent".
// Values may be parenthesised or bracketed tuples, so the comma split has to
// respect brackets.
#let _tag-fields(tag) = {
  let out = (:)
  for m in tag.matches(regex("(\\w+)=(\\([^)]*\\)|\\[[^\\]]*\\]|[^,]+)")) {
    out.insert(m.captures.at(0), m.captures.at(1).trim())
  }
  out
}
#let _fld(t, k) = t.at(k, default: "?")

// Title + a plain-English account of the mechanism, the second a function of the
// parsed tag so it names this instance's values and cells.  ONLY fields that
// belong to the family's latent — the ones every one of the k scenes shares —
// may be named; per-scene incidentals (which values happen to be misplaced this
// time round) are described by role instead, since the caption sits under one
// scene but is about the task.
#let info = (
  // minds-free substrate
  physics: ("Constant movement", t => [
    Value #_fld(t, "val") drifts one cell #_fld(t, "dir") per frame; nothing else
    on the grid changes.]),
  desire: ("Goal-directed movement", t => [
    Value #_fld(t, "av") wants value #_fld(t, "gv") and takes the shortest route
    to it, one step per frame. It sees the grid as it is, so nothing separates
    what it takes to be the case from what is.]),
  // independent extension: fork / sync without belief
  overlay: ("Overlay", t => [
    Value #_fld(t, "val") steps #_fld(t, "dir"), but the cells it leaves are
    never cleared, so it smears across the grid rather than moving through it.]),
  comet: ("Comet", t => [
    Value #_fld(t, "av") walks to value #_fld(t, "gv") as in goal-directed movement, but keeps
    every cell it has passed through: goal-directed motion and a trail, with no
    mind in it.]),
  registration: ("Registration", t => [
    Value #_fld(t, "val") is moved to the cell the template assigns it. Every
    other value on the canvas stays where it stands, including the ones the
    template disagrees about.]),
  // belief — the five theory-of-mind variants (all kind='belief' in the corpus)
  belief_wall: ("False belief", t => [
    Value #_fld(t, "av") heads for value #_fld(t, "gv") but believes a wall
    stands at #_fld(t, "pw"), so it bends round that cell and pays extra steps
    for it. No frame of the task contains a wall, and value #_fld(t, "bv") is
    standing on the cell in question throughout, untouched.]),
  belief_witness: ("Witness belief", t => [
    Value #_fld(t, "av") believes a wall stands at #_fld(t, "pw") and detours
    round it on its way to value #_fld(t, "gv"). Value #_fld(t, "aw"), heading
    for value #_fld(t, "gw"), walks straight through that same cell. The wall
    has to be attributed to one agent and withheld from the other.]),
  belief_goal: ("Goal displacement", t => [
    Value #_fld(t, "gv") never moves. Value #_fld(t, "av") walks to
    #_fld(t, "displaced_to") and stops there, because in its own model the goal
    has been shifted away from where it actually lies: it goes to where it
    believes the goal to be, as Sally goes to her own basket.]),
  belief_observers: ("Two observers", t => [
    Value #_fld(t, "gv") stays where it is. Value #_fld(t, "av") heads for
    #_fld(t, "displaced_to"), the cell where it believes the goal to be, while
    value #_fld(t, "observer") heads for the goal itself. One scene, a false
    belief attributed to the first agent and none to the second.]),
  belief_false_obstacle: ("False obstacle", t => [
    Value #_fld(t, "av") is mistaken twice over: it believes a wall stands at
    #_fld(t, "pw"), where value #_fld(t, "bv") is in fact standing, and that
    value #_fld(t, "gv") lies at #_fld(t, "displaced_to"). So it bends round an
    empty cell and settles one short of the goal. The wall at
    #_fld(t, "real_wall") is the real one, and stays on the grid throughout.]),
  // symmetric corners — one minds-free task per cube axis
  flee: ("Flee", t => [
    Value #_fld(t, "av") moves to whichever neighbouring cell is furthest from
    value #_fld(t, "hv"), and stops once no move gains it anything.]),
  deletion: ("Deletion", t => [
    Cell #_fld(t, "cell") is cleared, whatever was standing on it; the rest of
    the grid is untouched.]),
  denoise: ("Denoise", t => [
    Every cell holding value #_fld(t, "noise") is erased at once, wherever it
    lies; everything else is kept.]),
  obstacle: ("Obstacle", t => [
    A wall is put on the grid at #_fld(t, "pw"), and value #_fld(t, "av") walks
    round it to reach value #_fld(t, "gv"). The detour has the shape of the
    false-belief one, but here the grid itself shows what causes it.]),
  relocate: ("Relocation", t => [
    The wall at #_fld(t, "from") is cleared and a wall appears at
    #_fld(t, "to"). Nothing else on the grid moves.]),
  underlay: ("Underlay", t => [
    Value #_fld(t, "val") steps #_fld(t, "dir") and leaves a trail, but wherever
    the grid is already occupied the standing value wins: the trail stops at
    value #_fld(t, "bystander") instead of painting over it.]),
  perception: ("Perception", t => [
    The template is updated to say where value #_fld(t, "val") actually is on
    the canvas; everything else the template holds is left as it was. The world
    is read but not changed.]),
  multi_reg: ("Multi-registration", t => [
    Every value the template also carries is moved to the cell the template
    assigns it; values only the canvas has stay put. The program names no value
    at all — it applies to whatever the two grids share.]),
  reg_except: ("Registration-except", t => [
    Every value is snapped to its template cell except value
    #_fld(t, "anchor"), which keeps the position it has on the canvas: one
    object exempted from a rule the rest are subject to.]),
  inpaint: ("Inpainting", _ => [
    Wherever the canvas is empty the template's value is filled in; wherever the
    canvas already holds something, the canvas wins.]),
  readout: ("Readout", _ => [
    The template is returned as it stands and the canvas is thrown away: the
    answer is whatever the second grid says, however the first one looks.]),
  // pair-plumbing corners — one minds-free task per combinator complement
  wipe: ("Wipe", _ => [
    Every value on the grid is cleared in a single step, and the blank grid then
    stays blank.]),
  composite: ("Composite", t => [
    Value #_fld(t, "drift") steps #_fld(t, "dir") on one layer while value
    #_fld(t, "masked") is erased from the other, and the two layers are then
    flattened into a single grid.]),
  drift_reg: ("Drifting registration", t => [
    Value #_fld(t, "val") is moved to the cell the template assigns it while
    value #_fld(t, "drift") steps #_fld(t, "dir") at the same time: a
    registration carried out inside a moving image.]),
  map_update: ("Map update", t => [
    The two grids trade roles and everything is then snapped: the template takes
    on the canvas's position for every value they both hold, keeps value
    #_fld(t, "stale"), which only it has, and does not take up value
    #_fld(t, "fresh"), which only the canvas has.]),
)

// The built-in caption for a sample: bold title + [kind], then a sentence or two
// on the mechanism, named to the values in the grids below it.  Exposed so
// callers can build on / tweak it.
#let default-caption(s) = {
  let meta = info.at(s.kind, default: (s.kind, none))
  stack(dir: ttb, spacing: 4pt,
    stack(dir: ltr, spacing: 6pt,
      text(weight: "bold", meta.at(0)),
      text(fill: luma(150), size: 8pt, "[" + s.kind + "]")),
    ..if meta.at(1) != none {
      (block(width: 100%, text(size: 9pt, fill: luma(60),
                               (meta.at(1))(_tag-fields(s.tag)))),)
    } else { () },
  )
}

// ONE scene of a sample, rendered in the requested mode: in "path" the frames
// collapse onto a single arrowed grid (any non-frame channel, e.g. the fn_p_g
// template, stays beside it as an ordinary panel); in "frames" every panel is
// drawn in sequence.  A scene with fewer than two frames falls back to "frames" —
// there is no path to draw.  `label` (when given) names the scene on the path
// panel, or as a line above the frame row.
#let _render-scene(sc, size: 11pt, gap: 12pt, mode: "frames", show-steps: false,
                   label: none, posited: none) = {
  let (frames, extras) = _split-panels(sc)
  let path-mode = mode == "path" and frames.len() >= 2
  let body = if path-mode {
    let p0 = path-panel(frames.map(p => p.grid), frames.map(p => p.label),
                        size: size, show-steps: show-steps, posited: posited,
                        label: if label == none { auto } else { label })
    (p0,) + extras.map(p => panel(p, size: size, posited: posited))
  } else {
    sc.panels.map(p => panel(p, size: size, posited: posited))
  }
  let row = stack(dir: ltr, spacing: gap, ..body)
  // in frames mode the scene label has no panel header to ride on, so it gets its
  // own line above the row
  if label != none and not path-mode {
    stack(dir: ttb, spacing: 3pt, text(size: 8pt, fill: luma(110), label), row)
  } else { row }
}

// How many scenes to draw, and which way to lay them out.  `scenes: auto` (or a
// count >= k) takes the whole set; the default 1 keeps every existing figure a
// single trajectory.  Path scenes go side by side — that is the view the k-scene
// format is for, four grids of one task in a row — while frame scenes stack,
// since one scene is already a wide row of grids.
#let _pick-scenes(s, n) = {
  let all = _scenes-of(s)
  if n == auto or n == none or n >= all.len() { all } else { all.slice(0, n) }
}
#let _scene-dir(mode, dir) = if dir != auto { dir } else if mode == "path" { ltr } else { ttb }

// Erase one value from every frame of a scene.  `posited-scenes` uses this to show
// a program-stamped object ONLY in the scene whose geometry forces the program to
// stamp it: in the obstacle family the wall stands at the same cell in all four
// scenes, but it lies on the agent's direct path in only one of them, and that one
// scene is the whole reason the wall is in the program.  Drawing it in the other
// three invites the reader to hunt for work it is not doing there.
#let _strip-value(sc, v) = (..sc, panels: sc.panels.map(p =>
  (..p, grid: p.grid.map(row => row.map(x => if x == v { 0 } else { x })))))

// How many grids one scene occupies: in the path view its frames collapse to one,
// but a template channel still rides alongside (fn_p_g), so those scenes are twice
// as wide and only two of them fit in a row.
#let _scene-width(sc, mode) = {
  let (frames, extras) = _split-panels(sc)
  if mode == "path" and frames.len() >= 2 { 1 + extras.len() } else { sc.panels.len() }
}

// Scenes per row: four single-grid scenes fit a text column, so a wider scene
// wraps onto the next row instead of running off the page.  ttb never wraps.
#let _scene-cols(chosen, mode, dir) = {
  if dir != ltr { 1 } else {
    calc.max(1, calc.floor(4 / calc.max(1, _scene-width(chosen.first(), mode))))
  }
}

// Cell size when the caller did not pick one: the path view has a single grid per
// scene to spend space on and needs room for the arrows, but four of them in a row
// must still fit a text column, so a scene SET draws smaller than a lone scene.
#let _size-for-scenes(mode, size, n) = {
  if size != auto { size }
  else if mode != "path" { 11pt }
  else if n > 1 { 15pt } else { 20pt }
}

// Horizontal placement of a rendered block.  `center` (the default everywhere) is
// what a figure wants; pass `alignment: none` to get the raw block back and place
// it yourself.  Inside a #figure this is a no-op — figures centre their body — so
// it is safe either way.
#let _placed(body, alignment) = if alignment == none { body } else { align(alignment, body) }

// Render a single sample.  `caption` is the HEADER above the grids (title + the
// mechanism in words), not a figure caption:
//   none (default) → no header, grids only — the figure's own caption says what it
//                    is, so the header would only repeat it
//   auto           → the built-in default-caption(s)
//   <content>      → use your own header instead
// For a numbered caption BELOW, wrap the whole thing in Typst's own figure:
//   #figure(scenes-figure("belief_wall"), caption: [...]) <fig-x>
// `mode` is "frames" (the grid sequence t0 … tn) or "path" (one grid, arrows for
// the transitions).  `scenes` is how many of the task's k scenes to show (auto =
// all); with more than one, each is labelled "scene i".
// `posited` names a cell value the PROGRAM stamps rather than the scene supplies:
// it is drawn hollow and hatched instead of as a solid numbered cell, and
// `posited-scenes` (1-based indices into the drawn scenes, auto = all) restricts
// which scenes draw it at all — elsewhere it is erased from the frames.
#let render-sample(
  s,
  size: auto,
  caption: none,
  gap: 12pt,
  mode: "frames",
  show-steps: false,
  scenes: 1,
  scene-dir: auto,
  scene-gap: auto,
  scene-cols: auto,
  alignment: center,
  posited: none,
  posited-scenes: auto,
) = {
  let cap = if caption == auto { default-caption(s) } else { caption }
  let chosen = _pick-scenes(s, scenes)
  let sz = _size-for-scenes(mode, size, chosen.len())
  let sdir = _scene-dir(mode, scene-dir)
  let sgap = if scene-gap != auto { scene-gap } else if sdir == ltr { 16pt } else { 9pt }
  let cols = if scene-cols != auto { scene-cols } else { _scene-cols(chosen, mode, sdir) }
  let shows-posited(i) = (
    posited == none or posited-scenes == auto or posited-scenes.contains(i + 1))
  let bodies = chosen.enumerate().map(((i, sc)) => _render-scene(
    if shows-posited(i) { sc } else { _strip-value(sc, posited) },
    size: sz, gap: gap, mode: mode, show-steps: show-steps, posited: posited,
    label: if chosen.len() > 1 { "scene " + str(i + 1) } else { none }))
  _placed(block(breakable: false, stack(dir: ttb, spacing: 5pt,
    ..if cap != none { (cap,) } else { () },
    grid(columns: cols, column-gutter: sgap, row-gutter: 9pt,
         align: top + left, ..bodies),
  )), alignment)
}

// ── colour legend ─────────────────────────────────────────────────────────────
#let legend(size: 11pt) = stack(dir: ltr, spacing: 10pt,
  ..((0, "empty"), (1, "id 1"), (2, "id 2"), (3, "wall"), (4, "id 4"), (5, "id 5"))
    .map(e => stack(dir: ltr, spacing: 4pt,
      cell(e.at(0), size: size), text(size: 8pt, fill: luma(90), e.at(1)))),
  stack(dir: ltr, spacing: 4pt,
    posited-cell(size: size), text(size: 8pt, fill: luma(90), "posited by the program")))

// ── lookups over the loaded data ──────────────────────────────────────────────
#let sample-for(kind, data: task-data) = data.samples.find(s => s.kind == kind)

// ── the two entry points ──────────────────────────────────────────────────────

// Selective figure.  Pass any number of kind names (e.g. "physics", "belief");
// pass none to take every family in data order.  `dir: ltr` lays the chosen
// instances out side by side (handy for a compact one/two-up demo figure).
// `caption` is forwarded to each sample as its HEADER (none by default, auto =
// built-in title + mechanism, or custom content applied to every chosen sample) —
// for a numbered caption below, put the result in a #figure.
// `mode: "path"` collapses each trajectory onto a single arrowed grid; `scenes`
// says how many of the task's k scenes to draw (1 = a representative one, auto =
// the whole set).  `alignment` (default center) places the result on the page;
// `none` leaves it flush left.
#let task-figure(
  ..kinds,
  data: task-data,
  dir: ttb,
  spacing: 12pt,
  show-legend: false,
  size: auto,
  caption: none,
  gap: 12pt,
  mode: "frames",
  show-steps: false,
  scenes: 1,
  scene-dir: auto,
  scene-gap: auto,
  scene-cols: auto,
  alignment: center,
  posited: none,
  posited-scenes: auto,
) = {
  let want = kinds.pos()
  let chosen = if want.len() == 0 { data.samples } else {
    want.map(k => sample-for(k, data: data)).filter(s => s != none)
  }
  // stacked ttb, each sample is centred on its own; laid out ltr they must keep
  // their natural widths, so the row is centred as a whole instead
  let per-item = if dir == ltr { none } else { alignment }
  let body = chosen.map(s => render-sample(s, size: size, caption: caption, gap: gap,
                                           mode: mode, show-steps: show-steps,
                                           scenes: scenes, scene-dir: scene-dir,
                                           scene-gap: scene-gap, scene-cols: scene-cols,
                                           alignment: per-item,
                                           posited: posited, posited-scenes: posited-scenes))
  let items = if show-legend { (_placed(legend(), per-item),) + body } else { body }
  _placed(stack(dir: dir, spacing: spacing, ..items),
          if dir == ltr { alignment } else { none })
}

// The whole scene set of one task, side by side in the path view: k trajectories
// from k different starts, all of which the ONE latent program must reproduce.
// Sugar for task-figure(kind, mode: "path", scenes: auto).
#let scenes-figure(kind, mode: "path", scenes: auto, ..args) = task-figure(
  kind, mode: mode, scenes: scenes, ..args)

// Everything, grouped and titled by role — for the appendix.
#let groups = (
  ("Minds-free substrate",                                 ("physics", "desire")),
  ("Independent extension — fork / sync, no belief",       ("overlay", "comet", "registration")),
  ("Belief — theory of mind",
   ("belief_goal", "belief_observers", "belief_wall", "belief_witness",
    "belief_false_obstacle")),
  ("Symmetric corners — one minds-free task per cube axis",
   ("flee", "deletion", "denoise", "obstacle", "relocate", "underlay",
    "perception", "multi_reg", "reg_except", "inpaint", "readout")),
  ("Pair-plumbing corners — one minds-free task per combinator complement",
   ("wipe", "composite", "drift_reg", "map_update")),
)

// All families as a flat sequence, in the curated order of `groups`.  This one is a
// CATALOGUE rather than a figure, so it keeps the two defaults task-figure drops:
// every entry gets its built-in header (nothing else here says which family it is)
// and stays flush left, its headers lining up with the running text.  Pass
// `caption: none` / `alignment: center` for the figure-style treatment.
#let all-tasks(
  data: task-data,
  spacing: 9pt,
  size: auto,
  caption: auto,
  gap: 12pt,
  mode: "frames",
  show-steps: false,
  scenes: 1,
  scene-dir: auto,
  scene-gap: auto,
  scene-cols: auto,
  alignment: none,
) = {
  for k in groups.map(g => g.at(1)).flatten() {
    let s = sample-for(k, data: data)
    if s != none {
      render-sample(s, size: size, caption: caption, gap: gap,
                    mode: mode, show-steps: show-steps, scenes: scenes,
                    scene-dir: scene-dir, scene-gap: scene-gap, scene-cols: scene-cols,
                    alignment: alignment)
      v(spacing)
    }
  }
}

// ── standalone preview (discarded when this file is imported) ──────────────────
#set page(paper: "a4", margin: 1.6cm)
#set text(size: 10pt, font: "New Computer Modern")
#set par(justify: false)

#align(center, text(size: 15pt, weight: "bold", "Task families"))
#align(center, text(size: 9pt, fill: luma(120),
  "one representative scene each — " + str(task-data.size) + "×" + str(task-data.size) + " grids"))
#v(2pt)
#align(center, legend())
#v(6pt)
#line(length: 100%, stroke: 0.4pt + luma(200))

#all-tasks()

#pagebreak()
#align(center, text(size: 15pt, weight: "bold", "Task families — collapsed path view"))
#align(center, text(size: 9pt, fill: luma(120),
  "the same trajectories on one grid, an arrow per transition"))
#v(2pt)
#align(center, legend())
#v(6pt)
#line(length: 100%, stroke: 0.4pt + luma(200))

#all-tasks(mode: "path", show-steps: true)

#pagebreak()
#align(center, text(size: 15pt, weight: "bold", "Task families — the whole scene set"))
#align(center, text(size: 9pt, fill: luma(120),
  "every scene of one task per family, path view: one latent program, "
  + "k different starts, all of which a solution must reproduce"))
#v(2pt)
#align(center, legend())
#v(6pt)
#line(length: 100%, stroke: 0.4pt + luma(200))

#all-tasks(mode: "path", scenes: auto, spacing: 11pt)
