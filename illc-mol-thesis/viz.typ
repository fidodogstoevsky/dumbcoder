// Task-family visualisations — a small library + a standalone preview.
//
// USE AS A LIBRARY (selective figures in your paper):
//   #import "viz.typ": task-figure, all-tasks
//   #figure(task-figure("physics", "belief"), caption: [...])   // early demo
//   #all-tasks()                                                // appendix: everything
//
// COMPILE STANDALONE (preview every family):
//   typst compile viz.typ
//
// Importing only pulls the #let bindings; the preview at the bottom of this file
// is discarded in the importing document, so it is safe to `#import`.
//
// DATA: produced by file16.export_task_samples (run `python file16.py --cube
// --samples`).  Each sample is {kind, tag, T, panels:[{label, grid}]}; fn families
// list successive `unfold` frames, fn_p_g families list world | template | result.

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
)

#let cell(v, size: 11pt) = box(
  width: size, height: size,
  radius: 1.2pt, inset: 0pt,
  fill: palette.at(str(v), default: rgb("#cccccc")),
  stroke: 0.3pt + rgb("#c2c2c8"),
  align(center + horizon,
    if v != 0 { text(size: size * 0.55, fill: white, weight: "bold", str(v)) }),
)

#let render-grid(g, size: 11pt) = grid(
  columns: (size,) * g.at(0).len(),
  rows: (size,) * g.len(),
  gutter: 1pt,
  ..g.flatten().map(v => cell(v, size: size)),
)

// a labelled grid (one frame / channel)
#let panel(p, size: 11pt) = stack(dir: ttb, spacing: 3pt,
  text(size: 8pt, fill: luma(110), p.label),
  render-grid(p.grid, size: size),
)

// ── per-kind captions: human title + ground-truth program ─────────────────────
#let info = (
  physics:      ("Physics — a body moves",            "(step v d)"),
  desire:       ("True belief — seek a goal",              "(optimize (neg_dist gv) av)"),
  overlay:      ("Overlay — motion blur",             "(fork (step v d) overlay)"),
  registration: ("Registration — snap one object",    "(sync_to_world v)"),
  belief:       ("False belief — act on a stale map",
                 "(fork (compose (wall_at r c) (optimize (neg_dist gv) av)) (sync_to_world av))"),
  flee:         ("Flee — avoid a hazard",             "(optimize (distance hv) av)"),
  deletion:     ("Deletion — punch one hole",         "(clear_at r c)"),
  denoise:      ("Denoise — drop a noise value",      "(erase nv)"),
  perception:   ("Perception — record into the map",  "(sync_to_model v)"),
  multi_reg:    ("Multi-registration — snap all",     "sync_all"),
  reg_except:   ("Registration-except — all but one", "(sync_except a)"),
  inpaint:      ("Inpainting — fill holes, keep pixels", "underlay"),
  readout:      ("Readout — report the model",        "snd_gg"),
)

// The built-in caption for a sample: bold title + [kind], ground-truth program,
// and the per-instance tag.  Exposed so callers can build on / tweak it.
#let default-caption(s) = {
  let meta = info.at(s.kind, default: (s.kind, ""))
  stack(dir: ttb, spacing: 5pt,
    stack(dir: ltr, spacing: 6pt,
      text(weight: "bold", meta.at(0)),
      text(fill: luma(150), size: 8pt, "[" + s.kind + "]")),
    raw(meta.at(1)),
    ..if s.tag != "" { (text(size: 8pt, fill: luma(120), s.tag),) } else { () },
  )
}

// Render a single sample.  `caption` is optional:
//   auto (default) → the built-in default-caption(s)
//   none           → no caption, grids only
//   <content>      → use your own caption instead
#let render-sample(s, size: 11pt, caption: auto, gap: 12pt) = {
  let cap = if caption == auto { default-caption(s) } else { caption }
  block(breakable: false, stack(dir: ttb, spacing: 5pt,
    ..if cap != none { (cap,) } else { () },
    stack(dir: ltr, spacing: gap, ..s.panels.map(p => panel(p, size: size))),
  ))
}

// ── colour legend ─────────────────────────────────────────────────────────────
#let legend(size: 11pt) = stack(dir: ltr, spacing: 10pt,
  ..((0, "empty"), (1, "id 1"), (2, "id 2"), (3, "wall"), (4, "id 4"), (5, "id 5"))
    .map(e => stack(dir: ltr, spacing: 4pt,
      cell(e.at(0), size: size), text(size: 8pt, fill: luma(90), e.at(1)))))

// ── lookups over the loaded data ──────────────────────────────────────────────
#let sample-for(kind, data: task-data) = data.samples.find(s => s.kind == kind)

// ── the two entry points ──────────────────────────────────────────────────────

// Selective figure.  Pass any number of kind names (e.g. "physics", "belief");
// pass none to take every family in data order.  `dir: ltr` lays the chosen
// instances out side by side (handy for a compact one/two-up demo figure).
// `caption` is forwarded to each sample (auto = built-in, none = omit, or custom
// content applied to every chosen sample).
#let task-figure(
  ..kinds,
  data: task-data,
  dir: ttb,
  spacing: 12pt,
  show-legend: false,
  size: 11pt,
  caption: auto,
  gap: 12pt,
) = {
  let want = kinds.pos()
  let chosen = if want.len() == 0 { data.samples } else {
    want.map(k => sample-for(k, data: data)).filter(s => s != none)
  }
  let body = chosen.map(s => render-sample(s, size: size, caption: caption, gap: gap))
  let items = if show-legend { (legend(),) + body } else { body }
  stack(dir: dir, spacing: spacing, ..items)
}

// Everything, grouped and titled by role — for the appendix.
#let groups = (
  ("Minds-free substrate",                           ("physics", "desire")),
  ("Independent extension — fork / sync, no belief", ("overlay", "registration")),
  ("Belief — theory of mind",                        ("belief",)),
  ("Symmetric corners — one minds-free task per cube axis",
   ("flee", "deletion", "denoise", "perception",
    "multi_reg", "reg_except", "inpaint", "readout")),
)

// All families as a flat sequence (no group headings), each with its own caption,
// in the curated order of `groups`.
#let all-tasks(
  data: task-data,
  spacing: 9pt,
  size: 11pt,
  caption: auto,
  gap: 12pt,
) = {
  for k in groups.map(g => g.at(1)).flatten() {
    let s = sample-for(k, data: data)
    if s != none {
      render-sample(s, size: size, caption: caption, gap: gap)
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
  "one example trajectory each — " + str(task-data.size) + "×" + str(task-data.size) + " grids"))
#v(2pt)
#align(center, legend())
#v(6pt)
#line(length: 100%, stroke: 0.4pt + luma(200))

#all-tasks()
