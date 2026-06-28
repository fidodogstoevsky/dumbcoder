// world_tape.typ — render generated tasks as a timeline of world-states.
//
// A "task" is a single world evolving over time: a sequence of 2D integer
// matrices (frames), drawn as grids of colored cells laid out left→right.
//
//   #import "world_tape.typ": world-tape, grid-view, arc-colors
//   #world-tape((t0, t1, t2))

// ── Digit → color palette (ARC standard) ──────────────────────────
#let arc-colors = (
  rgb("#000000"), // 0 black
  rgb("#0074D9"), // 1 blue
  rgb("#2ECC40"), // 2 green
  rgb("#AAAAAA"), // 3 grey
  rgb("#FF4136"), // 4 red
  rgb("#FFDC00"), // 5 yellow
  rgb("#F012BE"), // 6 magenta
  rgb("#FF851B"), // 7 orange
  rgb("#7FDBFF"), // 8 cyan
  rgb("#870C25"), // 9 maroon
)

// ── One matrix → grid of colored cells ────────────────────────────
// `matrix` is an array of rows, each row an array of ints 0–9.
#let grid-view(matrix, cell: 16pt, gridline: 0.5pt + rgb("#333"), digits: false) = {
  let cols = if matrix.len() > 0 { matrix.at(0).len() } else { 0 }
  grid(
    columns: (cell,) * cols,
    rows: (cell,) * matrix.len(),
    ..matrix.flatten().map(d => rect(
      width: cell, height: cell, inset: 0pt,
      fill: arc-colors.at(d),
      stroke: gridline,
    )[
      #if digits {
        set align(center + horizon)
        // contrast: light digits on dark cells, dark on light
        set text(size: cell * 0.5, fill: if d in (0, 1, 9) { white } else { black })
        [#d]
      }
    ])
  )
}

// ── A task = one world-state timeline laid out horizontally ───────
// `frames` is an array of matrices; frames.at(0) = state at t0, etc.
#let world-tape(
  frames,
  cell: 16pt,
  gap: 10pt,
  arrow: false,          // draw a flow arrow between frames
  step-labels: true,    // label each frame t0, t1, …
  start: 0,             // first time index
  title: none,          // optional header shown above the grids
  desc: none,           // optional description shown under the header
  ..args,               // forwarded to grid-view (gridline, digits)
) = block(
  fill: rgb("#f6f8fa"),
  stroke: 0.5pt + rgb("#d0d7de"),
  radius: 10pt,
  inset: gap,
  {
  let sep = if arrow {
    align(horizon, text(size: 14pt, fill: rgb("#888"))[#sym.arrow.r])
  } else { none }

  let views = frames.enumerate().map(((i, f)) => {
    let g = grid-view(f, cell: cell, ..args)
    if step-labels {
      stack(dir: ttb, spacing: 4pt,
        align(center + top, g),
        align(center, text(size: 8pt, fill: rgb("#555"))[$t_#(start + i)$]),
      )
    } else { g }
  })

  // interleave separators between frames
  let parts = ()
  for (i, v) in views.enumerate() {
    if i > 0 and sep != none { parts.push(sep) }
    parts.push(v)
  }
  let tape = block(stack(dir: ltr, spacing: gap, ..parts.map(p => align(top, p))))

  // optional header + description above the tape
  let head = ()
  if title != none {
    head.push(text(size: 11pt, weight: "bold", fill: rgb("#24292f"))[#title])
  }
  if desc != none {
    head.push(text(size: 9pt, fill: rgb("#555"))[#desc])
  }
  if head.len() > 0 {
    stack(dir: ttb, spacing: 4pt, ..head, v(gap - 4pt), tape)
  } else {
    tape
  }
})
