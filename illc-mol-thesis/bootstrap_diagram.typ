// The bootstrap diagram — how the library of the combinator run was assembled.
//
//   #import "bootstrap_diagram.typ": bootstrap-diagram
//   #figure(bootstrap-diagram(), caption: [...]) <fig-bootstrap>
//
// One box per abstraction, placed in the column of the round whose compression
// step invents it.  The counts inside a box are the solutions that pay for the
// term THAT round (belief · non-mental, coloured as in the corpus-DL figure);
// the column header carries the cumulative solve counts the search phase feeds
// the round.  A solid arrow says the right-hand term is built on the left-hand
// one — its body instantiates it, and the search that found the solutions
// paying for it enumerated with it as an atom.  Dashed edges are the two
// weaker relations: the single-value commit fills the open constructor's
// commit hole (that is its belief use), and the early seek-and-edit policies
// are re-cut into the seek block.  Exact per-round counts: the appendix table.

#let _belief = rgb("#2a78d6")     // the corpus-DL figure's belief blue
#let _nonmental = rgb("#7a5bd0")  // its pooled non-mental violet
#let _ink = luma(70)
#let _muted = luma(140)

// "b · n" with the two counts in their family colours
#let _counts(b, n, size: 8pt) = text(size: size)[
  #text(fill: _belief, weight: "bold", str(b))
  #text(fill: _muted, " · ")
  #text(fill: _nonmental, weight: "bold", str(n))
]

// a term box: name above, its invention-round counts below
#let _tbox(name, b, n, w, h) = box(
  width: w, height: h, inset: (x: 4pt, y: 0pt), radius: 2.5pt,
  fill: luma(250), stroke: 0.6pt + luma(170),
  align(center + horizon, stack(dir: ttb, spacing: 3.5pt,
    par(leading: 3pt, text(size: 8pt, fill: _ink, name)),
    _counts(b, n))),
)

// the ghosted "later re-cut" box
#let _mbox(name, w, h) = box(
  width: w, height: h, inset: (x: 4pt, y: 0pt), radius: 2.5pt,
  stroke: (paint: luma(190), thickness: 0.6pt, dash: "dashed"),
  align(center + horizon, par(leading: 3pt, text(size: 7.5pt, fill: _muted, name))),
)

#let _arrow(from, to, dashed: false) = {
  let (x1, y1) = from
  let (x2, y2) = to
  let (dx, dy) = (x2 - x1, y2 - y1)
  let len = calc.sqrt(calc.pow(dx / 1pt, 2) + calc.pow(dy / 1pt, 2)) * 1pt
  let (ux, uy) = (dx / len, dy / len)
  let head = 4.5pt
  let paint = if dashed { luma(160) } else { luma(110) }
  place(top + left, line(
    start: from, end: (x2 - ux * head, y2 - uy * head),
    stroke: (paint: paint, thickness: 0.8pt, cap: "round",
             ..if dashed { (dash: "dashed") } else { (:) })))
  let ang = calc.atan2(dx / 1pt, dy / 1pt)
  place(top + left, dx: x2 - ux * head / 2, dy: y2 - uy * head / 2,
    box(width: 0pt, height: 0pt, place(center + horizon, rotate(ang,
      polygon(fill: paint, stroke: none,
        (0pt, 0pt), (0pt, head), (head, head / 2))))))
}

#let bootstrap-diagram() = {
  let (w1, w2, w3) = (118pt, 132pt, 118pt)      // column widths
  let (x1, x2, x3) = (0pt, 158pt, 330pt)        // column left edges
  let W = x3 + w3
  let H = 196pt

  // y of each box top; heights are fixed so arrow endpoints are known
  let hdr = 0pt
  let frame-y = 34pt;  let frame-h = 36pt       // derive-and-commit frame
  let commit-y = 80pt; let commit-h = 36pt      // single-value commit
  let m1-y = 126pt;    let m1-h = 24pt          // + 4 more, re-cut
  let pw-y = 34pt;     let pw-h = 36pt          // phantom-wall
  let dg-y = 80pt;     let dg-h = 36pt          // displaced-goal
  let oc-y = 126pt;    let oc-h = 28pt          // open constructor
  let m2-y = 164pt;    let m2-h = 24pt          // two-mover, re-cut
  let ta-y = 80pt;     let ta-h = 36pt          // two-actor
  let sb-y = 138pt;    let sb-h = 28pt          // seek block

  let header(x, w, round, b, n) = place(top + left, dx: x, dy: hdr,
    box(width: w, align(center, stack(dir: ttb, spacing: 3pt,
      text(size: 8.5pt, fill: _ink, weight: "bold", round),
      { text(size: 7.5pt, fill: _muted, "solved  "); _counts(b, n, size: 7.5pt) }))))

  let at(x, y, body) = place(top + left, dx: x, dy: y, body)
  let mid-r(x, w, y, h) = (x + w, y + h / 2)    // right-edge midpoint
  let mid-l(x, y, h) = (x, y + h / 2)           // left-edge midpoint

  box(width: W, height: H, {
    header(x1, w1, "round 1", 0, 148)
    header(x2, w2, "round 2", 75, 148)
    header(x3, w3, "round 3", 107, 148)

    at(x1, frame-y, _tbox([derive-and-commit\ frame], 0, 12, w1, frame-h))
    at(x1, commit-y, _tbox([single-value\ commit], 0, 12, w1, commit-h))
    at(x1, m1-y, _mbox([\+ 4 more abstractions,\ later re-cut], w1, m1-h))

    at(x2, pw-y, _tbox([phantom-wall\ constructor], 24, 0, w2, pw-h))
    at(x2, dg-y, _tbox([displaced-goal\ constructor], 22, 0, w2, dg-h))
    at(x2, oc-y, _tbox([open constructor], 9, 0, w2, oc-h))
    at(x2, m2-y, _mbox([\+ a two-mover abstraction,\ later re-cut], w2, m2-h))

    at(x3, ta-y, _tbox([two-actor\ constructor], 24, 0, w3, ta-h))
    at(x3, sb-y, _tbox([seek block], 23, 47, w3, sb-h))

    // the frame is instantiated by every private-copy constructor
    _arrow(mid-r(x1, w1, frame-y, frame-h), mid-l(x2, pw-y, pw-h))
    _arrow(mid-r(x1, w1, frame-y, frame-h), mid-l(x2, dg-y, dg-h))
    _arrow(mid-r(x1, w1, frame-y, frame-h), (x2, oc-y + oc-h / 2 - 5pt))
    // the commit fills the open constructor's commit hole
    _arrow(mid-r(x1, w1, commit-y, commit-h), (x2, oc-y + oc-h / 2 + 5pt), dashed: true)
    // the two-actor constructor is a seek composed onto the displaced-goal one
    _arrow(mid-r(x2, w2, dg-y, dg-h), mid-l(x3, ta-y, ta-h))
    // the early policies are re-cut into the seek block
    _arrow(mid-r(x2, w2, m2-y, m2-h), mid-l(x3, sb-y, sb-h), dashed: true)
  })
}
