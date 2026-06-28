// Re-exports the illc-mol-thesis package, overriding `mol-chapter` so the
// chapter title renders as "Chapter N: Name" on a single line.
#import "@preview/illc-mol-thesis:0.2.0": *

#let mol-chapter(body) = [
  #pagebreak()
  #hide(
    heading(body,
      hanging-indent: 0pt,
      level: 1,
      supplement: [Chapter])
  )
  #text(size: 24pt, weight: "bold")[
    #set par(first-line-indent: 0pt)
    Chapter #context counter(heading).display(): #body
  ]
]
