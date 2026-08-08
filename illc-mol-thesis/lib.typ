// Re-exports the illc-mol-thesis package, overriding `mol-chapter` so the
// hidden level-1 heading can carry a label. The package's version returns a
// content *sequence*, so `#mol-chapter("Results") <ch-results>` fails with
// "cannot reference sequence"; passing the label in attaches it to the heading
// itself, which is what `@ch-results` resolves against ("Chapter 5").
//
//   #mol-chapter("Results", lbl: <ch-results>)   ... then ... @ch-results
//
// Rendering is identical to the package's `mol-chapter`.
#import "@preview/illc-mol-thesis:0.2.0": *

#let mol-chapter(body, lbl: none) = [
  #pagebreak()
  #hide[
    #heading(body,
      hanging-indent: 0pt,
      level: 1,
      supplement: [Chapter])#lbl
  ]
  #text(size: 28pt, weight: "bold")[
    #set par(first-line-indent: 0pt)
    #text(size: 24pt, [Chapter #context counter(heading).display()])

    #body]
]
