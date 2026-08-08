#import "@preview/illc-mol-thesis:0.2.0": *

#show: mol-thesis

// The illc-mol-thesis template forces `cite(style: "alphanumeric")` and
// `bibliography(style: "elsevier-vancouver")`, which renders citations as
// [GCT24]. These two lines override that with author-date. Both must be set:
// the template's explicit cite style would otherwise win over the bibliography's.
#set cite(style: "apa")
#set bibliography(style: "apa")

#mol-titlepage(
  title: "Learning to Theorize in Mental Terms",
  author: "Gidon Kaminer",
  birth-date: "March 30st, 2000",
  birth-place: "New York, United States",
  defence-date: "August 28th, 2026",
  supervisors: ("Dr Fausto Carcassi",),
  committee: (
    "Dr Fausto Carcassi (supervisor)",
    "Dr Malvin Gattinger (chair)",
    "Dr Martha Lewis",
    "Dr Giorgio Sbardolini"),
  degree: "MSc in Logic"
)

#mol-abstract[
We study the acquisition of Theory of Mind.
]

#pagebreak()

#outline()
#include "chapter1.typ"
#include "chapter2.typ"
#include "chapter3.typ"
#include "chapter4.typ"
#include "chapter5.typ"
#include "chapter6.typ"
#include "chapter7.typ"
#include "appendix.typ"
#pagebreak()

#load-bib(read("refs2.bib"), main: true)