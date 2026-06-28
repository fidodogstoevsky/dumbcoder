#import "@preview/illc-mol-thesis:0.2.0": *

#show: mol-thesis

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
  ABSTRACT OF THE THESIS
  
  #lorem(150)
]

#outline()
#include "chapter1.typ"
#include "chapter2.typ"
#include "chapter3.typ"
#include "chapter4.typ"
#include "chapter5.typ"
#pagebreak()

#load-bib(read("references.bib"), main: true)
