#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Results and Discussion")

== Results

=== MDL-margin distribution

(phases 1 - 2).For every solved belief task, price the found program and each rival spelling under the final library.

*Plot*: histogram/CDF of margin (nats), one curve per belief variant. One figure kills the "maybe a non-mental program was almost as short" objection for the whole corpus.

=== Corpus-DL trajectory

Per ECD round: total description length of all solutions under the current library, plus library cost. 

*Plot*: total DL vs round (should fall), stacked or faceted by family — belief should show the steepest drop at the round fn_8 appears. This is the DreamCoder-style figure examiners will recognize.

=== Solve dynamics

*Plot 1*: cumulative solves vs round, one line per family (belief's S-curve vs the flat-at-ceiling non-mental families).

*Plot 2*: per-task solve time, round n vs round n+1, as a paired slope-graph or log-scale scatter — the 1100 s → 10 s collapse is currently buried in log text.

=== Per-variant belief coverage + combo

*Plot 1:* Bar chart of solved/total for wall/witness/goal/dual

*Plot 2*: heatmap of (gv,av) $times$ abstraction-used showing fn_8 tiling every cell.

=== Silo vs joint stitch

compress belief solutions alone vs pooled with the non-mental corpus. Is the constructor still found, and at what DL? Quantifies the file-16 claim that belief is an MDL win, not a silo artefact. Plot: library-DL and constructor-invention bar pairs.

== Discussion

out of the full 2×2×… cube of channel-direction / channel-target / z-order / scope variants, joint MDL still selects exactly the read-model→write-world, single-av corner for belief, and leaves the symmetric variants attached to their non-mental tasks.

pair apparatus was available to every task, but recruited by only the false belief ones. Those are the trajectories that can't be accounted for by only a single grid, so the search pays +~10 nats for the second grid only when nothing cheaper explains the data. Representationalist's criterion: posit the hidden world only if it pays for itself, only if its necessary. Availability is not use. The inference "this agent needs a belief" is made, per task, by cost-driven model selection. 

The `(grid,grid)` and arrow types aren't smuggled-in ToM, they're just the logical form that ToM requires. Intensionality is the coexistence of world and model. Any substrate expressive enough to even state "act on the world via a transformed model of it" must be able to hold two representations at once. And "two things held at once" is a product, in some guise (a pair, two registers, two tape regions, two variables). You cannot discover ToM in a substrate that can't hold two representations [for the same reason you can't discover addition in a language with no notion of "two numbers." The pair is to theory of mind what the integers are to arithmetic.]
And the pair is symmetric and content-free: `dup` makes `(w, w)`. There's nothing in it that says "the second one is the model, the believed world." The asymmetry that makes one copy real and the other believed is: transform the second (`mapsnd`, not `mapfst`), commit from model to world (`sync_to_world`). This asymmetry is entirely in the composition, which is the discovered part. The medium is neutral, it could go any way, as seen in the other tasks which do use those other primitives and wire them in other ways. But the system is able to discover this particular composition of these particular primitives to solve belief tasks, via programs that constitute a theory of mind.

Sure you can keep going. Decompose `dup` from a flat memory/tape with allocate/read/write, so "hold a second world-model" becomes a discovered pattern of buffer use. But be clear-eyed: that relocates the regress, it doesn't end it. "Allocate a buffer" is then the primitive, and it's no more or less mental than "form a pair" — the capacity-for-plurality is presupposed by intensionality at every level, because it's constitutive of intensionality. There is no substrate that discovers ToM "from nothing"; there's only substrates whose primitives are, or aren't, individually non-mental and general.

generality by reuse is what shows it
1. the same combinators used to create the belief abstraction are used in non-mental tasks
2. the library is also initialized with symmetric primitives to the ones used in the belief abstraction

so it's a domain-general library that the search happened to recruit for belief. 

== vs. real world

obviously this isn't how an infant learns theory of mind

I'm missing the _self_. I'm focusing on learning to distinguish between different entities in the domain, but not an infant learning to distinguish others from itself. For example, if I bite myself it hurts, but if I bite someone else it doesn't. maybe pain occurs only if I bite this but not that (extensional). or we have different states, different experiences, and sometimes someone else can be experiencing pain even when I'm not. 

I'm missing the scale. I'm focusing on a narrow timescale. in real infants this would unfold over a long time scale. 

== setup

to what extent is the ToM already present in the input data and DSL and structure

== Further work

Polymorphism