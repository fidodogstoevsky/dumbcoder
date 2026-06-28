#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Bayesian Model")

search over hypothesis space

== DreamCoder

=== Explore

=== Compress

=== Dream

- enumerate typed program trees by probability 
- enumeration weighed by description length
- goal is MDL
- wake/sleep library learning by compression
- I'm doing inductive synthesis on I/O pairs, not deductive

types
- I've got a simply-typed and monomorphic DSL
- the types are only used to prune enumeration
- so they just constrain which wirings are legal
- they don't constrain what the program must prove
- no generic composer
- type-pruned, example-driven, compression-based synthesis

goal:
- show that "belief"/ToM can be a discoverable compound under MDL rather than a built-in primitive
- 