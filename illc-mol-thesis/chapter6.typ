
#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Further work")

== Comparison to LLMs

I'm doing search in program-dynamics space, not token space. Synthesizing programs is logical and semantic discovery, unlike token prediction.  ECD enumerates typed program trees by probability and runs them on grids; there is no surface-syntax statistics anywhere.

== Polymorphism

We use a simply-typed monomorphic type system. For our purporses, types are really used only to prune enumeration.They constrain which compositions are legal, not what the program must prove. DreamCoder uses polymorphic types, so unification can rule out ill-typed branches early. But since our DSL has only $tilde$30 primitives our bottleneck isn't branching factor, so monomorphic dict lookup is sufficient. 

With monomorphic types, a stitch abstraction discovered at type `grid` can only be reused at `grid`. With polymorphism, an abstraction abstracted over $forall a$ can be reused across instantiations of any type $a$. Phase 3 hand-engineers this with t eh cons/nil stack leaving arity as a free parameter. If we had polymorphism it would just fall out of the type discipline. 

To do something like that, we'd replace bare string types with a small ADT:

- ground `('con', 'grid', [])`
- variable `('var', 'a')`
- applied `('con', 'pair', [t1, t2])`
- arrows `(args, ret)`

We'd implement Robinson unification 

== persistent models and interaction

currently an attributed grid state, a belief, exists only for that one frame, since we're just working with grid to grid transition functions. so do something where the agent actually has a persistent memory of its modified grid

and something where the agent can actually react to its environment, can write to the private model and read from it, can navigate based on obstacles, etc

== breaking down optimize

this primitive packs in a lot, it's a full bfs search. so try to rediscover that. this requires a persistent model (to keep track of visited cells) and a and interactions with the environemtn, conditionals (if wall, move away)

== probabilistic primitives

an agent that sometimes does this and sometimes that, etc

== vs. real world

obviously this isn't how an infant learns theory of mind

I'm missing the _self_. I'm focusing on learning to distinguish between different entities in the domain, but not an infant learning to distinguish others from itself. For example, if I bite myself it hurts, but if I bite someone else it doesn't. maybe pain occurs only if I bite this but not that (extensional). or we have different states, different experiences, and sometimes someone else can be experiencing pain even when I'm not. 

I'm missing the scale. I'm focusing on a narrow timescale. in real infants this would unfold over a long time scale. 

== richer belief content

== recursive mindreading

#load-bib(read("chapter1.bib") + read("chapter2.bib"))