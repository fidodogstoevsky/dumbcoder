
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

=== What is claimed, and what is not

Fixing the level fixes the scope of the claim, and it is worth being explicit about what falls outside it. We are not studying how a child in fact comes to learn theory of mind. That is the province of developmental and cognitive psychology, and nothing here should be read as a proposal about neural implementation or about the moment-to-moment course of a particular child's development.

Nor is the claim that children run the algorithm presented in @sec-dreamcoder. ECD, the wake-sleep library-learning system described there and implemented in Chapter 4, is offered as an algorithmic-level _realisation_ of the computational-level claim: a demonstration that the problem, so stated, is solvable by a mechanism that begins with no mental vocabulary at all. Its role in the argument is one of sufficiency rather than fidelity. If a learner equipped only with domain-general primitives and a preference for short programs comes to posit belief-like structure, then positing such structure is something a learner _can_ do rather than something it must be given --- and the nativist's inference from "no mechanism has been named" to "no mechanism exists" is blocked. Whether human learners use this mechanism, or some other one that solves the same problem, is a further question that this thesis does not settle.

There is a caveat to the tidy division of levels, and it matters for @sec-hbm. Many early Bayesian models addressed the computational level alone, characterising cognition as approximately optimal statistical inference in a fixed environment without reference to how the computation is carried out. The hierarchical models discussed later sit somewhere between the computational and the algorithmic: they describe cognition as approximately optimal inference in probabilistic models defined over a learner's subjective and dynamically growing mental representations of the world's structure, rather than over some objective and fixed world statistics @tenenbaum_how_2011. Once the hypothesis space is itself something the learner builds and rebuilds, the question of what the learner is doing can no longer be cleanly separated from the question of what it is doing it _with_. That entanglement is not a defect of the account; it is the part of the account that does the developmental work.

// --- source notes for this section, retained ---
// - identify the computational problem that the mind is trying to solve: the problem is find
//   the best explanation. we argue that best is determined logically by bayes' rule. so it's
//   doing bayesian updating. then the algorithmic level is the procedure at which it's
//   produced, ECD
// - engineering: building AI, building algorithms that solve computational problems
// - reverse-engineering: identify the abstract computational problems that the mind must
//   solve to do what it does. knowing that the mind solves these certain problems, what are
//   the AI systems we have that can solve problems like that. Then we can model the mind as
//   that kind of system.
// - bayesian models focus on the computational level, the logic of the problem, what the
//   information processing system is doing. [Marr 1982]
// - rational analysis framework. analyze cognition in terms of adaptive solutions to
//   environmental problems [Anderson 1990]  <- NOT YET IN chapter2.bib


#load-bib(read("chapter1.bib") + read("chapter2.bib"))