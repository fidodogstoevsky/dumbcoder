
#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Further work")

== Polymorphism

We use a simply-typed monomorphic type system. For our purporses, types are really used only to prune enumeration.They constrain which compositions are legal, not what the program must prove. DreamCoder uses polymorphic types, so unification can rule out ill-typed branches early. But since our DSL has only $tilde$30 primitives our bottleneck isn't branching factor, so monomorphic dict lookup is sufficient. 

With monomorphic types, a stitch abstraction discovered at type `grid` can only be reused at `grid`. With polymorphism, an abstraction abstracted over $forall a$ can be reused across instantiations of any type $a$. The grid-stack of @sec-arity would have to hand-engineer this, with a cons/nil constructor and a depth-polymorphic lift written out for each pair operation; if we had polymorphism it would just fall out of the type discipline.

To do something like that, we'd replace bare string types with a small ADT:

- ground `('con', 'grid', [])`
- variable `('var', 'a')`
- applied `('con', 'pair', [t1, t2])`
- arrows `(args, ret)`

We'd implement Robinson unification

== Channel arity as a free parameter <sec-arity>

The product $G times G$ of Types carries a commitment we should not be making on the learner's behalf. It says that a learner holding a private representation holds _exactly one_, and a one-model frame is precisely the kind of thing a theory of mind is supposed to have. If the answer to "why one world and one model, and not three?" is "because the type system says so", then part of the structure at issue has been fixed by the modeller rather than found by the learner. @sec-limits lists this as a limit of the results; this section says what removing it would take.

The design is a further weakening of the endowment of @primitives --- strictly less given, on the same corpus, with the same objective. The fixed pair is replaced by a recursive grid-stack,

```
gstack ::= () | (grid, *gstack)
```

with `nil` and `cons` as constructors, and every pair operation is replaced by a depth-polymorphic lift of itself that acts on the top of the stack. The type system of Types gains the arrows $G arrow.r G^*$, $G^* arrow.r G^*$ and $G^* arrow.r G$ to push, transform and project, and loses the arrows into and out of $G times G$ that fixed the arity at two. At depth 1 the lifts reproduce `fork` and `sync_to_world` exactly, so nothing expressible under either endowment of @primitives is lost; but the number of private channels a program opens is now something search selects rather than something the type system fixes.

The same monomorphism is also why the language needs a separate composer at each arrow of Types. The pieces belief is assembled from --- copying a grid, altering one channel of a pair, and composing two functions --- are defined at any objects, not just at $G$, so the four composers of @primitives are four copies of one operation the type system cannot state once. Nothing in the argument turns on this, but a language with polymorphic plumbing would collapse them into a single symbol, and the wiring belief needs would be that much less of the endowment.

The measurement is then per family: for each solved task, read off the maximum stack depth its program reaches, and compare that against the depth the family's ground-truth program uses. The corpus would need at least one non-mental family whose ground truth genuinely requires depth 2 --- a cross-channel blur, say --- so that arity 2 is known to be both expressible and selectable, and the physics and desire families should sit at arity 0, holding no private channel at all. The prediction the thesis's argument makes is that no family, mental or not, selects more than one private channel except the one whose ground truth demands two; and the quantity that would carry it is not the observation but the margin, the description length of the arity-1 encoding of a belief scene against the arity-2 encoding of the same scene, under base primitives and under the learned library.

If that came out, belief's single-model frame would be a discovery rather than a stipulation, and the last structural commitment in Types would be gone. If it did not --- if compression started stacking channels wherever the grammar allowed it --- what the runs of Chapter 5 found would still be a belief abstraction, but the claim that the learner also found its _frame_ would have to be withdrawn.

== persistent models and interaction <sec-persistent>

currently an attributed grid state, a belief, exists only for that one frame, since we're just working with grid to grid transition functions. so do something where the agent actually has a persistent memory of its modified grid

and something where the agent can actually react to its environment, can write to the private model and read from it, can navigate based on obstacles, etc

== breaking down optimize <sec-optimize>

this primitive packs in a lot, it's a full bfs search. so try to rediscover that. this requires a persistent model (to keep track of visited cells) and a and interactions with the environemtn, conditionals (if wall, move away)

== probabilistic primitives

an agent that sometimes does this and sometimes that, etc

Go from a LoT to a PLoT

== no sequencing



== vs. real world

obviously this isn't how an infant learns theory of mind

I'm missing the _self_. I'm focusing on learning to distinguish between different entities in the domain, but not an infant learning to distinguish others from itself. For example, if I bite myself it hurts, but if I bite someone else it doesn't. maybe pain occurs only if I bite this but not that (extensional). or we have different states, different experiences, and sometimes someone else can be experiencing pain even when I'm not. 

I'm missing the scale. I'm focusing on a narrow timescale. in real infants this would unfold over a long time scale. 

== richer belief content

== recursive mindreading

== POMDP

observation function



#load-bib(read("refs.bib"))