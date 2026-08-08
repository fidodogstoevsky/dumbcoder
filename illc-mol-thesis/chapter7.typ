#import "lib.typ": *

#mol-chapter("Conclusion", lbl: <ch-conclusion>)

In this thesis we studied the acquisition of Theory of Mind. In Chapter 1 we surveyed the developmental literature, where we identified belief attribution as a core capacity of ToM which is much studied experimentally. We then compared theoretical perspectives of ToM acqusition and argued that rational constructivism, which models ToM development as theory change, is the most compelling among the accounts. 

Rational constructivism posits that domain-specific conceptual vocabulary is built by domain-general inference over structured hypotheses, guided by a preference for simpler hypotheses (@constructivism). We identified that the nativist's strongest reply to the constructivist is to deny that mechanisms for learning structured domains can exist. So we set out to answer this charge by counterexample, via a model that can learn the structure of ToM. 

In Chapter 2 we explained the learning framework that we chose to base our model in. We justified our choice of representing hypotheses as programs, and showed how we can use Bayesian inference to synthesize programs, by taking the prior over them is the measure of their simplicity. We showed that similar inference methods can be used to learn the library itself, compressing recurring structure into new abstractions and adding them to the library. 

In Chapter 3 we surveyed BToM as the best available model of the mature capacity for ToM. We showed how an observer uses inverse planning to infer an agent's belief, inferring under which falsely believed model an agent's irrational behavior would be rational. We then identified the structure that a BToM modeler ordinarily stipulates (@stipulated), so that we could ensure that none of those features are smuggled into our own setup. 

In Chapter 4 we introduced our model of a learner that isn't endowed with the stipulations given in BToM. We created a corpus of grid-based tasks among which some require belief attribution. We gave our learner a primitive library of a planner, a symmetric pair type, and general-purpose functions over pairs. We showed how the learner could in principle compose its domain-general primitives into programs that attribute beliefs. And we spelled out the criteria under which our learner would be evaluated. 

In Chapter 5 we showed the results our implementation of the model yielded. Our learner's library of discovered abstractions satisfied most of the criteria. The abstractions open a private model of the world, make the model false, run an agent's planner against the model rather than the world, and publish back the agent's resulting movement.

The commit is spelled three different ways across the abstractions, yet in every one of them the value whose behavior reaches the world is the value whose reasoning was simulated. That correct binding makes the abstractions _attributions_ and not just counterfactuals. The falsehood is _somebody's_. 

The library's order of arrival is also evidence in and of itself. No belief tasks are solved in round one, then compression over the solutions to those non-mental tasks yields precisely the two capacities that belief needs: holding a second grid, and transfering a value between grids. One those are added as new abstractions, the belief families are solved in rounds two and three. 

The atomic control run confirms the reading from the other side. Handed the same two capacities outright as primitives, the learner is able to solve those same tasks a round earlier. This is in keeping with the theory theory's account of the developmental timeline (@explanandum): belief attribution arrives late because it is a composition whose components must themselves first be assembled.  

Our result is an existence claim. There exists a learning process that constructs terms encoding the structure of belief attribution, where the learner begins only with domain-general primitives. But our demonstration is quite limited: it covers one corpus and one initial library. The belief attributed is a thin, single-step divergence with no acquisition or revision, no metarepresentation, and no recursive mind reading. And of course, we make no claim that children actually learn ToM this way. Our claim is at the computational level.

Within its scope, though, the result is hopefully enough to keep the nativist at bay. Our learner was shown behavior and was equipped only to prefer short hypotheses. It added the belief attribution abstraction to its library because naming the structure "an agent acting on its own model of the world" made the observed behavior shorter to describe than descriptions told in terms of the world alone. If our learner is anything like a child, then Theory of Mind really is a _theory_. Its unobservables posited because the world is more compactly explained with them than without.

#load-bib(read("refs2.bib"))
