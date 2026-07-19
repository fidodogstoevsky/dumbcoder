#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Introduction: Theory of Mind")

== The nature and scope of theory of mind

The ice cream peddler's scratchy jingle got louder still as he brought his cart to a stop nearby. A child, hitherto engrossed in sandbox amusements, abandoned her playmates abruptly. She darted across the playground, hurtling at full speed towards a gaggle of parents. Absorbing the impact of a 20 kg cruise missile, the target continued his conversation unfazed by the child now tugging at his jeans. Only when he deigned to look down at him did he shriek and go pale. But the child barely had time to react, because just then she heard her name shouted from the other side of the playground. She turned to its source and ran towards a similar-looking man holding two ice cream cones.

You chuckle at the scene, observing from a nearby park bench. The girl, wanting ice cream, pestered a man she thought was her father. The man, undoubtedly accustomed to his own nagging kids, ignored the girl thinking it was his own child. Descriptions like these, in terms of desires, beliefs, and intentions, seem trivial and obvious. But there's something miraculous about them. How are you able to peer into other peoples' minds? Mental content is unobservable, all you see is a sequence of physical actions. And yet humans, even infants, are experts at inferring mental content rapidly and accurately.

Humans have the cognitive ability to impute mental states to others, to attribute a belief, desire, or intention to another person. This capacity is known as "theory of mind",  "folk psychology", the ability to "mindread", to "mentalize". It is integral to everything that makes us human: to cooperate with someone you need to understand what they think, what they want. To decode someone's words, you need to begin with what they're trying to (by the Rational Speech Act framework). To understand social roles like "helping" and "hindering", you need to attribute an intention to them. Humans make these inferences so naturally that it seems like not much of an impressive ability at all, and "mindreading" seems too fabulous a term for something so mundane. But the ubiquity of these inferences should make the human capacity for mentalizing even more impressive, not less.

In this thesis we study the question: how is this capacity acquired? How do humans, even at a young age, come to acquire this ability? We'll begin by 




=== the problem of other minds:
frame it as a fundamental challenge: minds are unobservable, yet humans (even infants) are experts at inferring their contents rapidly and accurately

=== the function of ToM:
disucss its role in social interaction, such as cooperation, communication (pragmatics, RSA), and understanding social roles like "helping" or "hindering"

== Developmental milestones (ToM scale)

=== early roots (0-12 months)

discuss evidence that infants as young as 6 months old interpret motion as goal-directed and apply a "teleological stance", an intuitive schema relating actions, goals, and environmental constraints (this also helps later to defend `optimize`) as non-intensional

ToM scale: Diverse desires

=== emergence of epistemic states (12-24 months)

review findings on infants' sensitivity to perceptual access (e.g. knowing an agent can only have a goal for an object they can see)

ToM scale: Knowledge access

=== "Sally-Anne" False belief task

milestone where children transition from reality-centric responses to representing false beliefs around age 4

ToM scale: False beliefs

=== second-order mentalizing

brief overview of more complex social reasoning that emerges later (e.g. "I think that you think"), where the representations have to be richer, the model has to be higher-dimensional

== Theoretical perspectives on ToM acquisition

=== Nativism and core knowledge

present the view that ToM is a built-in module, a core system [Leslie]. Discuss the "poverty of stimulus" argument, the data is too sparse for general learning. Which implies innate domain-specific constraints like the "principle of efficiency"

=== Empiricism and connectionism/associationism

contrast nativism with the view that ToM is learned through statistical patterns and bottom-up assocations from experience. Challenge for associationism: the structure exists, it's just not in the data. So how can a learner infer abstract concepts like "belief" if they aren't in the data? 

=== constructivism and the theory theory

Introduce the "child as scientist" metaphor. Argue that children don't just associate, they construct causal models, intuitive/folk theories of the mind that they revise in light of new evidence, similar to scientific progress. 

== Reconstructing constructivism

=== rational constructivism
the modern bridge. It adopts the constructivist idea of theory change but seeks a more formal mechanism than classic Piagetian accounts. 

=== the role of inductive bias
introduce the idea that learning "gets off the ground" by combining strong, potentially innate, domain-general mechanisms with minimal, skeletal domain-specific knowledge

_could be an analyzer which highlights events resulting from one’s own actions, making the latent concept of intervention more salient. Alternatively, an innate or early-developing agency-detector might help in identifying interventions resulting from the actions of intentional agents. Altogether this suggests a novel take on nativism—a “minimal nativism”— in which strong, but domain-general, inference and representational resources are aided by weaker, domain-specific perceptual input analyzers._ [Goodman 2011]

_where abstract knowledge is clearly constructed, such as intuitive biology, it has been observed that the most abstract domain knowledge often comes into place first, before specific knowledge (Wellman & Gelman, 1998). The blessing of abstraction provides a potential explanation of this observation as well.
Though we have argued that abstract knowledge about causality may be learnable, our results should also not be taken to support an entirely empiricist viewpoint. Our ideal learner possesses a rich language for expressing theories and a strong inductive learning mechanism. These are both significant innate structures, though ones that may be required for many learning tasks. In addition, we have shown that the domain-general mechanisms for learning and representation are greatly aided by a collection of domain-specific “perceptual input analyzers.” It may be ontogenetically cheap to build innate structures that make some intervention events salient, but quite expensive to build an innate abstract theory (or a comprehensive analyzer). Our simulations suggested that these analyzers need not be perfectly tuned to causality or cover all intervention events. There are a number of plausible candidates that have been previously suggested to support causal reasoning: animacy or agency detectors, Michottean event detectors, proprioception, etc. Since a powerful learning mechanism is present in human cognition, the most efficient route to abstract knowledge may be by bootstrapping from these simple, non-conceptual mechanisms. Thus we are suggesting a kind of minimal nativism: strong domain-general inference and representational resources, aided by weak domain-specific input analyzers._ [Goodman 2011]

=== transition to chapter 2

end by noting that while the theory theory provides a powerful metaphor, it lacks a precise computational backbone to explain how children search through the infinite space of possible theories. This sets up chapter 2 to introduce bayesian program induction (via DreamCoder) as that formal implementation. 

#load-bib(read("chapter1.bib") + read("chapter2.bib"))