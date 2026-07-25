#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Development of Theory of Mind")

== The nature and scope of theory of mind

The ice cream peddler's scratchy jingle got louder still as he brought his cart to a stop nearby. A child, hitherto engrossed in sandbox amusements, abandoned her playmates abruptly. She darted across the playground, hurtling at full speed towards a gaggle of parents. Absorbing the impact of a 20 kg cruise missile, the target continued his conversation unfazed by the girl now tugging at his jeans. Only when he deigned to look down at the child did he shriek and go pale. But the girl barely had time to react, because just then she heard her name shouted from the other side of the playground. She turned to its source and ran towards a similar-looking man whose outstretched arms held two ice cream cones.

You chuckle at the scene, observing from a nearby park bench. The girl, wanting ice cream, pestered a man she thought was her father. The man, undoubtedly accustomed to his own nagging kids, ignored the girl thinking it was his own child. Descriptions like these, in terms of desires, beliefs, and intentions, seem trivial and obvious. But there's something miraculous about them. How are you able to peer into other peoples' minds? Mental content is unobservable, all you see is a sequence of physical actions. And yet humans, even infants, are experts at inferring mental content rapidly and accurately.

Humans have the cognitive ability to impute mental states to others, to attribute a belief, desire, or intention to another person. This capacity is known as "theory of mind",  "folk psychology", the ability to "mindread", to "mentalize". It is integral to everything that makes us human: to cooperate with someone you need to understand what they think, what they want. To decode someone's words, you need to begin with what they're trying to (by the Rational Speech Act framework). To understand social roles like "helping" and "hindering", you need to attribute an intention to them. Humans make these inferences so naturally that it seems like not much of an impressive ability at all, and "mindreading" seems too fabulous a term for something so mundane. But the ubiquity of these inferences should make the human capacity for mentalizing even more impressive, not less.

In this thesis we study the question: how is this capacity acquired? How do humans, even at a young age, come to acquire this ability? We'll begin by 


#underline[in this section:]
- *the problem of other minds* framed as a fundamental challenge: minds are unobservable, yet humans (even infants) are experts at inferring their contents rapidly and accurately
- *function of ToM* its role in social interaction, such as cooperation, communication (pragmatics, RSA), and understanding social roles like "helping" or "hindering"

"Theory of Mind (ToM) refers to the cognitive capacity to attribute mental states to self or others."
Other names for the same capacity include:
- "commonsense psychology"
- "naive psychology"
- "folk psychology"
- "mindreading"
- "mentalizing"
@margolis_oxford_2012

mental states include things like
- perception
- bodily feelings
- emotional states
- propositional attitudes (beliefs, desires, hopes, intentions)
@margolis_oxford_2012

We'll be focusing on the attribution of propositional attitudes.



== Developmental milestones (ToM scale)

=== early roots (0-12 months)

discuss evidence that infants as young as 6 months old interpret motion as goal-directed and apply a "teleological stance", an intuitive schema relating actions, goals, and environmental constraints (this also helps later to defend `optimize`) as non-intensional

ToM scale: Diverse desires [Repacholi and Gopnik 1997 "Early reasoning about desires: evidence from 14 and 18 month olds"] 18 month olds can identify diverse desires and reason about them

Woodward 1998: attribution of object-directed goal in 5-6 month olds. Infants already see agentically

teleological stance [Gerg-ely, G., Nádasdy, Z., Csibra, G., & Biró, S. (1995). Taking the intentional stance at 12 months of age. Cognition, 56, 165–193]

=== emergence of epistemic states (12-24 months)

review findings on infants' sensitivity to perceptual access (e.g. knowing an agent can only have a goal for an object they can see)

ToM scale: Knowledge access

=== "Sally-Anne" False belief task

milestone where children transition from reality-centric responses to representing false beliefs around age 4

ToM scale: False beliefs

Description of the false belief task @wimmer_beliefsabout_nodate. Results: 40% of the 4-year olds and 90% of the 6 to 7 year olds correctly predicted Maxi would look at the location where he believes the chocolate is, rather than its actual location. When information access or lack thereof was made explicitly salient, it rose to 50% of 4-year olds. But children under 3.5 had quite low rates of solving, and their performance didn't improve when the scenario was made more explicit. So the conclusion is that around 3 or 4 years old, children achieve the ability to attribute minds. 

=== second-order mentalizing

brief overview of more complex social reasoning that emerges later (e.g. "I think that you think"), where the representations have to be richer, the model has to be higher-dimensional

== Theoretical perspectives on ToM acquisition

=== Nativism and core knowledge

view that ToM is a built-in module, a core system [Leslie]. Discuss the "poverty of stimulus" argument, the data is too sparse for general learning. Which implies innate domain-specific constraints like the "principle of efficiency"

[Berwick 2011] for poverty of stimulus, and Chomsky 1971

"On one side, a purely nativist response was to deny learning and to focus on characterizing the detailed representations that were already in place" @griffiths_bayesian_2024

=== Empiricism and connectionism/associationism

view that ToM is learned through statistical patterns and bottom-up assocations from experience.

the structure exists, it's just not in the data. So how can a learner infer abstract concepts like "belief" if they aren't in the data?

[McClelland 2010] on emergence

[McClelland & Rumelhart, 1986] and [Rogers & McClelland, 2004] on connectionism

statistical models of learning defined over large numerical vectors @tenenbaum_how_2011, @griffiths_bayesian_2024

"On another side, an empiricist response was to suggest that structured representations were not necessary, that learning (and the inferences that followed) was simply a bottom-up process of learning statistical associations" @griffiths_bayesian_2024

=== (rational) constructivism and the theory theory

Introduce the "child as scientist" metaphor. children don't just associate, they construct causal models, intuitive/folk theories of the mind that they revise in light of new evidence, similar to scientific progress. 

developmental (Piagetian) account of building up from smaller components

rational constructivism (reconstructing constructivism) adopts the constructivist idea of theory change but seeks a more formal mechanism than classic Piagetian accounts. 

I'm gonna end the section by noting that while the theory theory provides a powerful metaphor, it lacks a precise computational backbone to explain how children search through the infinite space of possible theories. Basically that the strength of the nativist argument lies in part in the supposed lack of algorithms for learning those structured representations. then this sets up chapter 2 to introduce bayesian program induction (via DreamCoder) as that formal implementation. 

#load-bib(read("chapter1.bib") + read("chapter2.bib"))