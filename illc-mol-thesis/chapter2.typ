#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Learning as Bayesian Program Synthesis")

== Levels of analysis and reverse engineering

_state that the thesis' claim is computational with ECD as an algorithmic representation_


- Marr's levels of analysis
- computational: what is the thing that the system is doing
- it's doing bayesian updating. it's finding the simplest theory. 
- algorithmic: the procedure by which the solution is produced


We're not studying how a child actually comes to learn theory of mind, that's the realm of developmental and cognitive psychology. 

identify the computational problem that the mind is trying to solve: the problem is find the best explanation. we argue that best is determined logically by bayes' rule. so it's doing bayesian updating. then the algorithmic level is the procedure at which it's produced, ECD

engineering: building AI, building algorithms that solve computational problems

reverse-engineering: identify the abstract computational problems that the mind must solve to do what it does. knowing that the mind solves these certain problems, what are the AI systems we have that can solve problems like that. Then we can model the mind as that kind of system. 

Start by thinking about the computational problem that the mind is solving. out of all the data, its task is to figure out the underlying data generation process, i.e. what's going on in the world. this is the problem of positing hypotheses, assigning prior probabilities to them, and updating the probabilities based on new data. Then Bayes' rule shows us the ideal solution to the problem. Then if we see that humans perform similarly to these algorithms, we have better evidence that it's this kind of problem solving @griffiths_bayesian_2024

bayesian models focus on the computational level, the logic of the problem, what the information processing system is doing. 


[Marr 1982]

Bayesian framework addresses the question: "how to update beliefs and make inferences in light of observed data". Starts with the logic of the inference when generalizing, i.e. what's happening computationally, rather than how it does it algorithmically. 
@perfors_learnability_2011


- computational level: "characterizes the problem that a cognitive system solves and the principles by which its solution can be computed from the available inputs in natural environments"
- algorithmic level: "describes the procedures executed to produce this solution and the representations or data structures over which the algorithms operate"
- implementation level: "specifies how these algorithms and data structures are instantiated in the circuits of a brain or machine" @tenenbaum_how_2011

reverse engineering means start at the computational level, understand what the system is doing, then get into the algorithm. So it's top-down, or function-first. @griffiths_bayesian_2024

rational analysis framework. analyze cognitionin terms of adaptive solutions to environmental problems [Anderson 1990]

== Bayesian inference: the logic of learning

explain prior/likelihood, size principle, Bayes as normative solution

== Programs as the hypothesis space

- Language of Thought
- compositionality/productivity
- lambda calculus
- Fodor/Goodman/Rule
- symbols vs. vectors

*bayesian inference over structured representations*

contrast the symbolic approach (structured, compositional) with the connectionist approach (unstructured, associative weights), arguing that programs capture the productivity and open-endedness of human thought

concepts can be embedded and composed etc.

defining probabilities over structured symbolic forms of representation like graphs, grammars, predicate logic, relational schemas, and functional programs @griffiths_bayesian_2024

only programs capture full breadth and depth of people's complex capabilities to udnerstand and execute algorithms[Goodman et al 2015]

with code we can model both procedural and declarative knowledge using a single format of representation @rule_precis_nodate

Lambda calculus as formal language for compositional semantics [Heim & Kratzer, 1998] [Steedman, 2000]

and for other learning tasks [Piantadosi, Goodman, Ellis, & Tenenbaum, 2008]
[Liang, Jordan, & Klein, 2009, 2010] [Zettlemoyer & Collins, 2005, 2007]



*why programs for theory of mind?*

By the classical theory of concept formation, concepts have a lot of qualities that are reminiscent of programs @bruner_study_2009
- concepts are represented compositionally
- concepts are logical combinations of features of objects
- concepts are rules for classifying objects
- concept learning is deducing correct classification rule

necessity of structured symbolic mental representations [Fodor 1975]

[Lake 2017]

how else would we capture/account for human's algorithmic abilities, if not with mental programs? [Goodman 2015]
- knowledge decomposes into concepts

mind is a program that corresponds to the world. and in fact that's how we can learn so quickly, [Baum 2003]

programs are the universal mental thing that can be instantiated in different mediums, languages, etc [Lupyan & Bergens 2016]

social reasoning requires recursive, compositional representations to handle nested mental states. There's no cleaner way of doing this than with programs. it's basically already a program.

learning is programming. learning a concept is building a program composed of lower level primitives, building an algorithm that does something. symbolic programs are the best formal knowledge representation @rule_precis_nodate

embedded queries for choices of another agent, intuitive psychology [Stuhlmüller & Goodman, 2013] [Goodman 2015]

Human thinking as computation, computational model [Newell et al 1958, Newell et al 1959]

Thinking is a production system [Lovett & Anderson 2005]

symbolic programs give learner the freedom to adopt any syntax that is useful [Gopnick & Wellman 2012]

Siskind gives the example `lift(x,y)=CAUSE(x,GO(y,UP))` for knowledge as compositional programs [Siskind 1996]

== Bayesian program synthesis

- Bayes over programs
- prior as program size/description length

Prior vs. Likelihood: learner trades off the simplicity of a program (the prior) against its fit to observed behavior (the likelihood)

Piantadosi 2012:
- likelihopod function that uses the size principle, penalize overly broad hypotheses [Tenenbaum 1999]
  - used for cross situational word learning [Frank, Goodman, and Tenenbaum (2007)]
  - used for solving subset problem in compositional semantics [Piantadosi et al. (2008)]
- prior is from rational rules model [Tenenbaum, Feldman, and Griffiths (2008)] which first linked probabilistic inference with formal, compositional, representations
  - assumes learners prefer simplicity

likelihood is: data have zero probability if they disagree with classificaiton rule, and constant probability otherwise [Goodman 2008]

Bayesian inference: rational framework for updating beliefs given observed data [Jaynes, 2003; Mackay, 2003]

background knowledge is constrained hyptohesis space, finer-grained knowledge is the prior degrees of belief in the hypotheses [Griffiths 2024]

_The likelihood reflects the fit between hypothe- sis and data, and the prior indicates the a priori plausibility of a hypothesis (which might decrease for hypotheses that have low frequency, are complicated, or seem otherwise improbable). The contribution of these two factors to the conclusions that we should draw is fairly natural and makes intuitive sense in a variety of contexts. Returning to an exam- ple introduced in chapter 1, if you see John coughing (your data d), you might consider three hypotheses about the cause of the cough: a cold (h 1 ), lung disease (h 2 ), or heartburn (h 3 ). You might rule out heartburn on the basis of fit, since it might only slightly increase the chance of coughing. A cold and lung disease both fit well with the cough—they increase the probability of coughing—but they differ in plausibility. Normally, a cold is far more com- mon than lung disease, and thus might be the hypothesis that you would select to explain the coughing. However, the plausibility of these two hypotheses might change if you were passing by a hospital and saw John coughing inside. All inductive inferences require con- sidering fit and plausibility. Bayes’ rule just tells you how they should be combined to reach a conclusion, using the common language of probability theory to determine the impact of each factor. In the context of cognitive science, priors become a useful way of describing the inductive_ [Griffiths 2004]

_Without the constraints imposed by the prior, no meaningful generalizations would be possible. Without the likelihood, nothing could be learned from multiple examples beyond simply eliminating inconsistent hypotheses. ... The prior determines which concepts count as “natural,” whereas the likelihood generates the specificity preference and determines how the strength of that preference—and thus the sharpness of generalization—increases as a function of the number of examples._ [Xu 2007]

since knowledge is programs, learning is program induction [Kitzelmann, 2009; Flener & Schmid, 2008; Gulwani et al., 2017]

== Learning the language itself

- HBMs
- overhypotheses
- blessing

=== Hierarchical Bayesian models

Blessing of abstraction, overhypotheses, HBMs

Explain how learning overhypotheses enables a learner to acquire the form of a theory (i.e. agents are rational) before learning specific goals

abstract knowledge can be acquired faster than specific facts, by pooling evidence across multiple scenarios. This could explain how children acquire core social constraints early in life. 

theory of mind is the overhypothesis that structures instances

early Bayesian models addressed only the computational level, just stated what the system can do (optimal statistical inference), without specifying how it does it. HBMs get closer to the algorithmic level, showing "cognition as approximately optimal inference in probabilistic models defined over a learner's subjective and dynamically growing mental representations of the world's structure, rather than some objective and fixed world statistics." @tenenbaum_how_2011

e.g. concept learning is set membership and causal reasoning is directed graph relations. but it's computationally prohibitive to list all the possible hypotheses, their priors and likelihoods, it's combinatorial. So there's higher level constraints like a bipartite graph or something (the mediacl example) @tenenbaum_how_2011

General reference for HBMs, assumptions at mulitple levels of abstraction [Gelman, Carlin, Stern, & Ru-bin, 1995]

ideal learner of abstract knowledge is one that learns over a hierarchy of models [Tenenbaum, Griffiths, & Kemp, 2006]

The blessing of abstraction: Learning high level abstractions can be faster than learning low-level instances @kemp_learning_2007

HBMs for domain-specific abstract causal knowledge @kemp_learning_2007 

HBMs for simple relational theories @kemp_discovery_2008

a learner who is simultaneously learning abstract and specific knowledge is almost as efficient as a learner with an innate (i.e. fixed) and correct abstract theory
@goodman_learning_2011

HBMs for domain general causality learning @goodman_learning_2011

"hypothesis spaces of hypothesis spaces, with priors on priors" @griffiths_bayesian_2024 So over a longer timescale you can learn the priors needed for a specific learning task.

The overhypotheses constrain the space of lower-level hypotheses

@kemp_discovery_2008 gives an example of how HBMs can be used to learn overhypotheses about feature-variability (e.g. the shape bias) which help to do categorization (the lower-level learning task)

@tenenbaum_how_2011 gives an example of HBMs for constraining directed graph models, which are then used for explaining observed events. First learn intervention-based causality, then use that to constrain inferences about instances of causality. 

introduce the idea that learning is bootstrapped and "gets off the ground" by combining strong, potentially innate, domain-general mechanisms with minimal, skeletal domain-specific knowledge

Minimal nativism: "strong but domain-general inference and representational resources are aided by weaker, domain-specific perceptual input analyzers" @goodman_learning_2011

first the most abstract domain knoweldge comes into place and then the specific knowledge [Wellman and Gelman 1998]

@goodman_learning_2011

blessing of abstraction also defined in @perfors_learnability_2011

== The DreamCoder algorithm

=== Wake: exploration/enumeration/search (Bayesian inference)

The wake phase solves the inverse problem: searching for the program (intention) that most likely generated the data (actions)

[Ellis 2020]
[Palmarini 2024]

=== Sleep: Abstraction (library learning, compression)

Explain how refactoring and compression automate the discovery of new social primitives, implementing the Theory Theory by allowing the learner to grow their own language of thought 

[Palmarini 2024]
[Ellis 2020]

=== Sleep: dreaming (amortized inference)

explain the neural recognition network as a way to "compile" slow, deliberative Bayesian search into fast, "intuitive" social reasoning.

#load-bib(read("chapter1.bib") + read("chapter2.bib"))