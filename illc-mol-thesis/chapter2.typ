#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Learning as Bayesian Program Synthesis")

== Levels of analysis and reverse engineering

Of the three positions surveyed in @perspectives, constructivism (the theory theory) comes closest to accounting for the features of theory of mind (@explanandum) that a theory of its acquisition ought to account for. But it's a mostly descriptive account: it doesn't specify which process does the theory construction, nor by what standards one theory is preferred to another. Without specifics, the nativist can say that no such constructive process exists, that it only looks like construction but is really just maturation of a native system. In this chapter we'll describe a model of a particular constructive process. 

//Without an answer, the nativist can reply that no such process exists, and that what looks like construction is really maturation of something already in place. The strength of that reply is parasitic on the absence of a mechanism, and the business of this chapter is to supply one.

We proceed at Marr's computational level of analysis of information-processing systems @marr_vision_2010, in the tradition of "reverse-engineering" the mind @griffiths_bayesian_2024. Whereas in engineering we start from a computational problem and build a system that solves it, in reverse-engineering we start with a system and figure out what computational problem it solves: what information is its input, what results are its output, and what counts as a solution. Then once the problem is stated in these abstract terms we can ask which of the systems we know how to build solve problems of that shape, and then we model the mind as a system of that kind @griffiths_bayesian_2024.

@perfors_learnability_2011

The problem solved 

So we should ask first what problem a mindreader is solving. @explanandum has already supplied most of the answer. The data available to the learner is a trajectory of bodies through space. What must be recovered is the process that generated that trajectory, and that process runs through states --- beliefs, desires, intentions --- that never appear in the data. The trajectory underdetermines the process: any behaviour is compatible with indefinitely many belief-desire pairs, and no amount of further observation eliminates all but one.

Stated that way, the problem is an instance of a familiar one. Given data, entertain hypotheses about the process that generated it, assign each a prior plausibility, and revise those plausibilities as the data comes in. And there is a known answer to the question of how such revision _ought_ to go: Bayes' rule specifies the ideal solution, and thereby generates predictions that can be checked against what people actually do @griffiths_bayesian_2024. If human judgements line up with those predictions, we have grounds for thinking that this is the problem being solved --- and a model we can use to build machines that solve it the same way.

That is the shape of the claim this thesis makes. The computational-level thesis is that acquiring a theory of mind is the problem of finding the best explanation of observed behaviour, where "best" is fixed not by taste but by the logic of Bayesian inference: a trade-off between the prior plausibility of an explanation and how well it fits the data. @sec-bayes makes that trade-off precise, and @sec-programs argues that the hypotheses being traded off are best understood as _programs_.

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

== Bayesian inference: the logic of learning <sec-bayes>

explain prior/likelihood, size principle, Bayes as normative solution

== Programs as the hypothesis space <sec-programs>

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

== Learning the language itself <sec-hbm>

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

== The DreamCoder algorithm <sec-dreamcoder>

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