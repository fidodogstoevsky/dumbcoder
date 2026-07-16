#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Bayesian Model")

== Bayesian Cognitive Modeling

=== Reverse-engineering the mind 

[Griffiths 1.3]

=== Marr's levels of analysis 

[Marr 1982]
[Perfors 2011]
[Tenenbaum 2011]
[Griffiths 2024]

=== Rational constructivism

[Griffiths 2024]
[Tenenbaum 2011]

== The language of thought as a representational medium

=== Beyond associations, to programs

contrast the symbolic approach (structured, compositional) with the connectionist approach (unstructured, associative weights), arguing that programs capture the productivity and open-endedness of human thought

_defined over large numerical vectors. Learning was seen as estimating strengths in an asso- ciative memory, weights in a neural network or parameters of a high-dimensional, nonlinear function (McClelland & Rumelhart, 1986; Rogers & McClelland, 2004). Bayesian cognitive models, in contrast, have had the most success with defining probabilities over more struc- tured symbolic forms of knowledge representations used in computer science and AI, such as graphs, grammars, predicate logic, relational schemas, and functional programs. Different forms of representation are used to capture people’s knowledge in different domains and tasks, as well as at different levels of abstraction. In learning words and concepts from examples, the knowledge that guides both children_ [Griffiths 2024]

_The Form of Abstract Knowledge Abstract knowledge provides essential constraints for learning, but in what form? This is just question 2. For complex cognitive tasks such as concept learning or causal reasoning, it is impossible to simply list every logically possible hypothesis along with its prior and likelihood. Some more sophisticated forms of knowledge representation must underlie the probabilistic generative models needed for Bayesian cognition.
In traditional associative or connectionist approaches, statistical models of learning were defined over large numerical vectors. Learning was seen as estimating strengths in an associative memory, weights in a neural network, or parameters of a high-dimensional nonlinear function (12, 14). Bayesian cognitive models, in contrast, have had most success defining probabilities over more structured symbolic forms of knowledge representations used in computer science and artificial intelligence, such as graphs, grammars, predicate logic, relational schemas, and functional programs. Different forms of representation are used to capture people’s knowledge in different domains and tasks and at different levels of abstraction._ [Tenenbaum 2011]

_operates over structured (Fodor, 1975; Fodor & Pylyshyn, 1988), probabilistic (Lake et al., 2017;
Goodman et al., 2015), program-like (Turing, 1936; Baum, 2004) representations. While there
have been many other proposals for modeling conceptual representations, only programs arguably
capture the full breadth and depth of people’s algorithmic abilities (Goodman et al., 2015). Code
can model both procedural and declarative information and allow them to interact seamlessly.
Universal programming languages integrate all this knowledge into a single formal representation._ [Rule 2020]

== Bayesian program induction as the learning engine

Prior vs. Likelihood: Introduce Bayes' Rule as the engine of learning, where the learner trades off the simplicity of a program (the prior) against its fit to observed behavior (the likelihood)

_compute this probability,
P(h|d) ∝ P(h)P(d|h), (1)
identifying the posterior probability as proportional to the product of the prior and the likelihood. This Bayesian posterior provides a rational analysis of inductive learning: a coherent integration of evidence and a priori knowledge into posterior beliefs (optimal within the specified learning context).
By viewing concept acquisition as an inductive problem, we may employ this Bayesian framework to describe the learning of a concept from examples. As described earlier, our hypothesis space is the collection of all phrases in a grammatically-generated concept language. Since each concept of this language is a classification rule, a natural likelihood is given by simply evaluating this rule on the examples: the data have zero probability if they disagree with the classification rule, and constant probability otherwise. However, to account for the unreliability of examples, we will allow non-zero probability for a data set, even if some examples are misclassified. That is, we assume that there is a small probability that any example is an outlier, which should be ignored. We will see in the next section that this outlier assumption combines with rule-evaluation to give a simple likelihood function, which decreases exponentially with the number of “misclassifications” of the examples._ [Goodman 2008]

_Bayesian inference gives a rational framework for updating beliefs about latent vari- ables in generative models given observed data (Jaynes, 2003; Mackay, 2003). Background knowledge is encoded through a constrained space of hypotheses H about possible values for the latent variables—candidate world structures that could explain the observed data. Finer-grained knowledge comes in the prior probabilities P(h) that specify the learner’s degree of belief in each hypothesis h prior to (or independent of) the observations. Bayes’ rule updates these prior probabilities to posterior probabilities P(h|d) conditional on the observed data d:_ [Griffiths 2024]

_natural interpretations in many problems. The likelihood reflects the fit between hypothe- sis and data, and the prior indicates the a priori plausibility of a hypothesis (which might decrease for hypotheses that have low frequency, are complicated, or seem otherwise improbable). The contribution of these two factors to the conclusions that we should draw is fairly natural and makes intuitive sense in a variety of contexts. Returning to an exam- ple introduced in chapter 1, if you see John coughing (your data d), you might consider three hypotheses about the cause of the cough: a cold (h 1 ), lung disease (h 2 ), or heartburn (h 3 ). You might rule out heartburn on the basis of fit, since it might only slightly increase the chance of coughing. A cold and lung disease both fit well with the cough—they increase the probability of coughing—but they differ in plausibility. Normally, a cold is far more com- mon than lung disease, and thus might be the hypothesis that you would select to explain the coughing. However, the plausibility of these two hypotheses might change if you were passing by a hospital and saw John coughing inside. All inductive inferences require con- sidering fit and plausibility. Bayes’ rule just tells you how they should be combined to reach a conclusion, using the common language of probability theory to determine the impact of each factor. In the context of cognitive science, priors become a useful way of describing the inductive_ [Griffiths 2004]

_Summary of the basic modeling framework. Although both priors and likelihoods can be understood on their own terms, it is only in combination that they explain how people can successfully learn the extensions of new words from just a few positive examples. Successful word learning requires both a constrained space of candidate hypotheses—provided by the prior—and the ability to reweight hypotheses according to how well they explain a set of observed examples—provided by the likelihood. Without the constraints imposed by the prior, no meaningful generalizations would be possible. Without the likelihood, nothing could be learned from multiple examples beyond simply eliminating inconsistent hypotheses. In particular, priors and likelihoods each contribute directly to the main pattern of generalization that we described in the introduction and that we look for in our experiments: Given just a single example of a novel kind label, generalization to other objects should be graded, but given several examples, learners should apply the word more discriminatingly, generalizing to all and only members of the most specific natural concept that spans the observed examples. The prior determines which concepts count as “natural,” whereas the likelihood generates the specificity preference and determines how the strength of that preference—and thus the sharpness of generalization—increases as a function of the number of examples._ [Xu 2007]

== Hierarchical Bayesian models

Explain how learning overhypotheses enables a learner to acquire the _form_ of a theory (i.e. agents are rational) before learning specific goals

_The formalism of hierarchical Bayesian modeling makes it possible to express the assumptions relating knowledge at multiple levels of abstraction (Gelman, Carlin, Stern, & Ru-bin, 1995), and Bayesian inference over such a model describes an ideal learner of abstract knowledge (Tenenbaum, Griffiths, & Kemp, 2006). Though real learning is undoubtedly resource-constrained, the dynamics of an ideal learner can uncover unexpected properties of what it is possible to learn from a given set of evidence. For instance, it has been reported (e.g. Kemp, Perfors, & Tenenbaum, 2007) that learning at the abstract level of a hierarchical Bayesian model is often surprisingly fast in relation to learning at the more specific levels. We term this effect the blessing of abstrac-tion1: abstract learning in an hierarchical Bayesian model is often achieved before learning in the specific systems it relies upon, and, as a result, a learner who is simultaneously learning abstract and specific knowledge is almost as efficient as a learner with an innate (i.e. fixed) and correct abstract theory. Hierarchical Bayesian models have been used before to study domain-specific abstract causal knowledge (Kemp, Goodman, & Tenenbaum, 2007), and simple relational theories (Kemp et al., 2008). Here we combine these approaches to study knowledge of causality at the most abstract, domain general level._
[Goodman 2011]

_trees, causal networks, and other forms of structure that people appear to know explicitly (Rogers & McClelland, 2004). Bayesian cognitive modelers have answered these challenges by combining the structured
knowledge representations described in this chapter with advanced methods from Bayesian statistics known as hierarchical Bayesian models (HBMs; Gelman, Carlin, Stern, & Rubin, 1995). HBMs address the origins of hypothesis spaces and priors by positing not just a single level of hypotheses to explain the data, but multiple levels: hypothesis spaces of hypothesis spaces, with priors on priors. Bayesian inference across all levels allows hypotheses and priors needed for a specific learning task to themselves be learned at larger or longer-time scales, at the same time as they constrain lower-level learning (see chapter 8). In machine learning and AI, HBMs have primarily been used for transfer learning or_
[Griffiths 2024]

_This paper suggests that hierarchical Bayesian models (Good, 1980; Gelman, Carlin, Stern & Rubin, 2003) can help to explain the computational principles which allow overhypotheses to be learned. Hierarchical Bayesian models (HBMs) include representations at multiple levels of abstraction, and show how knowledge can be acquired at levels quite remote from the data given by experience. To illustrate these points, we describe one of the simplest possible HBMs and use it to suggest how overhypotheses about feature-variability (e.g. the shape bias) are acquired and used to support categorization. We also present an extension of this basic model that groups categories into ontological kinds (e.g. objects and substances) and discovers the features and the patterns of feature variability that are characteristic of each kind._ [Kemp 2009]

_schema discovers the disease-symptom framework theory by assigning variables 1 to 6 to class C1, variables 7 to 16 to class C2, and a prior favoring only C1 → C2 links. These assignments, along with the effective number of classes (here, two), are inferred automatically via the Bayesian Occam's razor. Although this three-level model has many more degrees of freedom than the model in (B), learning is faster and more accurate. With n = 80 patients, the causal network is identified near perfectly. Even n = 20 patients are sufficient to learn the high-level C1→ C2 schema and thereby to limit uncertainty at the network level to just the question of which diseases cause which symptoms. (D) A HBM for learning an abstract theory of causality (62). At the highest level are laws expressed in first-order logic representing the abstract properties of causal relationships, the role of exogenous interventions in defining the direction of causality, and features that may mark an event as an exogenous intervention. These laws place constraints on possible directed graphical models at the level below, which in turn are used to explain patterns of observed events over variables. Given observed events from several different causal systems, each encoded in a distinct data matrix, and a hypothesis space of possible laws at the highest level, the model converges quickly on a correct theory of intervention-based causality and uses that theory to constrain inferences about the specific causal networks underlying the different systems at the level below._ [Tennenbaum 2011]

== The blessing of abstraction

abstract knowledge can be acquired faster than specific facts, by pooling evidence across multiple scenarios. This could explain how children acquire core social constraints early in life. 

_The formalism of hierarchical Bayesian modeling makes it possible to express the assumptions relating knowledge at multiple levels of abstraction (Gelman, Carlin, Stern, & Ru-bin, 1995), and Bayesian inference over such a model describes an ideal learner of abstract knowledge (Tenenbaum, Griffiths, & Kemp, 2006). Though real learning is undoubtedly resource-constrained, the dynamics of an ideal learner can uncover unexpected properties of what it is possible to learn from a given set of evidence. For instance, it has been reported (e.g. Kemp, Perfors, & Tenenbaum, 2007) that learning at the abstract level of a hierarchical Bayesian model is often surprisingly fast in relation to learning at the more specific levels. We term this effect the blessing of abstrac-tion1: abstract learning in an hierarchical Bayesian model is often achieved before learning in the specific systems it relies upon, and, as a result, a learner who is simultaneously learning abstract and specific knowledge is almost as efficient as a learner with an innate (i.e. fixed) and correct abstract theory. Hierarchical Bayesian models have been used before to study domain-specific abstract causal knowledge (Kemp, Goodman, & Tenenbaum, 2007), and simple relational theories (Kemp et al., 2008). Here we combine these approaches to study knowledge of causality at the most abstract, domain general level._ [Goodman 2011]


_Blessing of abstraction : The phenomenon whereby higher-level, more abstract knowl-_ [Perfors 2011]

== The DreamCoder algorithm

=== Wake: exploration/enumeration/search (Bayesian inference)

The wake phase solves the inverse problem: searching for the program (intention) that most likely generated the data (actions)

[Ellis 2020]

_D are unknown.
The marginal likelihood of the observed tasks is then given by pθD (X ) =
∏ x∈X
∑ ρ p(x | ρ)pθD (ρ), where p(x | ρ)
is the likelihood of x being produced, and hence solved, by ρ. To learn a good generative model that maximises the likelihood, we need to know which programs score highly under p(x | ρ)—i.e. solve our tasks. We have seen previously why this is challenging: discovering programs
that can account for the observed tasks requires search.
To help with search, a recognition (inference) model qϕD (ρ | x) is learnt to infer the programs that are most likely to solve a given task. The recognition model parameters ϕD map tasks to distributions over programs that, as with the generative model, specifies the probability that components part of the library D are used. Estimating ϕD is done using both (ρ, x) pairs sampled from the generative model (fantasies) and programs found to solve the observed tasks x ∈ X (replays). The recognition model is used to search for programs that solve a task x by enumerating programs in decreasing order of their probability under qϕD (ρ | x). Programs that solve x are stored in a task-specific set Bx._ [Palmarini 2024]

=== Sleep: Abstraction (library learning, compression)

Explain how refactoring and compression automate the discovery of new social primitives, implementing the Theory Theory by allowing the learner to grow their own language of thought 

_Discovering programs that may have produced the observed tasks (those in {Bx}x∈X ) now provides more data to infer the parameters θD generating them. Inferring θD entails choosing the library D whose components they control. Rather than maximise the likelihood directly, DREAM-CODER performs maximum a posteriori (MAP) inference using a prior over libraries D and parameters θD. Maximis-ing the MAP objective (which can only be approximated) w.r.t. D corresponds to updating D to include functions that best compress the discovered solutions. After updating D, parameters θD are updated to their MAP estimates._ [Palmarini 2024]

_its descendants (Wong, Ellis, Tenenbaum, & Andreas, 2022; Bowers et al., 2023) are recent AI approaches for Bayesian program learning inspired by Church and the PLoT hypothesis that show how such “library learning” can dramatically expand the repertoire of effectively learnable concepts, given a limited amount of computational resources. These systems work not only by adding newly learned concepts to the library, but also by abstracting out program components that are implicitly shared between previously learned concepts to identify the most explanatory and compact theory of a domain—effectively perform- ing a heuristic version of hierarchical Bayesian inference to learn the prior for generating novel concepts. LAPS (Language for Abstraction and Program Search) (Wong et al., 2022) extends DreamCoder to learn a joint prior on programs and natural-language translations of those programs, and models how—consistent with studies in cognitive development (Carey, 2009)—experience with natural language in a doman can bootstrap a learner’s theory acqui- sition and joint acquisition of word meanings and novel concepts, well beyond what could be gleaned merely from observed perceptual examples._ [Griffiths 2024]

=== Sleep: dreaming (amortized inference)

explain the neural recognition network as a way to "compile" slow, deliberative Bayesian search into fast, "intuitive" social reasoning.

== Learning theory of mind via inverse planning

=== agents as approimately rational planners

defining the generative model. agents move in an environment (MDP/POMDP) to maximize utility efficiency

_One alternative to a cue-based account is to use generative models of action choice, as in the Bayesian inverse planning (or “Bayesian theory-of-mind”) models of Baker, Saxe, and Tenenbaum (2009) or the “naive utility calculus” models of Jara-Ettinger, Gweon, Tenenbaum, and Schulz (2015) (See also Jern and Kemp (2015) and Tauber and Steyvers (2011), and a related alternative based on predictive coding from Kilner, Friston, and Frith (2007)). These models formalize explicitly mentalistic concepts such as ‘goal,’ ‘agent,’ ‘planning,’ ‘cost,’ ‘efficiency,’ and ‘belief,’ used to describe core psychological reasoning in infancy. They assume adults and children treat agents as approximately rational planners who choose the most efficient means to their goals. Planning computations may be formalized as solutions to Markov Decision Processes (or POMDPs), taking as input utility and belief functions defined over an agent’s state-space and the agent’s state-action transition functions, and returning a series of actions the agent should perform to most efficiently fulfill their goals (or maximize their utility). By simulating these planning processes, people can predict what agents might do next, or use inverse reasoning from observing a series of actions to infer the utilities and beliefs of agents in a scene. This is directly analogous to how simulation engines can be used for intuitive physics, to predict what will happen next in a scene or to infer objects’ dynamical properties from how they move. It yields similarly flexible reasoning abilities: Utilities and beliefs can be adjusted to take into account how agents might act for a wide range of novel goals and situations. Importantly, unlike in intuitive physics, simulation-based reasoning in intuitive psychology can be nested recursively to understand social interactions – we can think about agents thinking about other agents._ [Lake 2016]

_Humans are adept at inferring the mental states underlying other agents’ actions, such as goals, beliefs, desires, emotions and other thoughts. We propose a computational framework based on Bayesian inverse planning for modeling human action understanding. The framework represents an intuitive theory of intentional agents’ behavior based on the principle of rationality: the expectation that agents will plan approximately rationally to achieve their goals, given their beliefs about the world. The mental states that caused an agent’s behavior are inferred by inverting this model of rational planning using Bayesian inference, integrating the likelihood of the observed actions with the prior over mental states. This approach formalizes in precise probabilistic terms the essence of previous qualitative approaches to action understanding based on an ‘‘intentional stance” [Dennett, D. C. (1987). The intentional stance. Cambridge, MA: MIT Press] or a ‘‘teleological stance” [Gerg-ely, G., Nádasdy, Z., Csibra, G., & Biró, S. (1995). Taking the intentional stance at 12 months of age. Cognition, 56, 165–193]. In three psychophysical experiments using animated stimuli of agents moving in simple mazes, we assess how well different inverse planning models based on different goal priors can predict human goal inferences. The results provide quantitative evidence for an approximately rational inference mechanism in human goal inference within our simplified stimulus paradigm, and for the flexible nature of goal representations that human observers can adopt. We discuss the implications of our experimental results for human action understanding in real-world contexts, and suggest how our framework might be extended to capture other kinds of mental state inferences, such as inferences about beliefs, or inferring whether an entity is an intentional agent._ [Baker 2009]

_analog of expected utility theory used for understanding everyday behavior. Importantly, this does
not assume that humans necessarily plan or act rationally in all circumstances. Rather, we tac-
itly expect others to behave rationally (or approximately so) in particular situations, and perform
"explanation by rationalization" by attributing the mental states which make their behavior appear
most rational within the present context.
The Bayesian Theory of Mind (BToM) framework combines these principles governing the_ [Baker 2012]

_To address these questions, we formalize action understanding as a Bayesian inference prob-
lem. We model the intuitive causal relation between beliefs, goals and actions as rational proba-
bilistic planning in Markov decision problems (MDPs), and invert this relation using Bayes' rule
to infer agents' beliefs and goals from their actions. We test our framework with psychophysical
experiments in a simple setting that allows us to collect a large amount of fine-grained human
judgments to compare with the strong quantitative predictions of our models._ [Baker 2012]