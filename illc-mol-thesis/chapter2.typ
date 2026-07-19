#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Bayesian Model")

== Bayesian Cognitive Modeling

- Marr's levels of analysis
- computational: what is the thing that the system is doing
- it's doing bayesian updating. it's finding the simplest theory. 
- algorithmic: the procedure by which the solution is produced
- 

identify the computational problem that the mind is trying to solve: the problem is find the best explanation. we argue that best is determined logically by bayes' rule. so it's doing bayesian updating. then the algorithmic level is the procedure at which it's produced, ECD



=== Reverse-engineering the mind 


Introduce the philosophy of adopting an "engineering" approach to understand human behavior by identifying the abstract computational problems that the mind must solve

_This book is about how the human mind comes to understand the world—and ultimately, perhaps, how we humans may come to understand ourselves. Many disciplines, ranging from neuroscience to anthropology, share this goal—but the approach that we adopt here is quite specific. We adopt the framework of cognitive science, which aims to create such an understanding through reverse-engineering: using the mathematical and computational tools from the engineering project of creating artificial intelligence (AI) systems to better understand the operation of human thought. AI generates a rich and hugely diverse stream of hypotheses about how the human mind might work. But cognitive science does not just take AI as a source of inspiration. What we have learned about the mathematical and computa- tional underpinnings of human cognition can also help to build more human like intelligence in machines. The fields of AI and cognitive science were born together in the late 1950s, and grew up _
[Griffiths 2024]

_In presenting the Bayesian approach in this book, we have chosen to emphasize the fact that the underlying philosophy is one of reverse-engineering. To create a Bayesian model of cognition, we begin by thinking about the computational problem that the mind is solv- ing. This involves identifying the data available to learn from, specifying hypotheses about how those data are being generated, and assigning prior probabilities to those hypothe- ses. Bayes’ rule then indicates the ideal solution to that problem, generating predictions that can be compared against human behavior. If we see a correspondence between model and behavior—and particularly if it holds up as we run further experiments designed to test the model’s predictions—then we have a way to understand why people might be doing what they are doing, and a model that we can use to make machines that perform similarly. This reverse-engineering approach instantiates a view of how to make progress in cog-_ [Griffiths 2024]

=== Marr's levels of analysis 

bayesian models focus on the computational level, the logic of the problem, what the information processing system is doing. 


[Marr 1982]

_us about actual human minds. The paper ends with an appendix containing a glossary
and a collection of useful resources for those interested in learning more.
2 Bayesian Basics: Inductive generalization from
examples
The most basic question the Bayesian framework addresses is how to update beliefs and
make inferences in light of observed data. In the spirit of Marr’s (1982) computational-
level of analysis, it begins with understanding the logic of the inference made when
generalizing from examples, rather than the algorithmic steps or specific cognitive pro-_
[Perfors 2011]

_Lastly, the project of reverse-engineering the mind must unfold over multiple levels of analysis, only one of which has been our focus here. Marr (68) famously argued for analyses that integrate across three levels: The computational level characterizes the problem that a cognitive system solves and the principles by which its solution can be computed from the available inputs in natural environments; the algorithmic level describes the procedures executed to produce this solution and the representations or data structures over which the algorithms operate; and the implementation level specifies how these algorithms and data structures are instantiated in the circuits of a brain or machine. Many early Bayesian models addressed only the computational level, characterizing cognition in purely functional terms as approximately optimal statistical inference in a given environment, without reference to how the computations are carried out (25, 39, 69). The HBMs of learning and development discussed here target a view between the computational and algorithmic levels: cognition as approximately optimal inference in probabilistic models defined over a learner’s subjective and dynamically growing mental representations of the world’s structure, rather than some objective and fixed world statistics.
_
[Tenenbaum 2011]

_Marr (1982) argued that information-processing systems can be analyzed at three levels: the computational level characterizes the problem that a system solves and the principles by which its solution can be computed from the available inputs in natural environments; algorithmic-level analysis describes the procedures executed to produce this solution and the representations or data structures over which the algorithms operate; and the implementa- tion level specifies how these algorithms and data structures are instantiated in the circuits of a brain or machine. Reverse-engineering means beginning at the computational level, trying to understand the function of a system before diving into algorithms and implementation. For this reason, we have referred to it as a top-down or function-first approach (Griffiths et al., 2010). This idea was made explicit in Anderson’s (1990) framework for rational analysis, which focuses on analyzing cognition in terms of adaptive solutions to prob- lems posed by the environment, and resulted in several groundbreaking Bayesian models of cognition. Many early Bayesian models addressed only the computational level, characterizing cog-_
[Griffiths 2024]

=== Rational constructivism

position the framework as a middle way between nativism and empiricism, where domain-general statistical mechanisms construct domain-specific structural knowledge

_associations). Many developmental researchers rejected this choice altogether and pursued less formal approaches to describing the growing minds of children, under the headings of “constructivism” or the “theory theory” (Gopnik & Meltzoff, 1997). The potential to explain how people can genuinely learn with abstract structured knowledge may be the most salient feature of Bayesian cognitive models—the biggest reason for their popularity in some developmental circles (Gopnik & Tenenbaum, 2007; Griffiths, Chater, Kemp, Perfors, & Tenenbaum, 2010; Perfors, Tenenbaum, & Wonnacott, 2010; Griffiths, Sobel, Tenenbaum, & Gopnik, 2011b; Xu, 2019; Spelke, 2022), and the biggest target of skepticism from oth- ers, in both the traditional nativist and empiricist camps (Berwick, Pietroski, Yankama, & Chomsky, 2011; McClelland et al., 2010)._
[Griffiths 2024]

_chology that similarly grew out of a desire to answer the tension between representation and learning in a boundedly rational system. On one side, a purely nativist response was to deny learning and to focus on characterizing the detailed representations that were already in place. On another side, an empiricist response was to suggest that structured represen- tations were not necessary, that learning (and the inferences that followed) was simply a bottom-up process of learning statistical associations. Other debates played out in cognitive development as well. Some research tended to characterized children as “noisy” or “irra- tional” adults, while other research sought to demonstrate that children are efficient and effective rational learners. With the development of the probabilistic framework came a renewed interest in unifying_ [Griffiths 2024]

_Most importantly, the Bayesian approach lets us move beyond classic either-or dichotomies that have long shaped and limited debates in cognitive science: “empiricism versus nativism,” “domain-general versus domain-specific,” “logic versus probability,” “symbols versus statistics.” Instead we can ask harder questions of reverseengineering, with answers potentially rich enough to help us build more humanlike AI systems. How can domain-general mechanisms of learning and representation build domain-specific systems of knowledge? How can structured symbolic knowledge be acquired through statistical learning? The answers emerging suggest new ways to think about the development of a cognitive system. Powerful abstractions can be learned surprisingly quickly, together with or prior to learning the more concrete knowledge they constrain. Structured symbolic representations need not be rigid, static, hard-wired, or brittle. Embedded in a probabilistic framework, they can grow dynamically and robustly in response to the sparse, noisy data of experience._
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

=== $lambda$-calculus as universal formalism

explain why $lambda$-calculus is a natural choice for modeling concepts as stochastic functions that can be combined and recombined

=== why programs for theory of mind? 

_We begin by defining the stochastic λ-calculus, a mathematical system that is able to rep- resent complex generative models using only a few basic constructs. It is based on the λ-calculus, which is one of the principal models of deterministic computation. After intro- ducing this mathematical basis, we will add a number of features to make modeling easier, making this a more practical probabilistic programming language (PPL) for cognitive sci- ence. In particular, we will adopt the PPL Church (Goodman et al., 2012), which elegantly extends the sparse mathematical system of stochastic λ-calculus based on the Scheme dialect of Lisp. How should we begin to build a compositional formal system—a language—for express-_
[Griffiths 2024]

_some fundamental knowledge that is more than just associations between words and cardinalities.
We present a formal learning model which shows that statistical inference over a sufficiently powerful representational space can explain why children follow this developmental trajectory. The model uses several pieces of machinery, each of which has been independently proposed to explain cognitive phenomena in other domains. The representational system we use is lambda calculus, a formal language for compositional semantics (e.g., Heim & Kratzer, 1998; Steedman, 2000), computation more generally (Church, 1936), and other natural-language learning tasks (Piantadosi, Goodman, Ellis, & Tenenbaum, 2008; Liang, Jordan, & Klein, 2009, 2010; Zettlemoyer & Collins, 2005, 2007). The core inductive part of the model uses Bayesian statistics to formalize what inferences learners should make from data. This involves two key parts: a likelihood function which measures how well hypotheses fit observed data, and a prior which measures the complexity of individual hypotheses. We use simple and previously proposed forms of both. The model uses a likelihood function that uses the size principle (Tenenbaum, 1999) to penalize hypotheses which make overly broad predictions. Frank, Goodman, and Tenenbaum (2007) proposed that this type of likelihood function is important in cross-situa-tional word learning and Piantadosi et al. (2008) showed that it could solve the subset problem in learning compositional semantics. The prior is from the rational rules model of Goodman, Tenenbaum, Feldman, and Griffiths (2008), which first linked probabilistic inference with formal, compositional, representations. The prior assumes that learners prefer simplicity and re-use in compositional hypotheses and has been shown to be important in accounting for human rule-based concept learning._
[Piantadosi 2012]

_come to have meaning as CRS supposes, by virtue of how the structures they are mapped to act on other symbols. Learners are able to derive new facts by applying their internal expressions to each other in novel ways. As I show, this can give rise to rich systems of knowledge that span classes of computations and permit learners to extend a few simple observations into the domain of richer cognitive theories.
2 Combinatory Logic as a Language for Universal Isomorphism
A mathematical system known as combinatory logic provides the formal tool we’ll use to construct a universal isomorphism language as a hypothesized LOT. Combi-natory logic was developed in the early- and mid-1900s in order to allow logicians to work with expressions that did not require variables like “x” and “y”, yet had the same expressive power (Hindley and Seldin 1986). Combinatory logic’s usefulness is demonstrated by the fact that it was invented at least three independent times by mathematicians, includingMoses Schönfinkel, John vonNeumann, andHaskell Curry (Cardone and Hindley 2006). The main advantages of combinatory logic are its simplicity (allowing us to posit very minimal built-in machinery) and its power (allowing us to model symbols, structures, and relations). In cognitive research, combinatory logic is primarily seen in formal theories of natural language semantics (Steedman_
[Piantadosi 2019]

=== why programs for theory of mind?

social reasoning requires recursive, compositional representations to handle nested mental states. There's no cleaner way of doing this than with programs. it's basically already a program.

_The pattern of using an embedded query to capture the choices of another agent is a very general pattern for modeling intuitive psychology (Stuhlmüller & Goodman, 2013). We could write down the abstract structure schematically as: (define choice (lambda (belief state goal?)
(query (define action (action-prior)) action (goal? (belief state action)))))
where belief is taken to be the agent’s summary of the world dynamics (transitions from states to states, given actions), and goal? is a goal predicate on states picking out those that the agent desires. Of course many additional refinements and additions may be needed to build an adequate model of human intuitive psychology—agents form_
[Goodman 2015]

_Probabilistic Programs as a Unifying Language of Thought 467
probabilistic programming for Bayesian inverse planning to more abstract symbolic plan- ning settings, using the Gen PPL (Cusumano-Towner et al., 2019) and the Planning Domain Description Language to represent an agent’s world models, goals, and plans using Gen also enables explicit probabilistic modeling of a boundedly rational agent’s approximate plan- ning algorithm to make goal inference robust when the agent might be prone to planning mistakes of various kinds. Of course, many additional components will be needed to build fully adequate models of_
[Griffiths 2024]

_or the blocked agent is previously associated as negative, or...’
One alternative to a cue-based account is to use generative models of action choice, as in the Bayesian inverse planning (or “Bayesian theory-of-mind”) models of Baker, Saxe, and Tenenbaum (2009) or the “naive utility calculus” models of Jara-Ettinger, Gweon, Tenenbaum, and Schulz (2015) (See also Jern and Kemp (2015) and Tauber and Steyvers (2011), and a related alternative based on predictive coding from Kilner, Friston, and Frith (2007)). These models formalize explicitly mentalistic concepts such as ‘goal,’ ‘agent,’ ‘planning,’ ‘cost,’ ‘efficiency,’ and ‘belief,’ used to describe core psychological reasoning in infancy. They assume adults and children treat agents as approximately rational planners who choose the most efficient means to their goals. Planning computations may be formalized as solutions to Markov Decision Processes (or POMDPs), taking as input utility and belief functions defined over an agent’s state-space and the agent’s state-action transition functions, and returning a series of actions the agent should perform to most efficiently fulfill their goals (or maximize their utility). By simulating these planning processes, people can predict what agents might do next, or use inverse reasoning from observing a series of actions to infer the utilities and beliefs of agents in a scene. This is directly analogous to how simulation engines can be used for intuitive physics, to predict what will happen next in a scene or to infer objects’ dynamical properties from how they move. It yields similarly flexible reasoning abilities: Utilities and beliefs can be adjusted to take into account how agents might act for a wide range of novel goals and situations. Importantly, unlike in intuitive physics, simulation-based reasoning in intuitive psychology can be nested recursively to understand social interactions – we can think about agents thinking about other agents._ [Lake 2016]

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

=== agents as approximately rational planners

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

_Bayesian Models of Cognitive Development 513
Skeletal structured generative models are one route for embodying and implementing core knowledge, but other approaches are being developed as the AI and machine learn- ing community re-engages with findings from cognitive development. One such promising direction, specifically in intuitive physics, is to combine artificial neural networks or graphs with different minimal notions of objects, and to let the network discover the dynamics and interactions between objects (for several recent examples, see Mrowca et al., 2018; Battaglia et al., 2016, 2018; Chang, Ullman, Torralba, & Tenenbaum, 2016). A very different compu- tational approach, however, is try to recover the principles of core knowledge purely from vast amounts of empirical data (for recent examples in intuitive physics and psychology, respectively, see Piloto et al., 2018; Rabinowitz et al., 2018). Such blank-slate models on their own do not yet generalize well, but it is too early to say_ [Griffiths 2024]

=== Inverting the planner

Show how the observer uses Bayesian induction to work backward from a trajectory to the most likely goals and beliefs

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

_eral Bayesian treatment of “theory of mind” or “mentalizing”: the core human capacity to observe other agents’ actions and infer the beliefs, desires, and intentions that gave rise to them, assuming that those actions were the result of approximately rational goal-directed planning and decision-making processes in the agent’s mind. Chapter 14 develops the
Bayesian approach to modeling theory of mind with a focus on understanding the actions of agents moving around us and in our local environment, on relatively short spatial and tem- poral scales, as developed in seminal work by Baker, Saxe, and Tenenbaum (2009), Lucas, Griffiths, et al. (2014), Jern, Lucas, and Kemp (2017), Jara-Ettinger, Gweon, Schulz, and Tenenbaum (2016), Baker, Jara-Ettinger, Saxe, and Tenenbaum (2017), and Ullman et al. (2009). At its core, this framework posits that individuals infer the intentions and beliefs of others by observing their actions and then “inverting” the agents’ planning process. Instead of predicting future actions based on given intentions (as in forward planning and rational decision-making; see chapter 7), Bayesian inverse planning starts with the observed actions and works backwards to infer the most likely intentions and beliefs that could have produced those actions. To achieve this, the model uses Bayes’ rule, combining prior beliefs about the actor’s preferences and intentions with the likelihood of the observed actions, given that those intentions assume an approximately rationally planner, to produce a posterior distribu- tion over possible intentions. This method provides a structured way to interpret the behavior of others in terms of underlying mental states, giving us not only a candidate mechanism behind the human ability to understand and predict the actions of others, but also opening up a vast landscape of insights and modeling opportunities for human social cognition and the cultural basis of cognition more generally. Finally, the Bayesian approach can be used to shed light on some aspects of cultural_ [Griffiths 2024]

_tion (14.24)). Using the resulting policy as the generative model, joint belief (the probability distribution over states), desire (the latent reward function), and competence (the underly- ing cost function) can be inferred through Bayesian inference, where given an observed trajectory t,
p(B, R, C|t) ∝ p(t|B, R, C)p(B)p(R)p(C). (14.25)
Baker et al. (2017) developed and tested the model that we have just presented, asking adult participants to make joint inferences about agents’ beliefs and desires based on how they navigated an environment. Figure 14.11 shows several scenarios from this experiment, in which a hungry graduate student leaves their office to walk to lunch at one of three food trucks: Korean (K), Lebanese (L), or Mexican (M). There are two parking spots for the trucks (marked in yellow), and trucks can park in different spots on different days, or not show up at all, so the student may not know where each truck is parked and must plan carefully where to walk to get lunch from the best truck available as quickly as possible. Using a POMDP for a generative model, the agent’s desires can be captured using a reward function that represents their preferences over trucks, and the agent’s initial beliefs can be represented as a probability distribution over each of three partially observable world states: the Northeast parking spot being occupied by (1) Lebanese (L) or (2) Mexican (M), or (3) being empty (N for none). Finally, observations of the trucks are determined by line of sight, with a small probability of observation failure. Consider figure 14.11c, in which the student can initially see the Korean truck in the_ [Griffith 2024]

=== From skeletal knowledge to rich theories

show how DreamCoder can start with a skeletal planner (the principle of efficiency) and iteratively construct a library of complex social concepts like "helping", "hindering", or "chasing"

_Skeletal structured generative models are one route for embodying and implementing core knowledge, but other approaches are being developed as the AI and machine learn- ing community re-engages with findings from cognitive development. One such promising direction, specifically in intuitive physics, is to combine artificial neural networks or graphs with different minimal notions of objects, and to let the network discover the dynamics and interactions between objects (for several recent examples, see Mrowca et al., 2018; Battaglia et al., 2016, 2018; Chang, Ullman, Torralba, & Tenenbaum, 2016). A very different compu- tational approach, however, is try to recover the principles of core knowledge purely from vast amounts of empirical data (for recent examples in intuitive physics and psychology, respectively, see Piloto et al., 2018; Rabinowitz et al., 2018). Such blank-slate models on their own do not yet generalize well, but it is too early to say_ [Griffiths 2024]

_C.L. Baker et al. / Cognition 113 (2009) 329–349 349
Kautz, H., & Allen, J. (1986). Generalized plan recognition. In Proceedings of the fifth national conference on artificial intelligence (pp. 32–37).
Kemp, C., Perfors, A., & Tenenbaum, J. B. (2007). Learning overhypotheses with hierarchical Bayesian models. Developmental Science, 10(3), 307–321.
Körding, K. (1997). Decision theory: What ‘‘should” the nervous system do? Science, 318(5850), 606–610.
Liao, L., Fox, D., & Kautz, H. (2004). Learning and inferring transportation routines. In Proceedings of the nineteenth national conference on artificial intelligence (pp. 348–353). _ [Baker 2009]

_programming languages—as candidate mechanistic models for thinking. And finally, by drawing on hierarchical Bayesian frameworks for learning inductive constraints and learn- ing to learn, the PLoT lets us think in terms of programs as hypotheses for novel concepts, and probabilistic meta-programs that generate domain-specific languages of programs as powerful and dynamically evolvable hypothesis spaces and priors for inductive concept learning. Looking ahead, we should start by acknowledging that Church models, and PLoT models_ [Griffiths 2024]

_Each cycle the system is additionally tested on the domain’s
held-out tasks. Testing time is consistent across all domains: the system is provided 10 minutes per task to search for a solution using its current library and recognition model. Fig. 5A (Row 1) shows the percentage of test tasks solved in each cycle by all systems. Except for text editing, where performance remains comparable across all systems, utilizing the recognition model for chunking (dream decompiling) enables faster domain proficiency and enhanced generalization through the learnt library. This distinction is most evident during the intermediate cycles of learning, following similar performance in the initial iterations and before proceeding to converge again in the later iterations. Notably, this occurs despite all systems having solved a similar number of training tasks throughout (Fig. 6, Appendix C). At the respective peak differences (excluding text editing), DDC-PC outperforms DREAMCODER by 13.25% on average test_ [Palmarini 2024]

