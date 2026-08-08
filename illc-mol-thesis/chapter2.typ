#import "lib.typ": *

#show table.cell.where(y: 0): set text(style: "normal", weight: "bold")
#show table: set par(justify: false)
#set table(stroke: (_, y) => if y == 1 { (top: 0.9pt) } else if y > 1 { (top: 0.2pt) })

#mol-chapter("Learning as Bayesian Program Synthesis", lbl: <ch-learning>)

== The computational-level problem <sec-computational>

The learner observes a visual scene depicting a trajectory of bodies through space, and outputs an explanation of the process that generated the trajectory. The output process is formulated in terms of mental states (beliefs, desires, intentions) that don't appear in the data. The trajectory underdetermines the process, since any behaviour is compatible with indefinitely many belief-desire pairs (@explanandum). So no amount of further observation eliminates all but one.

The problem is of a familiar form: given data, entertain hypotheses about the process that generated it, assign each a prior plausibility, and revise those plausibilities as new data comes in. _Bayes' rule_ is the known answer to the normative question of how such revision ought to proceed. So the computational-level claim @marr_vision_2010 is that acquiring a theory of mind is the problem of finding the best explanation of observed behaviour, where the best explanation takes into account both its prior plausibility and how well it fits the data.

== Bayesian inference: the logic of learning <sec-bayes>

Say a learner observes data $d$, a trajectory of bodies through space. A learner's task is to choose among possible explanations $cal(H)$ a hypothesis $h$ that best explains the trajectory. The task is to choose the most plausible hypothesis given the data, i.e. to find the hypothesis whose posterior probability $P(h|d)$ is highest. In calculating this quantity, the learner should weigh two things. First, the prior probability $P(h)$: how plausible was hypothesis $h$ before any data $d$ was observed at all? Second, the likelihood $P(d|h)$: how well does hypothesis $h$ account for the data $d$, now that data is available? Bayes' rule specifies how to combine these considerations: 

$
P(h|d) = (P(d|h) P(h)) / (sum_(h' in cal(H)) P(d|h') P(h'))
$

Since the denominator is a normalising constant that doesn't depend on $h$, it can be ignored for the purpose of ranking hypotheses against each other. So we rewrite the equation as the proportion

$
  P(h|d) prop P(d|h) dot P(h)
$

The posterior probability  $P(h|d)$ of a hypothesis $h$ given data $d$ is proportional to the likelihood $P(d|h)$ of the data $d$ given hypothesis $h$ times the prior probability $P(h)$ of hypothesis $h$ before data $d$ was observed. Bayes' rule is a theorem of the probability calculus: given the basic commitment to representing degrees of belief as probabilities, it's the unique coherent way to revise them in light of evidence @jaynes_probability_2003 @mackay_information_2019. Having stated the computational problem the mind faces, Bayes' rule specifies the ideal solution.#footnote[To be clear this claim is at the computational level, not at the algorithmic level. Nobody is saying that a child computes posterior probabilities, or represents the hypothesis space $cal(H)$ explicitly, etc. The claim is that the child's behavior approximates the solution to an inference problem whose ideal form is described by Bayes' rule. It's not an algorithmic-level claim about the procedure by which the child does this, though we'll later introduce an algorithm for approximating this inference.]

The prior $P(h)$ expresses the a priori plausibility of a hypothesis, i.e. how likely the learner takes $h$ to be, independent of the current observations. So the prior encodes the learner's background knowledge. For example a hypothesis that is outright impossible has prior $P(h)=0$ so it's excluded from $cal(H)$ and no data observation can increase its posterior probability. The prior determines which explanations are more natural, so we can start to generalize from among the indefinitely many hypotheses consistent with the data.

Anything can be a prior, since it's just what encodes the learner's background knowledge, assumptions, and preferences. In this thesis the prior we'll focus on is the preference for simplicity. Hypotheses that are short, that reuse existing structure rather than stipulating new structure, will have higher priors than hypotheses that are long and idiosyncratic. This assumption is supported by accounts of human rule-based concept learning @goodman_rational_2008 @chater_simplicity_2003, and by the finding that the difficulty of learning a Boolean concept tracks the length of its shortest logical expression @feldman_minimization_2000. In @sec-programs we'll specify how the simplicity of a hypothesis is calculated, after we've said what hypotheses are made of.

The likelihood $P(d|h)$ expresses how probable the observed data would be if the hypothesis were true. In our setting a candidate hypothesis either explains the observed behaviour (i.e. it reproduces the trajectory exactly) or it does not. So the likelihood is all-or-nothing: the data have probability zero $P(d|h)=0$ if they disagree with the hypothesis, and constant probability otherwise @goodman_rational_2008. The binary likelihood can only ever eliminate hypotheses (likelihood $P(d|h)=0$ in the product yields posterior $P(h|d)=0)$. So once a set of hypotheses all fit the data perfectly (only those with constant likelihood remain), the likelihood can't distinguish between the remaining hypotheses and their ranking is decided entirely by the prior.

== Programs as the hypothesis space <sec-programs>

Bayes' rule doesn't specify what we should define our hypothesis space over, so we can choose the best medium for representing our hypotheses. Since we choose simplicity as our prior, hypotheses should have parts that can be counted so their relative size and complexity can be compared. And a hypothesis must be executable, since the indicator likelihood asks whether running a hypothesis reproduces the observed trajectory. A finite learner must have unbounded and systematic capacity (@explanandum), so the hypotheses must be generated by rules of combination @fodor_connectionism_1988. And also the hypothesis space must admit new terms, since we want to model how a learner acquires the terms that denote mental content (attitudes) even though they don't appear in the data.

These requirements rule out connectionist representations (@empiricism) where a hypothesis is a weight vector or a point in a high-dimensional parameter space @tenenbaum_how_2011. Though they're efficient to update, they lack a natural notion of countable constituent parts so we can't compare their relative sizes. What we need is a structured symbolic representation like a grammar, predicate logic, relational schemas, or functional programs @griffiths_bayesian_2024 @tenenbaum_how_2011. What we need is a language of thought @fodor_language_2010, where mental representations are symbolic objects combined by syntactic rules and evaluated as a function of their parts @bruner_study_1986 @carey_origin_2011 @siskind_computational_nodate.

Programs are the natural medium for our hypothesis representation. They are compositional, since the value of a composite expression is a function of the values of its parts. And a program admits any expression of the right type in any argument position of that type. They are executable, which is what the indicator likelihood requires. And they encode procedural and declarative knowledge in the same format @rule_precis_nodate.

Problem solving has been treated as symbol manipulation since the advent of the computational theory of mind which views the mind as a compact program that mirrors the structure of the world @newell_elements_1958 @baum_what_2004 @goodman_concepts_2014. A symbolic medium also leaves the learner free to adopt new vocabulary beyond what the modeler supplied @gopnik_reconstructing_2012. And furthermore, if knowledge is programs then learning is program induction @flener_introduction_2008 @schmid_aaip_nodate @gulwani_program_2017 which we'll soon make use of.

We adopt the $lambda$-calculus as our program representation formalism, since it's the standard choice for models of learning across domains @piantadosi_bayesian_nodate @zettlemoyer_learning_nodate @piantadosi_bootstrapping_2012 @schmidt_meaning_nodate. Most importantly it gives us $lambda$-abstraction as the operation of taking a recurring piece of structure and giving it a name. So this way a learner can come to possess a term that names something that isn't observable in the data, since the new term's content is fixed entirely by what it does in combination with the other terms.#footnote[Again, we don't claim the mind implements a $lambda$-calculus interpreter, only that its ToM competence is well described as inference over hypotheses that compose, execute, embed, and can be measured for size (which are all properties the $lambda$-calculus representation supplies).]

Since our programs are composed of symbols of a grammar, a distribution over a grammar's productions induces a distribution over expressions. Since the probability of an expression is the product of the probabilities of the choices made in generating it, longer expressions (involving more choices) get less probability mass. The cost of writing a message in a code optimal for that distribution is the negative log of its probability @shannon_mathematical_1948, so $h$'s description length under the given grammar is $"DL"(h)=-log P(h)$. So the prior is $P(h) prop e^(-"DL"(h))$. And we have an indicator likelihood for whether executing $h$ reproduces the observed behavior. Substituting into Bayes' rule, the posterior is computed as

$
P(h|d) prop e^(-"DL"(h)) dot bb(1)[h "solves" d]
$

So maximizing the posterior is just minimizing the description length of correct programs. The learner should adopt the program which has the shortest description length of all programs that solve the data (i.e. that reproduce the trajectory). This is basically the theory theory (@constructivism): the learner should prefer the simplest theory that accounts for the evidence.

== Program induction and the search problem <sec-search>

The trouble is that the optimization problem of searching through program space for one that fits the data is about as badly behaved as an optimization can be#footnote[Here we start to move from the computational level toward the algorithmic one, i.e. the procedure in which the computational problem is solved @marr_vision_2010 @tenenbaum_how_2011]. The space is infinite, since a grammar whose productions can nest inside one another generates unboundedly many expressions, and the number of expressions of a given size grows exponentially in that size. There's no gradient to follow, because programs are discrete: there's no meaningful sense in which one is slightly perturbed into another. And the binary likelihood doesn't give partial credit, so a program that differs from a correct one by a single symbol receives exactly the same zero-likelihood as a wrong program picked at random. So the search is blind: there's no signal anywhere in the space to hill-climb on, and no way to tell that one is getting warm.

To make the search less inefficient at least we can enumerate hypotheses in decreasing order of prior probability, checking each against the data until one fits, so given enough time we find a solution program. Since $P(h) prop e^(-"DL"(h))$, enumerating every program of description length at most $ell$ takes time on the order of $e^ell$. The time to find a solution is exponential in the description length of that solution under the current library.

Note that in @sec-programs we introduced description length as a way of ranking hypotheses, but this way it also determines which hypotheses are even reachable at all. The prior is simultaneously a measure of plausibility and a budget. A concept's being expressible in the language doesn't guarantee that it's learnable in the language within a reasonable budget. 

Description length is relative to the library available, so the same program may be long under one library and short under another that happens to contain the right symbols. So facing the search problem, we should just change the library so the target gets shorter. If we add symbols to the library that are useful for the domain, programs that were previously out of reach will fall within the budget since they can now be expressed more succinctly. This doesn't make the problem tractable in general, but it might make particular programs tractable when they weren't before. 

== Learning the language itself <sec-hbm>

If we're permitted to adjust the library whenever the search is going badly, then the modeler can make anything learnable by just adding a primitive that does the work. What we need is an account on which changing the library is not a device the modeler applies from outside but something the learner does as part of the same inference it was already doing.

In the setup (@sec-bayes) the learner is given a hypothesis space $cal(H)$ and a prior $P(h)$ which is the type-uniform distribution over the library's symbols. So revising the prior is just changing which symbols the learner has. When a structure recurs across the solutions found for a domain, that recurrence is evidence that the structure is useful. It can then be named, added to the library, and thereafter costs a single symbol wherever it appears. Programs that previously were prohibitively long can now be expressed more succinctly, since the complexity of the often-used structure is abstracted into a new named symbol.

Adding a symbol --- revising the library --- follows a similar inferential principle as in @sec-programs but applied a step further out. Given a corpus of observations $D = {d_1, ..., d_n}$, the goal is to find the best library $cal(L)$ for that corpus. The posterior $P(cal(L)|D)$ is given by:

$
P(cal(L)|D) prop P(cal(L)) dot product_(i=1)^n sum_h P(d_i|h) dot P(h|cal(L))
$

A library's prior $P(cal(L))$ is determined by its simplicity: a library with fewer and less-complex symbols has higher prior probability. Without this complexity cost the best library is always the one carrying a dedicated symbol for every task in the corpus, so it overfits to the specific corpus and won't generalize to similar corpora of the same domain. 

The product runs over the corpus of $n$ tasks $D = {d_1, ..., d_n}$. For a given task $d_i$ and a given hypothesis $h$, the likelihood $P(d_i|h)$ indicates whether executing program $h$ correctly produces $d_i$ (zero otherwise). And $P(h|cal(L))$ is the description length prior (@sec-programs) relative to a library, so $e^(-"DL"(h|cal(L)))$ is the cost of writing $h$ in $cal(L)$. Then the program $h$ is marginalized away, since a library just needs to make an adequate hypothesis available.

Because the likelihood is an indicator, every $h$ that fails to reproduce $d_i$ contributes nothing to the sum. So the inner sum collapses to the total prior mass the language places on programs that solve the task, $sum_(h "solves" d_i) P(h|cal(L))$. So the equation says that a good library is one under which solutions are cheap, and cheap for every task in the corpus at once. This is the posterior over libraries inferred in @ellis_dreamcoder_2020. 

Learning the library rather than a hypothesis within it has been done for intuitive theories generated by a probabilistic grammar @ullman_theory_2010, and for program fragments transferred from solved problems to unsolved ones @liang_learning_2010. DreamCoder adds that what gets learned is a library of named $lambda$-expressions, so that revising the library means acquiring new concepts rather than redistributing probability among symbols the learner already had @ellis_dreamcoder_2020.

But settling a library is an even worse optimization problem than program space search (@sec-search), since the space of candidate libraries is larger than the space of hypotheses under any one of them. DreamCoder approximates it by maximizing a lower bound on the posterior @ellis_dreamcoder_2020. In rounds, it alternates solving a batch of tasks under a fixed library, then revising the library in light of what the solutions turn out to have in common @dechter_bootstrap_2013. An abstraction found in a later round may be defined using abstractions found in earlier ones, so the library accumulates in layers.

== Our claim

In our picture of learning, "strong but domain-general inference and representational resources are aided by weaker, domain-specific perceptual input analyzers" @goodman_learning_2011. Of course something must be built-in, so our disagreement with the nativist is about what that must be. On the modular view what's innate includes the attitude concepts themselves, a proprietary representational format for ascribing them, and the agent-directed inferential machinery that operates on it @leslie_pretending_1994. On our view what's innate is a capacity to represent structured hypotheses, a preference for short ones, a procedure for searching, and a stock of low-level domain-general primitive operations. So anything specific to the mental is constructed.

The construction of a conceptual vocabulary is cumulative @carey_origin_2011 @piantadosi_bootstrapping_2012. An abstraction acquired at one stage becomes a term in the language in which the next stage's hypotheses are written, so at each stage more complex hypotheses can more easily be posited. This correctly predicts the developmental progression (@timeline): ToM is a layered vocabulary, and its components should become available in an order fixed by what each one is built out of.

Goal-directedness is available early because it's an abstraction over a great many observed pursuits. Belief is available late because it's an abstraction over goal-directed pursuit together with the ability to posit counterfactual "what-if"s, so it can't be assembled before its parts exist. The developmental delay (@explanandum) is a consequence of the compositional structure of what's being learned, so it's directly explained by the theory theory (@palmarini_bayesian_2024). 

We give the argument: if a learner equipped only with domain-general primitives and a preference for short programs comes to posit belief-like structure, then positing such structure is something a learner _can_ in principle do rather than something it must be given. So a counterexample blocks the nativist's inference from "no mechanism has been named" to "no mechanism exists". Whether human learners use this mechanism, or some other one that solves the same problem, is a further question that this thesis does not settle.

#load-bib(read("refs2.bib"))