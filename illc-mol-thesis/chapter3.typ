#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Bayesian Theory of Mind")



== Agent as planner <sec-planner>

In this chapter we first claim that Bayesian Theory of Mind (BToM) is the best available computational account of the mature ToM capacity described in Chapter 1. BToM formalises the intentional stance as inverse planning, and its posteriors quantitatively track human judgements. We then claim that in BToM the structure of belief attribution is stipulated by the modeler rather than inferred from the data.

To adopt the intentional stance towards a system is to treat it as a rational agent. First attribute to it the beliefs and desires it ought to have (given its situation and history), then predict its behavior by working out what a rational agent would do @dennett_intentional_2002. People adopt the intentional stance even towards two triangles and a circle moving about a rectangle, seeing them as chasing, hiding, bullying, and escaping each other @heider_experimental_1944. The intentional stance is a strategy that gets used because it regularly yields predictive success @dennett_intentional_2002.

Bayesian Theory of Mind (BToM) gives a computational account of this strategy, modeling the agent as a planner. The agent occupies one of a set of states, has available in each state a set of actions, and the actions move it between states according to a transition function. The planner takes as input a utility function and a model of the environment, both defined over the agent's state space, and returns the actions the agent should perform to maximize utility @lake_building_2016 @baker_action_2009. Even twelve-month-old infants encode actions as directed at a goal staet, and expect an agent to use the most efficeint means available to @timeline. 

== Inverse planning <sec-inverse>

A planning agent computes in the forward direction: given the model of the world $m$ and a goal $g$, find the optimal behavior (the trajectory $t$) for achieving that goal. So the agent computes $t = "optimize"(g,m)$, where "optimize" is a planning algorithm that finds the best path to $g$ which is available on $m$.

An observer watching an agent needs to compute in the backward direction: given the world model $m$ and the agent's trajectory $t$, find the goal that, assuming a rational agent, would yield such a behavior. The observer can simulate what the agent would have done if its goal were $g'$, calculating the resulting trajectory as $t'="optimize"(g',m)$ since the observer assumes the agent is rational. Then if the simulated trajectory $t'$ matches the actual trajectory $t$, then the hypothesis that $g'$ is the agent's goal (that $g = g'$) is correct. The process is analogous to modeling intuitive physics by running a physics engine forward under different settings and scoring the outcomes @lake_building_2016.

Using Bayes' rule (@sec-bayes), the learner finds the hypothesized goal that best explains the agent's behavior. Writing $g$ for the agent's goal, $m$ for its model of the environment, and $t$ for the observed agent's trajectory, the posterior is calculated as

$
P(g, m | t) prop P(t | g, m) dot P(g, m)
$

where the prior $P(g,m)$ encodes what the observer expects agents to want before seeing anything, and the likelihood $P(t|g,m)$ encodes the probability that an approximately rational agent with goal $g$ and model $m$ would move as observed (which is supplied by the planner) @baker_action_2009. It's been shown experimentally that people's predictions on behavioral stimuli are close to the model posteriors of this approximately rational inference mechanism @baker_action_2009.

== Belief as an input <sec-false-belief>

The posterior is joint over $(g,m)$, the agent's goal $g$ and a model $m$ of the environment in which it navigates. But $m$ doesn't have to be an accurate model of the actual state of affairs (the observer's world), it could be any possible world model. The observer can supply a counterfactual model $m'$ that diverges from the actual state of affairs, and then run the same posterior computation to see how the agent would behave if it were navigating in the counterfactual world $m'$. Then the same computation that yields sensible behavior from an accurate model would yield mistaken behavior from the inaccurate one.

This is belief attribution in essence, attributing a counterfactual world to an agent such that it acts according to the counterfactual model as its own private world. So we can ask: given an (irrational) trajectory, which model of the environment would've made those actions rational? Take the false-belief task (@timeline):  Sally puts her marble in the basket and leaves; in her absence the marble is moved to the box; and when she returns she searches the basket @wimmer_beliefsabout_nodate. Evaluated against the actual (observer's) world $m$, Sally searching the basket is irrational since the marble is in the box. But evaluated against her private model she had when she left $m'$ (that the marble is in the basket), her searching the basket is optimal. An observer that can represent the divergence of worlds can predict that Sally will look in the wrong location, and even which wrong location she'll look in @goodman_intuitive_nodate.

The inverted computation over counterfactual models broadly requires two steps. First, from her own model $m$ the observer derives the agent's private model by $m' = "derive"(m)$,  where "derive" modifies $m$ to account for the agent's imputed false belief, yielding counterfactual model $m'$. In the Sally--Anne example, in the observer's model $m$ (the real world) the marble is in the box, whereas Sally's model $m'$ (her private world) is the same as $m$ in every respect except that the marble is in the basket. 

The observer's second step is to run the agent's planner on its private model, to yield its trajectory (behavior) given that model. Rather than running $t'="optimize"(g',m)$ on her own model $m$ (as in @sec-inverse), the observer runs $t'="optimize"(g',m')$ on the agent's private model $m'$. So $t'$ is the agent's predicted trajectory, given that it pursues goal $g'$ along the most optimal path afforded by counterfactual model $m'$. 



where the derive carries whatever the agent is wrong about --- for Sally, it puts the marble back in the basket --- and $hat(t)$ is the trajectory the observer predicts, to be scored against the trajectory observed. Everything belief-specific lives in the derive. The planner is the planner of @sec-planner, handed $m'$ exactly as it would be handed $m$, indifferent to whether its input is anybody's actual world. This two-step decomposition --- a divergent model derived, a planner run against it unchanged --- is the structure whose acquisition the rest of this thesis is about: Chapter 4 transcribes it into its learner's language as the compound the search must assemble (@signature), and Chapter 5 reads the term the search finds against it, line by line.

To handle divergent models, the observer needs to run the planning computation inside itself: once against its own model $m$, to know what is actually the case, and once against the model $m'$ it attributes, to work out what the agent will do. This kind of nested computation is another reason that representing hypotheses as programs (@sec-programs) is especially fitting for the ToM domain @stuhlmuller_reasoning_2014. Regardless of whether a program takes $m$ or $m'$ as input, it's executed and evaluated in the same way. To attribute a mental state is to say something about the process by which an agent selects its actions, and a program is the natural way to write down that procedure.

== What is stipulated in BToM <stipulated>

BToM is a model of the ToM capacity of a mature adult @baker_action_2009 @baker_rational_2017. In this thesis we're interested in how that capacity is acquired, specifically whether a system without a hardcoded BToM can develop something akin to it. So when we introduce our model we'll need to show that our learner doesn't have a built-in BToM. To argue that, we first have to be precise about which features aren't given by the data but are stipulated by the framework. There are five, and @no-btom answers them in this order.

*The decomposition into agent, goal and belief.*
A scene depicting entities moving around doesn't inherently encode them as agents and goals, and it doesn't inherently encode their behavior as generated by two attitudes rather than one or five. But BToM decomposes the scene into an agent that has a goal and a belief, which the planner takes as separate arguments that vary and can be inferred independently of each other. The decomposition is in the architecture, not in the scene.

*The separation of the believed world from the actual world.*
The planner is evaluated against a world model $m'$ supplied as one of its inputs, and that input is permitted to diverge from the state of affairs the observer takes to obtain, $m eq.not m'$. This is the stipulation that makes false belief representable at all, and it too is made in the architecture rather than inferred from any scene.

*The space of admissible divergences.*
Given that $m$ and $m'$ may diverge, something has to specify which divergences are admissible. What's inferred under BToM is which particular world model $m'$ the agent holds, but the range it's drawn from is fixed in advance, along with the scenarios that produce it --- an occlusion, an absence while the world was rearranged. That a belief could be false, and how it came to be, is never itself in question.

*The space of goals.*
Likewise on the other input. The posterior over goals is inferred, but the space it ranges over is given, and so is the prior on it. Baker et al. compare several goal priors precisely because the choice matters and is the modeller's to make @baker_action_2009. What inference does is locate a point inside a space someone else drew.

*Rationality.*
That the agent plans approximately optimally toward its goal given its model. This is the one item on the list with independent developmental support, since the teleological stance is in place well before the first birthday (@sec-planner, @timeline), and it is correspondingly the one item this thesis also grants its learner.

The consequence is that inference in these models is parameter estimation within a fixed theory rather than construction of the theory's terms. That is exactly the distinction drawn in @explanandum: the underdetermination problem, to which Bayesian inversion is the answer, presupposes a learner who already possesses the concept _believes_ and has only to choose among ascriptions. A learner who lacks the concept cannot make an ascription at all, and no amount of inverting a planner will supply it, because the planner's shape is what encodes it.

The same fixity leaves the developmental ordering of @timeline outside the model's reach. Since the structure is complete from the start, every one of its parameters is available for inference on day one. There is nothing in the framework that makes a goal attribution cheaper, or earlier, or prior to a belief attribution, so there is nothing in it that predicts why the first is reliable before the first birthday and the second not until around four. The delay has to be explained by something the model doesn't contain — maturation, or performance demands masking an underlying competence — which is the same appeal to auxiliary hypotheses that @nativism was charged with in @perspectives, and it's available here for the same reason: the structure that would have to develop is assumed rather than derived.

Written out as a list, in fact, the stipulations are close to an inventory of what the modularity-nativist claims is innate. Attitude concepts introduced as such, a proprietary format in which they are ascribed to agents, and inferential machinery that operates on agents in particular @leslie_pretending_1994 — the difference is that BToM says it in probability theory and says it precisely. So the nativist can read the framework's success as evidence _for_ the position of @nativism rather than against it: here is the endowment, written down, and here is how closely a system so endowed matches human judgement. The reading isn't perverse, and it's what the models license, given that they make no claim at all about where the endowment comes from. What they establish is that a learner who ends up with this structure would attribute mental states as people do. Where such a learner could get the structure is a question they leave open, and it's the question of the next section.

== What this thesis asks

The five items of @stipulated have a shape we already have a name for. They are not hypotheses about any particular scene; they are hypotheses about what the hypotheses about scenes are like --- that behaviour is generated by an agent, that the agent has a goal and a model, that the model may diverge from the world, that the divergence is drawn from such-and-such a range. In the terms of @sec-hbm they are the learner's language rather than anything written in it: the vocabulary in which a particular attribution gets stated at all. BToM installs that vocabulary and infers underneath it. The question of this thesis is whether it can be inferred rather than installed --- whether the language, given to the framework by its modeller, is the kind of thing a learner could arrive at by the same inference it runs on everything else.

Putting it that way fixes what the model of the next chapter has to do. It cannot be handed any of the five. Its learner begins with grids, a planner, and domain-general combinators for building and consuming pairs --- no agent, no goal, no attitude, no channel that is anybody's --- and it has to assemble the structure out of them, in program space, under a description-length objective that knows nothing about which tasks are mental. This raises the bar past expressiveness. A language in which belief is merely _statable_ proves nothing, since the stipulated architecture is statable in any language rich enough to write it down. What has to be shown is that the structure is _selected_: that among the programs which reproduce the scenes, the one carrying the agency signature is the one compression keeps, and that it beats the non-mental rivals the very same library can express. @criteria states that test in advance and @no-btom takes the five stipulations in the order given here.

Two things are worth conceding before the model is on the table. The first is that one item of the five is granted rather than earned. Rationality comes in as the `optimize` primitive, on the grounds @sec-planner gave: the teleological stance is in place before the first birthday, three years before the capacity whose acquisition is being modelled, and on the best analysis of it it attributes no representation to anybody. @rationality defends the choice at length and does not pretend it is free.

The second concession is about what is being asked for. BToM's $m'$ is a belief in the full sense: a state the agent came to be in, that could have been otherwise, and that would be revised if the agent looked again. What the learner here is asked to find is thinner --- a divergent world model, attributed to an agent and acted on by that agent's policy, but with no account of how the agent came to hold it and no machinery for updating it. Nothing in a single-frame transition function represents perceptual access, or a record of what was witnessed, or an update on being shown the truth. That is a real gap between the discovered structure and the thing it models, and it is the most important one; @sec-persistent takes it up as further work.


#load-bib(read("refs.bib"))