#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Bayesian Theory of Mind")



== Agent as planner <sec-planner>

In this chapter we first claim that Bayesian Theory of Mind (BToM) is the best available computational account of the mature ToM capacity described in Chapter 1. BToM formalises the intentional stance as inverse planning, and its posteriors quantitatively track human judgements. We then claim that in BToM the structure of belief attribution is stipulated by the modeler rather than inferred from the data.

To adopt the intentional stance towards a system is to treat it as a rational agent. First attribute to it the beliefs and desires it ought to have (given its situation and history), then predict its behavior by working out what a rational agent would do @dennett_intentional_2002. People adopt the intentional stance even towards two triangles and a circle moving about a rectangle, seeing them as chasing, hiding, bullying, and escaping each other @heider_experimental_1944. The intentional stance is a strategy that gets used because it regularly yields predictive success @dennett_intentional_2002.

Bayesian Theory of Mind (BToM) gives a computational account of this strategy, modeling the agent as a planner. The agent occupies one of a set of states, has available in each state a set of actions, and the actions move it between states according to a transition function. The planner takes as input a utility function and a model of the environment, both defined over the agent's state space, and returns the actions the agent should perform to maximize utility @lake_building_2016 @baker_action_2009 @baker_bayesian_2012. Even twelve-month-old infants encode actions as directed at a goal staet, and expect an agent to use the most efficeint means available to @timeline @gergely_taking_1995 @gergely_teleological_2003 @csibra_obsessed_2007.

== Inverse planning <sec-inverse>

A planning agent computes in the forward direction: given the model of the world $m$ and a goal $g$, find the optimal behavior (the trajectory $t$) for achieving that goal. So the agent computes $t = "optimize"(g,m)$, where "optimize" is a planning algorithm that finds the best path to $g$ which is available on $m$, since the agent is rational (@sec-planner).

An observer watching an agent needs to compute in the backward direction: given the world model $m$ and the agent's trajectory $t$, find the goal that, assuming a rational agent, would yield such a behavior. The observer can simulate what the agent would have done if its goal were $g'$, calculating the resulting trajectory as $t'="optimize"(g',m)$ since the observer assumes the agent is rational. Then if the simulated trajectory $t'$ matches the actual trajectory $t$, then the hypothesis that $g'$ is the agent's goal (that $g = g'$) is correct. The process is analogous to modeling intuitive physics by running a physics engine forward under different settings and scoring the outcomes @lake_building_2016 @battaglia_simulation_2013.

Using Bayes' rule (@sec-bayes), the learner finds the hypothesized goal that best explains the agent's behavior. Writing $g$ for the agent's goal, $m$ for its model of the environment, and $t$ for the observed agent's trajectory, the posterior is calculated as

$
P(g, m | t) prop P(t | g, m) dot P(g, m)
$

where the prior $P(g,m)$ encodes what the observer expects agents to want before seeing anything, and the likelihood $P(t|g,m)$ encodes the probability that an approximately rational agent with goal $g$ and model $m$ would move as observed (which is supplied by the planner) @baker_action_2009. It's been shown experimentally that people's predictions on behavioral stimuli are close to the model posteriors of this approximately rational inference mechanism @baker_action_2009 @baker_rational_2017 @jara-ettinger_quantitative_2021.

== Belief as an input <sec-false-belief>

The posterior is joint over $(g,m)$, the agent's goal $g$ and a model $m$ of the environment in which it navigates. But $m$ doesn't have to be an accurate model of the actual state of affairs (the observer's world), it could be any possible world model. The observer can supply a counterfactual model $m'$ that diverges from the actual state of affairs, and then run the same posterior computation to see how the agent would behave if it were navigating in the counterfactual world $m'$. Then the same computation that yields sensible behavior from an accurate model would yield mistaken behavior from the inaccurate one.

This is belief attribution in essence, attributing a counterfactual world to an agent such that it acts according to the counterfactual model as its own private world. So we can ask: given an (irrational) trajectory, which model of the environment would've made those actions rational? Take the false-belief task (@timeline):  Sally puts her marble in the basket and leaves; in her absence the marble is moved to the box; and when she returns she searches the basket  @baron-cohen_does_1985 @wimmer_beliefsabout_nodate. Evaluated against the actual (observer's) world $m$, Sally searching the basket is irrational since the marble is in the box. But evaluated against her private model she had when she left $m'$ (that the marble is in the basket), her searching the basket is optimal. An observer that can represent the divergence of worlds can predict that Sally will look in the wrong location, and even which wrong location she'll look in @goodman_intuitive_nodate @baker_rational_2017.

To calculate the likelihood, the observer needs to simulate the predicted trajectory. The agent's private model $m'$ is a counterfactual derived from the observers as $m' = "derive"(m)$, where "derive" modifies $m$ to account for the agent's false belief which the observer imputes. In the Sally--Anne example, in the observer's model $m$ (the real world) the marble is in the box, whereas Sally's model $m'$ (her private world) is the same as $m$ in every respect except that the marble is in the basket. 

The observer's then needs to run the agent's planner on its private model, to yield its trajectory (behavior) given that model. Rather than running $t'="optimize"(g',m)$ on her own model $m$ (as in @sec-inverse), the observer runs $t'="optimize"(g',m')$ on the agent's private model $m'$. So $t'$ is the agent's predicted trajectory, given that it pursues goal $g'$ along the most optimal path afforded by counterfactual model $m'$. Then $t'$ can be compared to $t$, to see whether the predicted goal $g'$ and model $m'$ predict that a rational agent will take trajectory $t'$. 

The "optimize" planner is the same one from @sec-planner. It's indifferent to whether its input is the real world model $m$ or an incorrect world model $m'$. Most importantly, it's indifferent to whether its input model is _attributed to an agent_ or not. All it does is run a search algorithm on input data. The model is executed and evaluated the same way. So there is nothing belief-specific about the planner. 

== What is stipulated by the BToM modeler <stipulated>

BToM is a model of the ToM capacity of a mature adult @baker_action_2009 @baker_rational_2017. It's a model of the computational problem that a person who has ToM is solving when they are trying to understand a person's behavior. So the ToM capacity is specified by the modeler. In this thesis we're interested in how this capacity might be acquired. In Chapter 4 we'll introduce a model that (we'll argue) doesn't have the mature ToM capacity which is stipulated by the modeler of a BToM model. To argue that these features aren't stipulated, we'll state them explicitly here. 

Most obviously BToM assumes the rationality principle, that the agent plans approximately optimally toward its goal given its model @sec-planner. We assume this as well in our model, because efficiency-sensitive goal-directed navigation is not ToM @timeline. But a lot of other feature of the BToM model are so basic that they're easily taken for granted. For example in a BToM model, the visual scene is often already decomposed into an agent that has a goal $g$ and a belief $m$, with the modeler already stipulating the roles that different entities play. The planner takes them as separate arguments that vary and can be inferred independently of each other.

A BToM model stipulates the attitudes that the learner searches over, for example the space of goals. The posterior over goals is inferred but the space it ranges over (and the prior over the space) is given. Baker et al. compare several goal priors @baker_action_2009, because the modeler has to specify the space. The model space that the posterior ranges over, the fact that the planner's input model $m'$ is permitted to diverge from the observer's $m$, is similarly stipulated by the modeler. And the modeler has to specify the divergences between $m$ and $m'$. The BToM model infers which particular model $m'$ the agent holds, but the range its drawn from is fixed in advance. That a belief could be false is not itself in question. 

Inference in a BToM is parameter estimation within a fixed theory, but we're interested in construction of the theory's terms. 

The five items of @stipulated have a shape we already have a name for. They are not hypotheses about any particular scene; they are hypotheses about what the hypotheses about scenes are like --- that behaviour is generated by an agent, that the agent has a goal and a model, that the model may diverge from the world, that the divergence is drawn from such-and-such a range. In the terms of @sec-hbm they are the learner's language rather than anything written in it: the vocabulary in which a particular attribution gets stated at all. BToM installs that vocabulary and infers underneath it. The question of this thesis is whether it can be inferred rather than installed --- whether the language, given to the framework by its modeller, is the kind of thing a learner could arrive at by the same inference it runs on everything else.

Putting it that way fixes what the model of the next chapter has to do. It cannot be handed any of the five. Its learner begins with grids, a planner, and domain-general combinators for building and consuming pairs --- no agent, no goal, no attitude, no channel that is anybody's --- and it has to assemble the structure out of them, in program space, under a description-length objective that knows nothing about which tasks are mental. This raises the bar past expressiveness. A language in which belief is merely _statable_ proves nothing, since the stipulated architecture is statable in any language rich enough to write it down. What has to be shown is that the structure is _selected_: that among the programs which reproduce the scenes, the one carrying the agency signature is the one compression keeps, and that it beats the non-mental rivals the very same library can express. @criteria states that test in advance and @no-btom takes the five stipulations in the order given here.




#load-bib(read("refs.bib"))