#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Bayesian Theory of Mind")

== Generative model of an agent

- MDP/POMDP
- utility
- principle of rationality
- intentional stance formalized

agents as approximately rational actors, inverse planning

defining the generative model. agents move in an environment (MDP/POMDP) to maximize utility efficiency

Bayesian inverse planning [Baker, Saxe, and Tenenbaum (2009)]

“naive utility calculus” [Jara-Ettinger, Gweon, Tenenbaum, and Schulz (2015)]

Assume that the observer sees the agent as a rational planner. Then planning computation is solution to markov decision process.
- input: utility and belief functions defined over agent's state space and state-action transition functions
- output: series of actions the agent should perfoirm to maximize utility or fiulfill goals @lake_building_2016

then the observer can simualte what the agent would do if the agent had certain goals/policies etc, and thus infer those from what actually happens by comparing the simulation to reality. @lake_building_2016 analogy to physics simulations. Say I want to figure out 

First we have principle of rationality, expetation that agents will plan rationally given their world model. Then we reverseengineer what their mental states are that caused their behavior, based onthat principle. Formalizing Dennet's intentional stance. @baker_action_2009

Explanation by rationalization, Bayesian theory of mind framework [Baker 2012]

model causal relation betwene beliefs, goals, actions as rational probabilistic planning in markov decision problem. [baker 2012]

observer uses Bayesian induction to work backward from a trajectory to the most likely goals and beliefs

humans can easily observe agents' actions and then infer the beliefs desires and intentions that led to those actions, assuming that the agent is approximately rational and goal directed

[Baker, Saxe, and Tenenbaum (2009)], [Lucas, Griffiths, et al. (2014)], [Jern, Lucas, and Kemp (2017)], [Jara-Ettinger, Gweon, Schulz, and Tenenbaum (2016)]
[Baker, Jara-Ettinger, Saxe, and Tenenbaum (2017)]
[Ullman et al. (2009)]

Forward-planning/rational decisionmaking: given an agent's intentions etc, predict their future actions

inverse planning: given an agent's actions, what were their intentions/beliefs/world mdoel?

use Bayes rule to produce posterior over possibel mental states

Baker et al 2017

== Inverse planning and the naive utility calculus

- Baker 2009/2012/2017
- Jara-Ettinger
- inverse RL

== belief attribution and false belief in BToM

== What these models don't do

Bayesian Theory of Mind _is_ a theory of mind, which is handed to the model. The agent/goal/belief structure, including the separation of believed world from actual world, is stipulated by the modeler. So BToM explains mature performance, not acquisition. A nativist can read it as evidence _for_ innateness. 

== In this thesis

we run the bayesian engine detailed in chapter 2 on ToM tasks, and ask whether the structure ch3 stipulates is instead discoverable by compression. 

#load-bib(read("chapter1.bib") + read("chapter2.bib"))