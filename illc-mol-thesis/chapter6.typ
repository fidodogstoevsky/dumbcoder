
#import "lib.typ": *

#mol-chapter("Limitations and Further Work", lbl: <ch-further>)

== Ablation runs <sec-quantify>

We currently only have an existence claim that is quite specific to this particular corpus. So in further work we'll regenerate the tasks from the generators at new seeds and then rerun the learner with its final library on them. We'll also run multiple trials over different seeds.

Another experiment we plan to run is to vary the density of the task families in the corpus and measure the round at which the belief attribution abstraction enters the library and the description length it saves (@tab-dl-census). The constructivist reading predicts that a concept becomes available when earlier learning has made it cheap, so in this experiment we would test the ratios of each family of tasks and what abstractions they yield.

Another ablation concerns the discount we currently give to visible values. In enumeration our symbols are priced type-uniformly (@enumeration), with the exception that cell values visible in a task's first frame cost zero nats. We're pretty sure this isn't a problem. For example, the census of @tab-dl-census is priced with the exception removed. But the repricing cannot test the exception's effect on search itself. It can't test which programs enumeration reaches within budget, so which solutions compression is shown. So we would re-run the experiment without the exception, to see whether different abstractions are formed.

== Coverage and the budget frontier <sec-budget>

Our run doesn't solve all the tasks in the corpus. The run converges at 132 of 168 belief tasks, missing 21 of 48 goal-displacement tasks and 15 of 24 false-obstacle tasks. We argue that these misses are due to budget: the control is exactly as expressive as the combinator run's library, since it just hands the learner two pre-made compositions of primitives. Yet it solves 23 belief tasks that aren't solved in the main run, which indicates that it's lack of budget.  

Most solves finish well inside the budget, so the distribution of solved times is as consistent with the misses being out of reach as with their being out of time. Some programs are clearly expressible in our library but are seemingly unreachable by enumeration. Take a scene depicting two agents holding contradictory false beliefs, each detouring around its own phantom wall. That's a scene the language states without difficulty. But it names six latent literals across two nested derive-and-commit frames, and no amount of budget we could give it brought a single such task within reach.

So to settle this we would run a budget dose response experiment. We'd run the same corpus at 600, 1200 and 2400 s per task and beyond. So at some level of budget increase we'd see tasks being solved if the budget was the limiting factor, but any tasks held back by being inexpressible still wouldn't be solved. 

== Persistent models, interaction, and observation <sec-persistent>

The belief our learner attributes is thin, because it's forced that way by the interpreter. A hypothesis is a single-frame transition function $G times G$. So a program that opens a private channel must build it, use it and collapse it within one step (@space). Therefore, an attributed model exists only for the duration of one call. Nothing carries it to the next frame, nothing represents how the agent came to hold it, and nothing updates it when the agent looks again. BToM's $m'$ is a belief in the full sense. It's a state the agent came to be in, that could have been otherwise, and that would be revised on new observation @baker_action_2009. What our learner finds is just a divergent world model, attributed to an agent and acted on by that agent's policy alone, with no account of acquisition and no ability to revize a belief. 

So we would first extend our model by adding the capacity for persistence and interaction. We'd make the interpreter's state the pair rather than the grid, so that an attributed model is threaded from step to step and the program can write to it and read from it across frames. An agent could then genuinely react to its environment. For example it could discover an obstacle by walking into it, record it in its model, and route around it thereafter. As it is now, every divergence is re-posited from scratch at every time step. We actually had this pair-interpreter setup in an earlier version of our model, but we changed to the simpler interpreter. 

The second extension we would like to make is to add an observation. Nothing in the present setup represents perceptual access, so the derive just stipulates the divergence, it doesn't explain it. Then scenes with occlusion, and a primitive restricting what an agent's channel receives, would let a false belief be _caused_. The agent believes a wall is there _because_ it saw one before it was removed, and stops believing when it sees again.

With persistence and observation together, acquisition and revision become depictable in scenes. And then we can ask whether an MDL learner discovers not just attribution, but updating. Whether it discovers belief as a state with a history, which is a part of BToM but isn't represented in our setup.

== The missing arrow $C times C arrow.r U$ <sec-coord-utility>

No arrow among our primitives carries a coordinate to a utility $C times C arrow.r U$. Coordinates are first-class terminals in our domainL the wall stamp primitive and the cell clear primitive consume them, and the reader primitive produces them. But the utility type is inhabited only by the two value-directed distances, so the planner can be aimed at an object but never at a location. Nor does our library offer arithmetic on coordinates from which an offset like "two above" could be formed.

These two absences are what make the extensional reading of the goal-displacement family unstatable with this library, even though "at each step, move toward the cell two above value 2" is the most natural reading of a scene like that. So for that family, the task is solvable only by belief attribution because we didn't include the primitives in the library that would've made it expressable.

Our rationale for excluding $C times C arrow.r U$ is that we grant planner as a capacity of the teleological stance, and the teleological stance relates an action to a goal _object_, not to bare space. But of course this rationale doesn't cancel the suspicious effect. At least the false-wall families don't depend on a primitive being omitted, as their extensional rivals are fully expressible in the language (but no extensional program solves those tasks since they don't generalize across sccenes. So the rivals fail for the correct reason). 

To discharge this concession, we would go ahead and to the library $cal(L)_0$ a coordinate-valued utility and the offset arithmetic to aim it. If a belief-attribution abstraction still enters the library and still beats the walk-to-a-location program on description length, then the family's verdict is vindicated. If it does not, then the goal-displacement family cannot carry a belief claim and the weight falls entirely on the false-wall families. Until then, the goal-displacement results of @ch-results should be read with this condition. 

== Metarepresentation <sec-metarep>

Our learner lacks _metarepresentation_: the grasp of a mental state _as_ a representation, something with a referent and a mode of presenting it @perner_understanding_1991. So our learner can't misrepresent. There is no mode of presentation in a grid. We'd typically use substitution failures to mark an intensional context, but they have no counterpart in one grid's differing from another. So Perner would say that our discovered abstraction is at most at the situation-theoretic stage. It attributes a situation, but doesn't understand a representation as a representation. Our central claim survives this challenge, because it rests only on attribution. The divergent model is indeed _somebody's_, and only that somebody's movement is determined based on the model.

But doing more than that is hard, because grids are extensional objects. But the setup already contains the ingredient a mode of presentation needs, namely a representation with syntax. The derive is a program, and the attributed model is only its output. A learner that attributed the _derive_ itself rather than the grid would have the beginnings of an intensional context. Substituting one for the other could change the agent's behaviour in scenes where the derives come apart. But even if the programs compose and give use the attribution of a derive like that, it would be fiendishly hard to design task families that force this distinction. 

== Recursive mindreading <sec-recursive>

The two-observer family puts two agents in one scene and asks only whether the learner attributes a private model to the right one. What it never asks is how the two agents stand to _each other_. Nothing in the corpus depicts one agent modelling another, and nothing in the discovered abstraction could express it if it did: the constructor opens a private copy of the _grid_, and a grid has no room in it for somebody else's model.

The natural target is the group case. Shum et al. extend inverse planning to scenes where behaviour is generated by relations --- who is cooperating with whom, who is working against whom --- and represent those relations as compositions of a small set of team operators, applied recursively, so that arbitrary team structures are built out of a compact set of building blocks @shum_theory_2019. That is the representation this thesis's question should be put to next, and putting it there is the same move made one level up: their operators are given, as `fork` and `sync_to_world` were given in the atomic control, and the question is whether a learner shown enough scenes of agents helping and hindering each other would assemble them out of the plumbing it already has, or whether they have to be installed.

Two things would have to change first. The corpus would need scenes whose generating program cannot be written as two independent agent blocks side by side --- one agent's policy conditioned on where another agent is going, which the present single-frame transition functions cannot state (@sec-persistent) --- and the type system would need channel arity as a free parameter, a recursive stack of grids in place of the fixed pair, since an agent modelling an agent is a private channel nested inside a private channel and the pair $G times G$ permits exactly one. The prediction, if the thesis's argument is right, is that the depth of nesting selected by compression should track the depth the scenes actually require, and stop there.

#load-bib(read("refs2.bib"))
