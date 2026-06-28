#import "@preview/illc-mol-thesis:0.2.0": *

#mol-chapter("Discussion")

== Section

Main argument: if you give the system a corpus of tasks that humans would describe with mental vocabulary, that are produced by mental processes, then compression should yield primitives with mental operations

theory is representation of the world. 


You'd think that the more abstract framework concepts would be harder to learn. But those end up getting learned first and then instantiated for specifics, it makes it easier to learn the specific cases



start with free-floating belief, a grid representation. Then it gets tied down to a specific agent, as that agent's belief. you might think it'd start with an individual agent's belief and then gets abstracted into the collective. but it's kinda the reverse, starts with the general and then goes to specific. 

the important part is the internal state. does it generate a representation of the world. 

the goal isn't to solve these tasks. if that were the goal we'd use reinforcement learning or something. rather the goal is to build a conceptual library that's interpretable, a library that explains the domain. RL would solve the problems but 

this is a dataset that lends itself well to "crisp symbolic forms" (Ellis 2020)

this is implicitly a categorization problem, categorizing agents vs. objects. the categorization becomes salient in the structure of how they are represented. so it's not as clear as a predicate `is_agent`, rather if the thing is represented having a movement function and belief function then it's an agent. for example: Say a domain has entities $a,b,c,d,e$ where $a,b,c$ are agents and $d,e$ are inanimate objects. So we could classify them explicitly with membership predicates as $A={a,b,c}$ and $O={d,e}$. Or we could identify the class membership structurally by observing that $a,b,c$ appear in the "friends" 2-place predicate $F={(a,b),(b,c)}$ while $d,e$ only appear in the "lighter than 200 lbs" predicate $L={a,b,c,d}$ (and $e$ is some heavy thing). 

goal isn't simply to learn specific theories to explain individual scenes. rather goal is to learn the framework itself, the framework theory of theory of mind

first order:
- data $d in cal(D)$ is the initial grid observation, `x[0]` of one of the tasks
- hypothesis space $cal(H)$ is the program space
- hypothesis $h in cal(H)$ is a particular hypothesis
- likelihood $P(d|h)$ is whether the hypothesis produces the program, whether it solves the task. 1 if it does, 0 otherwise, since we just run the program and it's deterministic so it'll eithe produce the correct answer or not
- prior $P(h)$ is the length of the program, how likely the hypothesis is (the shorter the program the higher the prior)
- posterior $P(h|d)$ is the probability, given we've seen this particular grid $d$, that the underlying mechanism producing it (the hypothesis) is $h$

second order:
- data $cal(D)$ is all data
- hypothesis space $cal(L)$ is the set of possible DSLs
- hypothesis $ell in cal(L)$ is a particular DSL
- likelihood $P(cal(D)|ell)$ is the probability that solutions to tasks $d in cal(D)$ are found given that the DSL is $ell$. Say $ell={0,1}$. So $ell$ can generate any of the following task sets: $cal(D)_0={}$, $cal(D)_1={0}$, $cal(D)_2={1}$, $cal(D)_3={0,1}$. So $P(cal(D)_2|ell)=0.25$. But say $ell_1={1}$. So $P(cal(D)_2|ell_1)=0.5$. the likelihood is how tightly the DSL fits the dataset. if the DSL is quite broad i.e. it can generate solutions to tons of different task sets (like, if the DSL is pure untyped lambda calculus) then likelihood is quite low. But if the DSL is extremely narrow/specialized (strict typing to enforce particular compositions) such that it only generates solutions to one specific task set, then the likelihood of $cal(D)$ under $ell$ is quite high. 
- prior $P(ell)$ is the length/complexity of the DSL. a shorter/simpler DSL is likelier to be chosen from the space of possible DSLs. The DSL is the framework theory. A simpler framework theory has a higher prior than a complex one.
- posterior $P(ell|cal(D))$ is the probability that framework theory $ell$ is the correct explanation for observation set $cal(D)$. 