# Terminology convention (fixed 2026-08-08)

One word per referent. When drafting new prose, check against this table.

## Core objects

| Word | Reserved meaning | Never use it for |
|---|---|---|
| **primitive** | An element of the initial library 𝓛₀ — what the learner is *given*. "base primitive" for emphasis. In the atomic control, `fork`/`sync_to_world` *are* primitives (they're in that run's 𝓛₀). | Anything compression produced. No "new primitive", "learned primitive", "belief-attribution primitive". |
| **abstraction** | An element added to the library by compression — what the learner *constructs*. (λ-abstraction, the operation, survives in ch2 only.) | Elements of 𝓛₀. |
| **symbol** | Any library element (primitive or abstraction) when what matters is its being one unit of the language: pricing, enumeration, `fn_9`, table columns, "an abstraction found in round N is a single symbol in round N+1". | — |
| **token** | An *occurrence* of a symbol in a program — the unit of description-length cost. Defined once in ch4 §Search ("summed −log p of its tokens"). | Library elements themselves (types, not occurrences). |
| **term** | The theory-theory register ONLY: theoretical terms of a theory, conceptual terms, "mental terms" (the title). One bridging sentence (ch2 §Our claim) equates acquiring an abstraction with acquiring a term. | Library entries in results prose ("six terms" → "six abstractions"), coordinates ("terminals"), summands of the MDL objective ("summands"). |
| **terminal** | A leaf symbol: the ten cell values, five coordinates, four directions. | — |
| **compound** | A composition of several symbols that is not (yet) itself in the library — "the belief compound". Defined at ch4 §Composing belief attribution. | Library entries. |
| **combinator** | The pair-manipulating primitives specifically: "the pair combinators". | (Killed: "tuple functions", "functions over pairs".) |
| **function** | Only in genuinely type-theoretic phrases: transition function, utility function, "a function of type …". | A classifier for library elements. |

## The library as a whole

- **library** — the formal object 𝓛; the default word everywhere (posterior over libraries, library revision).
- **language** — informal, only when the point is expressiveness ("expressible in the language", "the language the hypotheses are written in").
- **vocabulary** — only the conceptual/child register (ch1–2: "conceptual vocabulary", "vocabulary of attitudes").
- **endowment** — only for a run's 𝓛₀ in the two-run comparison ("combinator endowment" vs "atomic endowment").
- **grammar** — confined to ch2's general discussion of generative hypothesis spaces.
- **inventory** — banned.
- **atom / atomic** — "atomic control" is the control run's proper name; never use "atom" as a common noun for library elements (say "a single symbol") — it collides with the run name.

## The system and its parts

- **the learner** — the whole system. "the search" / "enumeration" / "compression" are its phases, not agents ("the searcher", "the enumerator" banned).
- **observer** — only (a) the BToM observer in ch1/ch3, (b) ch4's didactic walkthroughs of what a hypothesis posits.
- **our model** — the modeling contribution as a whole, chapter-level framing only; prefer "the learner" inside technical prose (avoids collision with world model m).
- **the combinator run** vs **the atomic control** — the two runs' canonical names ("the main run" banned). Raw run IDs (`p1-nodream`, `p2-nodream`) appear in appendix captions only.

## The belief machinery

- **derive-and-commit frame** — the frame's only name. `fork` is mentioned solely as the atomic control's primitive that buys it whole. The three-step *process* is "derive-run-commit".
- **commit** — the noun for a G×G→G collapse; **publish** the verb for what a commit does; **single-value commit** (never "single-value publication").
- **private copy** — default name for the derived second grid. "second channel" for the type-level slot of the pair. "(private) model" only when the possessive/BToM correspondence is the point ("acts on *its* private model").

## The belief structure: five levels

Never let these trade names — they live at different levels:

| Level | Name | Grammar |
|---|---|---|
| Property of a program | **agency signature** | One value independently filling every slot the attribution needs (simulated on the copy, published to the world, withheld as content). Programs/abstractions **carry** it; it is never itself "found" or "added". |
| Expression | **belief compound** | A multi-symbol composition carrying the signature, not (yet) named by compression. A compound never "enters the library" — its abstraction does. |
| Library entry | **belief-attribution abstraction**; the named **constructors** | An abstraction whose body is a belief compound. |
| Scaffolding | **derive-and-commit frame**, **single-value commit** | The two target structures. NOT belief vocabulary — they're neutral plumbing that non-mental families pay for (the round-1 result depends on keeping this distinction). |
| Scene interpretation | **belief reading** | A hypothesis about a scene (ch6's behavioral test). |

## Composition words

- **composition** — generic act/result; unreserved.
- **compound** — a specific unnamed composition (in practice: the belief compound).
- **combinator** — the pair primitives in 𝓛₀, nothing else.
- **constructor** — proper-name suffix for discovered abstractions that return a whole transition function AND open a private copy (hence the atomic run has "five constructors" out of six: `fn_7` opens no copy and is "the bare seek policy").
- **frame** — as a *structure name*, only the derive-and-commit frame. Bare "frame" in the scene sense (a rendered time-slice: "first frame", "frame-by-frame", defined ch4 §space) is a separate, permitted sense — the two never appear without disambiguating context. **block** — only the seek block.
- **policy** — a transition function read as an agent's behavior (movement/seek policy). A template task's solution is "a commit", never a "commit policy".
- **hole** — an abstraction's open argument slot (the Stitch sense) and nothing else; the signature is one value filling several slots, so "shared hole" is banned.

## "model" (the possessive test)

The bare noun **model** belongs exclusively to the mental/attributed sense: an agent's (or the observer's) representation of the world — $m$, $m'$, "world model", "private model", "counterfactual model", "(world, model) pair". A model in this sense is always *somebody's*: if you cannot attach an owner or a variable to it, you are using the word for the system or a framework — rewrite it.

- Your system: **"the learner"** in technical prose; **"our setup"** when it's the design rather than the agent; "our model" tolerated only in chapter-level framing sentences with no world-model in scope. Never bare "the model" for the system.
- BToM as a theory: **"BToM"**, "the BToM framework", "a BToM account" — never "the/a BToM model".
- The second grid, mechanically: **"private copy"** / "second channel" (see above); "private model" only in sentences making the attribution reading.
- **"recognition model"** — DreamCoder's proper name, quarantined in ch4 §Inference. The verb ("BToM models the agent") and "the modeler" are unambiguous and fine.

### grid vs frame vs model (the register rule)

Three words for what is extensionally one object (a model *is* of type Grid); the register decides which:

- **BToM register** (ch3, and any ch4 sentence recapping @sec-false-belief with $m$/$m'$): the objects are **models** — BToM has no grids. Never "counterfactual grid" next to $m'$; the variable decides the word.
- **Mechanical register** (our programs, the interpreter, channels): **grid** / **private copy**. "Counterfactual grid" is correct here when the point is a grid with no owner yet (ch4 §signature: "if a counterfactual grid is to be _somebody's_").
- **Scene register**: **frame** — a grid as rendered at a time step.

A sentence that crosses registers takes the word of the object's home register: $m'$ is always a model, even in mechanical prose.

## Tasks

- **belief families / belief tasks** (the 168) vs **non-mental families** (the 148). "mental" appears once, in ch4's defining sentence, then switches.
- Canonical family names: **false-wall** (24), **goal-displacement** (48), **witness** (48), **two-observer** (24), **false-obstacle** (24). (Generator names belief_wall / belief_goal / belief_witness / belief_observers / false_obstacle map to these in that order.)
