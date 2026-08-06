# dumbcoder

Synthesising mental abstractions — belief, agency — from non-mental primitives, by
program induction plus library compression. A stripped-down reimplementation of
[DreamCoder](https://github.com/ellisk42/ec): wake-sleep rounds of type-directed
enumeration and [Stitch](https://github.com/mlb2251/stitch) library learning, with no
recognition model (see `ARCHITECTURE.md` §2.4).

This is the code for the MSc thesis in `illc-mol-thesis/`.

## Where to start

| Document | What it is |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the system works and why it is built this way. The architecture reference. |
| [`PRIMITIVES_AND_TASKS.md`](PRIMITIVES_AND_TASKS.md) | Every primitive (repr, type, semantics) and every task family (ground-truth program, what it probes, its filter). The experiment reference. |

## Running

```sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Two experiments are reported in the thesis. They share one corpus and one harness
(`experiment.run_phase`) and differ only in the primitives the searcher is handed:

```sh
python phase1.py          # atomic control: fork and sync_to_world are primitives
python phase2.py          # combinator endowment: both must be assembled from parts
python phase1.py --smoke  # tiny corpus, short timeouts (~3 min; a wiring check, not a result)
```

A full run is measured in CPU-hours, not minutes — `run.job` is the SLURM driver, and
its header explains the walltime sizing. `--t-fn N` sets the per-task search timeout
and `--ecd-iters N` the number of wake-sleep rounds.

A run writes `phase{1,2}_{run,traj,verdict}.json` into the directory it is launched
from. Those are **git-ignored working copies**: tracking them would mean every run
dirties the tree whose commit it is stamping into its own provenance. The citable
artifacts live under `hpc_run_results/<date>/<cell>/`, and the runs the thesis reports
are `hpc_run_results/aug4/p{1,2}-nodream-aug04/`.

## Analysis and figures

These consume a run's artifacts and emit the thesis figures. They can be run from
anywhere; each resolves the repo root from its own path.

```sh
python viz/belief_solved.py       # solve rate per belief variant
python viz/solve_dynamics.py      # cumulative-solve curve, per-task solve-time collapse
python viz/corpus_dl.py           # per-round total description length
python viz/mdl_margin.py          # prices belief against its non-mental rivals
python viz/plot_mdl_margin.py     # …and plots the result
python viz/agent_tiling.py        # (gv,av) x abstraction-used tiling
python viz/behavioral_probe.py    # the false-belief test on a held-out scene
python scaffold.py                # which shallower tasks/abstractions built up to belief
python rival_audit.py             # diagnostic: replays caught programs to check for leaks
```

All of them except `plot_mdl_margin.py` and `rival_audit.py` take `--decomposed`, to
read the phase-2 artifacts instead of phase 1, and `--smoke`, to read the `.smoke`
ones. `rival_audit.py` regenerates the corpus itself and consumes no run.

## Layout

| File | Role |
|---|---|
| `ecd.py` | The ECD architecture: the `Deltas` library, enumeration, `saturate_stitch` compression, the driver. |
| `dsl.py` | Types, the `Delta` tree, the interpreters, and every primitive. |
| `prims.py` | The primitive sets handed to the searcher. |
| `scenes.py` | A task as `k` trajectories sharing one latent program; task identity; `solves`. |
| `tasks.py` | Every task generator, and the closed-set certifications that keep non-mental rivals out. |
| `experiment.py` | The shared harness: corpus assembly, ground-truth verification, the usage census, artifact export. |
| `phase1.py`, `phase2.py` | The two reported runs (thin drivers over `run_phase`). |
| `phase3_arity.py` | Exploratory, **not a reported result** — the further-work experiment of thesis ch. 6. See its docstring. |
| `viz/` | Figure scripts. |
| `run.job`, `sync.sh` | HPC batch-run scaffolding. |
