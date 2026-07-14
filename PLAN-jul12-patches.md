# Patch plan: Jul-12 long-run verdict failures

Context: the Jul-12 HPC runs (`hpc_run_results/jul12-*`) showed the verdict is
anti-monotone in search budget — phase1-short (t-fn 1200, 4 rounds) passed all 5
checks; both long runs (t-fn 3600, 6 rounds) failed 3/5. Longer search finds
extensional shortcuts before the library forms the belief abstraction. This plan
patches the leaks so the verdict is budget-robust, then reruns at both budgets.

Diagnosis was empirical: `rival_audit.py` regenerates the corpus with the run's
exact seeds and replays the suspicious round-1 caught programs from the logs.

## Audit findings (2026-07-14)

| Caught program (Jul-12 logs) | Actually solves | Gap |
|---|---|---|
| R1 `step 9 right ∘ flee-self ∘ flee-goal` (idx 189730 ph1 / 25110 ph2) | goal-displacement (av=9, gv=4, (up,up)) | gdb battery never tests flee content or compose chains |
| `fork(wall(1,0), flee 9→8, sync 8)` | goal-displacement (av=8, gv=9, (up,up)) | wrong-CONTENT mental solve; `optimize(distance v) v` ≡ step-up via tie-break |
| `wall ∘ seek ∘ erase 3` (×4, ~349k) | scaffold (plain belief) | by-design transient-wall vulnerability, only reached at 3600s |
| `wall ∘ seek ∘ clear_at` (×3) | **false-obstacle** | fob admission has NO transient-wall battery — real leak |
| snd_gg belief solves | 0/168 scenes vulnerable to bare readout | `fork(f, snd_gg) ≡ f` is a DSL identity (dsl.py fork/_snd_gg); the solves wrap an inner `fork(policy, sync_all)` — `sync_all` (dsl.py:480) moves only values shared by both channels, so the inner phantom wall never renders |

Structural bugs found alongside:
- fob is silently excluded from the wall-based check and cube census because it
  carries `displaced_to` (experiment.py ~1141–1146, ~1204) — phase1-long's 21
  literal wall-based fob solves never counted; likely what flipped
  `belief_keeps_corner` False.
- `_belief_commit_form`'s extensional guard (experiment.py ~385–396) tests the
  canonical gt, which always reproduces x on fob → any fork+scope solution is
  auto-classified 'degenerate'; the "should be impossible" WARNING is the
  classifier being vacuous, not wrong data. Root cause:
  `_scope_complements_all_fail` (tasks_minds.py ~811) verified complements only
  against the canonical derive — a rival derive that leaves gv/3 unmoved in the
  model makes sync_all safe.

## (a) Kill the pure-compose rival on goal-displacement  ✅ DONE

Files: `tasks_minds.py`.
1. Add `distance`-seeks to `_physically_explainable` (~line 85; currently tests
   only `step` and `optimize(neg_dist)`).
2. New `_compose_rival_explainable(x, g)`: reject scenes reproduced by any
   compose chain of length ≤ 3 over `{step(v,d), optimize(neg_dist u, v),
   optimize(distance u, v)}` for scene values (~16 base fns → ≤ ~4.4k unfolds per
   candidate; offline cost, fine). Wire into gdb admission AND witness/dual
   admission (phase1-long's 3 wall-free witness/dual solves are what the cube
   census saw).
3. Optional tier — wrong-content mental rivals (R3-type flee-derive forks): add a
   flee arm to `_wall_explainable` (~line 388). RISK: `optimize(distance gv) gv`
   ≡ step-up, and `_GOAL_CONTENT_SPECS` is all-vertical, so this may kill most
   'up' scenes. Measure yield per spec after the change; fallback = reject only
   non-mental rivals (tiers 1–2) and document flee-spelling as an alternative
   spelling of the same belief content.

IMPLEMENTED:
1. `_physically_explainable` now sweeps BOTH `optimize(neg_dist u, v)` (seek) and
   `optimize(distance u, v)` (flee) for all scene-value pairs (self-seek skipped;
   self-flee kept — R1 used it).
2. `_compose_base_fns` + `_compose_rival_explainable(x, g, max_len=3)` enumerate
   every compose chain of length ≤ `max_len` over step / seek / flee atoms. Wired
   into gdb (max_len=3), and — at **max_len=2** — into witness and dual admission.
   The 4-value witness/dual sweep at length 3 is ~87k unfolds/scene (≈240s for 8
   witness tasks), untenable; length 2 (≈1980 unfolds) still catches the wall-free
   2-agent rivals (one physical fn per agent, either polarity) — a genuine 3-deep
   detour needs a wall, which clobbers the crossing witness — and runs in ~18s for
   the full 48/24-task quotas.
3. Flee arm ADDED to `_wall_explainable` (both `neg_dist` and `dist` policies in
   the wall-fork). MEASURED: goal-displacement still fills 48/48 with all four
   content specs represented (12/12/11/13 up-up/down-down/down/up), so the flee arm
   did NOT collapse yield — kept it. This rejects the R3 wall∘flee fork as a
   wall-content spelling of any polarity.

Verify: `rival_audit.py` extended with a regression gate — R1/R2/R3/C1/C3 (+ fob
erase-3s) now 0 hits on goal/witness/dual/fob; snd_gg 0/168; every family fills its
per-combo quota. Only scaffold retains E1–E4/C2 hits (by design; see (c) target 2).

## (b) snd_gg — canonicalize the vacuous wrapper (NOT a data patch)  ✅ DONE

Files: `dsl.py` (+ optional `ecd.py`).
1. Extend `simplify` (dsl.py:1039, which already does this kind of
   semantics-preserving rewrite) with: `fork(f, snd_gg) → f`, the decomposed
   spelling `pipe_gpg(compose_gp dup (mapsnd f)) snd_gg → f`, and collapse
   `fork(f, fst_gg)` (≡ identity). `_core_uses`/`_corner_uses` and the stitch
   corpus all pass through `simplify`, so this cleans census, cube check, and
   compression in one place.

   IMPLEMENTED: added `_strip_snd_projection` + both snd_gg rewrites to
   `simplify` (dsl.py). The `fork(f, fst_gg) → identity` collapse was DELIBERATELY
   OMITTED: there is no identity primitive in the library, so emitting one would
   break the stitch string round-trip (`tr(D, …)` re-parses simplified programs and
   would fail on an out-of-library `id_fn` token). It is also unnecessary — `fst_gg`
   is not in the cube-census complements set, and `fork(f, fst_gg)` returns the world
   unchanged, so it can only ever solve a constant-trajectory task, never a belief
   scene. Unit-tested (phase-1 atomic + phase-2 decomposed spellings; non-wrapper
   forks untouched).
2. After unwrapping, these solves surface as sync_all-committed beliefs →
   handled by (d).
3. Optional: filter snd_gg/fst_gg from the commit slot at enumeration time
   (saves budget; canonicalization alone restores correctness).

Verify: unit test that a hand-built wrapped tree simplifies to the inner
program; smoke run asserts snd_gg absent from the belief census row while the
readout family still solves via snd_gg.

## (c) Transient-wall rivals: fob admission battery; round-1 budget cap for scaffold  ✅ DONE

Target 1 — false-obstacle (real leak). Files: `tasks_minds.py`
(`make_false_obstacle_belief_tasks`, ~line 831). Add a rejection battery:
- `compose(wall_at(r,c), seek, clear_at(r,c))` and `… erase(3)` chains over all
  cells × seek targets (neg_dist AND distance content);
- `fork(goal-unmoved derive, sync_all / sync_except k)` sweeps — this arm
  repairs the `_scope_complements_all_fail` guarantee (it was derive-relative).
(erase-3 chains auto-fail once scenes require the real wall in every frame; the
clear_at(pw) ones are the live threat — C1–C3 confirmed.)

Target 2 — scaffold (unfixable in data, by design: on a single-agent scene the
transient wall is extensionally identical to the private-copy belief in every
frame — the reason witness tasks exist). The failure is ORDERING: at 3600s the
rival (~349k) lands in round 1 before the policy token exists (acute in phase2,
where decomposed fork is token-expensive). Fix: per-round t_fn schedule in
`run_phase` — round 1 capped near the calibrated slowest-solve (~1200s; the
log's own "--t-fn can drop toward slowest-solve" line is this calibration), full
3600s for dreamed rounds 2+. DreamCoder-standard; state it in the thesis as a
curriculum choice. (Alternatives: drop scaffold for long runs — worked in
phase1-long round 1 but phase2-long still needed it; or accept + disclose.)

IMPLEMENTED:
- Target 1: `_false_obstacle_rival_explainable(x, g)` — (i) transient real-wall
  battery `compose(compose(wall_at(pr,pc), seek), clear_at(pr,pc))` and `… erase(3)`
  over all cells × agents × targets (incl. beacon 3) × BOTH neg_dist/dist policies;
  (ii) scope-complement sweep `fork(derive', sync_all | sync_except k)` over
  goal/wall-preserving derives `{seek, wall_at∘seek}` — repairs the derive-relative
  gap in `_scope_complements_all_fail`. Wired after the scope check in
  `make_false_obstacle_belief_tasks`. Result: fob fills 24/24 in ~7s, and the
  Jul-12 C1–C3 / erase-3 rivals no longer reproduce ANY fob scene (audit: 0 fob hits).
- Target 2: `ROUND1_T_FN_CAP = 1200.0` + `t_fn_round1` param on `run_phase`
  (CLI `--t-fn-round1`). Round loop selects `round_t_fn = t_fn_round1 if it==1 else
  t_fn`; default `t_fn_round1 = min(t_fn, 1200)` — a no-op at t_fn ≤ 1200 (smoke/
  short), caps the first round at 1200s for the 3600s long runs. Round header prints
  the per-round budget + "(round-1 cap)" flag. Scaffold's transient-wall rivals stay
  in the audit BY DESIGN (unfixable in data); the cap keeps them out of round 1 so
  the belief policy token forms first. Verified: phase1 + phase2 `--smoke` reach
  VERDICT clean.

## (d) Classifier and census-scope fixes  ✅ DONE

Files: `experiment.py`, `ecd.py`.

IMPLEMENTED:
1. `_belief_commit_form` now runs a SOLUTION-relative swap (`_swap_scope_commit`):
   parse each fork / decomposed-fork, replace any scope-complement commit (sync_all /
   sync_except) with sync_to_world(av) for that fork's own actor (the seek's target),
   re-run. Reproduces x → 'degenerate'; doesn't → new class 'complement'. The verdict
   conjoins `no_belief_complement` (n_belief_complement == 0) as a hard FAILURE check;
   counts persisted in the artifact (belief_commit.complement, false_obstacle.complement,
   A.no_belief_complement) and printed in (A)/FORCED/VERDICT. Unit-tested all four
   classes incl. a genuine reproducing-complement (scope reproduces, literal swap fails).
2. Census scope: both `'displaced_to' not in m` / `'displaced_to' in m` exclusions
   replaced with `belief_variant(m) != 'belief_goal'` / `== 'belief_goal'`, so
   false-obstacle's literal wall solves now count (it carries displaced_to but is
   wall-based).
3. ecd.py `sample()` guards a zero / non-finite total (all `-inf` logprob row) with a
   uniform fallback — no more NaN cdf.
1. `_belief_commit_form` (~line 371): replace the canonical-gt guard with a
   commit-swap on the SOLUTION's own derive — parse the outermost fork/pipe_gpg
   split, swap the scope commit for sync_to_world(av) (or its register
   decomposition), re-run. Reproduces x → 'degenerate'. Doesn't → new class
   'complement' that the verdict counts as a FAILURE, not the agency commit.
2. Census scope: replace `'displaced_to' not in m` exclusions (~1141, ~1204)
   with `belief_variant(m) != 'belief_goal'` (tasks_minds.belief_variant already
   orders real_wall before displaced_to for exactly this reason). Fob's
   wall-based literal solves then count — probably flips phase1-long's two
   census failures on its own.
3. Guard the zero-sum `ps / ps.sum()` at ecd.py:1088 (both phase2 runs trained
   the recognition model through NaN warnings).

Verify: unit tests per class (literal / degenerate / complement / None) with
hand-built solutions; smoke run; WARNING path fires only on genuine 'complement'.

## (e) Rerun protocol

1. Preconditions: fix the Snellius conda env (`EnvironmentNameNotFound:
   dumbcoder` — every Jul-12 run fell back to unpinned python); commit the
   working-tree instrumentation + these patches; both phases `--smoke` locally;
   extended rival-audit → 0 hits.
2. Runs: phase1 + phase2 at `--t-fn 3600 --ecd-iters 6` (24h), PLUS short runs
   (1200/4) on the SAME commit — the Jul-12 short/long pairs were confounded
   (short = 283-task corpus without relocate, long = 299 with it); this rerun
   makes budget the only variable.
3. Success criteria: verdict all-True at BOTH budgets (budget-robustness is the
   claim that beats the 1200s pass); FORCED section 0 degenerate / 0 complement;
   no snd_gg in belief census; witness+dual cube row wall-based.
4. Afterwards: fetch `phase{1,2}_run.json` / `phase{1,2}_traj.json`; regenerate
   mdl_margin, corpus_dl, solve_dynamics, behavioral_probe (all pre-patch
   artifacts stale). Stretch: t-fn ∈ {600, 1200, 2400, 3600} sweep as the
   robustness figure.

Open item: confirm the snd_gg inner-sync_all reading by scp-ing the Jul-12
`phase2_run.json` from Snellius and replaying its stored base-prim solutions
(the sync_all source semantics already entail the mechanism).

## Execution order

(d) + (b) first — small, self-contained, and (d).2 alone may flip two of
phase1-long's three failed checks (tells us how much was measurement vs search).
Then (a) + (c) generator hardening. Then (e).

Status: [x] (a)  [x] (b)  [x] (c)  [x] (d)  [ ] (e)
