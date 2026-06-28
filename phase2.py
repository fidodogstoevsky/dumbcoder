"""Phase 2 — decomposed fork and sync.

The same cube and corpus as phase 1, but `fork` and `sync` are spelled out, so
belief must be *rediscovered* as a compound rather than handed over:

    fork          ≡ (pipe_gpg (compose_gp dup (mapsnd derive)) commit)
    sync_to_world ≡ (register (locate av) (place av))
    sync_to_model ≡ (via_swap (register (locate v) (place v)))

(The scope complements sync_all / sync_except fold over an unbounded value set and
stay atomic — the decomposition reaches exactly as far as the locate/place
vocabulary does.)  The run first proves the decomposition is numerically identical
to phase 1's atomic machine, then asks whether the same joint MDL still recovers the
agent constructor — now with `av` shared three ways (optimize + locate + place).

    python phase2.py                 full run (several ECD rounds)
    python phase2.py --smoke         fast smoke run
    python phase2.py --samples       also dump one example trajectory per family
    python phase2.py --ecd-iters 6   override number of wake-sleep rounds
    python phase2.py --t-fn 600      override per-task fn timeout (belief is the long pole)
"""

import sys

from experiment import run_phase, cli_kwargs

if __name__ == '__main__':
    run_phase(decomposed=True, **cli_kwargs(sys.argv))
