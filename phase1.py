"""Phase 1 — atomic fork and sync.

Runs the full symmetric cube over the mixed minds / minds-free corpus with `fork`
and `sync_to_world` as atomic primitives.  Demonstrates (a) that fork and sync are
reached for by non-mental families (overlay, registration) as well as belief, and
(b) that every symmetric complement is the natural tool of some non-mental family —
so belief's lone-asymmetric-corner choice is a discovered MDL win, not a stipulation.

    python phase1.py                 full run (several ECD rounds)
    python phase1.py --smoke         fast smoke run
    python phase1.py --samples       also dump one example trajectory per family
    python phase1.py --ecd-iters 6   override number of wake-sleep rounds
    python phase1.py --t-fn 600      override per-task fn timeout (belief is the long pole)
"""

import sys

from experiment import run_phase, cli_kwargs

if __name__ == '__main__':
    run_phase(decomposed=False, **cli_kwargs(sys.argv))
