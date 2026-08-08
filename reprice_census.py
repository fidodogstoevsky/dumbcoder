"""Reprice the per-family DL census (tab-dl-census) under the visible-value discount.

The published census (experiment.report_dl_census) prices every solved task's found
program and its library-rewritten form under the FINAL library with a plain
uniform-type prior — i.e. with the enumeration-time visible-value discount REMOVED.
Chapter 6 proposes testing whether belief's saving depends on that discount.  This
script prices the same (base, rewritten) pairs from a phase run's artifact under
three priors:

  * none — uniform type Q (this must REPRODUCE the published tab-dl-census);
  * half — every terminal ecd.content_boost would boost costs HALF its uniform
    nats (q = lp/2 instead of 0);
  * full — ecd.content_boost verbatim (visible cellvalue literals and real-wall
    coord terminals cost 0), the prior enumeration actually searched under.

The library is reconstructed exactly as viz/mdl_margin.py does: re-stitch the run's
searched sols, remap the run's invented-abstraction names, price with the shared _dl.
Run from the repo root of a worktree pinned at the run's commit:

    python reprice_census.py --run <dir>/phase2_run.json --decomposed
"""

import argparse
import json
import os

import numpy as np

from ecd import Deltas, saturate_stitch, mat_key_id, content_boost
from prims import make_symmetric_prims
from tasks import belief_variant
from experiment import verify_ground_truth, _ALL_KINDS
from viz.mdl_margin import build_corpus, uniform_type_q, _dl, _remap_names

VARIANTS = ('none', 'half', 'full')


def discounted_qs(D, Q_u, x):
    """The three priors for one task: uniform, half-discount, full content boost."""
    Q_full = content_boost(D, Q_u.clone(), x)
    boosted = (Q_full != Q_u)
    Q_half = Q_u.clone()
    Q_half[boosted] = Q_u[boosted] * 0.5
    return {'none': Q_u, 'half': Q_half, 'full': Q_full}


def summarize(rec):
    b, l = np.array(rec['dl_base']), np.array(rec['dl_lib'])
    return {'n': len(b),
            'dl_base_median': float(np.median(b)), 'dl_lib_median': float(np.median(l)),
            'saved_median': float(np.median(b - l)), 'saved_mean': float(np.mean(b - l)),
            'saved_total': float(np.sum(b - l))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='phase run artifact (phase{1,2}_run.json)')
    ap.add_argument('--decomposed', action='store_true')
    ap.add_argument('--out', default=None,
                    help='output JSON (default: dl_census_reprice[.decomposed].json '
                         'next to the run artifact)')
    args = ap.parse_args()

    with open(args.run) as f:
        art = json.load(f)
    if bool(art.get('decomposed')) != bool(args.decomposed):
        raise SystemExit(f"{args.run} is a phase {2 if art.get('decomposed') else 1} run, "
                         f"but --decomposed={args.decomposed}.")

    print(f"corpus: rebuilding (decomposed={args.decomposed}) ...")
    tasks = build_corpus(smoke=False, decomposed=args.decomposed)
    D = Deltas(make_symmetric_prims(decomposed=args.decomposed))
    verify_ground_truth(D, tasks)

    from ecd import tr
    sols = {kid: tr(D, s) for kid, s in art['sols'].items()}
    print(f"run: {args.run} ({len(sols)} searched programs); re-stitching ...")
    saturate_stitch(D, sols, iterations=6, max_arity=5)
    rewritten_of = _remap_names(dict(art.get('rewritten', {})), art.get('library', []), D)
    print(f"final library: {len(D)} tokens ({len(D.invented)} invented: "
          f"{[d.repr for d in D.invented]})")

    Q_u = uniform_type_q(D)
    per_kind = {v: {} for v in VARIANTS}
    per_belief_variant = {v: {} for v in VARIANTS}
    n_skipped = 0
    for x, m in tasks:
        kid = mat_key_id(x)
        base, lib = art['sols'].get(kid), rewritten_of.get(kid)
        if base is None or not lib:
            continue
        qs = discounted_qs(D, Q_u, x)
        try:
            prices = {v: (_dl(D, q, base), _dl(D, q, lib)) for v, q in qs.items()}
        except Exception:
            # same convention as report_dl_census: an unparseable rewritten form is a
            # stitch/library mismatch, not a magnitude — skip, don't misprice.
            n_skipped += 1
            continue
        for v, (dl_base, dl_lib) in prices.items():
            rec = per_kind[v].setdefault(m['kind'], {'dl_base': [], 'dl_lib': []})
            rec['dl_base'].append(dl_base)
            rec['dl_lib'].append(dl_lib)
            if m['kind'] == 'belief':
                bv = per_belief_variant[v].setdefault(
                    belief_variant(m), {'dl_base': [], 'dl_lib': []})
                bv['dl_base'].append(dl_base)
                bv['dl_lib'].append(dl_lib)
    if n_skipped:
        print(f"NOTE: {n_skipped} solved task(s) skipped (rewritten form failed to parse)")

    census = {v: {k: summarize(r) for k, r in per_kind[v].items()} for v in VARIANTS}
    by_variant = {v: {k: summarize(r) for k, r in per_belief_variant[v].items()}
                  for v in VARIANTS}

    hdr = (f"\n{'family':13s} {'n':>4s} |"
           + " |".join(f" {'no discount':>26s}" if v == 'none' else
                       f" {'half':>26s}" if v == 'half' else f" {'full (search prior)':>26s}"
                       for v in VARIANTS))
    sub = (f"{'':13s} {'':>4s} |"
           + " |".join(f" {'before':>8s} {'after':>8s} {'saved':>8s}" for _ in VARIANTS))
    print(hdr)
    print(sub)
    print("-" * len(sub))
    for kind in _ALL_KINDS:
        if kind not in census['none']:
            continue
        row = f"{kind:13s} {census['none'][kind]['n']:4d} |"
        for v in VARIANTS:
            s = census[v][kind]
            row += (f" {s['dl_base_median']:8.2f} {s['dl_lib_median']:8.2f}"
                    f" {s['saved_median']:8.2f} |")
        print(row + ('  <- belief' if kind == 'belief' else ''))

    print("\nbelief by variant (saved median):")
    for var in sorted(by_variant['none']):
        row = f"  {var:22s} {by_variant['none'][var]['n']:3d} |"
        for v in VARIANTS:
            row += f" {by_variant[v][var]['saved_median']:8.2f}"
        print(row)

    out_path = args.out or os.path.join(
        os.path.dirname(args.run) or '.',
        f"dl_census_reprice{'.decomposed' if args.decomposed else ''}.json")
    with open(out_path, 'w') as f:
        json.dump({'provenance': art.get('provenance'),
                   'source_run': args.run,
                   'decomposed': args.decomposed,
                   'library': [d.repr for d in D.invented],
                   'discount_variants': {
                       'none': 'uniform type Q (published tab-dl-census pricing)',
                       'half': 'boosted terminals cost half their uniform nats',
                       'full': 'ecd.content_boost verbatim (enumeration search prior)'},
                   'census_by_kind': census,
                   'belief_by_variant': by_variant}, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
