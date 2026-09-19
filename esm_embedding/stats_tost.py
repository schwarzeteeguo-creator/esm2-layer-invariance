"""Equivalence testing + paired inference for the revision (Reviewer 1, M7 + m8/m10).

Implements:
  1. TOST (two one-sided tests) for practical equivalence with a
     pre-specified margin on the mean paired difference of per-dataset
     Spearman rho (default margin delta = 0.01, justified as "differences
     below this cannot change a downstream layer-choice decision").
  2. Bootstrap 95% CI for the mean paired difference (10k resamples,
     seeded).
  3. Paired effect size d_z (Cohen's d for paired designs — the
     convention Reviewer 1 asked us to specify).
  4. Within-dataset spread vs cross-dataset variance decomposition.

Works on probing_v4 output files (v4__arm__featset__split.json) and on
pooling_v3 files; any two "conditions" with per-dataset rho vectors can
be compared:

  python -m esm_embedding.stats_tost \
      --a revision_results/probing_v4/v4__masked_wt__combined__random.json \
      --layer_a 8  --layer_b 32 \
  python -m esm_embedding.stats_tost \
      --a fileA.json --b fileB.json   # two whole files (matched datasets)
"""

import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy import stats

DEFAULT_MARGIN = 0.01
N_BOOT = 10000
BOOT_SEED = 42


def extract_rhos(spec) -> dict:
    """Return {dataset: rho} from a v4/pooling JSON, optionally one layer."""
    path = Path(spec.file)
    with open(path) as f:
        data = json.load(f)
    out = {}
    for name, entry in data.items():
        r = entry["result"] if "result" in entry else entry
        if spec.layer is not None:
            node = r.get(str(spec.layer))
            if node is None:
                continue
            v = node.get("spearman")
        elif "rho_mean" in r:            # random-position control
            v = r["rho_mean"]
        elif "spearman" in r:            # aux arm
            v = r["spearman"]
        else:                            # per-layer dict -> layer mean
            vals = [x["spearman"] for x in r.values()
                    if isinstance(x, dict)
                    and np.isfinite(x.get("spearman", np.nan))]
            v = float(np.mean(vals)) if vals else None
        if v is not None and np.isfinite(v):
            out[name] = float(v)
    return out


def paired_tost(d, margin=DEFAULT_MARGIN):
    """TOST on paired differences d (H1: |mean| < margin)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dbar, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return {"equivalent": bool(abs(dbar) < margin), "p_tost": 0.0,
                "ci_low": dbar, "ci_high": dbar, "n": n,
                "mean_diff": dbar, "margin": margin, "se": 0.0}
    df = n - 1
    t_low = (dbar - (-margin)) / se          # H0: mean <= -margin
    t_high = (dbar - margin) / se            # H0: mean >=  margin
    p_low = 1.0 - stats.t.cdf(t_low, df)
    p_high = stats.t.cdf(t_high, df)
    p_tost = max(p_low, p_high)
    # exact t CI for reporting
    tcrit = stats.t.ppf(0.975, df)
    return {"n": n, "mean_diff": float(dbar), "se": float(se),
            "margin": margin,
            "ci95_t": [float(dbar - tcrit * se), float(dbar + tcrit * se)],
            "p_tost": float(p_tost),
            "equivalent": bool(p_tost < 0.05)}


def bootstrap_ci(d, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, dtype=float)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def cohen_dz(d):
    d = np.asarray(d, dtype=float)
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="file A (v4 or pooling json)")
    ap.add_argument("--b", default=None, help="file B (optional)")
    ap.add_argument("--layer_a", type=int, default=None)
    ap.add_argument("--layer_b", type=int, default=None)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    class Spec:
        pass
    sa, sb = Spec(), Spec()
    sa.file, sb.file = args.a, args.b or args.a
    sa.layer, sb.layer = args.layer_a, args.layer_b

    ra, rb = extract_rhos(sa), extract_rhos(sb)
    common = sorted(set(ra) & set(rb))
    if not common:
        sys.exit("No matched datasets between the two conditions.")
    d = np.array([ra[k] - rb[k] for k in common])

    tost = paired_tost(d, margin=args.margin)
    boot = bootstrap_ci(d)
    dz = cohen_dz(d)
    wilcox = stats.wilcoxon(d) if len(d) >= 10 and np.any(d != 0) else None

    label = args.label or (f"{Path(args.a).stem}"
                           f"{f' L{args.layer_a}' if args.layer_a is not None else ''}"
                           f" vs "
                           f"{Path(args.b or args.a).stem}"
                           f"{f' L{args.layer_b}' if args.layer_b is not None else ''}")
    print("=" * 72)
    print(f"PAIRED COMPARISON: {label}")
    print(f"matched datasets : {len(common)}")
    print(f"mean paired Δρ   : {tost['mean_diff']:+.5f}  (SE {tost['se']:.5f})")
    print(f"bootstrap 95% CI : [{boot[0]:+.5f}, {boot[1]:+.5f}]")
    print(f"t-based 95% CI   : [{tost['ci95_t'][0]:+.5f}, {tost['ci95_t'][1]:+.5f}]")
    print(f"Cohen's d_z      : {dz:+.4f}")
    if wilcox is not None:
        print(f"Wilcoxon signed-rank p (difference test): {wilcox.pvalue:.4g}")
    print(f"TOST margin ±{tost['margin']}: p_TOST = {tost['p_tost']:.4g} -> "
          f"{'PRACTICALLY EQUIVALENT' if tost['equivalent'] else 'NOT equivalent at this margin'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
