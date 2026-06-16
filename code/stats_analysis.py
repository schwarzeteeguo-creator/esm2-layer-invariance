"""Statistical analysis for probing/pooling results.

Computes:
  - Bootstrap 95% CI for mean Spearman rho per layer
  - Cohen's d effect size for layer comparisons
  - Per-dataset Δρ distribution (multi-layer vs last-layer)
  - Computational cost comparison table
  - FDR-corrected pairwise tests

Usage:
  python -m esm_embedding.stats_analysis \
      --probing_results /path/to/layer_probing_xxx.json \
      --pooling_results /path/to/multiscale_results.json \
      --output_dir /path/to/stats_output
"""

import os, sys, json, argparse
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
from collections import defaultdict

def bootstrap_ci(values, n_bootstrap=10000, ci=95, seed=42):
    """Compute bootstrap confidence interval for mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    means = np.zeros(n_bootstrap)
    n = len(values)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        means[i] = np.mean(values[idx])
    alpha = (100 - ci) / 2
    return np.mean(values), np.percentile(means, alpha), np.percentile(means, 100 - alpha)

def cohens_d(x, y, paired=True):
    """Cohen's d effect size."""
    x, y = np.array(x), np.array(y)
    diff = x - y if paired else None
    if paired:
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    else:
        pooled_std = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
        d = (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
    return d

def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    bh_thresholds = (np.arange(1, n + 1) / n) * alpha
    significant = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= bh_thresholds[i]:
            significant[:i + 1] = True
            break
    result = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        result[idx] = significant[i]
    return result

def analyze_probing(results_path, output_dir):
    """Analyze per-layer probing results."""
    with open(results_path) as f:
        data = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_layers = 33
    layer_rhos = defaultdict(list)

    for name, r in data.items():
        for l_str, scores in r["layer_scores"].items():
            l = int(l_str)
            rho = scores.get("spearman")
            if rho is not None and not np.isnan(rho):
                layer_rhos[l].append(rho)

    print(f"\n{'='*70}")
    print(f"PROBING ANALYSIS: {Path(results_path).stem}")
    print(f"{'='*70}")
    print(f"Datasets: {len(data)}")

    # Bootstrap CI per layer
    print(f"\n{'Layer':>6} {'Mean':>10} {'95% CI low':>12} {'95% CI high':>12} {'N':>6}")
    print("-" * 52)
    for l in sorted(layer_rhos.keys()):
        if len(layer_rhos[l]) < 3: continue
        mean_rho, ci_low, ci_high = bootstrap_ci(layer_rhos[l])
        marker = " <-- final" if l == num_layers - 1 else ""
        print(f"  {l:3d}   {mean_rho:>10.4f}  {ci_low:>11.4f}  {ci_high:>11.4f}  {len(layer_rhos[l]):>5}{marker}")

    # Invariance score
    layer_means = [np.mean(layer_rhos[l]) for l in range(num_layers) if len(layer_rhos[l]) > 0]
    inv_score = np.std(layer_means)
    print(f"\nInvariance score: {inv_score:.6f}")

    # Best vs final layer
    best_l = max(range(num_layers), key=lambda l: np.mean(layer_rhos[l]) if len(layer_rhos[l]) > 0 else -999)
    final_l = num_layers - 1

    best_vals = np.array(layer_rhos[best_l])
    final_vals = np.array(layer_rhos[final_l])

    # Paired per-dataset comparison
    common_keys = []
    paired_best, paired_final = [], []
    for name in data:
        bs = data[name]["layer_scores"].get(str(best_l), {}).get("spearman", np.nan)
        fs = data[name]["layer_scores"].get(str(final_l), {}).get("spearman", np.nan)
        if not np.isnan(bs) and not np.isnan(fs):
            paired_best.append(bs)
            paired_final.append(fs)
            common_keys.append(name)

    paired_best = np.array(paired_best)
    paired_final = np.array(paired_final)

    t_stat, p_val = sp_stats.ttest_rel(paired_best, paired_final)
    w_stat, w_p = sp_stats.wilcoxon(paired_best, paired_final)
    cd = cohens_d(paired_best, paired_final, paired=True)

    print(f"\nBest layer: {best_l} (rho={np.mean(best_vals):.4f})")
    print(f"Final layer: {final_l} (rho={np.mean(final_vals):.4f})")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")
    print(f"Wilcoxon: W={w_stat:.1f}, p={w_p:.4f}")
    print(f"Cohen's d: {cd:.4f}")
    print(f"Delta ρ: {np.mean(paired_best - paired_final):+.6f}")

    # FDR correction for all layer pairs
    pvals_all = []
    pairs_all = []
    for l1 in range(num_layers):
        for l2 in range(l1 + 1, num_layers):
            vals1 = []
            vals2 = []
            for name in data:
                v1 = data[name]["layer_scores"].get(str(l1), {}).get("spearman", np.nan)
                v2 = data[name]["layer_scores"].get(str(l2), {}).get("spearman", np.nan)
                if not np.isnan(v1) and not np.isnan(v2):
                    vals1.append(v1)
                    vals2.append(v2)
            if len(vals1) >= 3:
                _, p = sp_stats.ttest_rel(vals1, vals2)
                pvals_all.append(p)
                pairs_all.append((l1, l2))

    fdr_sig = fdr_correction(pvals_all)
    n_sig_fdr = sum(fdr_sig)
    n_sig_nominal = sum(p < 0.05 for p in pvals_all)
    print(f"\nPairwise layer comparisons: {len(pvals_all)} total")
    print(f"  Nominally significant (p<0.05): {n_sig_nominal}")
    print(f"  FDR-significant (BH, α=0.05): {n_sig_fdr}")
    if n_sig_fdr > 0:
        for i in np.where(fdr_sig)[0]:
            print(f"    Layer {pairs_all[i][0]} vs {pairs_all[i][1]}: p={pvals_all[i]:.4f}")

    # Per-dataset Δρ distribution
    deltas_all = []
    for name in data:
        rhos = [data[name]["layer_scores"].get(str(l), {}).get("spearman", np.nan)
                for l in range(num_layers)]
        rhos = [r for r in rhos if not np.isnan(r)]
        if rhos:
            deltas_all.append(np.max(rhos) - np.min(rhos))

    deltas_all = np.array(deltas_all)
    print(f"\nWithin-dataset Δρ (max-min):")
    print(f"  Mean:   {np.mean(deltas_all):.5f}")
    print(f"  Median: {np.median(deltas_all):.5f}")
    print(f"  Max:    {np.max(deltas_all):.5f}")
    print(f"  % with Δρ < 0.01: {np.sum(deltas_all < 0.01) / len(deltas_all) * 100:.1f}%")

    # Save stats
    stats_out = {
        "n_datasets": len(data),
        "invariance_score": float(inv_score),
        "best_layer": best_l, "best_layer_rho": float(np.mean(best_vals)),
        "final_layer_rho": float(np.mean(final_vals)),
        "delta_rho": float(np.mean(paired_best - paired_final)),
        "paired_t": {"t": float(t_stat), "p": float(p_val)},
        "wilcoxon": {"W": float(w_stat), "p": float(w_p)},
        "cohens_d": float(cd),
        "n_pairwise_comparisons": len(pvals_all),
        "n_nominally_significant": int(n_sig_nominal),
        "n_fdr_significant": int(n_sig_fdr),
        "within_dataset_delta": {
            "mean": float(np.mean(deltas_all)),
            "median": float(np.median(deltas_all)),
            "max": float(np.max(deltas_all)),
        },
        "bootstrap_ci": {
            str(l): {"mean": float(np.mean(layer_rhos[l])),
                     "ci_low": float(bootstrap_ci(layer_rhos[l])[1]),
                     "ci_high": float(bootstrap_ci(layer_rhos[l])[2])}
            for l in range(num_layers) if len(layer_rhos[l]) >= 3
        }
    }

    stats_file = output_dir / f"stats_{Path(results_path).stem}.json"
    with open(stats_file, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nStats saved to {stats_file}")
    return stats_out


def analyze_pooling(results_path, output_dir):
    """Analyze multi-scale pooling results."""
    with open(results_path) as f:
        data = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["last_layer", "mean_20_33", "concat_6_14_20_26_33"]
    method_labels = {
        "last_layer": "Last Layer",
        "mean_20_33": "Mean L20-33",
        "concat_6_14_20_26_33": "Concat 5L",
    }

    readout_models = set()
    for name, r in data.items():
        for method in methods:
            mr = r.get("methods", {}).get(method, {})
            readout_models.update(k.replace("_rho", "") for k in mr if k.endswith("_rho"))

    print(f"\n{'='*70}")
    print(f"POOLING ANALYSIS: {Path(results_path).stem}")
    print(f"{'='*70}")
    print(f"Datasets: {len(data)}, Readout models: {readout_models}")

    for rm in sorted(readout_models):
        key = f"{rm}_rho"
        print(f"\n--- {rm.upper()} ---")
        method_means = {}
        for method in methods:
            vals = []
            for name, r in data.items():
                v = r.get("methods", {}).get(method, {}).get(key)
                if v is not None and not np.isnan(v):
                    vals.append(v)
            if vals:
                mean_v, ci_low, ci_high = bootstrap_ci(vals)
                method_means[method] = np.mean(vals)
                print(f"  {method_labels[method]:<20}: mean={mean_v:.4f} [{ci_low:.4f}, {ci_high:.4f}] (n={len(vals)})")

        # Paired delta vs last_layer
        if "last_layer" in method_means:
            baseline_val = method_means["last_layer"]
            for method in methods:
                if method == "last_layer": continue
                paired_deltas = []
                for name, r in data.items():
                    rl = r.get("methods", {}).get("last_layer", {}).get(key)
                    rm_val = r.get("methods", {}).get(method, {}).get(key)
                    if rl is not None and rm_val is not None and not np.isnan(rl) and not np.isnan(rm_val):
                        paired_deltas.append(rm_val - rl)
                if paired_deltas:
                    paired_deltas = np.array(paired_deltas)
                    delta_mean, delta_ci_low, delta_ci_high = bootstrap_ci(paired_deltas)
                    cd = cohens_d(np.array([d + 0 for d in paired_deltas]), np.zeros_like(paired_deltas), paired=True)
                    print(f"    Δ vs Last Layer: mean={delta_mean:.5f} [{delta_ci_low:.5f}, {delta_ci_high:.5f}]")
                    print(f"    % |Δ| < 0.002: {np.sum(np.abs(paired_deltas) < 0.002) / len(paired_deltas) * 100:.1f}%")


def compute_cost_table():
    """Compute computational cost comparison."""
    print(f"\n{'='*70}")
    print("COMPUTATIONAL COST ANALYSIS")
    print(f"{'='*70}")

    strategies = {
        "Last layer": {"dim": 1280, "layers": 1},
        "Mean L20-33 (pooled)": {"dim": 1280, "layers": 14},
        "Concat 5 layers": {"dim": 6400, "layers": 5},
        "All 33 layers (stored)": {"dim": 42240, "layers": 33},
    }

    n_mutations = 100_000
    bytes_per_float = 4

    print(f"\nFor a screen of {n_mutations:,} mutations (float32):")
    print(f"{'Strategy':<28} {'Dim':>8} {'Storage':>12} {'Rel. cost':>10}")
    print("-" * 62)
    baseline_storage = None
    for name, info in strategies.items():
        storage_gb = info["dim"] * n_mutations * bytes_per_float / (1024**3)
        if baseline_storage is None:
            baseline_storage = storage_gb
            rel = "1×"
        else:
            rel = f"{storage_gb / baseline_storage:.0f}×"
        print(f"  {name:<26} {info['dim']:>8} {storage_gb:>9.2f} GB {rel:>10}")

    print(f"\nFor a screen of 1,000,000 mutations:")
    print(f"{'Strategy':<28} {'Storage':>12}")
    print("-" * 42)
    for name, info in strategies.items():
        storage_gb = info["dim"] * 1_000_000 * bytes_per_float / (1024**3)
        print(f"  {name:<26} {storage_gb:>9.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probing_results", nargs="*", default=[],
                        help="Path(s) to probing result JSON files")
    parser.add_argument("--pooling_results", nargs="*", default=[],
                        help="Path(s) to pooling result JSON files")
    parser.add_argument("--output_dir", default="stats_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for probing_path in args.probing_results:
        if Path(probing_path).exists():
            analyze_probing(probing_path, args.output_dir)

    for pooling_path in args.pooling_results:
        if Path(pooling_path).exists():
            analyze_pooling(pooling_path, args.output_dir)

    compute_cost_table()

if __name__ == "__main__":
    main()
