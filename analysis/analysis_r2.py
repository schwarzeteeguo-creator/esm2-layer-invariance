"""Round-2 revision analyses (local, CPU-only).

A1  TOST recompute: all leakage-free best-vs-final layer comparisons with a
    single unified verdict rule (bootstrap 90% CI inside +/-0.01 margin).
A2  Pooling aggregation: full arm x split x strategy x readout means from
    pooling_v3 JSONs (includes the unreported random split).
A3  Paired bootstrap: (a) last layer vs each fusion strategy, (b) final-layer
    LightGBM vs Ridge.
A5  Cap / multi-site statistics from the raw ProteinGym CSVs, replicating
    load_dataset_rows (variant cap 10k seed 42, row subsample 5k seed 42)
    and the modulo/contiguous fold-straddling behaviour.

Usage:
  python revision_docs/analysis_r2.py            # everything
  python revision_docs/analysis_r2.py --only a1 # one part
"""

import argparse, json, sys, csv, math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SR = ROOT / "revision_results" / "server_results"
OUT = ROOT / "revision_results" / "r2_analysis"
OUT.mkdir(parents=True, exist_ok=True)

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
MARGIN = 0.01
N_BOOT = 10000
SEED = 42

PROBING_DIRS = {
    "ESM-2": SR / "probing_v4_esm2",
    "SaProt (mask)": SR / "probing_v4_saprot",
    "SaProt (3Di)": SR / "probing_v4_saprot_3di",
}
LEAK_FREE = ["modulo", "contiguous", "position_groupkfold"]
ARMS = ["masked_wt", "unmasked_wt", "mutant"]


def load_probing(model_dir, arm, split, featset="combined"):
    f = model_dir / f"v4__{arm}__{featset}__{split}.json"
    if not f.exists():
        return None
    with open(f) as fh:
        raw = json.load(fh)
    # dataset -> {layer(str): spearman}
    out = {}
    for name, entry in raw.items():
        curve = {}
        for layer, node in entry["result"].items():
            v = node.get("spearman")
            if v is not None and np.isfinite(v):
                curve[layer] = float(v)
        if curve:
            out[name] = curve
    return out


# ───────────────────────────── A1: TOST ─────────────────────────────

def boot_ci90(d, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    means = d[rng.randint(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return [float(np.percentile(means, 5)), float(np.percentile(means, 95))]


def tost_p(d, margin=MARGIN):
    d = np.asarray(d, float)
    n = len(d)
    dbar, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0 if abs(dbar) < margin else 1.0
    df = n - 1
    p_low = 1.0 - stats.t.cdf((dbar + margin) / se, df)
    p_high = stats.t.cdf((dbar - margin) / se, df)
    return float(max(p_low, p_high))


def t_ci90(d):
    d = np.asarray(d, float)
    n = len(d)
    dbar, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    tcrit = stats.t.ppf(0.95, n - 1)
    return [float(dbar - tcrit * se), float(dbar + tcrit * se)]


def run_a1():
    rows, degenerate = [], []
    for model, mdir in PROBING_DIRS.items():
        for arm in ARMS:
            for split in LEAK_FREE:
                per_ds = load_probing(mdir, arm, split)
                if not per_ds:
                    continue
                layer_means = {}
                for name, curve in per_ds.items():
                    for l, v in curve.items():
                        layer_means.setdefault(l, []).append(v)
                layer_means = {l: np.mean(v) for l, v in layer_means.items()}
                best = max(layer_means, key=layer_means.get)
                if best == "32":
                    degenerate.append((model, arm, split, layer_means["32"]))
                    continue
                pairs = []
                for name, curve in per_ds.items():
                    if "32" in curve and best in curve:
                        pairs.append((name, curve[best] - curve["32"]))
                d = np.array([p[1] for p in pairs])
                b = boot_ci90(d)
                t = t_ci90(d)
                verdict = ("Equivalent" if (b[0] >= -MARGIN and b[1] <= MARGIN)
                           else "Difference" if (b[0] > MARGIN or b[1] < -MARGIN)
                           else "Inconclusive")
                p = tost_p(d)
                wilcox = (stats.wilcoxon(d).pvalue
                          if len(d) >= 10 and np.any(d != 0) else None)
                rows.append({
                    "model": model, "arm": arm, "split": split,
                    "best_layer": int(best), "n": len(d),
                    "mean_delta": float(d.mean()),
                    "boot_ci90": b, "t_ci90": t,
                    "tost_p": p, "tost_verdict": "Equivalent" if p < 0.05 else "Not equiv.",
                    "verdict": verdict,
                    "dz": float(d.mean() / (d.std(ddof=1) + 1e-12)),
                    "wilcoxon_p": float(wilcox) if wilcox else None,
                })
    n_eq = sum(r["verdict"] == "Equivalent" for r in rows)
    n_inc = sum(r["verdict"] == "Inconclusive" for r in rows)
    n_diff = sum(r["verdict"] == "Difference" for r in rows)
    summary = {"n_tests": len(rows), "equivalent": n_eq,
               "inconclusive": n_inc, "difference": n_diff,
               "n_degenerate_best_is_L32": len(degenerate)}
    with open(OUT / "a1_tost.json", "w") as fh:
        json.dump({"summary": summary, "rows": rows,
                   "degenerate": [list(d) for d in degenerate]}, fh, indent=1)
    print("A1 TOST:", summary)
    for r in rows:
        b = r["boot_ci90"]
        print(f"  {r['model']:<15} {r['arm']:<12} {r['split']:<18} "
              f"L{r['best_layer']:<3} n={r['n']:<3} "
              f"Δ={r['mean_delta']:+.4f} CI90=[{b[0]:+.4f},{b[1]:+.4f}] "
              f"p_tost={r['tost_p']:.3f} -> {r['verdict']}")
    return rows, summary


# ───────────────────────── A2/A3: pooling ─────────────────────────

def pooling_rhos(model_file, arm="mutant"):
    """dataset -> (split, strategy, readout) -> rho  (combined feats only)."""
    with open(model_file) as fh:
        raw = json.load(fh)
    out = {}
    for name, dsr in raw.items():
        a = dsr.get("arms", {}).get(arm, {})
        flat = {}
        for split, strats in a.items():
            for strat_feat, entry in strats.items():
                if not strat_feat.endswith("__combined"):
                    continue
                strat = strat_feat.split("__")[0]
                rho = entry.get("rho", entry)  # attention nests under "rho"
                for r, v in rho.items():
                    val = v.get("spearman") if isinstance(v, dict) else None
                    if val is not None and np.isfinite(val):
                        flat[(split, strat, r)] = float(val)
        out[name] = flat
    return out


def paired_diff_table(rhos, split, ref_strategy, strategies, readouts):
    """Paired (strategy - ref) per dataset with bootstrap 90% CI."""
    tables = {}
    for strat in strategies:
        if strat == ref_strategy:
            continue
        for ro in readouts:
            pairs = [(n, v[(split, strat, ro)] - v[(split, ref_strategy, ro)])
                     for n, v in rhos.items()
                     if (split, strat, ro) in v and (split, ref_strategy, ro) in v]
            if len(pairs) < 5:
                continue
            d = np.array([p[1] for p in pairs])
            b = boot_ci90(d)
            wilcox = (stats.wilcoxon(d).pvalue
                      if np.any(d != 0) else None)
            tables[(strat, ro)] = {
                "n": len(d), "mean_delta": float(d.mean()),
                "boot_ci90": b,
                "wilcoxon_p": float(wilcox) if wilcox else None,
                "per_dataset": dict(pairs),
            }
    return tables


def run_a2_a3():
    strategies = ["last_layer", "mean_20_33", "concat_5L", "attention"]
    readouts = ["ridge", "rf", "mlp", "lightgbm"]
    report = {"means": {}, "fusion_vs_last": {}, "lgbm_vs_ridge": {}}
    for model_tag, sub in [("ESM-2", "pooling_v3_esm2"),
                           ("SaProt", "pooling_v3_saprot")]:
        f = next((SR / sub).glob("pooling_v3__*.json"))
        rhos = pooling_rhos(f)
        # A2: means per split/strategy/readout (mutant arm, combined)
        for split in ["random", "modulo", "contiguous", "position_groupkfold"]:
            for strat in strategies:
                for ro in readouts:
                    vals = [v[(split, strat, ro)] for v in rhos.values()
                            if (split, strat, ro) in v]
                    if vals:
                        report["means"].setdefault(model_tag, {}) \
                            .setdefault(split, {})[f"{strat}__{ro}"] = {
                            "mean": float(np.mean(vals)),
                            "sd": float(np.std(vals, ddof=1)),
                            "n": len(vals)}
        # A3a: fusion vs last layer (all splits, mutant arm)
        for split in ["random", "modulo", "contiguous"]:
            report["fusion_vs_last"][model_tag, split] = paired_diff_table(
                rhos, split, "last_layer",
                ["mean_20_33", "concat_5L", "attention"], readouts)
        # A3b: final-layer LightGBM vs Ridge (mutant arm)
        for split in ["random", "modulo", "contiguous"]:
            pairs = [(n, v[(split, "last_layer", "lightgbm")]
                          - v[(split, "last_layer", "ridge")])
                     for n, v in rhos.items()
                     if (split, "last_layer", "lightgbm") in v
                     and (split, "last_layer", "ridge") in v]
            d = np.array([p[1] for p in pairs])
            report["lgbm_vs_ridge"][model_tag, split] = {
                "n": len(d), "mean_delta": float(d.mean()),
                "boot_ci90": boot_ci90(d),
                "wilcoxon_p": float(stats.wilcoxon(d).pvalue)
                if np.any(d != 0) else None,
                "per_dataset": dict(pairs)}

    def stringify_keys(o):
        if isinstance(o, dict):
            return {"|".join(map(str, k)) if isinstance(k, tuple)
                    else str(k): stringify_keys(v) for k, v in o.items()}
        return o
    with open(OUT / "a2_a3_pooling.json", "w") as fh:
        json.dump(stringify_keys(report), fh, indent=1)

    # console digest: ridge + lightgbm, random vs modulo (mutant, combined)
    for model_tag in ["ESM-2", "SaProt"]:
        print(f"\nA2 {model_tag} (mutant arm, combined):")
        for split in ["random", "modulo"]:
            for ro in ["ridge", "lightgbm"]:
                line = f"  [{split:>7}/{ro:>9}] "
                for strat in strategies:
                    m = report["means"][model_tag].get(split, {}) \
                        .get(f"{strat}__{ro}", {}).get("mean")
                    line += f" {strat[:6]}={m:.3f}" if m else ""
                print(line)
        print(f"  fusion-vs-last (modulo, ridge):")
        for (strat, ro), t in report["fusion_vs_last"][
                (model_tag, "modulo")].items():
            if ro == "ridge":
                b = t["boot_ci90"]
                print(f"    {strat:<12} Δ={t['mean_delta']:+.4f} "
                      f"CI90=[{b[0]:+.4f},{b[1]:+.4f}] "
                      f"p={t['wilcoxon_p']:.3g}")
        t = report["lgbm_vs_ridge"][(model_tag, "modulo")]
        b = t["boot_ci90"]
        print(f"  last-layer lgbm-vs-ridge: Δ={t['mean_delta']:+.4f} "
              f"CI90=[{b[0]:+.4f},{b[1]:+.4f}] p={t['wilcoxon_p']:.3g}")
    return report


# ───────────────────── A5: caps & multi-site stats ─────────────────────

def parse_mutations(mut):
    out = []
    for token in mut.split(":"):
        if len(token) >= 3 and token[0] in AA_TO_IDX and token[-1] in AA_TO_IDX:
            try:
                out.append((token[0], int(token[1:-1]), token[-1]))
            except ValueError:
                continue
    return out


def run_a5():
    csv_dir = ROOT / "ProteinGym_DMS_data" / "DMS_ProteinGym_substitutions"
    # authoritative bookkeeping: the probing pipeline's own records
    bk_file = SR / "probing_v4_esm2" / "v4__mutant__combined__random.json"
    bk_all = json.load(open(bk_file))
    names = sorted(bk_all.keys())
    auth = {
        "n_capped_10k": sum(bool(b["capped"]) for b in bk_all.values()),
        "n_rows_at_5000": sum(b["n_rows"] == 5000 for b in bk_all.values()),
        "n_rows_lt_5000": sum(b["n_rows"] < 5000 for b in bk_all.values()),
        "total_analyzed_rows": sum(b["n_rows"] for b in bk_all.values()),
        "total_analyzed_variants": sum(b["n_variants"] for b in bk_all.values()),
    }
    stats_rows = []
    for name in names:
        f = csv_dir / f"{name}.csv"
        if not f.exists():
            stats_rows.append({"dataset": name, "missing_csv": True})
            continue
        variants = {}
        L = None
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                fit = row.get("DMS_score")
                mut = row.get("mutant", "")
                seq = row.get("mutated_sequence", "")
                try:
                    fit = float(fit)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fit):
                    continue
                if L is None and seq:
                    L = len(seq)
                subs = parse_mutations(mut)
                if not subs:
                    continue
                if L:
                    subs = [(a, p, b) for (a, p, b) in subs if 1 <= p <= L]
                    if not subs:
                        continue
                if seq and L and len(seq) != L:
                    continue
                variants[mut] = subs
        n_var = len(variants)
        capped = n_var > 10000
        if capped:  # seeded cap, same as load_dataset_rows
            rng = np.random.RandomState(SEED)
            keys = sorted(variants)
            keep = rng.choice(len(keys), 10000, replace=False)
            variants = {keys[i]: variants[keys[i]] for i in keep}
        rows_pos = [p for subs in variants.values() for (_, p, _) in subs]
        n_rows_raw = len(rows_pos)
        subsampled = n_rows_raw > 5000
        multi = {k: s for k, s in variants.items() if len(s) >= 2}
        n_multi_rows = sum(len(s) for s in multi.values())
        # fold straddling under modulo / contiguous (row fold = own position)
        straddle_mod = straddle_cont = 0
        for k, subs in multi.items():
            pos = [p for (_, p, _) in subs]
            if L:
                fm = {(p - 1) % 5 for p in pos}
                fc = {min(4, int((p - 1) * 5 // max(L, 1))) for p in pos}
                straddle_mod += len(fm) > 1
                straddle_cont += len(fc) > 1
        # cross-check vs probing bookkeeping
        bk = bk_all[name]
        stats_rows.append({
            "dataset": name, "n_variants_total": n_var,
            "capped_10k": bool(capped), "subsampled_5k_rows": bool(subsampled),
            "n_rows_after_cap": n_rows_raw,
            "bk_n_rows": bk.get("n_rows"), "bk_n_variants": bk.get("n_variants"),
            "bk_capped": bk.get("capped"),
            "multi_site_variants": len(multi),
            "multi_site_rows": n_multi_rows,
            "multi_site_row_pct_of_capped_rows":
                100.0 * n_multi_rows / max(n_rows_raw, 1),
            "multi_variants_straddling_modulo_folds": straddle_mod,
            "multi_variants_straddling_contiguous_folds": straddle_cont,
        })
    n_capped = sum(r.get("capped_10k", False) for r in stats_rows)
    n_sub = sum(r.get("subsampled_5k_rows", False) for r in stats_rows)
    tot_var = sum(r.get("n_variants_total", 0) for r in stats_rows)
    tot_multi = sum(r.get("multi_site_variants", 0) for r in stats_rows)
    tot_rows = sum(r.get("n_rows_after_cap", 0) for r in stats_rows)
    tot_multi_rows = sum(r.get("multi_site_rows", 0) for r in stats_rows)
    straddle_m = sum(r.get("multi_variants_straddling_modulo_folds", 0)
                     for r in stats_rows)
    straddle_c = sum(r.get("multi_variants_straddling_contiguous_folds", 0)
                     for r in stats_rows)
    # replication check (CSV vs LMDB bookkeeping; expected to differ where
    # the LMDB snapshot content differs from the v1.3 CSVs)
    mism = [r["dataset"] for r in stats_rows
            if not r.get("missing_csv")
            and (r["bk_capped"] != r["capped_10k"])]
    mism_rows = [r["dataset"] for r in stats_rows
                 if not r.get("missing_csv")
                 and (min(r["n_rows_after_cap"], 5000) != r["bk_n_rows"])]
    summary = {
        "authoritative_from_probing_bookkeeping": auth,
        "csv_replication": {
            "n_datasets": len(stats_rows),
            "datasets_capped_at_10k_variants": n_capped,
            "datasets_row_subsampled_to_5000": n_sub,
            "multi_site_variant_share_pct_postcap":
                100.0 * tot_multi / max(tot_var, 1),
            "multi_site_row_share_pct_postcap":
                100.0 * tot_multi_rows / max(tot_rows, 1),
            "multi_variants_straddling_modulo": straddle_m,
            "multi_variants_straddling_contiguous": straddle_c,
            "cap_flag_mismatches_vs_probing_json": mism,
            "row_count_mismatches_vs_probing_json": len(mism_rows),
            "note": "CSV(v1.3) replication; authoritative multi-site stats "
                    "recomputed from the LMDB snapshot on the server "
                    "(see b1_b2_b3 outputs).",
        },
    }
    with open(OUT / "a5_caps_multisite.json", "w") as fh:
        json.dump({"summary": summary, "rows": stats_rows}, fh, indent=1)
    print("\nA5:", json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["a1", "a2", "a5"], default=None)
    args = ap.parse_args()
    if args.only in (None, "a1"):
        run_a1()
    if args.only in (None, "a2"):
        run_a2_a3()
    if args.only in (None, "a5"):
        run_a5()


if __name__ == "__main__":
    main()
