"""Aggregate the single-substitution sensitivity run (B1) into the numbers
needed for manuscript Section 3.7 and Supplementary Table S11.

Reads the sharded probing_v4_single_* outputs (merged or shards), computes
per model-config x arm x split: layer-mean curves, best layer, mean paired
best-minus-final difference, and the real-3Di versus structure-masked
SaProt contrast at the final layer on the single-site subset. Compares each
quantity against the full-benchmark value it is a sensitivity check of.

Usage:
  python revision_docs/analysis_single_site.py [--local] [--outdir DIR]

--local: read from revision_results/single_site_local/ (fetched shards);
         default reads on the server via ssh+scp is NOT implemented, so
         fetch first or run on the server with --root.
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
CFG_DIRS = {
    "ESM-2": "probing_v4_single_esm2",
    "SaProt (mask)": "probing_v4_single_saprot",
    "SaProt (3Di)": "probing_v4_single_saprot_3di",
}
ARMS = ["masked_wt", "unmasked_wt", "mutant"]
SPLITS = ["random", "modulo", "contiguous", "position_groupkfold"]
# full-benchmark references (Table 2 of the manuscript, combined features)
FULL = {  # (model, arm, split): (best_layer, best_rho, L32_rho)
    ("ESM-2", "mutant", "modulo"): (31, 0.483, 0.479),
    ("ESM-2", "mutant", "contiguous"): (31, 0.365, 0.359),
    ("ESM-2", "mutant", "position_groupkfold"): (31, 0.477, 0.475),
    ("ESM-2", "masked_wt", "modulo"): (32, 0.459, 0.459),
    ("SaProt (mask)", "mutant", "modulo"): (30, 0.465, 0.460),
    ("SaProt (mask)", "mutant", "position_groupkfold"): (30, 0.461, 0.458),
    ("SaProt (3Di)", "mutant", "modulo"): (31, 0.531, 0.525),
    ("SaProt (3Di)", "mutant", "contiguous"): (31, 0.451, 0.445),
    ("SaProt (3Di)", "mutant", "position_groupkfold"): (31, 0.532, 0.524),
}


def load_curves(base: Path, arm: str, split: str):
    """dataset -> {layer:int -> rho} from shards + canonical file."""
    data = {}
    for f in sorted(base.glob(f"v4__{arm}__combined__{split}*.json")):
        if "merge" in f.name:
            continue
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            continue
        for name, e in d.items():
            curve = {int(k): v["spearman"] for k, v in
                     e["result"].items()
                     if isinstance(v, dict) and v.get("spearman") is not None
                     and np.isfinite(v["spearman"])}
            if curve:
                data[name] = curve
    return data


def layer_means(curves):
    lm = {}
    for c in curves.values():
        for l, v in c.items():
            lm.setdefault(l, []).append(v)
    return {l: float(np.mean(v)) for l, v in lm.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(
        ROOT / "revision_results" / "single_site_local"),
        help="directory containing the three probing_v4_single_* folders")
    ap.add_argument("--out", default=str(
        ROOT / "revision_results" / "r2_analysis" / "single_site_summary.json"))
    args = ap.parse_args()
    base = Path(args.root)

    report = {"configs": {}, "threedee_single_site": {},
              "coverage": {}}
    curves = {}
    for cfg, sub in CFG_DIRS.items():
        cdir = base / sub
        if not cdir.exists():
            print(f"[warn] missing {cdir}")
            continue
        for arm in ARMS:
            for split in SPLITS:
                cs = load_curves(cdir, arm, split)
                if not cs:
                    continue
                curves[(cfg, arm, split)] = cs
                lm = layer_means(cs)
                best = max(lm, key=lm.get)
                ds = [c[best] - c[32] for c in cs.values()
                      if best in c and 32 in c]
                rng = np.random.RandomState(42)
                boots = [np.mean(rng.choice(ds, len(ds))) for _ in range(10000)] \
                    if ds else []
                ci = [round(float(np.percentile(boots, 5)), 4),
                      round(float(np.percentile(boots, 95)), 4)] if boots else None
                entry = {
                    "n_datasets": len(cs),
                    "best_layer": best,
                    "best_rho": round(lm[best], 4),
                    "L32_rho": round(lm[32], 4),
                    "mean_best_minus_L32": round(float(np.mean(ds)), 4)
                    if ds else None,
                    "ci90_best_minus_L32": ci,
                }
                if (cfg, arm, split) in FULL:
                    fb, fbest, f32 = FULL[(cfg, arm, split)]
                    entry["full_best_layer"] = fb
                    entry["full_best_rho"] = fbest
                    entry["full_L32_rho"] = f32
                report["configs"][f"{cfg}|{arm}|{split}"] = entry
        report["coverage"][cfg] = len(
            load_curves(cdir, "mutant", "random"))

    # real-3Di vs structure-masked at L32, single-site subset (paired)
    for split in ["random", "modulo", "contiguous",
                  "position_groupkfold"]:
        a = curves.get(("SaProt (3Di)", "mutant", split), {})
        b = curves.get(("SaProt (mask)", "mutant", split), {})
        common = sorted(set(a) & set(b))
        if len(common) < 10:
            continue
        d = np.array([a[k][32] - b[k][32] for k in common])
        rep = {"n": len(d), "mean_delta": round(float(d.mean()), 4)}
        try:
            rep["wilcoxon_p"] = float(wilcoxon(d).pvalue)
        except ValueError:
            rep["wilcoxon_p"] = None
        report.setdefault("threedee_single_site", {})[split] = rep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps(report, indent=1)[:4000])


if __name__ == "__main__":
    main()
