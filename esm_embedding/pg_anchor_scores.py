"""Recompute published ProteinGym supervised scores on our 63 assays.

Reads the ProteinGym v1.3 supervised release (DMS_supervised_substitutions_
scores.zip), which ships per-assay out-of-fold predictions for published
supervised baselines under the official fold_random_5 / fold_modulo_5 /
fold_contiguous_5 schemes, and computes per-assay Spearman for every method
on the 63 assays used in this paper.

Usage (on the machine holding the data):
  python -m esm_embedding.pg_anchor_scores \
      --sup_dir <...>/DMS_substitutions_supervised_scores \
      --names_file <63 assay names, one per line> \
      --out anchors_supervised.json
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def assay_spearman(csv_path):
    cols = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        pred_cols = [c for c in r.fieldnames if c.endswith("_predictions")]
        for c in pred_cols:
            cols[c] = ([], [])
        ycol = "DMS_score"
        for row in r:
            try:
                y = float(row[ycol])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(y):
                continue
            for c in pred_cols:
                try:
                    p = float(row[c])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(p):
                    cols[c][0].append(p)
                    cols[c][1].append(y)
    out = {}
    for c, (p, y) in cols.items():
        if len(p) >= 10:
            rho, _ = spearmanr(y, p)
            if np.isfinite(rho):
                out[c[:-len("_predictions")]] = float(rho)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup_dir", required=True,
                    help="dir containing fold_random_5 / fold_modulo_5 / "
                         "fold_contiguous_5")
    ap.add_argument("--names_file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [l.strip() for l in open(args.names_file) if l.strip()]
    results = {}
    for fold in ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]:
        fold_dir = Path(args.sup_dir) / fold
        per_assay = {}
        for name in names:
            f = fold_dir / f"{name}.csv"
            if not f.exists():
                per_assay[name] = None
                continue
            per_assay[name] = assay_spearman(f)
        results[fold] = per_assay

    summary = {}
    for fold, per_assay in results.items():
        methods = sorted({m for d in per_assay.values() if d for m in d})
        summary[fold] = {}
        for m in methods:
            vals = [d[m] for d in per_assay.values() if d and m in d]
            summary[fold][m] = {"mean": float(np.mean(vals)),
                                "n": len(vals)}
    out = {"summary": summary, "per_assay": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
