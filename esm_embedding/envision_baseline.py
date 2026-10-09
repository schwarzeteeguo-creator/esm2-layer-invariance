"""Envision-style supervised baseline under leakage-free folds (R2, B3).

Public supervised-learning baseline requested by the reviewer: the Envision
feature family (Gray et al., Nature Methods 2018) evaluated under the SAME
modulo / contiguous / random folds as our probing experiments.

Features per substitution row (Envision's sequence-based set):
  - one-hot wild-type AA (20) and mutant AA (20)
  - BLOSUM62 substitution score, Grantham-style distance
  - PSSM profile column at the mutated position, from the dataset's
    ProteinGym MSA (log-odds vs background; 20) + MSA coverage/weight
    summary (conservation score). Datasets without a usable MSA fall back
    to non-PSSM features (flagged in the output).
Readout: gradient boosting regressor (Envision's published model family),
sklearn HistGradientBoostingRegressor with default depth/leaves, seeded.

Protocol: identical to probing_v4 — 5 folds from pg_splits, variant cap
10,000 (seed 42), row subsample 5,000 (seed 42), pooled out-of-fold
Spearman per dataset.

Usage:
  python -m es_embedding.envision_baseline \
      --data_dir <LMDB substitutions> --msa_dir <DMS_msa_files dir> \
      --output_dir revision_results/envision
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, json, time, argparse, gzip, warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_embedding.pg_splits import make_splits
from esm_embedding.probing_v4 import (load_dataset_rows, AA_LIST, AA_TO_IDX,
                                      BLOSUM62)

warnings.filterwarnings("ignore")

AA_BG = np.full(20, 1.0 / 20)   # uniform background for PSSM log-odds


def read_fasta_a3m(path):
    """Read (gzipped) a2m/a3i FASTA; returns list of gap-stripped sequences
    (lowercase = insertions removed, first record = query)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    seqs, cur = [], []
    try:
        with opener(path, "rt", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                        cur = []
                else:
                    cur.append(line)
        if cur:
            seqs.append("".join(cur))
    except OSError:
        return []
    # strip insertions (lowercase) and non-standard chars; keep alignment
    # columns relative to the query by removing lowercase letters only
    out = []
    for s in seqs:
        s = "".join(c for c in s if c.isupper() or c == "-" or c == ".")
        s = s.replace(".", "-")
        out.append(s)
    return out


def build_pssm(msa_path, wt):
    """PSSM [L, 20] + per-position conservation, from the assay MSA."""
    seqs = read_fasta_a3m(msa_path)
    seqs = [s for s in seqs if len(s) == len(wt)] or \
           [s for s in seqs if abs(len(s) - len(wt)) <= 2]
    if len(seqs) < 5:
        return None
    L = len(wt)
    counts = np.zeros((L, 20))
    for s in seqs:
        for i, c in enumerate(s[:L]):
            j = AA_TO_IDX.get(c, -1)
            if j >= 0:
                counts[i, j] += 1
    freq = (counts + AA_BG * 1.0) / (counts.sum(axis=1, keepdims=True) + 1.0)
    pssm = np.log(freq / AA_BG)              # log-odds
    coverage = counts.sum(axis=1) / max(len(seqs), 1)
    return pssm, coverage


def build_row_features(ds, pssm_pack):
    """Row-level Envision feature matrix [n_rows, d]."""
    rows, wt = ds["rows"], ds["wt"]
    feats = []
    for (vid, p, f, t, _y) in rows:
        wt1 = np.zeros(20); mt1 = np.zeros(20)
        if f in AA_TO_IDX:
            wt1[AA_TO_IDX[f]] = 1.0
        if t in AA_TO_IDX:
            mt1[AA_TO_IDX[t]] = 1.0
        blosum = float(BLOSUM62.get(f, {}).get(t, 0))
        grantham = abs(AA_TO_IDX.get(f, 0) - AA_TO_IDX.get(t, 0)) / 19.0
        pos_norm = p / max(ds["L"], 1)
        vec = [wt1, mt1, [blosum], [grantham], [pos_norm]]
        if pssm_pack is not None:
            pssm, coverage = pssm_pack
            vec.append(pssm[p - 1])
            vec.append([coverage[p - 1]])
        feats.append(np.concatenate(vec))
    return np.stack(feats).astype(np.float32)


def eval_gbm(X, y, positions, vids, L, scheme, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    splits = make_splits(positions, vids, L, scheme, n_folds=5, seed=seed)
    preds = np.full(len(y), np.nan)
    for tr, te in splits:
        m = HistGradientBoostingRegressor(random_state=seed)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    valid = np.isfinite(preds)
    if valid.sum() < 10:
        return {"spearman": None, "n": int(valid.sum())}
    rho, _ = spearmanr(y[valid], preds[valid])
    return {"spearman": float(rho), "n": int(valid.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--msa_dir", default=None,
                    help="ProteinGym DMS_msa_files directory (one .a2m/.a3m/"
                         ".gz per assay, named by assay or UniProt id)")
    ap.add_argument("--output_dir", default="revision_results/envision")
    ap.add_argument("--splits", nargs="+",
                    default=["random", "modulo", "contiguous",
                             "position_groupkfold"])
    ap.add_argument("--max_datasets", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "envision_per_dataset.json"
    results = {}
    if out_file.exists():
        results = json.load(open(out_file))
        print(f"Resuming: {len(results)} datasets done")

    data_dir = Path(args.data_dir)
    files = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if args.max_datasets:
        files = files[:args.max_datasets]

    for f in files:
        name = f.name
        if name in results:
            continue
        ds = load_dataset_rows(f, 10000, 5000, 42)
        if ds is None:
            continue
        pssm_pack, msa_src = None, "none"
        if args.msa_dir:
            cands = list(Path(args.msa_dir).glob(f"{name}*")) + \
                    list(Path(args.msa_dir).glob(f"*{name.split('_')[0]}*"))
            # ProteinGym MSA files are keyed by UniProt id (col 1 of the
            # DMS table); fall back to fuzzy match on the assay prefix
            if not cands and len(name.split("_")) > 1:
                uni = "_".join(name.split("_")[:2])
                cands = list(Path(args.msa_dir).glob(f"{uni}*"))
            for c in cands:
                pack = build_pssm(c, ds["wt"])
                if pack is not None:
                    pssm_pack, msa_src = pack, c.name
                    break
        t0 = time.time()
        X = build_row_features(ds, pssm_pack)
        y, pos, vids, L = (ds["fitness"], ds["positions"],
                           ds["variant_ids"], ds["L"])
        res = {sp: eval_gbm(X, y, pos, vids, L, sp) for sp in args.splits}
        results[name] = {"n_rows": ds["n_rows"], "n_variants": ds["n_variants"],
                         "capped": ds["capped"], "msa": msa_src,
                         "with_pssm": pssm_pack is not None, "result": res}
        with open(out_file, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"  {name} pssm={pssm_pack is not None} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # summary
    print("\n" + "=" * 70)
    for sp in args.splits:
        vals = [r["result"][sp]["spearman"] for r in results.values()
                if r["result"].get(sp, {}).get("spearman") is not None]
        if vals:
            print(f"envision {sp:<20} N={len(vals)} mean rho={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
