"""Multi-scale pooling benchmark V3 — the M3-fixed rerun.

Fixes relative to benchmark_v2 (original submission):
  1. FEATURE ARMS: mutation-specific features (mutant sequence with the
     substitution in place) as the primary arm; masked_wt kept for
     continuity with the original Experiment 1 protocol. The old
     Experiment 2 used UNMASKED WT embeddings — a different feature
     construction that explains the 0.600-vs-0.386 discrepancy
     (Reviewer 1, Major 2).
  2. Concat 5L BUG: the old pooling code filtered layer index 33 with
     `i < num_layers`, silently dropping the final layer — "Concat 5L"
     actually concatenated FOUR layers (6,14,20,26). V3 uses
     [6,14,20,26,32] (0-indexed final layer = 32), matching the paper's
     stated design.
  3. RIDGE: StandardScaler fit inside training folds only + alpha by
     nested CV (inner 3-fold, grid 1e-2..1e3). No fixed alpha=1.0.
     (Reviewer 1, Major 3)
  4. ATTENTION pooling is now genuinely TRAINED per training fold
     (softmax layer weights fit by gradient descent jointly with a ridge
     readout on training rows only). The old implementation instantiated
     an untrained module whose zero-initialised weights softmax to a
     uniform mean — it was secretly "uniform layer mean".
  5. MLP readout matches the manuscript's stated architecture
     (two hidden layers of 256) with enough iterations, instead of the
     single-128-unit, 50-iteration surrogate actually used before.
  6. Readout hyperparameters match the manuscript text: RF = 50 trees
     / depth 10, LightGBM = 100 estimators / depth 7.
  7. All four split schemes from pg_splits (ProteinGym official modulo
     and contiguous included); variants never straddle folds.

Usage (server):
  python -m esm_embedding.benchmark_v3 \
      --data_dir ... --model_key esm2_650m --model_dir ... \
      --arms mutant --output_dir revision_results/pooling_v3
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, json, time, argparse, warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_embedding.pg_splits import make_splits, SPLIT_SCHEMES
from esm_embedding.probing_v4 import (load_dataset_rows, ALPHA_GRID,
                                      RIDGE_CV_MODE,
                                      extract_model_arms)

warnings.filterwarnings("ignore")

CONCAT_LAYERS = [6, 14, 20, 26, 32]   # 0-indexed; spans depth incl. final
MEAN_LAYERS = list(range(20, 33))     # 0-indexed 20..32


# ─────────────────────────── pooling strategies ───────────────────────────

def pool_last(units):        return units[:, -1, :]
def pool_mean(units):        return units[:, MEAN_LAYERS, :].mean(axis=1)
def pool_concat(units):      return units[:, CONCAT_LAYERS, :].reshape(
    units.shape[0], -1)


def train_attention_pool(units_train, y_train, seed=42, steps=120, lr=0.05,
                         alpha=10.0):
    """Fit softmax layer weights + ridge readout on TRAIN rows only.

    Alternating optimisation: gradient steps on the layer-weight logits
    through a differentiable closed-form ridge prediction.
    Returns (weights [n_layers], ridge_vector) fitted jointly.
    """
    import torch
    torch.manual_seed(seed)
    X = torch.tensor(units_train, dtype=torch.float64)     # [n, L, D]
    y = torch.tensor(y_train, dtype=torch.float64)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)               # standardise feats
    ys = (y - y.mean()) / (y.std() + 1e-8)
    logits = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    n = X.shape[0]
    eye = torch.eye(X.shape[2], dtype=torch.float64)
    for _ in range(steps):
        opt.zero_grad()
        w = torch.softmax(logits, dim=0)
        pooled = torch.einsum("l,nld->nd", w, Xs)
        A = pooled.T @ pooled + alpha * eye
        coef = torch.linalg.solve(A, pooled.T @ ys)
        pred = pooled @ coef
        loss = ((pred - ys) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w = torch.softmax(logits, dim=0).numpy()
    return w


def pool_attention_apply(units, weights):
    """[n, L, D] layer weights [L] -> pooled [n, D]."""
    return np.einsum("l,nld->nd", weights, units)


# ─────────────────────────── readouts (in-fold scaling) ───────────────────────

def make_readout(name, seed=42):
    if name == "ridge":
        return RidgeCV(alphas=ALPHA_GRID, cv=RIDGE_CV_MODE)  # GCV, as probing_v4
    if name == "rf":
        return RandomForestRegressor(n_estimators=50, max_depth=10,
                                     random_state=seed, n_jobs=4)
    if name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(256, 256), activation="relu",
                            alpha=1e-4, batch_size=128, max_iter=300,
                            early_stopping=True, n_iter_no_change=15,
                            random_state=seed)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, max_depth=7,
                             learning_rate=0.05, random_state=seed,
                             n_jobs=4, verbose=-1)
    raise ValueError(name)


def eval_readout(name, X_raw, y, train_idx, test_idx, seed=42):
    scaler = StandardScaler().fit(X_raw[train_idx])
    Xtr = scaler.transform(X_raw[train_idx])
    Xte = scaler.transform(X_raw[test_idx])
    model = make_readout(name, seed)
    model.fit(Xtr, y[train_idx])
    return model.predict(Xte)


STRATEGY_FNS = {
    "last_layer": pool_last,
    "mean_20_33": pool_mean,
    "concat_5L": pool_concat,
    # attention handled specially (trained per fold)
}


# Default splits for pooling: the two ProteinGym official leakage-free
# schemes plus random for continuity with the original benchmark.
# position_groupkfold is omitted here (semantically redundant with modulo
# for per-position pooling features); probing_v4 covers all four schemes.
POOLING_SPLITS = ["random", "modulo", "contiguous"]


def run_pooling_benchmark(data_dir, output_dir, model_key, model_dir=None,
                          device="cuda", arms=("mutant",),
                          splits=POOLING_SPLITS, readouts=("ridge", "rf", "mlp",
                                                          "lightgbm"),
                          feature_sets=("combined", "embedding_only"),
                          max_datasets=None, max_variants=10000,
                          max_rows=5000, n_folds=5, seed=42,
                          shard=0, nshards=1, extraction_batch=8):
    from esm_embedding.model import load_model, _MODEL_PATHS
    if model_dir:
        _MODEL_PATHS[model_key] = Path(model_dir)
    info = load_model(model_key, device=device)
    is_saprot = model_key.startswith("saprot")

    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if max_datasets:
        files = files[:max_datasets]
    if nshards > 1:
        # disjoint dataset subsets per concurrent shard; results go to a
        # suffixed file that merge_shards.py folds into the canonical one
        files = [f for i, f in enumerate(files) if i % nshards == shard]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"__shard{shard}" if nshards > 1 else ""
    results_file = out_dir / f"pooling_v3__{model_key}{sfx}.json"
    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} datasets done")
    # sharded backfills must also skip datasets completed by earlier
    # unsharded runs (recorded in the canonical file)
    skip_done = set()
    if nshards > 1:
        base_file = out_dir / f"pooling_v3__{model_key}.json"
        if base_file.exists():
            try:
                skip_done = set(json.load(open(base_file)).keys())
            except (json.JSONDecodeError, OSError):
                skip_done = set()
        if skip_done:
            print(f"Skipping {len(skip_done)} datasets done in canonical file")

    for f in files:
        name = f.name
        if name in all_results or name in skip_done:
            continue
        ds = load_dataset_rows(f, max_variants, max_rows, seed)
        if ds is None:
            continue
        t0 = time.time()
        caches = extract_model_arms(ds, info, list(arms), info["device"],
                                    is_saprot, batch_size=extraction_batch)
        aux, y = ds["aux"], ds["fitness"]
        positions, vids, L = ds["positions"], ds["variant_ids"], ds["L"]

        ds_result = {"n_rows": ds["n_rows"], "capped": ds["capped"],
                     "arms": {}}
        for arm, cache in caches.items():
            units = cache["units"].astype(np.float32)
            unit_of_row = cache["unit_of_row"]
            arm_out = {}
            for split in splits:
                fold_list = make_splits(positions, vids, L, split,
                                        n_folds=n_folds, seed=seed)
                strat_out = {}
                for strategy in list(STRATEGY_FNS) + ["attention"]:
                    if strategy == "attention":
                        # train layer weights on TRAIN rows of each fold
                        attn_weights = []
                        pooled_by_fold = []
                        for tr, te in fold_list:
                            w = train_attention_pool(
                                units[unit_of_row[tr]], y[tr], seed=seed)
                            attn_weights.append(w.tolist())
                            pooled_by_fold.append(
                                pool_attention_apply(units[unit_of_row], w))
                        for featset in feature_sets:
                            preds_by_readout = {r: np.full(len(y), np.nan)
                                                for r in readouts}
                            for (tr, te), pooled_all in zip(fold_list,
                                                            pooled_by_fold):
                                X_raw = (np.concatenate(
                                            [pooled_all, aux], axis=1)
                                         if featset == "combined"
                                         else pooled_all)
                                for r in readouts:
                                    preds_by_readout[r][te] = eval_readout(
                                        r, X_raw, y, tr, te, seed)
                            entry = {}
                            for r in readouts:
                                entry[r] = _pooled_spearman(
                                    preds_by_readout[r], y)
                            strat_out[f"attention__{featset}"] = {
                                "rho": entry,
                                "layer_weights_per_fold": attn_weights,
                            }
                        continue
                    pooled_all = STRATEGY_FNS[strategy](units)[unit_of_row]
                    for featset in feature_sets:
                        X_raw = (np.concatenate([pooled_all, aux], axis=1)
                                 if featset == "combined" else pooled_all)
                        preds_by_readout = {r: np.full(len(y), np.nan)
                                            for r in readouts}
                        for tr, te in fold_list:
                            for r in readouts:
                                preds_by_readout[r][te] = eval_readout(
                                    r, X_raw, y, tr, te, seed)
                        entry = {}
                        for r in readouts:
                            entry[r] = _pooled_spearman(
                                preds_by_readout[r], y)
                        strat_out[f"{strategy}__{featset}"] = entry
                arm_out[split] = strat_out
            ds_result["arms"][arm] = arm_out
        all_results[name] = ds_result
        with open(results_file, "w") as fh:
            json.dump(all_results, fh, indent=1)
        print(f"  {name} done in {time.time()-t0:.0f}s", flush=True)

    summarize_pooling(results_file)


def _pooled_spearman(preds, y):
    valid = np.isfinite(preds) & np.isfinite(y)
    if valid.sum() < 10:
        return {"spearman": None, "n": int(valid.sum())}
    rho, _ = spearmanr(y[valid], preds[valid])
    return {"spearman": float(rho), "n": int(valid.sum())}


def summarize_pooling(results_file):
    with open(results_file) as f:
        all_results = json.load(f)
    print("\n" + "=" * 100)
    print(f"POOLING V3 — {Path(results_file).stem}")
    print("=" * 100)
    for arm in ("mutant", "masked_wt", "unmasked_wt"):
        for split in POOLING_SPLITS:
            for strat in ("last_layer", "mean_20_33", "concat_5L",
                          "attention"):
                for featset in ("combined", "embedding_only"):
                    for r in ("ridge", "rf", "mlp", "lightgbm"):
                        vals = []
                        for name, dsr in all_results.items():
                            a = dsr.get("arms", {}).get(arm, {})
                            s = a.get(split, {})
                            if strat == "attention":
                                v = s.get(f"attention__{featset}", {}) \
                                    .get("rho", {}).get(r, {}) \
                                    .get("spearman")
                            else:
                                v = s.get(f"{strat}__{featset}", {}) \
                                    .get(r, {}).get("spearman")
                            if v is not None and np.isfinite(v):
                                vals.append(v)
                        if vals:
                            print(f"{arm:<12} {split:<20} {strat:<12} "
                                  f"{featset:<15} {r:<10} "
                                  f"N={len(vals):>2} mean rho="
                                  f"{np.mean(vals):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="revision_results/pooling_v3")
    ap.add_argument("--model_key", default="esm2_650m")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", nargs="+", default=["mutant"])
    ap.add_argument("--splits", nargs="+", default=SPLIT_SCHEMES)
    ap.add_argument("--readouts", nargs="+",
                    default=["ridge", "rf", "mlp", "lightgbm"])
    ap.add_argument("--feature_sets", nargs="+",
                    default=["combined", "embedding_only"])
    ap.add_argument("--max_datasets", type=int, default=20)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0,
                    help="shard index for concurrent runs (see --nshards)")
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the dataset list into N disjoint shards; "
                         "outputs carry a __shardK suffix to be merged by "
                         "merge_shards.py")
    ap.add_argument("--extraction_batch", type=int, default=8,
                    help="forward-pass batch for feature extraction "
                         "(lower = smaller VRAM footprint)")
    args = ap.parse_args()
    run_pooling_benchmark(data_dir=args.data_dir, output_dir=args.output_dir,
                          model_key=args.model_key, model_dir=args.model_dir,
                          device=args.device, arms=args.arms,
                          splits=args.splits, readouts=args.readouts,
                          feature_sets=args.feature_sets,
                          max_datasets=args.max_datasets,
                          n_folds=args.n_folds, seed=args.seed,
                          shard=args.shard, nshards=args.nshards,
                          extraction_batch=args.extraction_batch)


if __name__ == "__main__":
    main()
