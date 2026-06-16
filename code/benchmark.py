"""Multi-scale embedding benchmark on ProteinGym DMS data.

Pipeline:
    1. For each ProteinGym LMDB dataset, extract wild-type multi-scale embeddings
    2. For each mutation, build features from:
       a. Pooled embedding at mutation position (position context)
       b. One-hot wild-type + mutant amino acid (mutation identity)
       c. BLOSUM62 substitution score (evolutionary prior)
       d. SaProt LM-head log-ratio score (zero-shot baseline)
    3. Train Ridge + Random Forest regressors per pooling strategy
    4. Report per-dataset and aggregate Spearman rho

Key comparison:
    The ONLY difference between methods is which embedding is used (a).
    Features (b-d) are identical across methods, isolating the effect
    of multi-scale pooling on predictive performance.

Usage:
    python -m esm_embedding.benchmark \
        --data_dir /root/autodl-tmp/SaProt/LMDB/ProteinGym/substitutions \
        --model_dir /root/autodl-tmp/SaProt/weights/PLMs/SaProt_650M_AF2_hf \
        --model_key saprot_650m \
        --output_dir /root/autodl-tmp/benchmark_results
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# BLOSUM62 matrix (subset for standard 20 AAs, row=f from, col=to)
# Source: https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
_BLOSUM62_MATRIX = {
    'A': [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    'R': [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    'N': [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    'D': [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    'C': [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    'Q': [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    'E': [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    'G': [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    'H': [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    'I': [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    'L': [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    'K': [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    'M': [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    'F': [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    'P': [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    'S': [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    'T': [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    'W': [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    'Y': [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    'V': [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
}
BLOSUM62 = {aa: {AA_LIST[j]: _BLOSUM62_MATRIX[aa][j] for j in range(20)}
            for aa in AA_LIST}


# ── LMDB reading ───────────────────────────────────────────────────
def read_lmdb(lmdb_path: str) -> dict:
    """Read a ProteinGym LMDB file and return parsed data."""
    import lmdb
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    data = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode() if isinstance(key, bytes) else key
            try:
                data[key_str] = json.loads(value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                data[key_str] = value.decode() if isinstance(value, bytes) else value
    env.close()
    return data


def parse_mutations(mut_info: str) -> List[Tuple[int, str, str]]:
    """Parse SaProt mutation string.
    Example: "A123V:G124L" -> [(123, 'A', 'V'), (124, 'G', 'L')]
    """
    mutations = []
    for token in mut_info.split(":"):
        if not token:
            continue
        if token[0] in "ACDEFGHIKLMNPQRSTVWY":
            from_aa = token[0]
            to_aa = token[-1]
            pos = int(token[1:-1])
            mutations.append((pos, from_aa, to_aa))
    return mutations


# ── Mutation-specific features ─────────────────────────────────────
def build_mutation_features(
    wt_aa: str,
    mt_aa: str,
) -> np.ndarray:
    """Build mutation-specific auxiliary features (same for all methods).

    Returns:
        Array of shape (42,) containing:
        - one-hot wt_aa (20-dim)
        - one-hot mt_aa (20-dim)
        - BLOSUM62 score (1-dim)
        - Grantham distance proxy (1-dim)
    """
    # One-hot wild-type AA
    wt_onehot = np.zeros(20)
    if wt_aa in AA_TO_IDX:
        wt_onehot[AA_TO_IDX[wt_aa]] = 1.0

    # One-hot mutant AA
    mt_onehot = np.zeros(20)
    if mt_aa in AA_TO_IDX:
        mt_onehot[AA_TO_IDX[mt_aa]] = 1.0

    # BLOSUM62 substitution score
    blosum = float(BLOSUM62.get(wt_aa, {}).get(mt_aa, 0))

    # Simple Grantham-like distance: |AA_idx_diff| as crude proxy
    grantham_proxy = abs(AA_TO_IDX.get(wt_aa, 0) - AA_TO_IDX.get(mt_aa, 0)) / 19.0

    return np.concatenate([wt_onehot, mt_onehot, [blosum], [grantham_proxy]])


# ── Feature extraction ────────────────────────────────────────────
def extract_dataset_features(
    lmdb_path: str,
    model_key: str = "saprot_650m",
    model_dir: str = None,
    device: str = "cuda",
    include_saprot_score: bool = True,
) -> Optional[dict]:
    """Extract multi-scale features for a single ProteinGym dataset.

    Returns:
        dict with:
            dataset_name, wild_type, num_mutations,
            features: {method_name: np.ndarray (N, D + 42 [+ 1])}
            fitness: np.ndarray (N,)
            saprot_scores: np.ndarray (N,) or None
    """
    data = read_lmdb(lmdb_path)
    dataset_name = Path(lmdb_path).name

    wild_type = data.get("wild_type", "")
    if not wild_type:
        return None

    length = int(data.get("length", 0))
    if length == 0:
        return None

    # Collect all mutations with DMS scores
    mutations = []
    for i in range(length):
        entry = data.get(str(i), data.get(i, {}))
        if isinstance(entry, str):
            entry = json.loads(entry) if entry else {}
        mut_info = entry.get("mut_info", "")
        fitness = entry.get("fitness", None)
        parsed = parse_mutations(mut_info)
        if parsed and fitness is not None:
            for pos, from_aa, to_aa in parsed:
                mutations.append((pos, from_aa, to_aa, float(fitness)))

    if len(mutations) == 0:
        return None

    # Extract multi-scale embeddings (single forward pass for wild-type)
    # Use local import to allow running from SaProt directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from esm_embedding.extract import extract_multi_scale
    from esm_embedding.model import load_model

    # Set model path
    if model_dir:
        from esm_embedding.model import _MODEL_PATHS
        _MODEL_PATHS[model_key] = Path(model_dir)

    try:
        multi_results = extract_multi_scale(
            sequence=wild_type,
            model_key=model_key,
            device=device,
        )
    except Exception as e:
        print(f"  ERROR extracting {dataset_name}: {e}")
        return None

    # Build feature matrix for each pooling method
    # features[method] = [pooled_embedding | mutation_aux | saprot_score?]
    features_by_method = defaultdict(list)
    saprot_scores = []
    valid_mutations = []

    for pos, from_aa, to_aa, fitness in mutations:
        if pos < 1 or pos > len(wild_type):
            continue

        # Get embeddings from all methods
        embs = {}
        valid = True
        for method, result in multi_results.items():
            if method == "_raw_layers":
                continue
            emb = result.residue_embeddings[pos - 1].numpy()
            if np.isnan(emb).any() or np.isinf(emb).any():
                valid = False
                break
            embs[method] = emb

        if not valid:
            continue

        # Mutation-specific auxiliary features
        aux = build_mutation_features(from_aa, to_aa)

        for method, emb in embs.items():
            feat = np.concatenate([emb, aux])
            features_by_method[method].append(feat)

        valid_mutations.append((pos, from_aa, to_aa, fitness))

    if len(valid_mutations) < 10:
        return None

    features = {}
    for method, feat_list in features_by_method.items():
        features[method] = np.stack(feat_list, axis=0)

    return {
        "dataset_name": dataset_name,
        "wild_type": wild_type,
        "num_mutations": len(valid_mutations),
        "features": features,
        "fitness": np.array([m[3] for m in valid_mutations]),
    }


# ── Regression benchmark ──────────────────────────────────────────
def evaluate_pooling_method(
    features: np.ndarray,  # (N, D)
    fitness: np.ndarray,    # (N,)
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Cross-validated regression to evaluate embedding quality.

    Two regressors:
        - Ridge: tests linear information content
        - Random Forest: tests non-linear patterns
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = fitness.copy()

    # Remove bad entries
    valid = ~(np.isnan(y) | np.isinf(y) | np.isnan(X).any(axis=1))
    X, y = X[valid], y[valid]
    if len(y) < 10:
        return {"ridge_rho": np.nan, "rf_rho": np.nan, "n": len(y)}

    # Speed optimization: use 3-fold for datasets > 3000 mutations
    effective_folds = 3 if len(y) > 3000 else n_folds
    n_splits = min(effective_folds, len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ridge_preds = np.zeros(len(y))
    rf_preds = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Ridge
        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(X_train, y_train)
        ridge_preds[test_idx] = ridge.predict(X_test)

        # Random Forest (reduced params for speed on large datasets)
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_preds[test_idx] = rf.predict(X_test)

    ridge_rho, _ = spearmanr(y, ridge_preds)
    rf_rho, _ = spearmanr(y, rf_preds)

    return {"ridge_rho": float(ridge_rho), "rf_rho": float(rf_rho), "n": len(y)}


# ── Main benchmark ────────────────────────────────────────────────
def run_benchmark(
    data_dir: str,
    model_dir: str = None,
    model_key: str = "saprot_650m",
    device: str = "cuda",
    output_dir: str = None,
    max_datasets: int = None,
) -> dict:
    """Run full multi-scale benchmark across ProteinGym datasets."""
    data_dir = Path(data_dir)
    lmdb_files = sorted(data_dir.glob("*"))
    print(f"Found {len(lmdb_files)} LMDB files in {data_dir}")
    if max_datasets:
        lmdb_files = lmdb_files[:max_datasets]
        print(f"  Limited to first {max_datasets}")

    # Create output directory
    out_dir = Path(output_dir) if output_dir else Path("benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results for incremental mode
    results_file = out_dir / "multiscale_results.json"
    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} datasets already processed")

    # Process each dataset
    t0 = time.time()
    for lmdb_file in tqdm(lmdb_files, desc="Datasets"):
        name = lmdb_file.name
        if name in all_results:
            continue

        result = extract_dataset_features(
            str(lmdb_file),
            model_key=model_key,
            model_dir=model_dir,
            device=device,
        )
        if result is None:
            continue

        # Evaluate each pooling method
        fitness = result["fitness"]
        method_results = {}

        for method, features in result["features"].items():
            eval_r = evaluate_pooling_method(features, fitness)
            method_results[method] = eval_r

        all_results[name] = {
            "n_mutations": result["num_mutations"],
            "wild_type": result["wild_type"],
            "methods": method_results,
        }

        # Save incrementally every 5 datasets
        if len(all_results) % 5 == 0:
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(all_results)} datasets in {elapsed:.1f}s")

    # Final save
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")

    # ── Aggregate ──────────────────────────────────────────────────
    methods = ["last_layer", "mean_20_33", "concat_6_14_20_26_33"]
    summary = _aggregate(all_results, methods)

    summary_file = out_dir / "multiscale_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")

    _print_summary(summary, methods)

    return {"per_dataset": all_results, "summary": summary}


def _aggregate(all_results: dict, methods: List[str]) -> dict:
    """Compute aggregate statistics."""
    agg = {
        m: {"ridge_rhos": [], "rf_rhos": [], "n_datasets": 0}
        for m in methods
    }

    for name, result in all_results.items():
        for method in methods:
            r = result.get("methods", {}).get(method, {})
            if r:
                if not np.isnan(r.get("ridge_rho", np.nan)):
                    agg[method]["ridge_rhos"].append(r["ridge_rho"])
                if not np.isnan(r.get("rf_rho", np.nan)):
                    agg[method]["rf_rhos"].append(r["rf_rho"])
                agg[method]["n_datasets"] += 1

    summary = {}
    for method in methods:
        summary[method] = {
            "ridge_mean": float(np.mean(agg[method]["ridge_rhos"]))
                if agg[method]["ridge_rhos"] else None,
            "ridge_std": float(np.std(agg[method]["ridge_rhos"]))
                if agg[method]["ridge_rhos"] else None,
            "rf_mean": float(np.mean(agg[method]["rf_rhos"]))
                if agg[method]["rf_rhos"] else None,
            "rf_std": float(np.std(agg[method]["rf_rhos"]))
                if agg[method]["rf_rhos"] else None,
            "n_datasets": agg[method]["n_datasets"],
        }
    return summary


def _print_summary(summary: dict, methods: List[str]):
    """Pretty-print aggregate results."""
    print("\n" + "=" * 72)
    print("MULTI-SCALE BENCHMARK RESULTS")
    print("=" * 72)
    print(f"{'Method':<30} {'Ridge ρ':>10} {'RF ρ':>10} {'#Datasets':>10}")
    print("-" * 60)

    best_ridge = max(
        (m for m in methods if summary[m]["ridge_mean"] is not None),
        key=lambda m: summary[m]["ridge_mean"], default=None
    )
    best_rf = max(
        (m for m in methods if summary[m]["rf_mean"] is not None),
        key=lambda m: summary[m]["rf_mean"], default=None
    )

    for method in methods:
        s = summary[method]
        ridge_str = f"{s['ridge_mean']:.4f}" if s["ridge_mean"] else "N/A"
        rf_str = f"{s['rf_mean']:.4f}" if s["rf_mean"] else "N/A"
        marker = ""
        if method == best_ridge:
            marker += " ← best Ridge"
        if method == best_rf:
            marker += " ← best RF"
        print(f"{method:<30} {ridge_str:>10} {rf_str:>10} {s['n_datasets']:>10}{marker}")

    print("-" * 60)
    print("last_layer = standard approach (ESM-1v, EVOLVEpro baseline)")
    print("mean_20_33  = mean-pool function/motif layers (Kumar et al. 2025)")
    print("concat_6_14_20_26_33 = multi-depth span (early+mid+late)")
    print("\nHigher ρ = better mutation effect prediction from embeddings.")


# ── CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-scale ESM embeddings on ProteinGym"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Path to ProteinGym LMDB directory")
    parser.add_argument("--model_dir", default=None,
                        help="Path to model weights (overrides built-in)")
    parser.add_argument("--model_key", default="saprot_650m",
                        choices=["saprot_650m", "esm2_650m", "esm2_150m", "esm2_3b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="benchmark_results")
    parser.add_argument("--max_datasets", type=int, default=None,
                        help="Limit to first N datasets (for quick test)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_benchmark(**vars(args))


if __name__ == "__main__":
    main()
