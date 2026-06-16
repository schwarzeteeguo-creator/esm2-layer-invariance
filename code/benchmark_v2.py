"""Enhanced pooling benchmark V2 — MLP + LightGBM readout.

Adds to benchmark.py:
  - MLP (2-layer, ReLU) readout
  - LightGBM readout
  - All four readout models run on same extracted features
  - Drop-in replacement for benchmark.py with same CLI

Usage:
  python -m esm_embedding.benchmark_v2 \
      --data_dir ... --model_dir ... --model_key saprot_650m \
      --readout_models ridge rf mlp lightgbm \
      --output_dir ...
"""

# CRITICAL: Force single-threaded BLAS/OpenMP to prevent thread explosion
# Must be set BEFORE any numpy/scipy/sklearn/lightgbm imports
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys, json, time, argparse, warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

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
BLOSUM62 = {aa: {AA_LIST[j]: _BLOSUM62_MATRIX[aa][j] for j in range(20)} for aa in AA_LIST}


def read_lmdb(lmdb_path):
    import lmdb
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    data = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            k = key.decode() if isinstance(key, bytes) else key
            try: data[k] = json.loads(value)
            except: data[k] = value.decode() if isinstance(value, bytes) else value
    env.close()
    return data


def parse_mutations(mut_info):
    muts = []
    for token in mut_info.split(":"):
        if token and token[0] in "ACDEFGHIKLMNPQRSTVWY":
            try:
                muts.append((int(token[1:-1]), token[0], token[-1]))
            except (ValueError, IndexError):
                continue
    return muts


def build_mutation_features(wt_aa, mt_aa):
    wt = np.zeros(20); mt = np.zeros(20)
    if wt_aa in AA_TO_IDX: wt[AA_TO_IDX[wt_aa]] = 1.0
    if mt_aa in AA_TO_IDX: mt[AA_TO_IDX[mt_aa]] = 1.0
    blosum = float(BLOSUM62.get(wt_aa, {}).get(mt_aa, 0))
    grantham = abs(AA_TO_IDX.get(wt_aa, 0) - AA_TO_IDX.get(mt_aa, 0)) / 19.0
    return np.concatenate([wt, mt, [blosum], [grantham]])


def extract_dataset_features(lmdb_path, model_key="saprot_650m", model_dir=None, device="cuda"):
    """Extract multi-scale features (shared across all readout models)."""
    import torch
    data = read_lmdb(lmdb_path)
    name = Path(lmdb_path).name
    wild_type = data.get("wild_type", "")
    if not wild_type: return None
    length = int(data.get("length", 0))
    if length == 0: return None

    mutations = []
    for i in range(length):
        entry = data.get(str(i), data.get(i, {}))
        if isinstance(entry, str): entry = json.loads(entry) if entry else {}
        mut_info = entry.get("mut_info", "")
        fitness = entry.get("fitness", None)
        parsed = parse_mutations(mut_info)
        if parsed and fitness is not None:
            for pos, from_aa, to_aa in parsed:
                mutations.append((pos, from_aa, to_aa, float(fitness)))

    if len(mutations) < 10: return None

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from esm_embedding.extract import extract_multi_scale
    from esm_embedding.model import load_model, _MODEL_PATHS

    if model_dir:
        _MODEL_PATHS[model_key] = Path(model_dir)

    try:
        multi_results = extract_multi_scale(sequence=wild_type, model_key=model_key, device=device)
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return None

    features_by_method = defaultdict(list)
    valid_mutations = []

    for pos, from_aa, to_aa, fitness in mutations:
        if pos < 1 or pos > len(wild_type): continue
        embs = {}
        valid = True
        for method, result in multi_results.items():
            if method == "_raw_layers": continue
            emb = result.residue_embeddings[pos - 1].numpy()
            if np.isnan(emb).any() or np.isinf(emb).any():
                valid = False; break
            embs[method] = emb
        if not valid: continue
        aux = build_mutation_features(from_aa, to_aa)
        for method, emb in embs.items():
            feat = np.concatenate([emb, aux])
            features_by_method[method].append(feat)
        valid_mutations.append((pos, from_aa, to_aa, fitness))

    if len(valid_mutations) < 10: return None

    features = {method: np.stack(feat_list, axis=0) for method, feat_list in features_by_method.items()}
    return {"dataset_name": name, "num_mutations": len(valid_mutations),
            "features": features, "fitness": np.array([m[3] for m in valid_mutations])}


def evaluate_with_readouts(features, fitness, readout_models, n_folds=5, seed=42):
    """Evaluate embedding features with multiple readout models.

    Args:
        readout_models: list of str, subset of {"ridge", "rf", "mlp", "lightgbm"}
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = fitness.copy()
    valid = ~(np.isnan(y) | np.isinf(y) | np.isnan(X).any(axis=1))
    X, y = X[valid], y[valid]
    if len(y) < 10:
        return {m: np.nan for m in readout_models}

    n_splits = min(3 if len(y) > 3000 else n_folds, len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    preds = {m: np.zeros(len(y)) for m in readout_models}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        if "ridge" in readout_models:
            m = Ridge(alpha=1.0, random_state=seed)
            m.fit(X_train, y_train)
            preds["ridge"][test_idx] = m.predict(X_test)

        if "rf" in readout_models:
            m = RandomForestRegressor(n_estimators=50, max_depth=10,
                                      min_samples_leaf=10, random_state=seed, n_jobs=1)
            m.fit(X_train, y_train)
            preds["rf"][test_idx] = m.predict(X_test)

        if "mlp" in readout_models:
            m = MLPRegressor(hidden_layer_sizes=(256,), activation='relu',
                             alpha=0.001, batch_size=64, max_iter=100,
                             early_stopping=True, random_state=seed)
            m.fit(X_train, y_train)
            preds["mlp"][test_idx] = m.predict(X_test)

        if "lightgbm" in readout_models:
            try:
                from lightgbm import LGBMRegressor
                m = LGBMRegressor(n_estimators=100, max_depth=7, learning_rate=0.05, n_jobs=4,
                                  min_child_samples=20, random_state=seed, verbose=-1)
                m.fit(X_train, y_train)
                preds["lightgbm"][test_idx] = m.predict(X_test)
            except ImportError:
                preds["lightgbm"][test_idx] = np.nan

    results = {}
    for model_name in readout_models:
        try:
            rho, _ = spearmanr(y, preds[model_name])
            results[f"{model_name}_rho"] = float(rho) if not np.isnan(rho) else None
        except:
            results[f"{model_name}_rho"] = None
    results["n"] = len(y)
    return results


def run_benchmark_v2(data_dir, model_dir=None, model_key="saprot_650m",
                     device="cuda", output_dir="benchmark_results_v2",
                     max_datasets=None, readout_models=None):
    if readout_models is None:
        readout_models = ["ridge", "rf", "mlp", "lightgbm"]

    data_dir = Path(data_dir)
    lmdb_files = sorted(data_dir.glob("*"))
    print(f"Found {len(lmdb_files)} LMDB files")
    print(f"Model: {model_key}, Readout: {readout_models}")
    if max_datasets: lmdb_files = lmdb_files[:max_datasets]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "multiscale_results_v2.json"
    all_results = {}
    if results_file.exists():
        with open(results_file) as f: all_results = json.load(f)
        print(f"Resuming: {len(all_results)} already processed")

    t0 = time.time()
    for lmdb_file in tqdm(lmdb_files, desc="Datasets"):
        name = lmdb_file.name
        # Only skip if ALL requested readout models already have valid results
        if name in all_results:
            existing = all_results[name].get("methods", {})
            if existing:
                sample = next(iter(existing.values()))
                missing = [rm for rm in readout_models
                          if f"{rm}_rho" not in sample or sample[f"{rm}_rho"] is None]
                if not missing:
                    continue  # all readouts done

        result = extract_dataset_features(str(lmdb_file), model_key=model_key,
                                          model_dir=model_dir, device=device)
        if result is None: continue

        fitness = result["fitness"]
        method_results = {}
        for method, features in result["features"].items():
            eval_r = evaluate_with_readouts(features, fitness, readout_models)
            method_results[method] = eval_r

        # Merge new results into existing entry (don't overwrite previous readouts!)
        if name in all_results:
            old_methods = all_results[name].get("methods", {})
            for method, new_evals in method_results.items():
                if method in old_methods:
                    old_methods[method].update(new_evals)
                else:
                    old_methods[method] = new_evals
            all_results[name]["methods"] = old_methods
        else:
            all_results[name] = {"n_mutations": result["num_mutations"], "methods": method_results}

        if len(all_results) % 5 == 0:
            with open(results_file, "w") as f: json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(all_results)} datasets in {elapsed:.0f}s")

    with open(results_file, "w") as f: json.dump(all_results, f, indent=2)

    # Aggregate
    methods = ["last_layer", "mean_20_33", "concat_6_14_20_26_33"]
    print("\n" + "=" * 90)
    print(f"POOLING BENCHMARK V2 — {model_key}")
    print("=" * 90)
    header = f"{'Method':<30}"
    for rm in readout_models: header += f" {rm:>12}"
    print(header)
    print("-" * (30 + 12 * len(readout_models)))

    for method in methods:
        row = f"{method:<30}"
        for rm in readout_models:
            vals = []
            for name, r in all_results.items():
                v = r.get("methods", {}).get(method, {}).get(f"{rm}_rho")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            if vals:
                row += f" {np.mean(vals):>11.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Save summary
    summary = {}
    for method in methods:
        summary[method] = {}
        for rm in readout_models:
            vals = []
            for name, r in all_results.items():
                v = r.get("methods", {}).get(method, {}).get(f"{rm}_rho")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            if vals:
                summary[method][rm] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    with open(out_dir / "multiscale_summary_v2.json", "w") as f: json.dump(summary, f, indent=2)

    return {"per_dataset": all_results, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--model_key", default="saprot_650m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="benchmark_results_v2")
    parser.add_argument("--max_datasets", type=int, default=None)
    parser.add_argument("--readout_models", nargs="+",
                        default=["ridge", "rf", "mlp", "lightgbm"],
                        choices=["ridge", "rf", "mlp", "lightgbm"])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_benchmark_v2(**vars(args))

if __name__ == "__main__":
    main()
