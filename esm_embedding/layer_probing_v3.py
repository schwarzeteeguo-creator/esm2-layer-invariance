"""Per-layer probing V3 — COMPREHENSIVE.

New in V3:
  - Multi-model: SaProt_650M_AF2 AND standard ESM-2 650M
  - Ablation modes: combined / embedding_only / auxiliary_only
  - Split modes: random (default) / position_level
  - Outputs structured results for downstream statistics

Usage:
  # SaProt probing + all ablations:
  python -m esm_embedding.layer_probing_v3 --model_key saprot_650m --data_dir ... --model_dir ...

  # Standard ESM-2 probing:
  python -m esm_embedding.layer_probing_v3 --model_key esm2_650m --data_dir ... --model_dir ...
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
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


def read_lmdb(path):
    import lmdb
    env = lmdb.open(path, readonly=True, lock=False)
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
                muts.append((token[0], int(token[1:-1]), token[-1]))
            except (ValueError, IndexError):
                continue
    return muts


def build_mutation_features(from_aa, to_aa):
    wt = np.zeros(20); mt = np.zeros(20)
    if from_aa in AA_TO_IDX: wt[AA_TO_IDX[from_aa]] = 1.0
    if to_aa in AA_TO_IDX: mt[AA_TO_IDX[to_aa]] = 1.0
    blosum = float(BLOSUM62.get(from_aa, {}).get(to_aa, 0))
    grantham = abs(AA_TO_IDX.get(from_aa, 0) - AA_TO_IDX.get(to_aa, 0)) / 19.0
    return np.concatenate([wt, mt, [blosum], [grantham]])


def extract_masked_features(lmdb_path, model, tokenizer, device, is_saprot):
    """Extract per-layer hidden states at MASKED positions.

    For SaProt: uses "AA#" tokens with structure mask "M#"
    For ESM-2: uses space-separated AA tokens with "<mask>" token
    """
    import torch
    data = read_lmdb(lmdb_path)
    name = Path(lmdb_path).name
    wild_type = data.get("wild_type", "")
    if not wild_type: return None
    length = int(data.get("length", 0))
    if length == 0: return None

    muts_by_pos: Dict[int, List[Tuple[str, str, float]]] = {}
    for i in range(length):
        entry = data.get(str(i), data.get(i, {}))
        if isinstance(entry, str):
            entry = json.loads(entry) if entry else {}
        mut_info = entry.get("mut_info", "")
        fitness = entry.get("fitness", None)
        parsed = parse_mutations(mut_info)
        if parsed and fitness is not None:
            for from_aa, pos, to_aa in parsed:
                if 1 <= pos <= len(wild_type):
                    muts_by_pos.setdefault(pos, []).append((from_aa, to_aa, float(fitness)))

    if len(muts_by_pos) < 5: return None

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    pos_features = {}

    for pos in tqdm(sorted(muts_by_pos.keys()), desc=f"  {name}", leave=False):
        if is_saprot:
            sa_tokens = [f"{aa}#" for aa in wild_type]
            sa_tokens[pos - 1] = tokenizer.mask_token  # "M#"
            token_str = " ".join(sa_tokens)
        else:
            aa_tokens = list(wild_type)
            aa_tokens[pos - 1] = tokenizer.mask_token  # "<mask>"
            token_str = " ".join(aa_tokens)

        inputs = tokenizer(token_str, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        all_hidden = list(outputs.hidden_states)
        transformer_hidden = all_hidden[1:]  # skip embedding layer
        token_pos = pos  # <cls> at position 0

        layer_states = np.zeros((num_layers, hidden_dim), dtype=np.float32)
        for l in range(num_layers):
            layer_states[l] = transformer_hidden[l][0, token_pos, :].cpu().numpy()

        pos_features[pos] = layer_states

    all_mutations = []
    for pos, muts in muts_by_pos.items():
        for from_aa, to_aa, fitness in muts:
            all_mutations.append((pos, from_aa, to_aa, fitness))

    N = len(all_mutations)
    MAX_TOTAL = 10000
    if N > MAX_TOTAL:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, MAX_TOTAL, replace=False)
        all_mutations = [all_mutations[i] for i in idx]
        N = MAX_TOTAL

    layer_features = np.zeros((N, num_layers, hidden_dim), dtype=np.float32)
    aux_features = np.zeros((N, 42), dtype=np.float32)
    fitness_arr = np.zeros(N, dtype=np.float32)
    positions = np.zeros(N, dtype=np.int32)

    for idx, (pos, from_aa, to_aa, fit) in enumerate(all_mutations):
        layer_features[idx] = pos_features[pos]
        aux_features[idx] = build_mutation_features(from_aa, to_aa)
        fitness_arr[idx] = fit
        positions[idx] = pos

    return {
        "dataset_name": name, "num_mutations": N,
        "num_layers": num_layers, "hidden_dim": hidden_dim,
        "layer_features": layer_features, "aux_features": aux_features,
        "fitness": fitness_arr, "positions": positions,
    }


def evaluate_layer(layer_emb, aux, fitness, positions=None,
                   n_folds=3, seed=42, max_samples=5000,
                   split_mode="random", ablation="combined"):
    """Evaluate one layer with configurable split and feature ablation.

    Args:
        split_mode: "random" or "position_level"
        ablation: "combined", "embedding_only", or "auxiliary_only"
    """
    # Build feature matrix based on ablation mode
    if ablation == "combined":
        X_raw = np.concatenate([layer_emb, aux], axis=1)
    elif ablation == "embedding_only":
        X_raw = layer_emb
    elif ablation == "auxiliary_only":
        X_raw = aux
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    y = fitness.copy()
    valid = ~(np.isnan(y) | np.isinf(y) | np.isnan(X_raw).any(axis=1))
    X, y = X_raw[valid], y[valid]

    if positions is not None:
        positions = positions[valid]

    if len(y) < 10:
        return {"spearman": np.nan, "n": len(y)}

    if len(y) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y), max_samples, replace=False)
        X, y = X[idx], y[idx]
        if positions is not None:
            positions = positions[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_splits = min(n_folds, len(y), 5)
    preds = np.zeros(len(y))

    if split_mode == "position_level" and positions is not None:
        kf = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in kf.split(X, groups=positions):
            ridge = Ridge(alpha=1.0, random_state=seed)
            ridge.fit(X[train_idx], y[train_idx])
            preds[test_idx] = ridge.predict(X[test_idx])
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            ridge = Ridge(alpha=1.0, random_state=seed)
            ridge.fit(X[train_idx], y[train_idx])
            preds[test_idx] = ridge.predict(X[test_idx])

    rho, _ = spearmanr(y, preds)
    return {"spearman": float(rho), "n": len(y)}


def run_layer_probing_v3(
    data_dir, model_dir, model_key="saprot_650m",
    device="cuda", output_dir="layer_probing_results_v3",
    max_datasets=None,
    split_mode="random",       # "random" or "position_level"
    ablations=None,             # list of ablation modes, e.g. ["combined", "embedding_only", "auxiliary_only"]
):
    """Run per-layer probing with multiple ablations and optional position-level split."""
    import torch
    from esm_embedding.model import load_model, _MODEL_PATHS

    if ablations is None:
        ablations = ["combined"]

    is_saprot = model_key.startswith("saprot")
    _MODEL_PATHS[model_key] = Path(model_dir) if not str(model_dir).startswith("facebook/") else model_dir

    data_dir = Path(data_dir)
    lmdb_files = sorted(data_dir.glob("*"))
    print(f"Found {len(lmdb_files)} LMDB files")
    if max_datasets:
        lmdb_files = lmdb_files[:max_datasets]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_key}...")
    info = load_model(model_key, device=device)
    model = info["model"]
    tokenizer = info["tokenizer"]
    dev = info["device"]
    num_layers = info["num_layers"]
    print(f"  {num_layers} layers, {info['hidden_dim']}-dim, is_saprot={is_saprot}")

    # Per-ablation results files
    results_files = {}
    all_results = {}
    for ablation in ablations:
        suffix = f"_{split_mode}_{ablation}"
        results_file = out_dir / f"layer_probing_{model_key}{suffix}.json"
        results_files[ablation] = results_file
        if results_file.exists():
            with open(results_file) as f:
                all_results[ablation] = json.load(f)
            print(f"Resuming {ablation}: {len(all_results[ablation])} already processed")
        else:
            all_results[ablation] = {}

    t0 = time.time()
    for lmdb_file in tqdm(lmdb_files, desc="Datasets"):
        name = lmdb_file.name

        # Check if ALL ablations are done for this dataset
        all_done = all(name in all_results.get(ablation, {}) for ablation in ablations)
        if all_done:
            continue

        data = extract_masked_features(str(lmdb_file), model, tokenizer, dev, is_saprot)
        if data is None:
            continue

        layer_feat = data["layer_features"]
        aux = data["aux_features"]
        fitness = data["fitness"]
        positions = data["positions"]

        for ablation in ablations:
            if name in all_results[ablation]:
                continue

            layer_scores = {}
            for l in range(num_layers):
                result = evaluate_layer(
                    layer_feat[:, l, :], aux, fitness,
                    positions=positions,
                    split_mode=split_mode,
                    ablation=ablation,
                )
                layer_scores[str(l)] = result

            all_results[ablation][name] = {
                "n_mutations": data["num_mutations"],
                "layer_scores": layer_scores,
            }

        del data
        torch.cuda.empty_cache()

        if len(all_results[ablations[0]]) % 3 == 0:
            for ablation in ablations:
                with open(results_files[ablation], "w") as f:
                    json.dump(all_results[ablation], f, indent=2)

    elapsed = time.time() - t0
    print(f"\nProcessed up to {len(all_results[ablations[0]])} datasets in {elapsed:.0f}s")

    # Save final results
    for ablation in ablations:
        with open(results_files[ablation], "w") as f:
            json.dump(all_results[ablation], f, indent=2)

    # Aggregate per ablation
    for ablation in ablations:
        res = all_results[ablation]
        layer_rhos = {l: [] for l in range(num_layers)}
        for name, r in res.items():
            for l_str, scores in r["layer_scores"].items():
                l = int(l_str)
                if not np.isnan(scores.get("spearman", np.nan)):
                    layer_rhos[l].append(scores["spearman"])

        print(f"\n{'='*65}")
        print(f"RESULTS: {model_key} | split={split_mode} | ablation={ablation}")
        print(f"{'='*65}")
        print(f"{'Layer':>6}  {'Mean Rho':>10}  {'Std':>8}  {'N':>6}")
        best_l, best_r = num_layers - 1, -999
        for l in range(num_layers):
            if not layer_rhos[l]: continue
            mean_r = np.mean(layer_rhos[l])
            std_r = np.std(layer_rhos[l])
            n = len(layer_rhos[l])
            marker = ""
            if l == num_layers - 1: marker = "  <-- final"
            if mean_r > best_r: best_r = mean_r; best_l = l
            if l == best_l: marker += "  <-- BEST"
            print(f"  {l:3d}   {mean_r:>10.4f}  {std_r:>8.4f}  {n:>6}{marker}")

        if layer_rhos[num_layers - 1]:
            final_r = np.mean(layer_rhos[num_layers - 1])
            layer_means = [np.mean(layer_rhos[l]) for l in range(num_layers) if layer_rhos[l]]
            inv_score = np.std(layer_means)
            print("-" * 65)
            print(f"Best layer: {best_l} (rho={best_r:.4f})")
            print(f"Final layer: {num_layers-1} (rho={final_r:.4f})")
            print(f"Delta: {best_r - final_r:+.4f}")
            print(f"Invariance score: {inv_score:.6f}")

        summary = {
            "model_key": model_key, "split_mode": split_mode, "ablation": ablation,
            "best_layer": best_l, "best_layer_rho": float(best_r),
            "final_layer_rho": float(final_r) if layer_rhos[num_layers - 1] else None,
            "invariance_score": float(inv_score) if layer_rhos[num_layers - 1] else None,
        }
        summary_file = out_dir / f"summary_{model_key}_{split_mode}_{ablation}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    return {"per_dataset": all_results, "summaries": "see output dir"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_key", default="saprot_650m",
                        choices=["saprot_650m", "esm2_650m"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="layer_probing_results_v3")
    parser.add_argument("--max_datasets", type=int, default=None)
    parser.add_argument("--split_mode", default="random",
                        choices=["random", "position_level"])
    parser.add_argument("--ablations", nargs="+",
                        default=["combined"],
                        choices=["combined", "embedding_only", "auxiliary_only"])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_layer_probing_v3(**vars(args))

if __name__ == "__main__":
    main()
