"""Per-layer probing V2 — CORRECTED.

Key fix: Instead of extracting WT hidden states (which only capture position-level
conservation), we MASK each mutation position and run a forward pass. This captures
mutation-context information — the model's "expectation" at the mutated site.

For each dataset:
    1. Collect unique mutation positions
    2. For each position, mask it with SaProt "M#" tokens, forward pass
    3. Extract ALL 33 layer hidden states at the masked position
    4. Train Ridge per layer: [hidden_state | aux_features] → DMS fitness
    5. Report per-layer Spearman rho

This is the correct design — hidden states reflect mutation-specific context.
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
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
            muts.append((token[0], int(token[1:-1]), token[-1]))
    return muts


def build_mutation_features(from_aa, to_aa):
    wt = np.zeros(20); mt = np.zeros(20)
    if from_aa in AA_TO_IDX: wt[AA_TO_IDX[from_aa]] = 1.0
    if to_aa in AA_TO_IDX: mt[AA_TO_IDX[to_aa]] = 1.0
    blosum = float(BLOSUM62.get(from_aa, {}).get(to_aa, 0))
    grantham = abs(AA_TO_IDX.get(from_aa, 0) - AA_TO_IDX.get(to_aa, 0)) / 19.0
    return np.concatenate([wt, mt, [blosum], [grantham]])


def extract_masked_features(lmdb_path, model, tokenizer, device):
    """Extract per-layer hidden states at MASKED positions.

    For each unique mutation position:
        1. Mask that position in the SA sequence
        2. Forward pass
        3. Extract all 33 layer hidden states at the masked token

    Returns features for ALL mutations (mutations at same position share hidden states).
    """
    import torch
    data = read_lmdb(lmdb_path)
    name = Path(lmdb_path).name
    wild_type = data.get("wild_type", "")
    if not wild_type: return None
    length = int(data.get("length", 0))
    if length == 0: return None

    # Collect mutations grouped by position
    muts_by_pos: Dict[int, List[Tuple[str, str, float]]] = {}
    for i in range(length):
        entry = data.get(str(i), data.get(i, {}))
        if isinstance(entry, str): entry = json.loads(entry) if entry else {}
        mut_info = entry.get("mut_info", "")
        fitness = entry.get("fitness", None)
        parsed = parse_mutations(mut_info)
        if parsed and fitness is not None:
            for from_aa, pos, to_aa in parsed:
                if 1 <= pos <= len(wild_type):
                    muts_by_pos.setdefault(pos, []).append((from_aa, to_aa, float(fitness)))

    if len(muts_by_pos) < 5: return None

    # Build SA token sequence using "M#" format
    sa_tokens = [f"{aa}#" for aa in wild_type]
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    # Storage: position → (layer_hidden_states dict)
    pos_features = {}  # pos → np.ndarray (num_layers, hidden_dim)

    for pos in tqdm(sorted(muts_by_pos.keys()), desc=f"  {name}", leave=False):
        masked = sa_tokens.copy()
        masked[pos - 1] = tokenizer.mask_token
        token_str = " ".join(masked)
        inputs = tokenizer(token_str, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        all_hidden = list(outputs.hidden_states)
        transformer_hidden = all_hidden[1:]  # skip embedding
        token_pos = pos  # <cls> at 0

        layer_states = np.zeros((num_layers, hidden_dim), dtype=np.float32)
        for l in range(num_layers):
            layer_states[l] = transformer_hidden[l][0, token_pos, :].cpu().numpy()

        pos_features[pos] = layer_states

    # Build mutation-level feature matrix
    all_mutations = []
    for pos, muts in muts_by_pos.items():
        for from_aa, to_aa, fitness in muts:
            all_mutations.append((pos, from_aa, to_aa, fitness))

    N = len(all_mutations)
    MAX_TOTAL = 10000  # prevent CPU OOM on massive datasets (e.g. HIS7_YEAST)
    if N > MAX_TOTAL:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, MAX_TOTAL, replace=False)
        all_mutations = [all_mutations[i] for i in idx]
        N = MAX_TOTAL

    layer_features = np.zeros((N, num_layers, hidden_dim), dtype=np.float32)
    aux_features = np.zeros((N, 42), dtype=np.float32)
    fitness_arr = np.zeros(N, dtype=np.float32)

    for idx, (pos, from_aa, to_aa, fit) in enumerate(all_mutations):
        layer_features[idx] = pos_features[pos]
        aux_features[idx] = build_mutation_features(from_aa, to_aa)
        fitness_arr[idx] = fit

    return {
        "dataset_name": name, "num_mutations": N,
        "num_layers": num_layers, "hidden_dim": hidden_dim,
        "layer_features": layer_features, "aux_features": aux_features,
        "fitness": fitness_arr,
    }


def evaluate_layer(layer_emb, aux, fitness, n_folds=3, seed=42, max_samples=5000):
    X_raw = np.concatenate([layer_emb, aux], axis=1)
    y = fitness.copy()
    valid = ~(np.isnan(y) | np.isinf(y) | np.isnan(X_raw).any(axis=1))
    X, y = X_raw[valid], y[valid]
    if len(y) < 10: return {"spearman": np.nan, "n": len(y)}

    # Subsample for large datasets to keep runtime manageable
    if len(y) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y), max_samples, replace=False)
        X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_splits = min(n_folds, len(y), 5)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(X[train_idx], y[train_idx])
        preds[test_idx] = ridge.predict(X[test_idx])
    rho, _ = spearmanr(y, preds)
    return {"spearman": float(rho), "n": len(y)}


def run_layer_probing(data_dir, model_dir, device="cuda", output_dir="layer_probing_results_v2", max_datasets=None):
    import torch
    from esm_embedding.model import load_model, _MODEL_PATHS

    _MODEL_PATHS["saprot_650m"] = Path(model_dir)
    data_dir = Path(data_dir)
    lmdb_files = sorted(data_dir.glob("*"))
    print(f"Found {len(lmdb_files)} LMDB files")
    if max_datasets: lmdb_files = lmdb_files[:max_datasets]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model ONCE
    print("Loading model...")
    info = load_model("saprot_650m", device=device)
    model = info["model"]
    tokenizer = info["tokenizer"]
    dev = info["device"]
    num_layers = info["num_layers"]
    print(f"  {num_layers} layers, {info['hidden_dim']}-dim")

    # Resume
    results_file = out_dir / "layer_probing_v2_results.json"
    all_results = {}
    if results_file.exists():
        with open(results_file) as f: all_results = json.load(f)
        print(f"Resuming: {len(all_results)} already processed")

    t0 = time.time()
    for lmdb_file in tqdm(lmdb_files, desc="Datasets"):
        name = lmdb_file.name
        if name in all_results: continue

        data = extract_masked_features(str(lmdb_file), model, tokenizer, dev)
        if data is None: continue

        layer_feat = data["layer_features"]
        aux = data["aux_features"]
        fitness = data["fitness"]

        layer_scores = {}
        for l in range(num_layers):
            result = evaluate_layer(layer_feat[:, l, :], aux, fitness)
            layer_scores[str(l)] = result

        all_results[name] = {"n_mutations": data["num_mutations"], "layer_scores": layer_scores}

        # Free GPU memory between datasets to avoid OOM
        del data
        torch.cuda.empty_cache()

        if len(all_results) % 3 == 0:
            with open(results_file, "w") as f: json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(all_results)} datasets in {elapsed:.0f}s")

    with open(results_file, "w") as f: json.dump(all_results, f, indent=2)

    # Aggregate
    layer_rhos = {l: [] for l in range(num_layers)}
    for name, r in all_results.items():
        for l_str, scores in r["layer_scores"].items():
            l = int(l_str)
            if not np.isnan(scores.get("spearman", np.nan)):
                layer_rhos[l].append(scores["spearman"])

    print("\n" + "=" * 65)
    print("PER-LAYER PROBING V2 (Masked Position Features)")
    print("=" * 65)
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

    final_r = np.mean(layer_rhos[num_layers - 1]) if layer_rhos[num_layers - 1] else 0
    print("-" * 65)
    print(f"\nBest layer: {best_l} (rho={best_r:.4f})")
    print(f"Final layer: {num_layers - 1} (rho={final_r:.4f})")
    print(f"Delta: {best_r - final_r:+.4f}")
    if abs(final_r) > 1e-6:
        print(f"Relative gain: {(best_r - final_r) / abs(final_r) * 100:+.1f}%")

    summary = {
        "best_layer": best_l, "best_layer_rho": float(best_r),
        "final_layer_rho": float(final_r),
        "per_layer": {str(l): {"mean": float(np.mean(layer_rhos[l])), "std": float(np.std(layer_rhos[l])), "n": len(layer_rhos[l])}
                      for l in range(num_layers) if layer_rhos[l]}
    }
    with open(out_dir / "layer_probing_v2_summary.json", "w") as f: json.dump(summary, f, indent=2)
    return {"per_dataset": all_results, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="layer_probing_results_v2")
    parser.add_argument("--max_datasets", type=int, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_layer_probing(**vars(args))

if __name__ == "__main__":
    main()
