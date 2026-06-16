"""Per-layer probing: which ESM-2 layer encodes the most mutation-effect information?

For each ProteinGym dataset:
    1. Run SaProt ONCE, save all 33 layers' hidden states
    2. For each mutation, extract per-layer embedding (1280-dim) at mutation position
    3. For each layer independently, train Ridge regression → predict DMS fitness
    4. Report layer-wise Spearman rho

Key question: Is the final layer optimal for mutation effect prediction?
Kumar et al. (2025) found layers 20-33 outperform the final layer by 32%
for kinase functional prediction. We test this on ProteinGym DMS data.

Usage:
    python -m esm_embedding.layer_probing \
        --data_dir /root/autodl-tmp/SaProt/LMDB/ProteinGym/substitutions \
        --model_dir /root/autodl-tmp/SaProt/weights/PLMs/SaProt_650M_AF2_hf \
        --output_dir /root/autodl-tmp/layer_probing_results \
        --max_datasets 5
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# BLOSUM62 matrix
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
    import lmdb
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    data = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            k = key.decode() if isinstance(key, bytes) else key
            try:
                data[k] = json.loads(value)
            except:
                data[k] = value.decode() if isinstance(value, bytes) else value
    env.close()
    return data


def parse_mutations(mut_info: str):
    mutations = []
    for token in mut_info.split(":"):
        if not token or token[0] not in "ACDEFGHIKLMNPQRSTVWY":
            continue
        mutations.append((token[0], int(token[1:-1]), token[-1]))
    return mutations


def build_mutation_features(from_aa, to_aa):
    wt = np.zeros(20); mt = np.zeros(20)
    if from_aa in AA_TO_IDX: wt[AA_TO_IDX[from_aa]] = 1.0
    if to_aa in AA_TO_IDX: mt[AA_TO_IDX[to_aa]] = 1.0
    blosum = float(BLOSUM62.get(from_aa, {}).get(to_aa, 0))
    grantham = abs(AA_TO_IDX.get(from_aa, 0) - AA_TO_IDX.get(to_aa, 0)) / 19.0
    return np.concatenate([wt, mt, [blosum], [grantham]])


# ── Per-layer feature extraction ──────────────────────────────────
def extract_layer_features(
    lmdb_path: str,
    model_dir: str,
    device: str = "cuda",
) -> Optional[dict]:
    """Extract per-layer hidden states at each mutation position.

    Runs model ONCE, saves all 33 layers' hidden states at mutation positions.

    Returns:
        dict with:
            dataset_name, num_mutations, num_layers (33), hidden_dim (1280),
            layer_features: np.ndarray (N_mutations, 33, 1280),
            aux_features: np.ndarray (N_mutations, 42),
            fitness: np.ndarray (N_mutations,)
    """
    import torch
    import lmdb
    from esm_embedding.model import load_model, tokenize_sequence, _MODEL_PATHS

    _MODEL_PATHS["saprot_650m"] = Path(model_dir)

    data = read_lmdb(lmdb_path)
    dataset_name = Path(lmdb_path).name
    wild_type = data.get("wild_type", "")
    if not wild_type:
        return None

    length = int(data.get("length", 0))
    if length == 0:
        return None

    # Collect all mutations
    mutations = []
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
                    mutations.append((pos, from_aa, to_aa, float(fitness)))

    if len(mutations) < 10:
        return None

    # Load model and do ONE forward pass
    model_info = load_model("saprot_650m", device=device)
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    dev = model_info["device"]
    num_layers = model_info["num_layers"]
    hidden_dim = model_info["hidden_dim"]

    inputs = tokenize_sequence(tokenizer, wild_type)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract per-layer per-position hidden states
    # hidden_states = (embedding, layer_0, ..., layer_32) — skip embedding
    all_hidden = list(outputs.hidden_states)
    transformer_hidden = all_hidden[1:]

    # For each layer: (1, seq_len+2, D) → squeeze batch → strip <cls>/<eos> → (L, D)
    layer_states = []
    for h in transformer_hidden:
        seq_states = h.squeeze(0)[1:-1].cpu().numpy()  # (seq_len, D)
        layer_states.append(seq_states)
    layer_states = np.stack(layer_states, axis=0)  # (33, seq_len, D)

    # Build feature matrices
    N = len(mutations)
    layer_features = np.zeros((N, num_layers, hidden_dim), dtype=np.float32)
    aux_features = np.zeros((N, 42), dtype=np.float32)
    fitness = np.zeros(N, dtype=np.float32)

    for idx, (pos, from_aa, to_aa, fit) in enumerate(mutations):
        layer_features[idx] = layer_states[:, pos - 1, :]  # (33, D)
        aux_features[idx] = build_mutation_features(from_aa, to_aa)
        fitness[idx] = fit

    return {
        "dataset_name": dataset_name,
        "num_mutations": N,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "layer_features": layer_features,
        "aux_features": aux_features,
        "fitness": fitness,
    }


# ── Per-layer evaluation ──────────────────────────────────────────
def evaluate_layer(
    layer_emb: np.ndarray,   # (N, D)
    aux: np.ndarray,          # (N, 42)
    fitness: np.ndarray,      # (N,)
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Evaluate one layer's embedding for DMS prediction.

    Feature = [layer_embedding | aux_features] (D + 42 dims)
    """
    X_raw = np.concatenate([layer_emb, aux], axis=1)
    y = fitness.copy()

    # Filter invalid
    valid = ~(np.isnan(y) | np.isinf(y) | np.isnan(X_raw).any(axis=1))
    X, y = X_raw[valid], y[valid]
    if len(y) < 10:
        return {"spearman": np.nan, "n": len(y)}

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_splits = min(n_folds, len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X):
        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(X[train_idx], y[train_idx])
        preds[test_idx] = ridge.predict(X[test_idx])

    rho, _ = spearmanr(y, preds)
    return {"spearman": float(rho), "n": len(y)}


# ── Main ──────────────────────────────────────────────────────────
def run_layer_probing(
    data_dir: str,
    model_dir: str,
    device: str = "cuda",
    output_dir: str = "layer_probing_results",
    max_datasets: int = None,
) -> dict:
    """Full per-layer probing benchmark."""
    data_dir = Path(data_dir)
    lmdb_files = sorted(data_dir.glob("*"))
    print(f"Found {len(lmdb_files)} LMDB files")
    if max_datasets:
        lmdb_files = lmdb_files[:max_datasets]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume incremental
    results_file = out_dir / "layer_probing_results.json"
    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} already processed")

    t0 = time.time()
    for lmdb_file in tqdm(lmdb_files, desc="Datasets"):
        name = lmdb_file.name
        if name in all_results:
            continue

        dataset_data = extract_layer_features(
            str(lmdb_file), model_dir=model_dir, device=device
        )
        if dataset_data is None:
            continue

        layer_feat = dataset_data["layer_features"]  # (N, 33, D)
        aux = dataset_data["aux_features"]
        fitness = dataset_data["fitness"]
        num_layers = dataset_data["num_layers"]

        layer_scores = {}
        for l in range(num_layers):
            result = evaluate_layer(layer_feat[:, l, :], aux, fitness)
            layer_scores[str(l)] = result

        all_results[name] = {
            "n_mutations": dataset_data["num_mutations"],
            "layer_scores": layer_scores,
        }

        if len(all_results) % 5 == 0:
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(all_results)} datasets in {elapsed:.0f}s")

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Aggregate: per-layer mean ρ across datasets ────────────────
    num_layers = 33
    layer_rhos = {l: [] for l in range(num_layers)}

    for name, r in all_results.items():
        for l_str, scores in r["layer_scores"].items():
            l = int(l_str)
            if not np.isnan(scores.get("spearman", np.nan)):
                layer_rhos[l].append(scores["spearman"])

    print("\n" + "=" * 65)
    print("PER-LAYER PROBING RESULTS (Mean Spearman rho)")
    print("=" * 65)
    print(f"{'Layer':>6}  {'Mean Rho':>10}  {'Std':>8}  {'#Datasets':>10}  Note")
    print("-" * 65)

    best_layer = max(layer_rhos, key=lambda l: np.mean(layer_rhos[l])
                     if layer_rhos[l] else -999)
    last_layer_rho = np.mean(layer_rhos[32]) if layer_rhos[32] else 0

    for l in range(num_layers):
        if not layer_rhos[l]:
            continue
        mean_r = np.mean(layer_rhos[l])
        std_r = np.std(layer_rhos[l])
        n = len(layer_rhos[l])

        note = ""
        if l == 32:
            note = "<-- last layer (standard baseline)"
        if l == best_layer:
            note += " <-- BEST"

        print(f"  {l:3d}   {mean_r:>10.4f}  {std_r:>8.4f}  {n:>10}  {note}")

    print("-" * 65)
    print(f"\nBest layer: {best_layer} (rho={np.mean(layer_rhos[best_layer]):.4f})")
    print(f"Last layer (32): rho={last_layer_rho:.4f}")
    if best_layer != 32:
        delta = np.mean(layer_rhos[best_layer]) - last_layer_rho
        print(f"Delta (best - last): {delta:+.4f}")
        print(f"Relative gain: {delta / abs(last_layer_rho) * 100:+.1f}%")

    # Save aggregate summary
    summary = {
        "best_layer": best_layer,
        "best_layer_rho": float(np.mean(layer_rhos[best_layer])),
        "last_layer_rho": float(last_layer_rho),
        "per_layer": {
            str(l): {
                "mean": float(np.mean(layer_rhos[l])),
                "std": float(np.std(layer_rhos[l])),
                "n": len(layer_rhos[l]),
            }
            for l in range(num_layers) if layer_rhos[l]
        }
    }
    summary_path = out_dir / "layer_probing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return {"per_dataset": all_results, "summary": summary}


# ── CLI ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-layer probing on ProteinGym"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="layer_probing_results")
    parser.add_argument("--max_datasets", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_layer_probing(**vars(args))


if __name__ == "__main__":
    main()
