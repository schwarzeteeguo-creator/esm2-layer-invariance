"""Layer cosine similarity analysis for Bioinformatics paper.

Computes pairwise cosine similarity between all 33 ESM-2 layers' hidden states,
averaged across multiple ProteinGym datasets. Produces data for the key figure
showing that layers are representationally distinct but information-equivalent.

Output:
    layer_similarity.json — 33×33 matrix + per-dataset stats
"""

import os, sys, json, argparse, warnings
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
warnings.filterwarnings("ignore")


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


def compute_layer_similarity(lmdb_path, model, tokenizer, device, n_positions=50):
    """Compute 33×33 layer cosine similarity for one dataset."""
    data = read_lmdb(lmdb_path)
    wild_type = data.get("wild_type", "")
    if not wild_type: return None
    length = len(wild_type)

    # Tokenize with M# format
    sa_tokens = [f"{aa}#" for aa in wild_type]
    token_str = " ".join(sa_tokens)
    inputs = tokenizer(token_str, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    all_hidden = list(outputs.hidden_states)
    transformer_hidden = all_hidden[1:]  # skip embedding, 33 layers
    num_layers = len(transformer_hidden)

    # Sample positions (skip <cls> at 0 and <eos> at end)
    seq_len = transformer_hidden[0].shape[1]
    valid_positions = list(range(1, min(seq_len - 1, length + 1)))
    if len(valid_positions) > n_positions:
        rng = np.random.RandomState(42)
        positions = sorted(rng.choice(valid_positions, n_positions, replace=False))
    else:
        positions = valid_positions

    # Compute similarity matrix
    sim_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            sims = []
            for p in positions:
                h_i = transformer_hidden[i][0, p, :]
                h_j = transformer_hidden[j][0, p, :]
                cos_sim = torch.nn.functional.cosine_similarity(
                    h_i.unsqueeze(0), h_j.unsqueeze(0)
                ).item()
                sims.append(cos_sim)
            sim_matrix[i, j] = np.mean(sims)

    # Key metrics
    neighbor_sims = [sim_matrix[i, i+1] for i in range(num_layers - 1)]
    distant_sim = sim_matrix[0, num_layers - 1]

    return {
        "dataset": Path(lmdb_path).name,
        "n_positions": len(positions),
        "sim_matrix": sim_matrix.tolist(),
        "mean_neighbor_sim": float(np.mean(neighbor_sims)),
        "std_neighbor_sim": float(np.std(neighbor_sims)),
        "layer0_vs_last": float(distant_sim),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="layer_similarity_results")
    parser.add_argument("--max_datasets", type=int, default=10)
    parser.add_argument("--n_positions", type=int, default=50)
    args = parser.parse_args()

    from esm_embedding.model import load_model, _MODEL_PATHS
    _MODEL_PATHS["saprot_650m"] = Path(args.model_dir)

    data_dir = Path(args.data_dir)
    lmdb_files = sorted(data_dir.glob("*"))
    print(f"Found {len(lmdb_files)} datasets")
    if args.max_datasets:
        lmdb_files = lmdb_files[:args.max_datasets]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("Loading model...")
    info = load_model("saprot_650m", device=args.device)
    model = info["model"]
    tokenizer = info["tokenizer"]
    device = info["device"]
    num_layers = info["num_layers"]
    print(f"  {num_layers} layers, {info['hidden_dim']}-dim")

    # Compute per-dataset similarity
    all_results = {}
    agg_matrix = np.zeros((num_layers, num_layers))
    n_valid = 0

    for lmdb_file in tqdm(lmdb_files, desc="Datasets"):
        name = lmdb_file.name
        result = compute_layer_similarity(
            str(lmdb_file), model, tokenizer, device, args.n_positions
        )
        if result is None:
            continue
        all_results[name] = {
            "mean_neighbor_sim": result["mean_neighbor_sim"],
            "layer0_vs_last": result["layer0_vs_last"],
        }
        agg_matrix += np.array(result["sim_matrix"])
        n_valid += 1

    agg_matrix /= n_valid
    print(f"\nAveraged over {n_valid} datasets")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYER COSINE SIMILARITY ANALYSIS")
    print("=" * 60)
    print(f"Mean neighbor similarity:  {agg_matrix.diagonal(1).mean():.4f}")
    print(f"Layer 0 vs Layer 1:        {agg_matrix[0, 1]:.4f}")
    print(f"Layer 0 vs Layer 16:       {agg_matrix[0, 16]:.4f}")
    print(f"Layer 0 vs Layer 32:       {agg_matrix[0, 32]:.4f}")
    print(f"Layer 16 vs Layer 32:      {agg_matrix[16, 32]:.4f}")

    # Correlation between similarity and layer distance
    distances = []
    similarities = []
    for i in range(num_layers):
        for j in range(i+1, num_layers):
            distances.append(abs(i - j))
            similarities.append(agg_matrix[i, j])
    from scipy.stats import pearsonr
    r, p = pearsonr(distances, similarities)
    print(f"\nCorrelation(layer_distance, cosine_sim): r={r:.4f}, p={p:.2e}")

    # Per-dataset variance
    neighbor_var = np.var([r["mean_neighbor_sim"] for r in all_results.values()])
    distant_var = np.var([r["layer0_vs_last"] for r in all_results.values()])
    print(f"\nCross-dataset variance:")
    print(f"  Neighbor sim:  std={np.sqrt(neighbor_var):.4f}")
    print(f"  Layer 0 vs 32: std={np.sqrt(distant_var):.4f}")

    # ── Save ────────────────────────────────────────────────────
    output = {
        "num_layers": num_layers,
        "num_datasets": n_valid,
        "n_positions_per_dataset": args.n_positions,
        "sim_matrix": agg_matrix.tolist(),
        "per_dataset": all_results,
        "stats": {
            "mean_neighbor_sim": float(agg_matrix.diagonal(1).mean()),
            "layer0_vs_1": float(agg_matrix[0, 1]),
            "layer0_vs_16": float(agg_matrix[0, 16]),
            "layer0_vs_32": float(agg_matrix[0, 32]),
            "layer16_vs_32": float(agg_matrix[16, 32]),
            "distance_sim_correlation": float(r),
        }
    }

    with open(out_dir / "layer_similarity.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_dir / 'layer_similarity.json'}")


if __name__ == "__main__":
    main()
