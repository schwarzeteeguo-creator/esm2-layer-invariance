"""Phase E benchmark: AF3/Boltz-1 structural confidence features for mutation effect prediction.

Compares three ablation modes:
    - seq_only:    [embedding | aux(42)]           — Phase A baseline
    - seq+struct:  [embedding | struct(12) | aux]  — Phase E primary model
    - struct_only: [struct(12) | aux]              — lower bound

Reuses benchmark_v2.py's evaluate_with_readouts() and feature-building pattern.
Adds structural features as an additional feature channel that is concatenated
before the readout model.

Usage:
    python -m esm_embedding.benchmark_phase_e \
        --data_dir ProteinGym/LMDB/substitutions \
        --model_dir weights/PLMs/SaProt_650M_AF2_hf \
        --boltz_cache_dir boltz_cache \
        --protein_list phase_e_proteins.csv \
        --ablation_modes seq_only seq+struct struct_only \
        --readout_models ridge rf mlp lightgbm \
        --output_dir phase_e_results
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import csv

# Single-threaded for server safety
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"
# Limit PyTorch native threading
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

# ── Import shared utilities from Phase A ──────────────────────────────

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from esm_embedding.structure import extract_structural_features, FEATURE_DIM
from esm_embedding.benchmark_v2 import (
    read_lmdb, parse_mutations, build_mutation_features,
    BLOSUM62, AA_LIST, AA_TO_IDX,
    evaluate_with_readouts,  # Reuse existing CV evaluation
)

# ── Core Phase E feature extraction ───────────────────────────────────

def load_boltz_cache(protein_id: str, cache_dir: str) -> Optional[dict]:
    """Load cached Boltz-1 result from .npz file.

    Returns dict with plddt, pae, iptm, ptm or None if not found.
    """
    cache_path = Path(cache_dir) / f"{protein_id}.npz"
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        return {
            "plddt": data["plddt"],
            "pae": data["pae"],
            "iptm": float(data.get("iptm", 0.0)),
            "ptm": float(data.get("ptm", 0.0)),
        }
    except Exception as e:
        print(f"  WARNING: Failed to load cache {cache_path}: {e}")
        return None


def extract_dataset_features_phase_e(
    lmdb_path: str,
    model_key: str = "esm2_650m",
    model_dir: str = None,
    device: str = "cuda",
    boltz_cache_dir: str = None,
    structural_features: Optional[dict] = None,
) -> Optional[dict]:
    """Extract features with optional structural confidence augmentation.

    Args:
        lmdb_path: Path to LMDB dataset
        model_key: Model to use for embeddings
        model_dir: Directory containing model weights
        device: 'cuda' or 'cpu'
        boltz_cache_dir: Directory with cached Boltz-1 .npz files
        structural_features: Pre-loaded structural features dict (overrides cache)

    Returns:
        dict with keys: dataset_name, num_mutations, features (dict method->np.array),
        fitness (np.array), has_structure (bool)
    """
    import torch
    from esm_embedding.extract import extract_multi_scale
    from esm_embedding.model import load_model, _MODEL_PATHS

    # Load LMDB
    data = read_lmdb(lmdb_path)
    name = Path(lmdb_path).name
    wild_type = data.get("wild_type", "")
    if not wild_type:
        return None

    length = int(data.get("length", 0))
    if length == 0:
        return None

    # Parse mutations
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

    if len(mutations) < 10:
        return None

    # Load structural confidence from cache
    struct_feat = None
    has_structure = False
    if boltz_cache_dir:
        # Use DMS_id (dataset name) to find cache
        protein_id = name  # LMDB filename is the DMS_id
        boltz_data = load_boltz_cache(protein_id, boltz_cache_dir)
        if boltz_data is not None:
            positions = [m[0] for m in mutations]  # 1-indexed positions
            struct_feat = extract_structural_features(
                plddt=boltz_data["plddt"],
                pae=boltz_data["pae"],
                positions=positions,
                iptm=boltz_data.get("iptm", 0.0),
            )
            has_structure = True

    if structural_features is not None:
        struct_feat = structural_features
        has_structure = True

    # Extract ESM embeddings
    if model_dir:
        _MODEL_PATHS[model_key] = Path(model_dir)

    try:
        multi_results = extract_multi_scale(
            sequence=wild_type, model_key=model_key, device=device)
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return None

    # Build feature matrices per pooling method
    features_by_method = defaultdict(list)
    valid_mutations = []

    for pos, from_aa, to_aa, fitness in mutations:
        if pos < 1 or pos > len(wild_type):
            continue
        embs = {}
        valid = True
        for method, result in multi_results.items():
            if method == "_raw_layers":
                continue
            if pos - 1 >= len(result.residue_embeddings):
                valid = False
                break
            emb = result.residue_embeddings[pos - 1].numpy()
            if np.isnan(emb).any() or np.isinf(emb).any():
                valid = False
                break
            embs[method] = emb
        if not valid:
            continue
        aux = build_mutation_features(from_aa, to_aa)
        for method, emb in embs.items():
            feat = np.concatenate([emb, aux])
            features_by_method[method].append(feat)
        valid_mutations.append((pos, from_aa, to_aa, fitness))

    if len(valid_mutations) < 10:
        return None

    features = {
        method: np.stack(feat_list, axis=0)
        for method, feat_list in features_by_method.items()
    }

    return {
        "dataset_name": name,
        "num_mutations": len(valid_mutations),
        "features": features,
        "fitness": np.array([m[3] for m in valid_mutations]),
        "has_structure": has_structure,
        "struct_features": struct_feat,
        "valid_positions": np.array([m[0] for m in valid_mutations]),
    }


def run_benchmark_phase_e(
    data_dir: str,
    model_dir: str = None,
    model_key: str = "esm2_650m",
    device: str = "cuda",
    output_dir: str = "phase_e_results",
    boltz_cache_dir: str = None,
    max_datasets: int = None,
    readout_models: List[str] = None,
    ablation_modes: List[str] = None,
    protein_list_csv: str = None,
):
    """Run Phase E benchmark with structural feature ablation.

    Args:
        data_dir: Directory containing LMDB ProteinGym data
        boltz_cache_dir: Directory with cached Boltz-1 predictions (.npz)
        protein_list_csv: CSV of selected proteins (from select_phase_e_proteins.py)
        ablation_modes: List of modes to run: ["seq_only", "seq+struct", "struct_only"]
    """
    if readout_models is None:
        readout_models = ["ridge", "rf", "mlp", "lightgbm"]
    if ablation_modes is None:
        ablation_modes = ["seq_only", "seq+struct", "struct_only"]

    data_dir = Path(data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load protein list if provided (to filter which datasets to run)
    target_ids = None
    if protein_list_csv:
        with open(protein_list_csv, newline="") as f:
            reader = csv.DictReader(f)
            target_ids = {row["DMS_id"] for row in reader}
        print(f"Target proteins from list: {len(target_ids)}")

    # Find LMDB files
    lmdb_files = sorted(data_dir.glob("*"))
    # Filter out directories that were intentionally skipped (large datasets etc.)
    lmdb_files = [f for f in lmdb_files if not f.name.endswith(".SKIP_LARGE")]
    if target_ids:
        lmdb_files = [f for f in lmdb_files if f.name in target_ids]

    print(f"Datasets to process: {len(lmdb_files)}")
    print(f"Model: {model_key}")
    print(f"Readout models: {readout_models}")
    print(f"Ablation modes: {ablation_modes}")
    print(f"Boltz cache: {boltz_cache_dir}")
    if max_datasets:
        lmdb_files = lmdb_files[:max_datasets]

    # Resume from existing results
    results_file = out_dir / "phase_e_results.json"
    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} datasets already processed")

    t0 = time.time()
    struct_count = 0
    n_total = len(lmdb_files)

    for i, lmdb_file in enumerate(tqdm(lmdb_files, desc="Datasets")):
        name = lmdb_file.name
        t_dataset = time.time()

        # Check if already processed with all ablations
        if name in all_results:
            entry = all_results[name]
            # Skip if marked as skipped (large dataset etc.)
            if entry.get("_skipped") or entry.get("_skipped_large"):
                continue
            # Extract ablation modes from keys like "last_layer__seq_only"
            existing_modes = set()
            for key in entry.get("ablations", {}).keys():
                if "__" in key:
                    existing_modes.add(key.rsplit("__", 1)[-1])
            # Only require modes this dataset can actually have
            has_struct = entry.get("has_structure", False)
            required = ablation_modes if has_struct else ["seq_only"]
            if set(required).issubset(existing_modes):
                continue

        # Extract features
        t_embed = time.time()
        result = extract_dataset_features_phase_e(
            str(lmdb_file),
            model_key=model_key,
            model_dir=model_dir,
            device=device,
            boltz_cache_dir=boltz_cache_dir,
        )
        if result is None:
            print(f"  [{i+1}/{n_total}] {name}: SKIP (no data)", flush=True)
            continue

        fitness = result["fitness"]
        has_struct = result.get("has_structure", False)
        struct_feat = result.get("struct_features")
        if has_struct:
            struct_count += 1

        embed_time = time.time() - t_embed
        n_mut = result["num_mutations"]
        struct_tag = "+S" if has_struct else "-S"

        # Build ablation feature matrices
        ablation_features = {}
        methods = list(result["features"].keys())
        n_abl = len(methods) * (3 if has_struct and struct_feat is not None else 1)

        for method in methods:
            emb_feat = result["features"][method]  # (N, D+42)

            # seq_only: embedding + aux (baseline)
            ablation_features[(method, "seq_only")] = emb_feat

            # seq+struct / struct_only: only if structure available
            if has_struct and struct_feat is not None:
                # seq+struct: [embedding | struct | aux]
                D = emb_feat.shape[1] - 42  # embedding dimension
                emb_part = emb_feat[:, :D]
                aux_part = emb_feat[:, D:]
                struct_part = struct_feat[:emb_feat.shape[0], :]  # align samples

                ablation_features[(method, "seq+struct")] = np.concatenate(
                    [emb_part, struct_part, aux_part], axis=1)

                # struct_only: [struct | aux]
                ablation_features[(method, "struct_only")] = np.concatenate(
                    [struct_part, aux_part], axis=1)

        # Evaluate
        print(f"  [{i+1}/{n_total}] {name}: {n_mut}mut {struct_tag} "
              f"embed={embed_time:.1f}s evaluating {n_abl}abl...", flush=True)
        t_eval = time.time()
        n_ablations = len(ablation_features)
        method_ablations = {}
        for j, ((method, ablation), features) in enumerate(ablation_features.items()):
            eval_r = evaluate_with_readouts(features, fitness, readout_models)
            method_ablations[f"{method}__{ablation}"] = eval_r
        eval_time = time.time() - t_eval

        # Store
        if name in all_results:
            all_results[name]["ablations"].update(method_ablations)
        else:
            all_results[name] = {
                "n_mutations": result["num_mutations"],
                "has_structure": has_struct,
                "ablations": method_ablations,
            }

        dataset_time = time.time() - t_dataset
        print(f"  [{i+1}/{n_total}] {name}: DONE eval={eval_time:.1f}s total={dataset_time:.1f}s", flush=True)

        # Save after each dataset
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(all_results)} datasets in {elapsed:.0f}s")
    print(f"  With structural features: {struct_count}/{len(all_results)}")

    # Save final results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Aggregate analysis ──────────────────────────────────────────
    methods = ["last_layer", "mean_20_33", "concat_6_14_20_26_33"]
    print("\n" + "=" * 100)
    print(f"PHASE E BENCHMARK — {model_key}")
    print("=" * 100)

    for ablation in ablation_modes:
        print(f"\n── {ablation} ──")
        header = f"{'Method':<30}"
        for rm in readout_models:
            header += f" {rm:>12}"
        print(header)
        print("-" * (30 + 12 * len(readout_models)))

        for method in methods:
            row = f"{method:<30}"
            for rm in readout_models:
                vals = []
                for name, r in all_results.items():
                    key = f"{method}__{ablation}"
                    v = r.get("ablations", {}).get(key, {}).get(f"{rm}_rho")
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        vals.append(v)
                if vals:
                    row += f" {np.mean(vals):>11.4f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

    # ── Ablation delta analysis ──────────────────────────────────────
    if "seq_only" in ablation_modes and "seq+struct" in ablation_modes:
        print("\n" + "=" * 100)
        print("ABLATION DELTA: (seq+struct) - (seq_only)")
        print("=" * 100)

        for method in methods:
            for rm in readout_models:
                deltas = []
                for name, r in all_results.items():
                    k1 = f"{method}__seq+struct"
                    k2 = f"{method}__seq_only"
                    v1 = r.get("ablations", {}).get(k1, {}).get(f"{rm}_rho")
                    v2 = r.get("ablations", {}).get(k2, {}).get(f"{rm}_rho")
                    if (v1 is not None and v2 is not None
                        and not np.isnan(v1) and not np.isnan(v2)):
                        deltas.append(v1 - v2)
                if deltas:
                    deltas = np.array(deltas)
                    mean_d = np.mean(deltas)
                    std_d = np.std(deltas)
                    n_pos = np.sum(deltas > 0)
                    print(f"  {method:<28} {rm:<10}: "
                          f"Δ={mean_d:+.5f} ± {std_d:.4f}  "
                          f"positive: {n_pos}/{len(deltas)} "
                          f"({n_pos/len(deltas)*100:.1f}%)")

    # Save summary
    summary = {"model_key": model_key, "ablation_modes": ablation_modes,
               "readout_models": readout_models, "n_datasets": len(all_results),
               "n_with_structure": struct_count}
    with open(out_dir / "phase_e_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {"per_dataset": all_results, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--model_key", default="saprot_650m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="phase_e_results")
    parser.add_argument("--boltz_cache_dir", default="boltz_cache")
    parser.add_argument("--max_datasets", type=int, default=None)
    parser.add_argument("--readout_models", nargs="+",
                        default=["ridge", "rf", "mlp", "lightgbm"],
                        choices=["ridge", "rf", "mlp", "lightgbm"])
    parser.add_argument("--ablation_modes", nargs="+",
                        default=["seq_only", "seq+struct", "struct_only"],
                        choices=["seq_only", "seq+struct", "struct_only"])
    parser.add_argument("--protein_list_csv", default=None,
                        help="CSV file with selected protein list")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_benchmark_phase_e(**vars(args))


if __name__ == "__main__":
    main()
