"""Logit Lens with ESM-2: Apply LM head to EACH transformer layer.

Unlike SaProt, ESM-2 uses standard single-letter AA tokens — no structure
info needed. This allows clean logit lens experiments without the
out-of-distribution issue of SaProt's structure-masked tokens.

Key question: Do intermediate layers produce better zero-shot mutation
scores than the final layer when projected through the LM head?

Usage:
    python -m esm_embedding.logit_lens_esm2 \
        --data_dir /root/autodl-tmp/SaProt/LMDB/ProteinGym/substitutions \
        --output_dir /root/autodl-tmp/logit_lens_esm2_results \
        --max_datasets 5
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ── LMDB reading ───────────────────────────────────────────────────
def read_lmdb(path: str) -> dict:
    import lmdb
    env = lmdb.open(path, readonly=True, lock=False)
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


def parse_mutations(mut_info: str) -> List[Tuple[str, int, str]]:
    """Parse "A123V:G124L" -> [('A', 123, 'V'), ('G', 124, 'L')]"""
    muts = []
    for token in mut_info.split(":"):
        if token and token[0] in "ACDEFGHIKLMNPQRSTVWY":
            muts.append((token[0], int(token[1:-1]), token[-1]))
    return muts


# ── Core: Logit Lens with ESM-2 ────────────────────────────────────
def run_logit_lens_dataset(
    lmdb_path: str,
    model,
    tokenizer,
    lm_head,
    num_layers: int,
    device: str = "cuda",
) -> Optional[dict]:
    """Run ESM-2 logit lens on one ProteinGym dataset.

    Uses standard ESM-2 tokenizer with single-letter AA tokens.
    Scoring: log P(mt_aa | context) - log P(wt_aa | context)
    """
    data = read_lmdb(lmdb_path)
    name = Path(lmdb_path).name
    wild_type = data.get("wild_type", "")
    if not wild_type:
        return None

    length = int(data.get("length", 0))
    if length == 0:
        return None

    dev = device

    # ── Collect mutations, grouped by position ─────────────────────
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
                    muts_by_pos.setdefault(pos, []).append(
                        (from_aa, to_aa, float(fitness))
                    )

    if len(muts_by_pos) < 5:
        return None

    # ── Per-layer scores ───────────────────────────────────────────
    layer_scores: Dict[int, List[Tuple[float, float]]] = {
        l: [] for l in range(num_layers)
    }

    # Build token sequence: "M A D E ..." (space-separated single AAs)
    wt_tokens = list(wild_type)  # ['M', 'A', 'D', 'E', ...]

    for pos in tqdm(sorted(muts_by_pos.keys()), desc=f"  {name}", leave=False):
        # Create masked version
        masked_tokens = wt_tokens.copy()
        masked_tokens[pos - 1] = tokenizer.mask_token

        token_str = " ".join(masked_tokens)
        inputs = tokenizer(token_str, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        # Verify the mask position is correct
        # ESM tokenizer: <cls> A A <mask> A A <eos>
        # mask should be at position `pos` (1-indexed, <cls> at 0)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: (embedding, layer_0, ..., layer_N)
        all_hidden = list(outputs.hidden_states)
        layer_hidden = all_hidden[1:]  # skip embedding layer
        token_pos = pos  # <cls> at 0, so 1-indexed pos maps directly

        for l_idx in range(num_layers):
            h = layer_hidden[l_idx][0, token_pos, :]  # (hidden_dim,)
            logits = lm_head(h)  # (vocab_size,)

            for from_aa, to_aa, fitness in muts_by_pos[pos]:
                # ESM-2 has proper single-letter AA tokens
                wt_id = tokenizer.convert_tokens_to_ids(from_aa)
                mt_id = tokenizer.convert_tokens_to_ids(to_aa)

                # Skip if AA is unknown (shouldn't happen with ESM-2)
                if wt_id == tokenizer.unk_token_id or mt_id == tokenizer.unk_token_id:
                    continue

                log_probs = torch.log_softmax(logits, dim=-1)
                score = (log_probs[mt_id] - log_probs[wt_id]).item()

                if not np.isnan(score) and not np.isinf(score):
                    layer_scores[l_idx].append((score, fitness))

    # ── Compute per-layer Spearman rho ─────────────────────────────
    layer_rhos = {}
    for l in range(num_layers):
        if len(layer_scores[l]) < 10:
            layer_rhos[str(l)] = {"spearman": np.nan, "n": len(layer_scores[l])}
            continue
        s = np.array([x for x, _ in layer_scores[l]])
        f = np.array([y for _, y in layer_scores[l]])
        valid = ~(np.isnan(s) | np.isinf(s) | np.isnan(f) | np.isinf(f))
        if valid.sum() < 10:
            layer_rhos[str(l)] = {"spearman": np.nan, "n": int(valid.sum())}
            continue
        rho, _ = spearmanr(s[valid], f[valid])
        layer_rhos[str(l)] = {"spearman": float(rho), "n": int(valid.sum())}

    return {
        "dataset_name": name,
        "n_positions": len(muts_by_pos),
        "layer_rhos": layer_rhos,
    }


# ── Main ──────────────────────────────────────────────────────────
def run_logit_lens_benchmark(
    data_dir: str,
    device: str = "cuda",
    output_dir: str = "logit_lens_esm2_results",
    max_datasets: int = None,
    model_path: str = None,
):
    from transformers import EsmForMaskedLM, EsmTokenizer

    data_dir = Path(data_dir)
    lmdb_files = sorted(data_dir.glob("*"))
    print(f"Found {len(lmdb_files)} LMDB files")
    if max_datasets:
        lmdb_files = lmdb_files[:max_datasets]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ESM-2 model
    if model_path is None:
        model_path = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading ESM-2 from: {model_path}")
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    lm_head = model.lm_head
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"  {num_layers} layers, {hidden_dim}-dim, vocab={vocab_size}")
    print(f"  Mask token: {repr(tokenizer.mask_token)} (id={tokenizer.mask_token_id})")

    # Verify single-letter AA tokens exist
    for aa in "MADE":
        tid = tokenizer.convert_tokens_to_ids(aa)
        status = "OK" if tid != tokenizer.unk_token_id else "MISSING!"
        print(f"  {aa} -> id={tid} ({status})")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.0f}M")

    # Resume
    results_file = out_dir / "logit_lens_esm2_results.json"
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

        result = run_logit_lens_dataset(
            str(lmdb_file), model, tokenizer, lm_head, num_layers, device=device
        )
        if result is None:
            continue

        all_results[name] = {
            "n_positions": result["n_positions"],
            "layer_rhos": result["layer_rhos"],
        }

        # Save checkpoint every 3 datasets
        if len(all_results) % 3 == 0:
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(all_results)} datasets in {elapsed:.0f}s")

    # Save final results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Aggregate ──────────────────────────────────────────────────
    layer_rhos_agg = {l: [] for l in range(num_layers)}

    for name, r in all_results.items():
        for l_str, scores in r["layer_rhos"].items():
            l = int(l_str)
            if not np.isnan(scores.get("spearman", np.nan)):
                layer_rhos_agg[l].append(scores["spearman"])

    print("\n" + "=" * 65)
    print("ESM-2 LOGIT LENS RESULTS")
    print("=" * 65)
    print(f"{'Layer':>6}  {'Mean Rho':>10}  {'Std':>8}  {'#Datasets':>10}")

    best_layer = num_layers - 1
    best_rho = -999
    final_rho = 0

    for l in range(num_layers):
        if not layer_rhos_agg[l]:
            continue
        mean_r = np.mean(layer_rhos_agg[l])
        std_r = np.std(layer_rhos_agg[l])
        n = len(layer_rhos_agg[l])

        marker = ""
        if l == num_layers - 1:
            marker = "  <-- final layer (standard ESM-2 zero-shot)"
            final_rho = mean_r
        if mean_r > best_rho:
            best_rho = mean_r
            best_layer = l

        print(f"  {l:3d}   {mean_r:>10.4f}  {std_r:>8.4f}  {n:>10}{marker}")

    print("-" * 65)
    print(f"\nBest layer: {best_layer} (rho={best_rho:.4f})")
    print(f"Final layer ({num_layers - 1}): rho={final_rho:.4f}")
    print(f"Delta: {best_rho - final_rho:+.4f}")
    if abs(final_rho) > 1e-6:
        print(f"Relative gain: {(best_rho - final_rho) / abs(final_rho) * 100:+.1f}%")

    # Compare to known baselines
    print(f"\nESM-2 standard zero-shot (from literature): rho=0.475")
    print(f"SaProt standard zero-shot (from Phase 1):  rho=0.477")

    # Save summary
    summary = {
        "model": "ESM-2 650M (facebook/esm2_t33_650M_UR50D)",
        "num_layers": num_layers,
        "num_datasets": len(all_results),
        "best_layer": best_layer,
        "best_layer_rho": float(best_rho),
        "final_layer_rho": float(final_rho),
        "delta": float(best_rho - final_rho),
        "per_layer": {
            str(l): {
                "mean": float(np.mean(layer_rhos_agg[l])),
                "std": float(np.std(layer_rhos_agg[l])),
                "n": len(layer_rhos_agg[l]),
            }
            for l in range(num_layers) if layer_rhos_agg[l]
        }
    }
    with open(out_dir / "logit_lens_esm2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="logit_lens_esm2_results")
    parser.add_argument("--max_datasets", type=int, default=None)
    parser.add_argument("--model_path", default=None, help="Local path to ESM-2 model (default: HF hub)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_logit_lens_benchmark(**vars(args))


if __name__ == "__main__":
    main()
