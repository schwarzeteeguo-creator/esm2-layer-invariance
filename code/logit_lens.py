"""Logit Lens: Apply SaProt LM head to EACH transformer layer.

Key question: Do intermediate layers, when projected through the final-layer
LM head, produce better zero-shot mutation scores than the final layer?

Method:
    1. Tokenize WT sequence as structure-masked tokens ("M# A# D#" format)
    2. For each unique mutation position, mask it and run one forward pass
    3. Extract hidden states for ALL 33 layers at the masked position
    4. Apply `model.lm_head()` to each layer's hidden state → pseudo-logits
    5. Score via marginalized probability over structure tokens:
       log(sum(P(all A* tokens))) - log(sum(P(all M* tokens)))
    6. Report per-layer Spearman rho vs DMS fitness

Usage:
    python -m esm_embedding.logit_lens \
        --data_dir /root/autodl-tmp/SaProt/LMDB/ProteinGym/substitutions \
        --model_dir /root/autodl-tmp/SaProt/weights/PLMs/SaProt_650M_AF2_hf \
        --output_dir /root/autodl-tmp/logit_lens_results \
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

# Standard 20 amino acids
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


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


# ── Token mapping for marginalized scoring ─────────────────────────
def build_aa_to_token_ids(tokenizer) -> Dict[str, List[int]]:
    """Build mapping: amino acid → list of all token IDs matching that AA.

    SaProt vocabulary structure:
      - Each token is a 2-char string: first=AA, second=structure_class
      - e.g., "Mp" (AA=M, struct=p), "M#" (AA=M, struct=unknown)
      - 20 structure classes + "#" (unknown/mask) = 21 structure tokens per AA
      - Total: 20 AA × 21 structure = 420 + 1 "#"×21 = 441 structure-aware tokens

    For marginalized scoring, we sum probabilities over all structure
    variants of the same amino acid.
    """
    aa_to_ids = {aa: [] for aa in AA_LIST}

    for token, token_id in tokenizer.get_vocab().items():
        if len(token) == 2 and token[0] in AA_LIST and token[1] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            # token like "Mp", "A#", "Dv" — first char is AA, second is structure
            aa = token[0]
            aa_to_ids[aa].append(token_id)

    # Also try with tok_str variant
    if len(aa_to_ids.get("M", [])) == 0:
        # Try alternative: convert each token through the tokenizer
        for aa in AA_LIST:
            ids = []
            for struct_char in "#pynwrqhgdvltmfsaeikc":
                tok = f"{aa}{struct_char}"
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid != tokenizer.unk_token_id:
                    ids.append(tid)
            aa_to_ids[aa] = ids

    # Verify
    for aa in AA_LIST:
        n = len(aa_to_ids[aa])
        if n == 0:
            print(f"  WARNING: No tokens found for AA '{aa}'")
        elif n != 21:
            print(f"  NOTE: AA '{aa}' has {n} tokens (expected 21)")

    return aa_to_ids


def marginalized_log_prob(
    logits: torch.Tensor,
    aa: str,
    aa_to_ids: Dict[str, List[int]],
) -> float:
    """Compute log(sum(P(all tokens for this AA))) from logits.

    Args:
        logits: raw logits vector (vocab_size,)
        aa: single-letter amino acid (e.g., "M")
        aa_to_ids: mapping from AA to list of token IDs

    Returns:
        log(sum(softmax(logits)[ids])) — marginalized log-probability
    """
    token_ids = aa_to_ids.get(aa, [])
    if not token_ids:
        return float("-inf")

    # Log-sum-exp trick for numerical stability
    # log(sum(exp(logits_i))) = log_sum_exp(logits_i) - log_sum_exp(all_logits)
    log_probs = torch.log_softmax(logits, dim=-1)
    aa_log_probs = log_probs[token_ids]  # log probabilities for all tokens of this AA
    log_total = torch.logsumexp(aa_log_probs, dim=-1)  # log(sum of probs)
    return log_total.item()


def direct_log_prob(
    logits: torch.Tensor,
    aa: str,
    tokenizer,
) -> float:
    """Compute log(P(aa#)) — direct probability of the structure-masked token."""
    tok = f"{aa}#"
    tid = tokenizer.convert_tokens_to_ids(tok)
    if tid == tokenizer.unk_token_id:
        return float("-inf")
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[tid].item()


# ── Core: Logit Lens ───────────────────────────────────────────────
def run_logit_lens_dataset(
    lmdb_path: str,
    model_info: dict,
    aa_to_ids: Dict[str, List[int]],
    device: str = "cuda",
) -> Optional[dict]:
    """Run logit lens on one ProteinGym dataset.

    Uses structure-masked tokens (e.g., "M#" for AA=M) since ProteinGym
    data does not include structural information. Scoring is done via
    marginalized probability over all structure variants per amino acid.

    Returns:
        dict with per-layer zero-shot scores and Spearman rho.
    """
    data = read_lmdb(lmdb_path)
    name = Path(lmdb_path).name
    wild_type = data.get("wild_type", "")
    if not wild_type:
        return None

    length = int(data.get("length", 0))
    if length == 0:
        return None

    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    lm_head = model.lm_head
    num_layers = model_info["num_layers"]
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

    # ── Build structure-masked token sequence ──────────────────────
    # Format: "M# A# D# E#" — each position is AA + structure-mask
    sa_tokens = [f"{aa}#" for aa in wild_type]

    # ── Per-layer scores ───────────────────────────────────────────
    layer_scores: Dict[int, List[Tuple[float, float]]] = {
        l: [] for l in range(num_layers)
    }
    # Also store direct "#"-token scores for comparison
    direct_layer_scores: Dict[int, List[Tuple[float, float]]] = {
        l: [] for l in range(num_layers)
    }

    for pos in tqdm(sorted(muts_by_pos.keys()), desc=f"  {name}", leave=False):
        # Create masked version (1-indexed pos → token position pos, since <cls> at 0)
        masked_tokens = sa_tokens.copy()
        masked_tokens[pos - 1] = tokenizer.mask_token

        token_str = " ".join(masked_tokens)
        inputs = tokenizer(token_str, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: (embedding, layer_0, ..., layer_N)
        all_hidden = list(outputs.hidden_states)
        layer_hidden = all_hidden[1:]  # skip embedding layer
        token_pos = pos  # 1-indexed pos maps directly (<cls> at 0)

        for l_idx in range(num_layers):
            h = layer_hidden[l_idx][0, token_pos, :]  # (hidden_dim,)
            logits = lm_head(h)  # (vocab_size,)

            for from_aa, to_aa, fitness in muts_by_pos[pos]:
                # === Marginalized scoring (primary) ===
                log_prob_wt = marginalized_log_prob(logits, from_aa, aa_to_ids)
                log_prob_mt = marginalized_log_prob(logits, to_aa, aa_to_ids)

                if not np.isinf(log_prob_wt) and not np.isinf(log_prob_mt):
                    score = log_prob_mt - log_prob_wt
                    layer_scores[l_idx].append((score, fitness))

                # === Direct "#" scoring (comparison) ===
                log_prob_wt_d = direct_log_prob(logits, from_aa, tokenizer)
                log_prob_mt_d = direct_log_prob(logits, to_aa, tokenizer)

                if not np.isinf(log_prob_wt_d) and not np.isinf(log_prob_mt_d):
                    score_d = log_prob_mt_d - log_prob_wt_d
                    direct_layer_scores[l_idx].append((score_d, fitness))

    # ── Compute per-layer Spearman rho ─────────────────────────────
    def compute_rhos(scores_dict):
        rhos = {}
        for l in range(num_layers):
            if len(scores_dict[l]) < 10:
                rhos[str(l)] = {"spearman": np.nan, "n": len(scores_dict[l])}
                continue
            s = np.array([x for x, _ in scores_dict[l]])
            f = np.array([y for _, y in scores_dict[l]])
            valid = ~(np.isnan(s) | np.isinf(s) | np.isnan(f) | np.isinf(f))
            if valid.sum() < 10:
                rhos[str(l)] = {"spearman": np.nan, "n": int(valid.sum())}
                continue
            rho, _ = spearmanr(s[valid], f[valid])
            rhos[str(l)] = {"spearman": float(rho), "n": int(valid.sum())}
        return rhos

    layer_rhos = compute_rhos(layer_scores)
    direct_rhos = compute_rhos(direct_layer_scores)

    return {
        "dataset_name": name,
        "n_positions": len(muts_by_pos),
        "layer_rhos": layer_rhos,
        "layer_rhos_direct": direct_rhos,
    }


# ── Main ──────────────────────────────────────────────────────────
def run_logit_lens_benchmark(
    data_dir: str,
    model_dir: str,
    device: str = "cuda",
    output_dir: str = "logit_lens_results",
    max_datasets: int = None,
):
    from esm_embedding.model import load_model, _MODEL_PATHS

    _MODEL_PATHS["saprot_650m"] = Path(model_dir)

    data_dir = Path(data_dir)
    lmdb_files = sorted(data_dir.glob("*"))
    print(f"Found {len(lmdb_files)} LMDB files")
    if max_datasets:
        lmdb_files = lmdb_files[:max_datasets]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("Loading model...")
    model_info = load_model("saprot_650m", device=device)
    print(f"  {model_info['num_layers']} layers, {model_info['hidden_dim']}-dim")

    # Build AA → token IDs mapping
    print("Building AA → token ID mapping...")
    aa_to_ids = build_aa_to_token_ids(model_info["tokenizer"])
    total_mapped = sum(len(ids) for ids in aa_to_ids.values())
    print(f"  Mapped {total_mapped} token IDs across {len(aa_to_ids)} amino acids")

    # Resume
    results_file = out_dir / "logit_lens_results.json"
    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} already processed")

    t0 = time.time()
    num_layers = model_info["num_layers"]

    for lmdb_file in tqdm(lmdb_files, desc="Datasets"):
        name = lmdb_file.name
        if name in all_results:
            continue

        result = run_logit_lens_dataset(
            str(lmdb_file), model_info, aa_to_ids, device=device
        )
        if result is None:
            continue

        all_results[name] = {
            "n_positions": result["n_positions"],
            "layer_rhos": result["layer_rhos"],
            "layer_rhos_direct": result["layer_rhos_direct"],
        }

        if len(all_results) % 3 == 0:
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(all_results)} datasets in {elapsed:.0f}s")

    # Save final results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Aggregate ──────────────────────────────────────────────────
    def aggregate_layer_rhos(results, key="layer_rhos"):
        layer_rhos_agg = {l: [] for l in range(num_layers)}
        for name, r in results.items():
            lr = r.get(key, {})
            for l_str, scores in lr.items():
                l = int(l_str)
                if not np.isnan(scores.get("spearman", np.nan)):
                    layer_rhos_agg[l].append(scores["spearman"])
        return layer_rhos_agg

    for scoring_name, rhos_key in [
        ("MARGINALIZED (sum over structure variants)", "layer_rhos"),
        ("DIRECT (M# token only)", "layer_rhos_direct"),
    ]:
        layer_rhos_agg = aggregate_layer_rhos(all_results, rhos_key)

        print("\n" + "=" * 65)
        print(f"LOGIT LENS RESULTS — {scoring_name}")
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
                marker = "  <-- final layer"
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

    # Save summary
    summary = {
        "model": "SaProt_650M_AF2",
        "scoring": "marginalized_over_structure_variants",
        "num_layers": num_layers,
        "num_datasets": len(all_results),
        "marginalized": {
            "best_layer": best_layer,
            "per_layer": {
                str(l): {
                    "mean": float(np.mean(layer_rhos_agg[l])),
                    "std": float(np.std(layer_rhos_agg[l])),
                    "n": len(layer_rhos_agg[l]),
                }
                for l in range(num_layers) if layer_rhos_agg[l]
            }
        }
    }
    with open(out_dir / "logit_lens_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="logit_lens_results")
    parser.add_argument("--max_datasets", type=int, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_logit_lens_benchmark(**vars(args))


if __name__ == "__main__":
    main()
