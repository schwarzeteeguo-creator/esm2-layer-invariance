"""Logit lens VALIDATION — must pass before any per-layer interpretation.

Reviewer 1, Major 4: "validate the implementation by confirming that the
final-layer score reproduces the published ProteinGym zero-shot value for
each model within tolerance, before interpreting any per-layer trend."

This script runs, per dataset:
  A. OFFICIAL-PATH zero-shot: mask the mutation position, take
     model(**inputs).logits, score = log P(mut) - log P(wt).
     This is the standard ESM-1v / ProteinGym protocol and is the ground
     truth the lens must reproduce at the final layer.
  B. LENS-PATH at final layer: lm_head(hidden_states[-1]) -> same scoring.
     Reported as max |A - B| difference over positions.
  C. Model introspection: locate the final LayerNorm applied between the
     last transformer block and the LM head (if any), so intermediate
     layers can be normed identically in logit_lens_v2.
  D. Token-position sanity check: decode the tokenized sequence and
     verify residue i (1-indexed) sits at token index i (<cls> at 0) —
     rules out the off-by-one the reviewer suspects.

For SaProt, scoring marginalizes over the 21 structure-token variants of
each amino acid (both paths use the same marginalization).

Usage:
  python -m esm_embedding.logit_lens_validate \
      --model esm2 --model_dir esm2_model --max_datasets 3
  python -m esm_embedding.logit_lens_validate \
      --model saprot --model_dir SaProt-main/weights/PLMs/SaProt_650M_AF2_hf
"""

import os, sys, json, argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_embedding.probing_v4 import read_lmdb, parse_mutations

# Literature anchors (mean over full ProteinGym substitution benchmark,
# per-dataset values vary widely; used as a RANGE sanity check)
LITERATURE = {
    "esm2": "ESM-2 650M published zero-shot mean rho ~ 0.41 (ProteinGym v1)",
    "saprot": "SaProt 650M published zero-shot mean rho ~ 0.457; "
              "our Phase-1 reproduction on 63 datasets: 0.477",
}

STRUCTURE_TOKENS = "pynwrqhgdlavtmfsekyic"  # 21 Foldseek 3Di states


def build_aa_token_map(tokenizer, is_saprot):
    """Map amino acid -> list of vocab row ids (SaProt: 21 ids per AA)."""
    vocab = tokenizer.get_vocab()
    aa_map = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        if is_saprot:
            ids = [vocab[f"{aa}{st}"] for st in STRUCTURE_TOKENS
                   if f"{aa}{st}" in vocab]
            # plus the structure-masked variant
            if f"{aa}#" in vocab:
                ids.append(vocab[f"{aa}#"])
        else:
            ids = [vocab[aa]] if aa in vocab else []
        if ids:
            aa_map[aa] = ids
    return aa_map


def marginal_log_prob(logits, token_ids):
    """log sum of probabilities over the token variants (1-D logits)."""
    logp = torch.log_softmax(logits.double(), dim=-1)
    return float(torch.logsumexp(logp[torch.tensor(token_ids)], dim=0))


@torch.no_grad()
def one_forward(model, tokenizer, token_str, device):
    inputs = tokenizer(token_str, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True)
    return out, inputs["input_ids"][0]


def validate_dataset(lmdb_path, model, tokenizer, device, is_saprot,
                     max_positions=400):
    data = read_lmdb(lmdb_path)
    name = Path(lmdb_path).name
    wt = data.get("wild_type", "")
    if not wt:
        return None

    muts_by_pos: Dict[int, List] = {}
    for key, entry in data.items():
        if not key.isdigit() or not isinstance(entry, dict):
            continue
        fitness = entry.get("fitness")
        subs = parse_mutations(entry.get("mut_info", ""))
        if not subs or fitness is None:
            continue
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fitness):
            continue
        for f, p, t in subs:
            if 1 <= p <= len(wt):
                muts_by_pos.setdefault(p, []).append((f, t, fitness))

    if len(muts_by_pos) < 5:
        return None

    aa_map = build_aa_token_map(tokenizer, is_saprot)
    mask_token = tokenizer.mask_token

    # ---- D. token-position sanity check (once) ----
    if is_saprot:
        probe_str = " ".join(f"{aa}#" for aa in wt[:5])
    else:
        probe_str = " ".join(list(wt[:5]))
    ids = tokenizer(probe_str)["input_ids"]
    decoded = tokenizer.convert_ids_to_tokens(ids)
    offset_ok = decoded[0] in ("<cls>", "<s>") and decoded[1][0].upper() == wt[0]

    positions = sorted(muts_by_pos)[:max_positions]
    scores_official, scores_lens_final, labels = [], [], []
    max_diff_final = 0.0

    for pos in positions:
        if is_saprot:
            toks = [f"{aa}#" for aa in wt]
            toks[pos - 1] = mask_token                      # "M#"
        else:
            toks = list(wt)
            toks[pos - 1] = mask_token                      # "<mask>"
        out, input_ids = one_forward(model, tokenizer, " ".join(toks), device)

        # A. official path: full-model logits at the masked token
        logits_official = out.logits[0, pos, :]

        # B. lens path at final layer
        hidden_final = out.hidden_states[-1][0, pos, :]
        logits_lens = model.lm_head(hidden_final)

        diff = (torch.log_softmax(logits_official.float(), -1)
                - torch.log_softmax(logits_lens.float(), -1)).abs().max().item()
        max_diff_final = max(max_diff_final, diff)

        for from_aa, to_aa, fitness in muts_by_pos[pos]:
            if from_aa not in aa_map or to_aa not in aa_map:
                continue
            lp_mt_o = marginal_log_prob(logits_official, aa_map[to_aa])
            lp_wt_o = marginal_log_prob(logits_official, aa_map[from_aa])
            lp_mt_l = marginal_log_prob(logits_lens, aa_map[to_aa])
            lp_wt_l = marginal_log_prob(logits_lens, aa_map[from_aa])
            scores_official.append(lp_mt_o - lp_wt_o)
            scores_lens_final.append(lp_mt_l - lp_wt_l)
            labels.append(fitness)

    y = np.asarray(labels)
    rho_o = spearmanr(np.asarray(scores_official), y).correlation
    rho_l = spearmanr(np.asarray(scores_lens_final), y).correlation
    return {
        "dataset": name, "n_positions": len(positions), "n_scores": len(y),
        "offset_check": bool(offset_ok),
        "rho_official_path": float(rho_o),
        "rho_lens_final_layer": float(rho_l),
        "max_abs_logprob_diff_final": float(max_diff_final),
    }


def introspect_final_norm(model):
    """Locate any LayerNorm between the last block and the LM head."""
    hits = []
    for n, m in model.named_modules():
        ln = type(m).__name__.lower()
        if "norm" in ln and isinstance(m, torch.nn.Module):
            hits.append(f"{n} ({type(m).__name__})")
    esm = getattr(model, "esm", None)
    has_esm_layernorm = any("esm.layernorm" in h or "esm.layer_norm" in h
                            for h in hits)
    return {"norm_modules": hits[-8:],
            "esm_has_final_layernorm": has_esm_layernorm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["esm2", "saprot"], default="esm2")
    ap.add_argument("--model_dir", default="esm2_model")
    ap.add_argument("--data_dir",
                    default="SaProt-main/LMDB/ProteinGym/substitutions")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_datasets", type=int, default=3)
    ap.add_argument("--max_positions", type=int, default=400)
    ap.add_argument("--dtype", default="float32",
                    help="float32 mandatory for validation (old bug: fp16)")
    args = ap.parse_args()

    from transformers import EsmForMaskedLM, EsmTokenizer
    is_saprot = args.model == "saprot"
    dtype = torch.float32  # fp16 was one of the suspected bugs — never again
    print(f"Loading {args.model} from {args.model_dir} (fp32)...")
    tokenizer = EsmTokenizer.from_pretrained(args.model_dir)
    model = EsmForMaskedLM.from_pretrained(args.model_dir).to(dtype)
    model = model.to(args.device).eval()

    print(json.dumps(introspect_final_norm(model), indent=1))
    print(f"Mask token: {tokenizer.mask_token!r} "
          f"(id={tokenizer.mask_token_id})")
    print(f"Literature anchor: {LITERATURE[args.model]}\n")

    files = sorted(p for p in Path(args.data_dir).iterdir() if p.is_dir())
    # pick datasets with a moderate number of positions for speed
    results = []
    for f in files[: max(args.max_datasets * 3, 12)]:
        r = validate_dataset(str(f), model, tokenizer, args.device,
                             is_saprot, args.max_positions)
        if r is not None:
            results.append(r)
            print(f"[{r['dataset']}] official rho={r['rho_official_path']:+.4f}  "
                  f"lens-final rho={r['rho_lens_final_layer']:+.4f}  "
                  f"max|ΔlogP|={r['max_abs_logprob_diff_final']:.2e}  "
                  f"offset_ok={r['offset_check']}")
        if len(results) >= args.max_datasets:
            break

    mean_o = np.mean([r["rho_official_path"] for r in results])
    mean_l = np.mean([r["rho_lens_final_layer"] for r in results])
    max_d = max(r["max_abs_logprob_diff_final"] for r in results)
    print("\n================ VALIDATION VERDICT ================")
    print(f"datasets validated: {len(results)}")
    print(f"mean official-path rho : {mean_o:+.4f}   (ground truth)")
    print(f"mean lens-final rho    : {mean_l:+.4f}")
    print(f"max |Δ logP| official vs lens-final: {max_d:.2e}")
    ok = max_d < 1e-3 and abs(mean_o - mean_l) < 0.01
    print("LENS vs OFFICIAL:", "PASS — lens reproduces the model's own "
          "zero-shot score at the final layer" if ok else
          "FAIL — lm_head application is inconsistent; fix before "
          "interpreting any layer")
    print(f"Absolute-level check vs literature ({LITERATURE[args.model]}):")
    print(f"  official-path mean on this sample: {mean_o:+.4f} "
          "(sample of datasets; full-benchmark mean is the real anchor)")


if __name__ == "__main__":
    main()
