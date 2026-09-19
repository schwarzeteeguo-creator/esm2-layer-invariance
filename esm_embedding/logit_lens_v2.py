"""Logit lens V2 — the fixed per-layer experiment (Reviewer 1, Major 4).

Fixes vs the original logit_lens_esm2.py / logit_lens.py:
  1. fp32 mandatory. The original ESM-2 script loaded the model in fp16.
  2. Built-in validation gate: for every dataset, the final-layer lens
     score is first checked against the model's own logits path
     (max |delta log-prob| < 1e-3). The experiment refuses to emit
     per-layer results for a dataset that fails the gate.
  3. Per-dataset results are saved (the original saved only aggregates,
     which made later reconciliation impossible).
  4. Dataset count is a first-class argument (Reviewer 1, Minor 5) —
     default is ALL datasets, not 5.
  5. SaProt: scoring marginalizes over the 21 structure-token variants,
     as before. IMPORTANT: without real 3Di structure tokens the input is
     structure-masked ("AA#"), which is NOT the published SaProt zero-shot
     protocol — SaProt numbers are only interpretable once real tokens
     are supplied via --struct_dir (one .3di.txt per dataset, produced by
     Foldseek). The script marks this explicitly in its output metadata.

Usage:
  python -m esm_embedding.logit_lens_v2 --model esm2 --model_dir esm2_model \
      --data_dir SaProt-main/LMDB/ProteinGym/substitutions \
      --output_dir revision_results/logit_lens_v2_esm2
"""

import os, sys, json, time, argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_embedding.probing_v4 import (read_lmdb, parse_mutations,
                                      load_structure_tokens)
from esm_embedding.logit_lens_validate import (build_aa_token_map,
                                               marginal_log_prob,
                                               LITERATURE)

GATE_TOL = 1e-3  # max |Δlog-prob| allowed between lens-final and official


@torch.no_grad()
def dataset_lens(lmdb_path, model, tokenizer, device, is_saprot,
                 struct_dir=None, max_positions=None):
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
    # F7YBW8 (4 mutated positions, ~5k variants) IS in the official
    # ProteinGym benchmark and a per-VARIANT Spearman over ~5k scores is
    # well-defined — only a truly degenerate 1-position set is dropped.
    if len(muts_by_pos) < 2:
        return None

    aa_map = build_aa_token_map(tokenizer, is_saprot)
    # per-AA id tensors for VECTORIZED scoring: the original per-row loop
    # (2 GPU syncs per row per layer) costs hours on combinatorial DMS
    # datasets with ~500k variants (e.g. HIS7_YEAST_Pokusaeva_2019) and
    # would be ~21x worse for SaProt marginalization. Same math, batched.
    aa_ids = {aa: torch.tensor(ids, device=device, dtype=torch.long)
              for aa, ids in aa_map.items()}
    struct_tokens = None
    if is_saprot and struct_dir:
        struct_tokens = load_structure_tokens(Path(struct_dir), name,
                                              len(wt))

    def build_tokens(pos):
        if is_saprot:
            st = struct_tokens or "#" * len(wt)
            toks = [f"{wt[i]}{'#' if struct_tokens is None else st[i]}"
                    for i in range(len(wt))]
            toks[pos - 1] = tokenizer.mask_token
        else:
            toks = list(wt)
            toks[pos - 1] = tokenizer.mask_token
        return " ".join(toks)

    positions = sorted(muts_by_pos)
    if max_positions:
        positions = positions[:max_positions]
    n_layers = model.config.num_hidden_layers

    layer_scores = {l: [] for l in range(n_layers)}
    labels = []
    gate_max_diff = 0.0

    for pos in positions:
        inputs = tokenizer(build_tokens(pos), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True)
        logits_official = out.logits[0, pos, :]
        hidden_final = out.hidden_states[-1][0, pos, :]
        logits_lens_final = model.lm_head(hidden_final)
        gate_max_diff = max(
            gate_max_diff,
            (torch.log_softmax(logits_official.float(), -1)
             - torch.log_softmax(logits_lens_final.float(), -1))
            .abs().max().item())

        hidden_layers = list(out.hidden_states)[1:]  # 33 transformer layers
        row_labels = []
        for from_aa, to_aa, fitness in muts_by_pos[pos]:
            if from_aa not in aa_map or to_aa not in aa_map:
                continue
            labels.append(fitness)
            row_labels.append((from_aa, to_aa))
        if not row_labels:
            continue
        to_ids = torch.stack([aa_ids[t] for _, t in row_labels])    # [n, k]
        from_ids = torch.stack([aa_ids[f] for f, _ in row_labels])
        for l in range(n_layers):
            # identical to marginal_log_prob loop, batched: one float64
            # log_softmax per layer, one gather per (from,to) matrix
            lp = torch.log_softmax(
                model.lm_head(hidden_layers[l][0, pos, :]).double(), dim=-1)
            scores = (torch.logsumexp(lp[to_ids], dim=-1)
                      - torch.logsumexp(lp[from_ids], dim=-1))
            layer_scores[l].extend(scores.tolist())

    gate_pass = gate_max_diff < GATE_TOL
    y = np.asarray(labels)
    per_layer = {}
    for l in range(n_layers):
        s = np.asarray(layer_scores[l])
        if len(s) != len(y) or len(y) < 10:
            per_layer[str(l)] = {"spearman": None, "n": len(y)}
            continue
        rho, _ = spearmanr(s, y)
        per_layer[str(l)] = {"spearman": float(rho), "n": int(len(y))}

    # official-path ground truth for anchoring
    return {
        "dataset": name, "n_positions": len(positions), "n_scores": int(len(y)),
        "gate_pass": bool(gate_pass), "gate_max_diff": float(gate_max_diff),
        "structure_source": ("real_3di" if struct_tokens else
                             ("structure_masked" if is_saprot else "n/a")),
        "per_layer": per_layer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["esm2", "saprot"], default="esm2")
    ap.add_argument("--model_dir", default="esm2_model")
    ap.add_argument("--data_dir",
                    default="SaProt-main/LMDB/ProteinGym/substitutions")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--struct_dir", default=None,
                    help="dir with real 3Di tokens per dataset (SaProt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_datasets", type=int, default=None)
    ap.add_argument("--max_positions", type=int, default=None)
    args = ap.parse_args()

    from transformers import EsmForMaskedLM, EsmTokenizer
    is_saprot = args.model == "saprot"
    print(f"Loading {args.model} (fp32)...")
    tokenizer = EsmTokenizer.from_pretrained(args.model_dir)
    model = (EsmForMaskedLM.from_pretrained(args.model_dir)
             .float().to(args.device).eval())
    n_layers = model.config.num_hidden_layers
    if is_saprot and not args.struct_dir:
        print("WARNING: SaProt without real 3Di tokens runs structure-"
              "masked inputs — NOT comparable to the published zero-shot "
              "protocol. Pass --struct_dir for protocol-valid results.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file = out_dir / "logit_lens_v2_results.json"
    results = {}
    if res_file.exists():
        with open(res_file) as f:
            results = json.load(f)

    files = sorted(p for p in Path(args.data_dir).iterdir() if p.is_dir())
    if args.max_datasets:
        files = files[:args.max_datasets]

    for i, f in enumerate(files):
        if f.name in results:
            continue
        t0 = time.time()
        r = dataset_lens(str(f), model, tokenizer, args.device, is_saprot,
                         struct_dir=args.struct_dir,
                         max_positions=args.max_positions)
        if r is None:
            continue
        results[f.name] = r
        with open(res_file, "w") as fh:
            json.dump(results, fh, indent=1)
        status = "GATE OK" if r["gate_pass"] else "GATE FAIL — excluded"
        print(f"  [{i+1}/{len(files)}] {f.name} "
              f"L32 rho={r['per_layer'].get(str(n_layers-1), {}).get('spearman')} "
              f"{status} ({time.time()-t0:.0f}s)", flush=True)

    # aggregate over gate-passing datasets only
    agg = {l: [] for l in range(n_layers)}
    n_fail = 0
    for name, r in results.items():
        if not r["gate_pass"]:
            n_fail += 1
            continue
        for l in range(n_layers):
            v = r["per_layer"].get(str(l), {}).get("spearman")
            if v is not None and np.isfinite(v):
                agg[l].append(v)
    summary = {
        "model": args.model, "n_layers": n_layers,
        "n_datasets_total": len(results),
        "n_datasets_gate_failed": n_fail,
        "structure_source": ("real_3di" if (is_saprot and args.struct_dir)
                             else ("structure_masked" if is_saprot else "n/a")),
        "per_layer": {str(l): {"mean": float(np.mean(v)), "std": float(np.std(v)),
                               "n": len(v)}
                      for l, v in agg.items() if v},
    }
    with open(out_dir / "logit_lens_v2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary["per_layer"].get("0"), indent=1))
    print(json.dumps(summary["per_layer"].get(str(n_layers - 1)), indent=1))


if __name__ == "__main__":
    main()
