"""Linear CKA + cosine similarity across layers (Reviewer 1, Major 8).

Uncentred cosine between raw hidden states conflates a change of basis
with a change of information. Linear CKA (Kornblith et al., 2019) is
basis-invariant: if distant layers are linearly decodable from one
another, CKA stays high even when raw cosine approaches zero.

For each dataset: one unmasked wild-type forward pass, sample N random
residue positions, collect all-layer hidden states [N, 33, D]; compute
the 33x33 cosine matrix (replicating the original Figure 2 analysis)
AND the 33x33 linear CKA matrix. Average over datasets; save both plus
headline contrasts (adjacent layers, layer 0 vs 32) and the
similarity-vs-distance correlation with an explicit unit of analysis
(unique layer pairs excluding the diagonal — Reviewer 1, Minor 7).

Usage:
  python -m esm_embedding.cka --model esm2 --model_dir esm2_model \
      --data_dir SaProt-main/LMDB/ProteinGym/substitutions \
      --output_dir revision_results/cka_esm2 --n_datasets 10
"""

import os, sys, json, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_embedding.probing_v4 import read_lmdb


def linear_cka(X, Y):
    """Linear CKA between two [n, d] activation matrices (column-centered)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    den = (np.linalg.norm(Xc.T @ Xc, ord="fro")
           * np.linalg.norm(Yc.T @ Yc, ord="fro"))
    return float(num / (den + 1e-12))


def cosine_mean(X, Y):
    """Mean uncentred cosine between matched rows of X and Y [n, d]."""
    num = (X * Y).sum(axis=1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1)
    return float(np.mean(num / (den + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["esm2", "saprot"], default="esm2")
    ap.add_argument("--model_dir", default="esm2_model")
    ap.add_argument("--data_dir",
                    default="SaProt-main/LMDB/ProteinGym/substitutions")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_datasets", type=int, default=10)
    ap.add_argument("--n_positions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import EsmForMaskedLM, EsmTokenizer

    is_saprot = args.model == "saprot"
    tokenizer = EsmTokenizer.from_pretrained(args.model_dir)
    model = (EsmForMaskedLM.from_pretrained(args.model_dir)
             .float().to(args.device).eval())
    n_layers = model.config.num_hidden_layers

    files = sorted(p for p in Path(args.data_dir).iterdir() if p.is_dir())
    rng = np.random.RandomState(args.seed)

    cos_mats, cka_mats = [], []
    used = []
    for f in files:
        if len(used) >= args.n_datasets:
            break
        data = read_lmdb(str(f))
        wt = data.get("wild_type", "")
        if not wt or len(wt) < 60:
            continue
        # sample positions with enough context
        cand = np.arange(2, len(wt) - 1)
        if len(cand) < args.n_positions:
            continue
        pos = rng.choice(cand, args.n_positions, replace=False)
        pos.sort()
        if is_saprot:
            toks = " ".join(f"{aa}#" for aa in wt)  # structure-masked
        else:
            toks = " ".join(list(wt))
        inputs = tokenizer(toks, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = list(out.hidden_states)[1:]  # [33] x [1, T, D]
        acts = np.stack([h[0, pos.astype(int), :].cpu().numpy()
                         for h in hidden])    # [L, N, D]

        cos = np.zeros((n_layers, n_layers))
        cka = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(i, n_layers):
                c = cosine_mean(acts[i], acts[j])
                k = linear_cka(acts[i], acts[j])
                cos[i, j] = cos[j, i] = c
                cka[i, j] = cka[j, i] = k
        cos_mats.append(cos)
        cka_mats.append(cka)
        used.append(f.name)
        print(f"  {f.name}: cos[L0,L32]={cos[0,-1]:+.3f} "
              f"cka[L0,L32]={cka[0,-1]:+.3f}")

    cos_avg = np.mean(cos_mats, axis=0)
    cka_avg = np.mean(cka_mats, axis=0)

    # similarity vs layer distance — explicit unit of analysis:
    # unique unordered pairs (i<j), diagonal excluded
    dists, cos_vals, cka_vals = [], [], []
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            dists.append(j - i)
            cos_vals.append(cos_avg[i, j])
            cka_vals.append(cka_avg[i, j])
    r_cos = float(np.corrcoef(dists, cos_vals)[0, 1])
    r_cka = float(np.corrcoef(dists, cka_vals)[0, 1])

    adj_cos = float(np.mean([cos_avg[i, i + 1] for i in range(n_layers - 1)]))
    adj_cka = float(np.mean([cka_avg[i, i + 1] for i in range(n_layers - 1)]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cosine_matrix_avg.npy", cos_avg)
    np.save(out_dir / "cka_matrix_avg.npy", cka_avg)
    summary = {
        "model": args.model, "n_datasets": len(used),
        "n_positions": args.n_positions, "datasets": used,
        "adjacent_cos": adj_cos, "adjacent_cka": adj_cka,
        "L0_vs_L32_cos": float(cos_avg[0, -1]),
        "L0_vs_L32_cka": float(cka_avg[0, -1]),
        "pearson_r_cos_vs_distance_unique_pairs": r_cos,
        "pearson_r_cka_vs_distance_unique_pairs": r_cka,
        "unit_of_analysis": "unique layer pairs i<j, diagonal excluded",
        "note": "CKA is basis-invariant; high distant-layer CKA with "
                "near-zero distant-layer cosine indicates a change of "
                "coordinates with preserved linear information.",
    }
    with open(out_dir / "cka_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
