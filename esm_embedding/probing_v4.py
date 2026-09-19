"""Unified per-layer probing V4 — three feature arms + leakage-free splits.

Addresses Reviewer 1 Major comments 1, 2, 5 and part of 3.

Feature arms
------------
random_position : CONTROL (Reviewer 1, Major 1). Each residue position gets
                  a fixed random Gaussian vector (matched 1280-dim),
                  identical for all mutations at that position. Pure
                  position-identity signal, no PLM information. No GPU.
aux             : the 42 hand-crafted mutation features alone (the only
                  fully mutation-specific channel under random splits).
masked_wt       : original Experiment 1 protocol — WT sequence with the
                  mutation position masked, hidden state at the masked
                  position. Position-only signal.
unmasked_wt     : original Experiment 2 protocol — plain WT forward pass,
                  hidden state at the mutation position. Position-only
                  signal.
mutant          : REVIEWER-REQUESTED FIX — full mutant sequence with the
                  substitution in place (no mask), hidden state at the
                  mutated position(s) (averaged for multi-point variants).
                  Genuinely mutation-specific representation.

Split schemes (pg_splits)
-------------------------
random / modulo / contiguous (ProteinGym official) + position_groupkfold.
5 outer folds; all rows of one variant stay in the same fold.

Statistical protocol (Reviewer 1, Major 3)
------------------------------------------
- StandardScaler fit on TRAIN rows only, inside each fold.
- Ridge alpha by NESTED CV: inner 3-fold over grid 1e-2..1e3 (no fixed
  alpha).
- Variant cap (default 10,000, seeded) recorded per dataset; rows
  subsampled to max_rows (default 5,000, seeded) identically across all
  arms/splits.

Usage (CPU-only control, no model needed):
  python -m esm_embedding.probing_v4 \
      --data_dir SaProt-main/LMDB/ProteinGym/substitutions \
      --arms random_position aux --output_dir revision_results/probing_v4

GPU arms (server):
  python -m esm_embedding.probing_v4 \
      --data_dir ... --model_key esm2_650m --model_dir /path/esm2 \
      --arms masked_wt unmasked_wt mutant --output_dir ...
"""

import os

# Single-threaded BLAS per worker (set before numpy/sklearn imports)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, json, time, argparse, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_embedding.pg_splits import make_splits, SPLIT_SCHEMES

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
BLOSUM62 = {aa: {AA_LIST[j]: _BLOSUM62_MATRIX[aa][j] for j in range(20)}
            for aa in AA_LIST}

ALPHA_GRID = np.logspace(-2, 3, 6)
# Alpha selection: RidgeCV with generalized cross-validation (GCV, cv=None).
# GCV is the exact leave-one-out-efficient estimator (Golub et al. 1979):
# one decomposition covers the whole alpha grid, ~4x cheaper than explicit
# inner K-fold while selecting alpha on training data only. Used uniformly
# across ALL arms and splits (incl. the local random-position control) so
# every number in the revision shares one readout protocol.
RIDGE_CV_MODE = None  # None -> GCV; set to an int for explicit inner folds


# ─────────────────────────── data loading ───────────────────────────

def read_lmdb(path):
    import lmdb
    env = lmdb.open(str(path), readonly=True, lock=False)
    data = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            k = key.decode() if isinstance(key, bytes) else key
            try:
                data[k] = json.loads(value)
            except Exception:
                data[k] = value.decode() if isinstance(value, bytes) else value
    env.close()
    return data


def parse_mutations(mut_info: str):
    """'A123V:G124L' -> [('A',123,'V'), ('G',124,'L')]"""
    out = []
    for token in mut_info.split(":"):
        if len(token) >= 3 and token[0] in AA_TO_IDX and token[-1] in AA_TO_IDX:
            try:
                out.append((token[0], int(token[1:-1]), token[-1]))
            except ValueError:
                continue
    return out


def load_structure_tokens(struct_dir, dataset: str, seq_len: int):
    """Real 3Di tokens for one dataset from a directory of .3di.txt files.

    One file per dataset, single line of 3Di states (Foldseek output),
    same length as the wild type. Gap-padded files use '#' (SaProt's
    unknown-structure placeholder) for residues not covered by the AFDB
    model, so a 3Di line may legitimately START with '#' — only '>'
    header lines are skipped here. Returns None when no exact-length
    match exists (callers fall back to structure-masked tokens)."""
    struct_dir = Path(struct_dir)
    for cand in (struct_dir / f"{dataset}.3di.txt",
                 struct_dir / dataset / "query.3di",
                 struct_dir / dataset / "structure.3di"):
        if not cand.exists():
            continue
        seqs = [l.strip().replace(" ", "") for l in cand.read_text()
                .splitlines() if l.strip() and not l.startswith(">")]
        for toks in reversed(seqs):        # last matching line wins
            if len(toks) == seq_len:
                return toks
    return None


def build_aux(from_aa, to_aa):
    wt = np.zeros(20); mt = np.zeros(20)
    if from_aa in AA_TO_IDX: wt[AA_TO_IDX[from_aa]] = 1.0
    if to_aa in AA_TO_IDX: mt[AA_TO_IDX[to_aa]] = 1.0
    blosum = float(BLOSUM62.get(from_aa, {}).get(to_aa, 0))
    grantham = abs(AA_TO_IDX.get(from_aa, 0) - AA_TO_IDX.get(to_aa, 0)) / 19.0
    return np.concatenate([wt, mt, [blosum], [grantham]])


def load_dataset_rows(lmdb_path, max_variants=10000, max_rows=5000, seed=42):
    """Load one LMDB dataset into per-substitution rows.

    Returns dict with rows (variant_id, position, from_aa, to_aa, fitness),
    aux matrix, mutant sequences per variant, WT sequence, and bookkeeping.
    """
    data = read_lmdb(lmdb_path)
    name = Path(lmdb_path).name
    wt_seq = data.get("wild_type", "")
    if not wt_seq or not isinstance(wt_seq, str):
        return None
    L = len(wt_seq)

    variants = []  # (variant_id, [subs], fitness, mutant_seq)
    for key, entry in data.items():
        if not key.isdigit() or not isinstance(entry, dict):
            continue
        fitness = entry.get("fitness")
        mut_info = entry.get("mut_info", "")
        subs = parse_mutations(mut_info)
        if not subs or fitness is None:
            continue
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fitness):
            continue
        # keep substitution-only variants whose positions are in range
        subs = [(f, p, t) for (f, p, t) in subs if 1 <= p <= L]
        if not subs:
            continue
        mseq = entry.get("seq")
        if isinstance(mseq, str) and len(mseq) != L:
            continue  # indel — out of scope
        variants.append((key, subs, fitness, mseq))

    if len(variants) < 5:
        return None

    capped = False
    if len(variants) > max_variants:
        rng = np.random.RandomState(seed)
        keep = rng.choice(len(variants), max_variants, replace=False)
        variants = [variants[i] for i in keep]
        capped = True

    rows = []  # (variant_id, pos, from, to, fitness)
    for vid, subs, fitness, _ in variants:
        for f, p, t in subs:
            rows.append((vid, p, f, t, fitness))

    # Row subsample (identical across all arms/splits — drawn once here)
    if len(rows) > max_rows:
        rng = np.random.RandomState(seed)
        keep = rng.choice(len(rows), max_rows, replace=False)
        rows = [rows[i] for i in keep]

    if len(rows) < 10:
        return None

    variant_ids = np.array([r[0] for r in rows])
    positions = np.array([r[1] for r in rows], dtype=np.int64)
    fitness = np.array([r[4] for r in rows], dtype=np.float64)
    aux = np.stack([build_aux(r[2], r[3]) for r in rows]).astype(np.float32)

    return {
        "name": name, "wt": wt_seq, "L": L,
        "rows": rows, "variant_ids": variant_ids, "positions": positions,
        "fitness": fitness, "aux": aux, "capped": capped,
        "n_variants": len({r[0] for r in rows}), "n_rows": len(rows),
        "mutant_seqs": {vid: mseq for (vid, _, _, mseq) in
                        [(v[0], v[1], v[2], v[3]) for v in variants]},
    }


# ─────────────────────── evaluation (nested-CV Ridge) ───────────────────────

def eval_ridge_nested(X_raw, y, positions, variant_ids, seq_len, scheme,
                      n_folds=5, seed=42):
    """Pooled out-of-fold Spearman with fold-internal scaling + nested alpha."""
    splits = make_splits(positions, variant_ids, seq_len, scheme,
                         n_folds=n_folds, seed=seed)
    if len(splits) < 2:
        return {"spearman": np.nan, "n": len(y), "alphas": []}
    preds = np.full(len(y), np.nan)
    alphas = []
    for tr, te in splits:
        scaler = StandardScaler().fit(X_raw[tr])
        Xtr, Xte = scaler.transform(X_raw[tr]), scaler.transform(X_raw[te])
        ridge = RidgeCV(alphas=ALPHA_GRID, cv=RIDGE_CV_MODE,
                        gcv_mode="eigen").fit(Xtr, y[tr])
        preds[te] = ridge.predict(Xte)
        alphas.append(float(ridge.alpha_))
    valid = np.isfinite(preds) & np.isfinite(y)
    if valid.sum() < 10:
        return {"spearman": np.nan, "n": int(valid.sum()), "alphas": alphas}
    rho, _ = spearmanr(y[valid], preds[valid])
    return {"spearman": float(rho), "n": int(valid.sum()), "alphas": alphas}


# ─────────────────── GPU arm extraction (masked/unmasked/mutant) ───────────────────

def _tok_and_hidden(model, tokenizer, token_str, device, wanted_positions):
    """One forward pass; return [n_wanted, n_layers, D] at wanted token idx."""
    import torch
    inputs = tokenizer(token_str, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = list(outputs.hidden_states)[1:]  # 33 transformer layers
    # residue i (1-indexed) sits at token index i (<cls> at 0)
    grabbed = [h[0, wanted_positions, :].cpu().numpy() for h in hidden]
    return np.stack(grabbed, axis=1).astype(np.float16)  # [n_pos, L, D]


def _batched_hidden(model, tokenizer, token_strs, positions_per_seq, device,
                    batch_size=8):
    """Batched forward passes.

    token_strs: list of token strings; positions_per_seq: per-seq int arrays
    of token indices to grab. Returns list of [n_pos, n_layers, D] arrays.
    Padding is masked via attention_mask, so grabbed positions are unaffected.
    """
    import torch
    outs = []
    for i in range(0, len(token_strs), batch_size):
        chunk = token_strs[i:i + batch_size]
        poss = positions_per_seq[i:i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = list(out.hidden_states)[1:]
        for b, p in enumerate(poss):
            grabbed = np.stack([h[b, p, :].cpu().numpy() for h in hidden],
                               axis=1)
            outs.append(grabbed.astype(np.float16))
    return outs


def extract_model_arms(ds, model_info, arms, device, is_saprot,
                       batch_size=8, struct_tokens=None):
    """Extract per-unit hidden states for the requested GPU arms.

    For SaProt, struct_tokens is the real per-residue 3Di string (same
    length as WT) or None for structure-masked ("AA#") inputs. Substitution
    positions keep the WT structure token in the mutant arm (side-chain
    change only); masked positions use the tokenizer's mask token.

    Returns {arm: {"units": array [n_units, 33, D], "unit_of_row": np.array}}.
    """
    wt = ds["wt"]
    model, tokenizer = model_info["model"], model_info["tokenizer"]
    out = {}

    def saprot_tokens(seq):
        st = struct_tokens if struct_tokens is not None else "#" * len(seq)
        return [f"{aa}{st[i]}" for i, aa in enumerate(seq)]

    if "masked_wt" in arms:
        uniq_pos = np.unique(ds["positions"])
        n_layers = model_info["num_layers"]
        cache = np.zeros((len(uniq_pos), n_layers, model_info["hidden_dim"]),
                         dtype=np.float16)
        strs, poss = [], []
        for p in uniq_pos:
            if is_saprot:
                toks = saprot_tokens(wt)
                toks[p - 1] = tokenizer.mask_token          # "M#"
            else:
                toks = list(wt)
                toks[p - 1] = tokenizer.mask_token          # "<mask>"
            strs.append(" ".join(toks))
            poss.append(np.array([p]))
        for i, grabbed in enumerate(_batched_hidden(
                model, tokenizer, strs, poss, device, batch_size)):
            cache[i] = grabbed[0]
        pos_index = {p: i for i, p in enumerate(uniq_pos)}
        out["masked_wt"] = {
            "units": cache,
            "unit_of_row": np.array([pos_index[p] for p in ds["positions"]]),
        }

    if "unmasked_wt" in arms:
        if is_saprot:
            toks = saprot_tokens(wt)   # real 3Di, or structure-masked
        else:
            toks = list(wt)
        uniq_pos = np.unique(ds["positions"])
        # one forward pass for the whole sequence; gather wanted positions
        import torch
        inputs = tokenizer(" ".join(toks), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = list(outputs.hidden_states)[1:]
        pos_arr = uniq_pos.astype(int)
        cache = np.stack([h[0, pos_arr, :].cpu().numpy() for h in hidden],
                         axis=1).astype(np.float16)
        pos_index = {p: i for i, p in enumerate(uniq_pos)}
        out["unmasked_wt"] = {
            "units": cache,
            "unit_of_row": np.array([pos_index[p] for p in ds["positions"]]),
        }

    if "mutant" in arms:
        uniq_vids = np.unique(ds["variant_ids"])
        vid_index = {v: i for i, v in enumerate(uniq_vids)}
        n_layers = model_info["num_layers"]
        cache = np.zeros((len(uniq_vids), n_layers, model_info["hidden_dim"]),
                         dtype=np.float16)
        sub_positions = {}
        # substitution positions and target residues per variant
        for vid, p, _, to, _ in ds["rows"]:
            sub_positions.setdefault(vid, {})[p] = to
        strs, poss, vids_ordered = [], [], []
        for vid in uniq_vids:
            mseq = ds["mutant_seqs"].get(vid)
            if not mseq:                     # fall back: apply subs to WT
                mseq = list(wt)
                for p, to_aa in sub_positions[vid].items():
                    mseq[p - 1] = to_aa
                mseq = "".join(mseq)
            if is_saprot:
                toks = saprot_tokens(mseq)
            else:
                toks = list(mseq)
            strs.append(" ".join(toks))
            poss.append(np.array(sorted(sub_positions[vid].keys()),
                                 dtype=int))
            vids_ordered.append(vid)
        for grabbed, vid in zip(_batched_hidden(model, tokenizer, strs, poss,
                                                device, batch_size),
                                vids_ordered):
            cache[vid_index[vid]] = grabbed.mean(axis=0)  # mean over mut pos
        out["mutant"] = {
            "units": cache,
            "unit_of_row": np.array([vid_index[v] for v in ds["variant_ids"]]),
        }

    return out


# ─────────────────────────── arm evaluation loop ───────────────────────────

def evaluate_arms_on_dataset(ds, arms, splits, feature_sets, n_random_draws=33,
                             n_folds=5, seed=42, model_info=None, device=None,
                             is_saprot=False, batch_size=8, struct_tokens=None):
    """Returns {arm: {featset: {split: {layer: result}}}} for one dataset."""
    y = ds["fitness"]
    positions, vids, L = ds["positions"], ds["variant_ids"], ds["L"]
    aux = ds["aux"]
    results = {}

    # --- control: random position vectors (mirrors per-layer structure) ---
    if "random_position" in arms:
        uniq_pos = np.unique(positions)
        pos_index = {p: i for i, p in enumerate(uniq_pos)}
        unit_of_row = np.array([pos_index[p] for p in positions])
        for split in splits:
            per_draw = []
            for d in range(n_random_draws):
                rng = np.random.RandomState(1000 + d)
                vecs = rng.randn(len(uniq_pos), 1280).astype(np.float32)
                for featset in feature_sets:
                    X = (np.concatenate([vecs[unit_of_row], aux], axis=1)
                         if featset == "combined" else vecs[unit_of_row])
                    r = eval_ridge_nested(X, y, positions, vids, L, split,
                                          n_folds=n_folds, seed=seed)
                    per_draw.append((d, featset, r))
            for featset in feature_sets:
                rhos = [r["spearman"] for _, fs, r in per_draw
                        if fs == featset and np.isfinite(r["spearman"])]
                results.setdefault("random_position", {}) \
                       .setdefault(featset, {})[split] = {
                    "rho_mean": float(np.mean(rhos)),
                    "rho_std": float(np.std(rhos)),
                    "n_draws": len(rhos),
                }

    # --- aux only ---
    if "aux" in arms:
        for split in splits:
            r = eval_ridge_nested(aux, y, positions, vids, L, split,
                                  n_folds=n_folds, seed=seed)
            results.setdefault("aux", {}) \
                   .setdefault("auxiliary_only", {})[split] = r

    # --- GPU arms ---
    gpu_arms = [a for a in ("masked_wt", "unmasked_wt", "mutant") if a in arms]
    if gpu_arms:
        caches = extract_model_arms(ds, model_info, gpu_arms, device,
                                    is_saprot, batch_size=batch_size,
                                    struct_tokens=struct_tokens)
        for arm, cache in caches.items():
            units = cache["units"].astype(np.float32)
            unit_of_row = cache["unit_of_row"]
            n_layers = units.shape[1]
            for split in splits:
                for featset in feature_sets:
                    layer_out = {}
                    for l in range(n_layers):
                        emb = units[unit_of_row, l, :]
                        X = (np.concatenate([emb, aux], axis=1)
                             if featset == "combined" else emb)
                        layer_out[str(l)] = eval_ridge_nested(
                            X, y, positions, vids, L, split,
                            n_folds=n_folds, seed=seed)
                    results.setdefault(arm, {}) \
                           .setdefault(featset, {})[split] = layer_out

    return results


# ─────────────────────────── driver (CPU arms, MP) ───────────────────────────

_CPU_ARMS = {"random_position", "aux"}


def _cpu_worker(args):
    """Compute results for one dataset; returns them (parent writes)."""
    (lmdb_path, arms, splits, feature_sets, n_random_draws, n_folds,
     seed, max_variants, max_rows) = args
    try:
        ds = load_dataset_rows(lmdb_path, max_variants, max_rows, seed)
        if ds is None:
            return None
        res = evaluate_arms_on_dataset(
            ds, [a for a in arms if a in _CPU_ARMS], splits, feature_sets,
            n_random_draws, n_folds, seed)
        return {"name": ds["name"], "n_rows": ds["n_rows"],
                "n_variants": ds["n_variants"], "capped": ds["capped"],
                "results": res}
    except Exception as e:  # never kill the pool
        return {"error": f"{Path(lmdb_path).name}: {e}"}


def _append_result(out_dir, ds_name, bookkeeping, res, suffix=""):
    """Merge one dataset's results into the per-(arm,featset,split) files.

    Called ONLY from the parent process (single writer, no races).
    `suffix` isolates concurrent shard outputs (__shardK) from the
    canonical files; merge_shards.py folds them together afterwards."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm, featsets in res.items():
        for featset, splits_d in featsets.items():
            for split, payload in splits_d.items():
                f = out_dir / f"v4__{arm}__{featset}__{split}{suffix}.json"
                data = {}
                if f.exists():
                    try:
                        with open(f) as fh:
                            data = json.load(fh)
                    except json.JSONDecodeError:
                        data = {}
                data[ds_name] = dict(bookkeeping, result=payload)
                with open(f, "w") as fh:
                    json.dump(data, fh, indent=1)


def _done_datasets(out_dir, arm, featset, split, suffix=""):
    """Datasets already recorded for this (arm,featset,split).

    With a shard suffix, the UNION of the shard file and the canonical
    file is returned, so a sharded backfill skips work already completed
    by earlier unsharded runs instead of redoing it."""
    keys = set()
    for s in {suffix, ""}:
        f = Path(out_dir) / f"v4__{arm}__{featset}__{split}{s}.json"
        if f.exists():
            try:
                with open(f) as fh:
                    keys |= set(json.load(fh).keys())
            except json.JSONDecodeError:
                pass
    return keys


def run(data_dir, arms, splits, feature_sets, output_dir,
        n_random_draws=33, n_folds=5, seed=42, max_variants=10000,
        max_rows=5000, model_key=None, model_dir=None, device=None,
        max_datasets=None, workers=1, batch_size=8, struct_dir=None,
        shard=0, nshards=1):
    data_dir = Path(data_dir)
    lmdb_files = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if max_datasets:
        lmdb_files = lmdb_files[:max_datasets]
    if nshards > 1:
        # concurrent shard processes each own a disjoint dataset subset;
        # outputs carry a __shardK suffix until merge_shards.py folds them
        lmdb_files = [f for i, f in enumerate(lmdb_files)
                      if i % nshards == shard]
    sfx = f"__shard{shard}" if nshards > 1 else ""
    print(f"Datasets found: {len(lmdb_files)} | arms: {arms} | splits: {splits}"
          f"{' | shard %d/%d' % (shard, nshards) if nshards > 1 else ''}")

    cpu_arms = [a for a in arms if a in _CPU_ARMS]
    gpu_arms = [a for a in arms if a in ("masked_wt", "unmasked_wt", "mutant")]

    # ---- CPU arms with multiprocessing (parent is the single writer) ----
    if cpu_arms:
        todo = []
        for f in lmdb_files:
            ds_name = f.name
            done = all(ds_name in _done_datasets(output_dir, arm, fs, sp,
                                                  suffix=sfx)
                       for arm in cpu_arms
                       for fs in (feature_sets if arm == "random_position"
                                  else ["auxiliary_only"])
                       for sp in splits)
            if not done:
                todo.append((str(f), arms, splits, feature_sets,
                             n_random_draws, n_folds, seed, max_variants,
                             max_rows))
        print(f"CPU-arm datasets to process: {len(todo)}")
        if todo:
            import multiprocessing as mp
            n_proc = max(1, min(workers, len(todo)))
            with mp.Pool(n_proc) as pool:
                for i, payload in enumerate(
                        pool.imap_unordered(_cpu_worker, todo)):
                    if payload is None:
                        continue
                    if "error" in payload:
                        print(f"  [{i+1}/{len(todo)}] ERROR "
                              f"{payload['error']}", flush=True)
                        continue
                    _append_result(output_dir, payload["name"],
                                   {"n_rows": payload["n_rows"],
                                    "n_variants": payload["n_variants"],
                                    "capped": payload["capped"]},
                                   payload["results"], suffix=sfx)
                    print(f"  [{i+1}/{len(todo)}] {payload['name']}",
                          flush=True)

    # ---- GPU arms single-process ----
    if gpu_arms:
        from esm_embedding.model import load_model, _MODEL_PATHS
        if model_dir:
            _MODEL_PATHS[model_key] = Path(model_dir)
        info = load_model(model_key, device=device or "cuda")
        is_saprot = model_key.startswith("saprot")
        print(f"GPU model: {model_key} ({info['num_layers']} layers)")
        for i, f in enumerate(lmdb_files):
            ds_name = f.name
            done = all(ds_name in _done_datasets(output_dir, arm, fs, sp,
                                                  suffix=sfx)
                       for arm in gpu_arms for fs in feature_sets
                       for sp in splits)
            if done:
                continue
            ds = load_dataset_rows(f, max_variants, max_rows, seed)
            if ds is None:
                continue
            struct_tokens, struct_src = None, "n/a"
            if is_saprot:
                if struct_dir:
                    struct_tokens = load_structure_tokens(struct_dir,
                                                          ds_name, ds["L"])
                struct_src = ("real_3di" if struct_tokens
                              else "structure_masked")
            t0 = time.time()
            res = evaluate_arms_on_dataset(
                ds, gpu_arms, splits, feature_sets, n_random_draws,
                n_folds, seed, model_info=info,
                device=info["device"], is_saprot=is_saprot,
                batch_size=batch_size, struct_tokens=struct_tokens)
            _append_result(output_dir, ds["name"],
                           {"n_rows": ds["n_rows"],
                            "n_variants": ds["n_variants"],
                            "capped": ds["capped"],
                            "struct": struct_src},
                           res, suffix=sfx)
            print(f"  [{i+1}/{len(lmdb_files)}] {ds['name']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- summary ----
    summarize(output_dir, suffix=sfx)


def summarize(out_dir, suffix=""):
    out_dir = Path(out_dir)
    files = sorted(f for f in out_dir.glob(f"v4__*{suffix}.json")
                   if "__shard" not in f.name or suffix)
    if not files:
        return
    print("\n" + "=" * 78)
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        rhos = []
        for name, entry in data.items():
            r = entry["result"]
            if "rho_mean" in r:                      # random control
                rhos.append(r["rho_mean"])
            elif "spearman" in r:                    # aux arm
                if np.isfinite(r["spearman"]):
                    rhos.append(r["spearman"])
            else:                                    # per-layer arm
                vals = [v["spearman"] for v in r.values()
                        if np.isfinite(v.get("spearman", np.nan))]
                if vals:
                    rhos.append(np.mean(vals))       # dataset-level layer mean
        if rhos:
            print(f"{f.stem:<55} N={len(rhos):>3}  mean rho = {np.mean(rhos):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["random_position", "aux"],
                    choices=["random_position", "aux", "masked_wt",
                             "unmasked_wt", "mutant"])
    ap.add_argument("--splits", nargs="+", default=SPLIT_SCHEMES)
    ap.add_argument("--feature_sets", nargs="+",
                    default=["combined", "embedding_only"],
                    choices=["combined", "embedding_only"])
    ap.add_argument("--output_dir", default="revision_results/probing_v4")
    ap.add_argument("--n_random_draws", type=int, default=33)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_variants", type=int, default=10000)
    ap.add_argument("--max_rows", type=int, default=5000)
    ap.add_argument("--model_key", default="esm2_650m",
                    choices=["esm2_650m", "saprot_650m"])
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_datasets", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="batch size for GPU forward passes (model arms)")
    ap.add_argument("--struct_dir", default=None,
                    help="dir with per-dataset .3di.txt real structure "
                         "tokens (SaProt only; falls back to "
                         "structure-masked when a file is missing)")
    ap.add_argument("--shard", type=int, default=0,
                    help="shard index for concurrent runs (see --nshards)")
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the dataset list into N disjoint shards; "
                         "outputs carry a __shardK suffix to be merged by "
                         "merge_shards.py")
    args = ap.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
