"""R3 reviewer comment 1: share of analysed rows that come from multi-site
variants (counting rows whose sibling rows were dropped by the 5,000-row
subsample), and how many test rows have a sibling row of the same variant in
training under each position-based split."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from esm_embedding.probing_v4 import load_dataset_rows, read_lmdb, parse_mutations
from esm_embedding.pg_splits import make_splits

data_dir = ROOT / "SaProt-main/LMDB/ProteinGym/substitutions"
out = {"per_dataset": {}}
tot = dict(rows=0, rows_multi=0, rows_multi_with_sibling=0, test_rows_shared={})
for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
    ds = load_dataset_rows(str(d))
    if ds is None:
        continue
    raw = read_lmdb(str(d))
    nsub = {}
    for k, e in raw.items():
        if k.isdigit() and isinstance(e, dict):
            nsub[k] = len(parse_mutations(e.get("mut_info", "")))
    vids = ds["variant_ids"]
    is_multi = np.array([nsub.get(v, 1) > 1 for v in vids])
    uniq, cnt = np.unique(vids, return_counts=True)
    c = dict(zip(uniq, cnt))
    has_sib = np.array([c[v] > 1 for v in vids])
    rec = dict(rows=len(vids), rows_multi=int(is_multi.sum()),
               rows_multi_with_sibling=int(has_sib.sum()))
    for scheme in ("modulo", "contiguous", "position_groupkfold"):
        shared = 0
        for tr, te in make_splits(ds["positions"], vids, ds["L"], scheme):
            trv = set(vids[tr].tolist())
            shared += int(sum(v in trv for v in vids[te]))
        rec[f"test_rows_with_variant_in_train_{scheme}"] = shared
        tot["test_rows_shared"][scheme] = tot["test_rows_shared"].get(scheme, 0) + shared
    out["per_dataset"][ds["name"]] = rec
    for k in ("rows", "rows_multi", "rows_multi_with_sibling"):
        tot[k] += rec[k]
tot["n_datasets_with_multi"] = sum(r["rows_multi"] > 0 for r in out["per_dataset"].values())
tot["pct_rows_multi"] = 100 * tot["rows_multi"] / tot["rows"]
tot["pct_rows_multi_with_sibling"] = 100 * tot["rows_multi_with_sibling"] / tot["rows"]
rows_in_multi_ds = sum(r["rows"] for r in out["per_dataset"].values() if r["rows_multi"] > 0)
tot["rows_in_multi_datasets"] = rows_in_multi_ds
out["summary"] = tot
dst = ROOT / "revision_results/r3_analysis"
dst.mkdir(exist_ok=True)
json.dump(out, open(dst / "multisite_share_r3.json", "w"), indent=1)
print(json.dumps(tot, indent=1))
for n, r in out["per_dataset"].items():
    if r["rows_multi"]:
        print(n, r)
