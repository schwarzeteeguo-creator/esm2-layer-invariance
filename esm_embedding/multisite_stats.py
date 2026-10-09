"""LMDB multi-site / cap statistics for the Methods section (R2).

Counts, per dataset and pooled: total variants, 10k-variant cap incidence,
multi-site variant and row shares, and how many multi-site variants have
rows landing in more than one fold under the modulo / contiguous schemes
(row fold = its own substitution position, as implemented in pg_splits).

Usage:
  python -m esm_embedding.multisite_stats --data_dir <LMDB substitutions> \
      --out results/r2_multisite_lmdb.json
"""
import sys, json, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_embedding.probing_v4 import (load_dataset_rows, read_lmdb,
                                      parse_mutations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows_out = []
    tot_var = tot_rows = tot_multi = tot_multi_rows = str_m = str_c = 0
    n_capped = n_sub = 0
    for f in sorted(Path(args.data_dir).iterdir()):
        if not f.is_dir():
            continue
        raw = read_lmdb(str(f))
        wt = raw.get("wild_type", "")
        L = len(wt) if isinstance(wt, str) else 0
        variants = []
        for k, e in raw.items():
            if not k.isdigit() or not isinstance(e, dict):
                continue
            subs = parse_mutations(e.get("mut_info", ""))
            try:
                fit = float(e.get("fitness"))
            except (TypeError, ValueError):
                continue
            if subs and np.isfinite(fit):
                variants.append(subs)
        n_var = len(variants)
        capped = n_var > 10000
        multi = [s for s in variants if len(s) >= 2]
        n_rows = sum(len(s) for s in variants)
        n_multi_rows = sum(len(s) for s in multi)
        sm = sc = 0
        for subs in multi:
            pos = [p for (_, p, _) in subs]
            if L:
                sm += len({(p - 1) % 5 for p in pos}) > 1
                sc += len({min(4, int((p - 1) * 5 // L)) for p in pos}) > 1
        ds = load_dataset_rows(str(f), 10000, 5000, 42)
        rows_out.append({"dataset": f.name, "n_variants_total": n_var,
                         "capped_10k": capped,
                         "multi_site_variants": len(multi),
                         "multi_site_rows": n_multi_rows,
                         "rows_after_cap": n_rows,
                         "straddle_modulo": sm, "straddle_contiguous": sc,
                         "analyzed_rows": ds["n_rows"] if ds else None,
                         "analyzed_variants": ds["n_variants"] if ds else None})
        tot_var += n_var
        tot_rows += n_rows
        tot_multi += len(multi)
        tot_multi_rows += n_multi_rows
        str_m += sm
        str_c += sc
        n_capped += capped
        n_sub += (n_rows > 5000)
    summary = {"n_datasets": len(rows_out),
               "capped_10k": n_capped, "subsampled_rows": n_sub,
               "multi_site_variant_pct": 100.0 * tot_multi / max(tot_var, 1),
               "multi_site_row_pct": 100.0 * tot_multi_rows / max(tot_rows, 1),
               "multi_variants_straddle_modulo": str_m,
               "multi_variants_straddle_contiguous": str_c}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows_out},
              open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
