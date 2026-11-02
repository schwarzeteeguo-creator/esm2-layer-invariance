"""R3 comment 1 (pooling): paired fusion-minus-last-layer differences under the
modulo split after excluding the datasets that contain multi-site variants
(A4_HUMAN_Seuma_2021 and DLG4_HUMAN_Faure_2021 in the 20-dataset subset)."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
d = json.load(open(ROOT / "revision_results/r2_analysis/a2_a3_pooling.json"))
ms = json.load(open(ROOT / "revision_results/r3_analysis/multisite_share_r3.json"))["per_dataset"]
multi = {n for n, r in ms.items() if r["rows_multi"] > 0}
def boot(x, seed=42):
    r = np.random.RandomState(seed)
    m = x[r.randint(0, len(x), (10000, len(x)))].mean(1)
    return [float(v) for v in np.percentile(m, [5, 95])]
out = {}
for k in ("ESM-2|modulo", "SaProt|modulo"):
    out[k] = {}
    for s, r in d["fusion_vs_last"][k].items():
        x = np.array([v for n, v in r["per_dataset"].items() if n not in multi])
        out[k][s] = {"n": len(x), "mean_delta": float(x.mean()), "boot_ci90": boot(x),
                     "mean_delta_all20": r["mean_delta"]}
json.dump(out, open(ROOT / "revision_results/r3_analysis/pooling_excl_multisite_r3.json", "w"), indent=1)
for k, v in out.items():
    better = sum(r["boot_ci90"][0] > 0 for r in v.values())
    worse = sum(r["boot_ci90"][1] < 0 for r in v.values())
    print(k, "n=18; fusion significantly better:", better, "worse:", worse)
