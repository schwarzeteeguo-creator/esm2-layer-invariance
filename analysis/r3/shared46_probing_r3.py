"""R3 minor 7: leakage-free probing (combined features) restricted to the 46
assays shared with ProteinGym v1.3, for like-for-like comparison with the
published supervised anchors. Also reports the single-substitution data."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision_docs"))
import analysis_r2 as A

zs = json.load(open(ROOT / "revision_results/r2_analysis/r2_anchors_zeroshot.json"))
def shared_names(obj):
    if isinstance(obj, dict) and "per_assay" in obj: return list(obj["per_assay"])
    for k in ("shared_assays", "shared", "assays_shared"):
        if isinstance(obj, dict) and k in obj: return obj[k]
    return None
shared = shared_names(zs)
if shared is None:
    raise SystemExit("keys: " + str(list(zs)[:20]))
shared = set(shared)
SS = ROOT / "revision_results/single_site_local"
dirs = {"full": A.PROBING_DIRS,
        "single": {"ESM-2": SS / "probing_v4_single_esm2",
                   "SaProt (mask)": SS / "probing_v4_single_saprot",
                   "SaProt (3Di)": SS / "probing_v4_single_saprot_3di"}}
res = {}
for tag, dd in dirs.items():
    for model, mdir in dd.items():
        for arm in A.ARMS:
            for split in ["random"] + A.LEAK_FREE:
                per = A.load_probing(mdir, arm, split)
                if not per: continue
                for subset, keep in (("all", None), ("shared46", shared)):
                    p = {n: c for n, c in per.items() if keep is None or n in keep}
                    lm = {}
                    for c in p.values():
                        for l, v in c.items(): lm.setdefault(l, []).append(v)
                    lm = {l: float(np.mean(v)) for l, v in lm.items()}
                    best = max(lm, key=lm.get)
                    res[f"{tag}|{subset}|{model}|{arm}|{split}"] = dict(
                        n=len(p), best=int(best), best_rho=lm[best], L32=lm["32"])
dst = ROOT / "revision_results/r3_analysis"; dst.mkdir(exist_ok=True)
json.dump(res, open(dst / "shared46_probing_r3.json", "w"), indent=1)
for k, v in res.items():
    if "|random" in k: continue
    print(f"{k:60s} n={v['n']:2d} L{v['best']:2d} best={v['best_rho']:.3f} L32={v['L32']:.3f}")
