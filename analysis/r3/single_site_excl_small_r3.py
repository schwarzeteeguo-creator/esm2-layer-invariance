"""R3 sensitivity: single-substitution probing after excluding the two datasets
that retain fewer than 50 single-substitution variants (F7YBW8_MESOW_Aakre_2015,
37; GCN4_YEAST_Staller_induction_2018, 33). Writes
revision_results/r3_analysis/single_site_excl_small_r3.json."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision_docs"))
import analysis_r2 as A

SS = ROOT / "revision_results/single_site_local"
SINGLE = {"ESM-2": SS / "probing_v4_single_esm2",
          "SaProt (mask)": SS / "probing_v4_single_saprot",
          "SaProt (3Di)": SS / "probing_v4_single_saprot_3di"}
DROP = {"F7YBW8_MESOW_Aakre_2015", "GCN4_YEAST_Staller_induction_2018"}


def boot(d):
    r = np.random.RandomState(A.SEED)
    m = d[r.randint(0, len(d), (A.N_BOOT, len(d)))].mean(1)
    return [float(x) for x in np.percentile(m, [5, 95])]


rows = {}
for model, dd in SINGLE.items():
    for arm in A.ARMS:
        for sp in A.LEAK_FREE:
            per = {n: c for n, c in A.load_probing(dd, arm, sp).items() if n not in DROP}
            lm = {}
            for c in per.values():
                for l, v in c.items():
                    lm.setdefault(l, []).append(v)
            lm = {l: float(np.mean(v)) for l, v in lm.items()}
            b = max(lm, key=lm.get)
            rec = dict(n=len(per), best=int(b), best_rho=lm[b], L32=lm["32"])
            if b != "32":
                d = np.array([c[b] - c["32"] for c in per.values() if b in c and "32" in c])
                rec.update(mean=float(d.mean()), boot90=boot(d))
            rows[f"{model}|{arm}|{sp}"] = rec
tested = [r for r in rows.values() if "mean" in r]
summary = dict(max_mean=max(r["mean"] for r in tested), max_upper=max(r["boot90"][1] for r in tested),
               best_layers=sorted({r["best"] for r in rows.values()}))
out = dict(dropped=sorted(DROP), summary=summary, rows=rows)
json.dump(out, open(ROOT / "revision_results/r3_analysis/single_site_excl_small_r3.json", "w"), indent=1)
print(json.dumps(summary))
