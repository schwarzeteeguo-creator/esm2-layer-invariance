"""R3 reviewer comment 2: (a) show the previous-version intervals were 95%
percentile bootstrap intervals of the same per-dataset differences; (b) report
how many verdicts sit within 0.001 of the margin and how robust they are to the
interval method; (c) apply the same verdict rule to the single-substitution
comparisons."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision_docs"))
import analysis_r2 as A

def boot_pct(d, lo, hi, seed=A.SEED):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    m = d[rng.randint(0, len(d), (A.N_BOOT, len(d)))].mean(axis=1)
    return [float(np.percentile(m, lo)), float(np.percentile(m, hi))]

def verdict(ci, M=A.MARGIN):
    if ci[0] >= -M and ci[1] <= M: return "Equivalent"
    if ci[0] > M or ci[1] < -M: return "Difference"
    return "Inconclusive"

def best_vs_final(dirs):
    out = []
    for model, mdir in dirs.items():
        for arm in A.ARMS:
            for split in A.LEAK_FREE:
                per = A.load_probing(mdir, arm, split)
                if not per: continue
                lm = {}
                for c in per.values():
                    for l, v in c.items(): lm.setdefault(l, []).append(v)
                lm = {l: np.mean(v) for l, v in lm.items()}
                best = max(lm, key=lm.get)
                if best == "32": continue
                d = np.array([c[best] - c["32"] for c in per.values() if best in c and "32" in c])
                b90 = boot_pct(d, 5, 95); b95 = boot_pct(d, 2.5, 97.5); t90 = A.t_ci90(d)
                out.append(dict(model=model, arm=arm, split=split, best=int(best), n=len(d),
                                mean=float(d.mean()), boot90=b90, boot95=b95, t90=t90,
                                tost_p=A.tost_p(d), v_boot90=verdict(b90), v_t90=verdict(t90),
                                v_tost="Equivalent" if A.tost_p(d) < 0.05 else "Not equivalent",
                                upper_gap_to_margin=float(A.MARGIN - b90[1])))
    return out

full = best_vs_final(A.PROBING_DIRS)
SS = ROOT / "revision_results/single_site_local"
single = best_vs_final({"ESM-2": SS / "probing_v4_single_esm2",
                        "SaProt (mask)": SS / "probing_v4_single_saprot",
                        "SaProt (3Di)": SS / "probing_v4_single_saprot_3di"})
def tally(rows, key):
    from collections import Counter
    return dict(Counter(r[key] for r in rows))
res = {"full": full, "single": single,
       "summary": {"full_n": len(full), "full_boot90": tally(full, "v_boot90"),
                   "full_t90": tally(full, "v_t90"), "full_tost": tally(full, "v_tost"),
                   "full_equiv_within_0.001_of_margin": sum(r["v_boot90"] == "Equivalent" and r["upper_gap_to_margin"] < 0.001 for r in full),
                   "single_n": len(single), "single_boot90": tally(single, "v_boot90"),
                   "single_t90": tally(single, "v_t90"),
                   "single_max_mean": max(r["mean"] for r in single),
                   "single_max_upper90": max(r["boot90"][1] for r in single)}}
dst = ROOT / "revision_results/r3_analysis"; dst.mkdir(exist_ok=True)
json.dump(res, open(dst / "equivalence_check_r3.json", "w"), indent=1)
print(json.dumps(res["summary"], indent=1))
f = lambda c: f"[{c[0]:+.4f},{c[1]:+.4f}]"
print("\nFULL")
for r in full:
    print(f"{r['model']:14s}{r['arm']:12s}{r['split']:20s}L{r['best']} {r['mean']:+.4f} b90{f(r['boot90'])} b95{f(r['boot95'])} t90{f(r['t90'])} {r['v_boot90']:12s} t:{r['v_t90']}")
print("\nSINGLE")
for r in single:
    print(f"{r['model']:14s}{r['arm']:12s}{r['split']:20s}L{r['best']} {r['mean']:+.4f} b90{f(r['boot90'])} {r['v_boot90']}")
