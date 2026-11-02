"""All numbers for the R3 manuscript (plan A).

Primary analysis: the wild-type arms use the full data in every split; the
mutant arm uses the full data under the random split (whole variants share a
fold) and single-substitution variants only under the position-based splits,
where rows of a multi-site variant would share one mutant-arm embedding and one
label across folds (reviewer R2.1).

Writes revision_results/r3_analysis/r3_numbers.json.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision_docs"))
import analysis_r2 as A

SS = ROOT / "revision_results/single_site_local"
FULL = A.PROBING_DIRS
SINGLE = {"ESM-2": SS / "probing_v4_single_esm2",
          "SaProt (mask)": SS / "probing_v4_single_saprot",
          "SaProt (3Di)": SS / "probing_v4_single_saprot_3di"}
MODELS = list(FULL)
SPLITS = ["random"] + A.LEAK_FREE
SHARED = set(json.load(open(ROOT / "revision_results/r2_analysis/r2_anchors_zeroshot.json"))["per_assay"])


def primary_dir(model, arm, split):
    if arm == "mutant" and split != "random":
        return SINGLE[model], "single"
    return FULL[model], "full"


def curve_stats(per):
    lm = {}
    for c in per.values():
        for l, v in c.items():
            lm.setdefault(l, []).append(v)
    lm = {l: float(np.mean(v)) for l, v in lm.items()}
    best = max(lm, key=lm.get)
    return lm, best


def boot(d, lo=5, hi=95):
    r = np.random.RandomState(A.SEED)
    m = d[r.randint(0, len(d), (A.N_BOOT, len(d)))].mean(1)
    return [float(np.percentile(m, lo)), float(np.percentile(m, hi))]


def verdict(ci, M=A.MARGIN):
    if ci[0] >= -M and ci[1] <= M:
        return "Equivalent"
    if ci[0] > M or ci[1] < -M:
        return "Difference"
    return "Inconclusive"


out = {"table2": {}, "equivalence": [], "degenerate": [], "table4": {}, "ranges": {},
       "anchors_this_work": {}, "inflation": {}}

# ---- Table 2 and equivalence (primary analysis)
for model in MODELS:
    for arm in A.ARMS:
        for split in SPLITS:
            mdir, src = primary_dir(model, arm, split)
            per = A.load_probing(mdir, arm, split)
            lm, best = curve_stats(per)
            key = f"{model}|{arm}|{split}"
            out["table2"][key] = dict(source=src, n=len(per), best=int(best),
                                      best_rho=lm[best], L32=lm["32"],
                                      L0=lm["0"], L16=lm["16"],
                                      curve=[lm[str(i)] for i in range(33)])
            per46 = {n: c for n, c in per.items() if n in SHARED}
            lm46, best46 = curve_stats(per46)
            out["table2"][key].update(n46=len(per46), best46=int(best46),
                                      best_rho46=lm46[best46], L32_46=lm46["32"])
            if split == "random":
                continue
            if best == "32":
                out["degenerate"].append(key)
                continue
            d = np.array([c[best] - c["32"] for c in per.values() if best in c and "32" in c])
            b = boot(d)
            t = A.t_ci90(d)
            out["equivalence"].append(dict(
                model=model, arm=arm, split=split, source=src, best=int(best), n=len(d),
                mean=float(d.mean()), boot90=b, t90=t, tost_p=A.tost_p(d),
                dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), verdict=verdict(b),
                verdict_t=verdict(t), upper_gap=float(A.MARGIN - b[1])))

eq = out["equivalence"]
lf_keys = [k for k in out["table2"] if not k.endswith("random")]
out["equivalence_summary"] = dict(
    n_tests=len(eq), n_degenerate=len(out["degenerate"]),
    equivalent=sum(r["verdict"] == "Equivalent" for r in eq),
    inconclusive=sum(r["verdict"] == "Inconclusive" for r in eq),
    difference=sum(r["verdict"] == "Difference" for r in eq),
    equivalent_t=sum(r["verdict_t"] == "Equivalent" for r in eq),
    equivalent_tost=sum(r["tost_p"] < 0.05 for r in eq),
    equivalent_at_margin=sum(r["verdict"] == "Equivalent" and r["upper_gap"] < 0.0005 for r in eq),
    max_mean=max(r["mean"] for r in eq), max_upper=max(r["boot90"][1] for r in eq),
    max_mean_wt=max(r["mean"] for r in eq if r["arm"] != "mutant"),
    max_upper_wt=max(r["boot90"][1] for r in eq if r["arm"] != "mutant"),
    best_layers=sorted({out["table2"][k]["best"] for k in lf_keys}),
    n_best_30_32=sum(30 <= out["table2"][k]["best"] <= 32 for k in lf_keys),
    n_leakfree=len(lf_keys))

# ---- Table 4: real 3Di minus structure-masked at layer 32 (primary sources)
for arm in A.ARMS:
    for split in SPLITS:
        dm, _ = primary_dir("SaProt (mask)", arm, split)
        d3, src = primary_dir("SaProt (3Di)", arm, split)
        pm, p3 = A.load_probing(dm, arm, split), A.load_probing(d3, arm, split)
        names = [n for n in p3 if n in pm and "32" in p3[n] and "32" in pm[n]]
        d = np.array([p3[n]["32"] - pm[n]["32"] for n in names])
        w = stats.wilcoxon(d[d != 0]).pvalue if np.any(d != 0) else 1.0
        out["table4"][f"{arm}|{split}"] = dict(source=src, n=len(d), mean=float(d.mean()),
                                               wilcoxon_p=float(w))

# ---- ranges of leakage-free probing on 63 and on 46 assays
lf = {k: out["table2"][k] for k in lf_keys}
out["ranges"] = dict(
    all63_best=[min(v["best_rho"] for v in lf.values()), max(v["best_rho"] for v in lf.values())],
    all63_L32=[min(v["L32"] for v in lf.values()), max(v["L32"] for v in lf.values())],
    shared46_best=[min(v["best_rho46"] for v in lf.values()), max(v["best_rho46"] for v in lf.values())],
    shared46_L32=[min(v["L32_46"] for v in lf.values()), max(v["L32_46"] for v in lf.values())])

# ---- "this work" rows for Supplementary Table S1B: final-layer mutant arm on
# single substitutions (ProteinGym's supervised folds are defined for single
# substitutions), 46 shared assays
for model in ("ESM-2", "SaProt (3Di)"):
    row = {}
    for split in ("random", "modulo", "contiguous"):
        per = A.load_probing(SINGLE[model], "mutant", split)
        per46 = {n: c for n, c in per.items() if n in SHARED and "32" in c}
        row[split] = dict(n=len(per46), L32=float(np.mean([c["32"] for c in per46.values()])))
    out["anchors_this_work"][model] = row

# ---- random-split inflation (random minus leakage-free, final layer, primary)
for model in MODELS:
    for arm in A.ARMS:
        r = out["table2"][f"{model}|{arm}|random"]["L32"]
        out["inflation"][f"{model}|{arm}"] = {s: r - out["table2"][f"{model}|{arm}|{s}"]["L32"]
                                             for s in A.LEAK_FREE}
infl = [v for d in out["inflation"].values() for v in d.values()]
out["inflation_range"] = [min(infl), max(infl)]

dst = ROOT / "revision_results/r3_analysis"
json.dump(out, open(dst / "r3_numbers.json", "w"), indent=1)

if __name__ == "__main__":
    f = lambda c: f"[{c[0]:+.4f},{c[1]:+.4f}]"
    print("TABLE 2 (primary)")
    for k, v in out["table2"].items():
        print(f"  {k:38s} {v['source']:6s} n={v['n']:2d} best L{v['best']:2d} {v['best_rho']:.3f} "
              f"L32 {v['L32']:.3f} | L0 {v['L0']:.3f} L16 {v['L16']:.3f}")
    print("EQUIVALENCE", json.dumps(out["equivalence_summary"]))
    for r in eq:
        print(f"  {r['model']:14s}{r['arm']:12s}{r['split']:20s}{r['source']:7s}L{r['best']} "
              f"{r['mean']:+.4f} b90{f(r['boot90'])} t90{f(r['t90'])} TOSTp={r['tost_p']:.3f} "
              f"dz={r['dz']:.2f} {r['verdict']}")
    print("  degenerate:", out["degenerate"])
    print("TABLE 4")
    for k, v in out["table4"].items():
        print(f"  {k:24s} {v['source']:6s} n={v['n']} {v['mean']:+.3f} p={v['wilcoxon_p']:.2g}")
    print("RANGES", json.dumps(out["ranges"]))
    print("ANCHORS", json.dumps(out["anchors_this_work"]))
    print("INFLATION range", out["inflation_range"])
    for k, v in out["inflation"].items():
        print("  ", k, {s: round(x, 3) for s, x in v.items()})
