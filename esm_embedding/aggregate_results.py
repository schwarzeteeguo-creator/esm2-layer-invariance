"""Aggregate all revision results into MAIN_TABLE.md.

Covers: probing main table (3 model lanes x 3 GPU arms x 4 splits),
random-position control, TOST equivalence tests, SaProt real3Di vs
structure-masked paired comparison, logit-lens curves, CKA, pooling
(partial until finished — rerun this script to refresh).

Usage:  python -m esm_embedding.aggregate_results [results_root]
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from esm_embedding.stats_tost import paired_tost, bootstrap_ci

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "revision_results/server_results")
CONTROL = Path("revision_results/probing_v4_gcv")
OUT_MD = ROOT.parent / "MAIN_TABLE.md"

SPLITS = ["random", "modulo", "contiguous", "position_groupkfold"]
LEAK_FREE = ["modulo", "contiguous", "position_groupkfold"]
ARMS = ["masked_wt", "unmasked_wt", "mutant"]
MODELS = [("probing_v4_esm2", "ESM-2", "n/a"),
          ("probing_v4_saprot", "SaProt (struct-masked)", "structure_masked"),
          ("probing_v4_saprot_3di", "SaProt (real 3Di)", "real_3di")]


def load_probe(model_dir, arm, split, featset="combined"):
    """-> {ds: {layer: spearman}}, bookkeeping {ds: struct}"""
    f = ROOT / model_dir / f"v4__{arm}__{featset}__{split}.json"
    if not f.exists():
        return {}, {}
    data = json.load(open(f))
    curves, meta = {}, {}
    for ds, e in data.items():
        res = e["result"]
        vals = {int(l): r["spearman"] for l, r in res.items()
                if isinstance(r, dict) and r.get("spearman") is not None
                and np.isfinite(r["spearman"])}
        if vals:
            curves[ds] = vals
            meta[ds] = e.get("struct", "n/a")
    return curves, meta


def layer_means(curves, keep=None):
    """mean per layer over datasets (optionally filtered set)."""
    out = {}
    for ds, vals in curves.items():
        if keep is not None and ds not in keep:
            continue
        for l, v in vals.items():
            out.setdefault(l, []).append(v)
    return {l: float(np.mean(v)) for l, v in out.items() if len(v) >= 40}, \
           {l: len(v) for l, v in out.items()}


def fmt(x):
    return f"{x:+.3f}" if np.isfinite(x) else "  —  "


def main():
    md = ["# Revision 主表（自动生成）", ""]
    headline = []

    # ---------- control ----------
    md += ["## 1. 位置泄漏控制（随机位置向量查找表，本地 GCV 协议，n=63）", "",
           "| 特征 | random | modulo | contiguous | GroupKFold |",
           "|---|---|---|---|---|"]
    ctrl = {}
    for arm, fs, label in [
            ("random_position", "combined", "随机位置向量 + aux"),
            ("random_position", "embedding_only", "随机位置向量 alone"),
            ("aux", "auxiliary_only", "aux alone")]:
        row = {}
        for sp in SPLITS:
            f = CONTROL / f"v4__{arm}__{fs}__{sp}.json"
            d = json.load(open(f))
            vals = [e["result"].get("rho_mean", e["result"].get("spearman"))
                    for e in d.values()]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            row[sp] = (float(np.mean(vals)), len(vals))
            ctrl[(arm, fs, sp)] = {k: e["result"] for k, e in d.items()}
        md.append(f"| {label} | " + " | ".join(
            f"{fmt(row[sp][0])} (n={row[sp][1]})" for sp in SPLITS) + " |")
        if arm == "random_position" and fs == "combined":
            headline.append(f"查找表对照: random {row['random'][0]:+.3f} → "
                            f"leakage-free "
                            f"{np.mean([row[s][0] for s in LEAK_FREE]):+.3f}")
    md += ["", "random vs modulo 配对 Wilcoxon（随机位置向量+aux）:",
           ""]
    a = {k: v.get("rho_mean") for k, v in ctrl[
        ("random_position", "combined", "random")].items()}
    b = {k: v.get("rho_mean") for k, v in ctrl[
        ("random_position", "combined", "modulo")].items()}
    common = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in common])
    _, p = wilcoxon(d)
    md.append(f"- n={len(common)}, 平均跌落 {d.mean():+.3f}, "
              f"Wilcoxon p={p:.1e}")
    md.append("")

    # ---------- probing main ----------
    md += ["## 2. Probing 主表（三臂 × 四划分；combined 特征集；"
           "格式：均值(最优层→L32)）", "",
           "| 模型 | 臂 | random | modulo | contiguous | GroupKFold |",
           "|---|---|---|---|---|---|"]
    store = {}   # (model_dir, arm, split) -> (curves, meta, means)
    for model_dir, mlabel, _ in MODELS:
        for arm in ARMS:
            cells = []
            for sp in SPLITS:
                curves, meta = load_probe(model_dir, arm, sp)
                means, _ = layer_means(curves)
                if not means:
                    cells.append("—")
                    continue
                best = max(means, key=means.get)
                n = len(curves)
                cells.append(f"{fmt(means[best])} (L{best}→{fmt(means[32])}) n={n}")
                store[(model_dir, arm, sp)] = (curves, meta, means)
            md.append(f"| {mlabel} | {arm} | " + " | ".join(cells) + " |")
    md.append("")
    for (model_dir, arm, sp), (curves, meta, means) in store.items():
        if sp == "modulo" and arm == "mutant":
            best = max(means, key=means.get)
            headline.append(f"probing mutant@modulo [{model_dir}]: "
                            f"L{best} {means[best]:+.3f}")

    # ---------- TOST ----------
    md += ["## 3. TOST 等效检验（leakage-free 划分；固定层=均值曲线最优层，"
           "与 L32 配对；边际 Δρ=0.01）", "",
           "| 模型 | 臂 | 划分 | 最优层 | n | mean Δ(best−final) | 90% CI | TOST p | 结论 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for model_dir, mlabel, _ in MODELS:
        for arm in ARMS:
            for sp in LEAK_FREE:
                key = (model_dir, arm, sp)
                if key not in store:
                    continue
                curves, meta, means = store[key]
                best = max(means, key=means.get)
                pairs = [(c[32] - c[best]) * -1 for c in curves.values()
                         if 32 in c and best in c]   # best - final
                d = np.array(pairs)
                if len(d) < 30 or best == 32:
                    continue
                tost = paired_tost(d, margin=0.01)
                lo, hi = bootstrap_ci(d)
                concl = ("等效" if tost["p_tost"] < 0.05 else
                         ("差异>边际" if abs(d.mean()) > 0.01 else "不确定"))
                md.append(f"| {mlabel} | {arm} | {sp} | L{best} | {len(d)} | "
                          f"{fmt(d.mean())} | [{lo:+.3f}, {hi:+.3f}] | "
                          f"{tost['p_tost']:.3f} | {concl} |")
    md.append("")

    # ---------- real3Di vs masked ----------
    md += ["## 4. SaProt 真3Di vs 结构掩码（同数据集同层配对，L32，"
           "combined）", "",
           "| 臂 | 划分 | n | mean Δ(3di−mask) | Wilcoxon p |",
           "|---|---|---|---|---|"]
    for arm in ARMS:
        for sp in SPLITS:
            c3, _ = load_probe("probing_v4_saprot_3di", arm, sp)
            cm, _ = load_probe("probing_v4_saprot", arm, sp)
            common = sorted(set(c3) & set(cm))
            d = np.array([c3[k][32] - cm[k][32] for k in common
                          if 32 in c3[k] and 32 in cm[k]])
            if len(d) < 30:
                continue
            _, p = wilcoxon(d) if np.any(d != 0) else (0, 1.0)
            md.append(f"| {arm} | {sp} | {len(d)} | {fmt(d.mean())} | {p:.1e} |")
    md.append("")

    # ---------- lens ----------
    md += ["## 5. Logit lens（零样本，n=63，门禁全过）", ""]
    for name, label in [("logit_lens_v2_esm2", "ESM-2"),
                        ("logit_lens_v2_saprot_3di", "SaProt real-3Di")]:
        f = ROOT / name / "logit_lens_v2_summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        pl = {int(k): v for k, v in s["per_layer"].items()}
        layers = sorted(pl)
        best = max(pl, key=lambda l: pl[l]["mean"])
        curve = " ".join(f"{pl[l]['mean']:+.2f}" for l in layers[::4])
        md.append(f"- **{label}**: L0 {pl[0]['mean']:+.3f} → 最优 L{best} "
                  f"**{pl[best]['mean']:+.3f}** → L32 {pl[32]['mean']:+.3f}；"
                  f"曲线（每4层采样）: {curve}")
        headline.append(f"lens {label}: 末层最优 {pl[32]['mean']:+.3f}")
    md.append("")

    # ---------- CKA ----------
    md += ["## 6. CKA（Kornblith 线性 CKA vs cosine，10 数据集 × 50 位置）", ""]
    for d, label in [("cka_esm2", "ESM-2"), ("cka_saprot", "SaProt")]:
        f = ROOT / d / "cka_summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        md.append(f"- **{label}** (n={s.get('n_datasets')}): 相邻层 "
                  f"CKA {s['adjacent_cka']:.3f} / cos {s['adjacent_cos']:.3f}；"
                  f"L0↔L32 CKA {s['L0_vs_L32_cka']:.3f} / "
                  f"cos {s['L0_vs_L32_cos']:.3f}")
    md.append("")

    # ---------- pooling (partial ok) ----------
    md += ["## 7. Pooling（n=20/模型设计；本节随进度自动刷新）", ""]
    for d, label in [("pooling_v3_esm2", "ESM-2"),
                     ("pooling_v3_saprot", "SaProt")]:
        f = ROOT / d / f"pooling_v3__{d.split('_')[2]}_650m.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        n = len(s)
        md.append(f"### {label}（已完成 {n}/20）")
        # mutant arm, modulo split, ridge + attention, combined
        rows = []
        for strat in ["last_layer", "mean_20_33", "concat_5L", "attention"]:
            for read in ["ridge", "mlp", "lightgbm"]:
                vals = []
                for name, dsr in s.items():
                    try:
                        node = dsr["arms"]["mutant"]["modulo"][
                            f"{strat}__combined"]
                        # attention nests readouts under "rho"
                        if strat == "attention" and "rho" in node:
                            node = node["rho"]
                        v = node[read]["spearman"]
                    except (KeyError, TypeError):
                        continue
                    if v is not None and np.isfinite(v):
                        vals.append(v)
                if len(vals) >= 5:
                    rows.append(f"| {strat} | {read} | {np.mean(vals):+.3f} "
                                f"(n={len(vals)}) |")
        md += ["| 策略 | 读出 | mutant@modulo ρ |", "|---|---|---|"] + rows
        md.append("")

    md += ["---", "生成: " + __import__("datetime").datetime.now().strftime(
        "%F %T"), "注: TOST 固定层选择=均值曲线最优层（m9 披露）；"
           "pooling n=20/模型为原投稿设计。"]

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"written: {OUT_MD}")
    print("\nHEADLINES:")
    for h in headline:
        print(" -", h)


if __name__ == "__main__":
    main()
