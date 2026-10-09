"""Build supplementary_v3.tex from the R2 analysis outputs.

All tables are numbered (Table S1..S11) with labels so the main text can
reference them unambiguously. S10/S11 are placeholders until the server
results (published anchors, Envision baseline, single-site sensitivity)
land.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R2 = ROOT / "revision_results" / "r2_analysis"
SR = ROOT / "revision_results" / "server_results"

a1 = json.load(open(R2 / "a1_tost.json"))
pool = json.load(open(R2 / "a2_a3_pooling.json"))
ms = json.load(open(R2 / "r2_multisite_lmdb.json"))
cka = {"ESM-2": json.load(open(R2 / "cka200_esm2_summary.json")),
       "SaProt": json.load(open(R2 / "cka200_saprot_summary.json"))}
cka50 = {"ESM-2": json.load(open(SR / "cka_esm2" / "cka_summary.json")),
         "SaProt": json.load(open(SR / "cka_saprot" / "cka_summary.json"))}

PROBE_DIRS = {"ESM-2": SR / "probing_v4_esm2",
              "SaProt (mask)": SR / "probing_v4_saprot",
              "SaProt (3Di)": SR / "probing_v4_saprot_3di"}
SPLITS4 = ["random", "modulo", "contiguous", "position_groupkfold"]
SPLIT_NAMES = {"modulo": "Modulo", "contiguous": "Contiguous",
               "position_groupkfold": "Pos.-grouped", "random": "Random"}
ARM_NAMES = {"masked_wt": "masked\\_wt", "unmasked_wt": "unmasked\\_wt",
             "mutant": "mutant"}


def esc(s):
    return s.replace("_", "\\_")


def fmt(x, nd=3):
    return f"{x:+.{nd}f}"


def _load(mdir, arm, split):
    f = mdir / f"v4__{arm}__combined__{split}.json"
    d = json.load(open(f))
    out = {}
    for name, e in d.items():
        out[name] = {int(k): v["spearman"] for k, v in e["result"].items()
                     if v.get("spearman") is not None}
    return out


# ── S1: assay composition ─────────────────────────────────────────
def table_s1():
    rows = sorted(ms["rows"], key=lambda r: r["dataset"])
    s = ms["summary"]
    lines = [r"\begin{table}[H]\centering\scriptsize",
             r"\caption{Assay composition of the 63-dataset benchmark "
             r"(SaProt LMDB snapshot, contained in ProteinGym v1.3). "
             r"Multi-site variants ($\geq$2 substitutions) dominate the "
             r"row count; caps follow the fixed protocol (10{,}000 "
             r"variants, then 5{,}000 rows, seed 42).}",
             r"\label{tab:S1assays}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Assay & Variants & Multi-site & Rows after & Analyzed & "
             r"Variant & Row \\",
             r" & total & (\%) & cap & rows & cap & ceiling \\",
             r"\midrule"]
    for r in rows:
        pct = 100.0 * r["multi_site_variants"] / max(r["n_variants_total"], 1)
        lines.append(
            f"{esc(r['dataset'])} & {r['n_variants_total']:,} & "
            f"{pct:.0f} & {r['rows_after_cap']:,} & "
            f"{r['analyzed_rows'] if r['analyzed_rows'] else '--'} & "
            f"{'yes' if r['capped_10k'] else ''} & "
            f"{'yes' if (r['analyzed_rows'] or 0) >= 5000 else ''} \\\\")
    lines += [r"\midrule",
              f"Total / pooled & {sum(r['n_variants_total'] for r in rows):,}"
              f" & {s['multi_site_variant_pct']:.1f} & "
              f"{sum(r['rows_after_cap'] for r in rows):,} & "
              f"{sum(r['analyzed_rows'] or 0 for r in rows):,} & "
              f"{s['capped_10k']} & 26 \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── S2: full 33-layer curves, one table per model x arm ──────────
def table_s2():
    parts = []
    for model, mdir in PROBE_DIRS.items():
        for arm in ["masked_wt", "unmasked_wt", "mutant"]:
            cols = {sp: _load(mdir, arm, sp) for sp in SPLITS4}
            layer_means = {sp: {} for sp in SPLITS4}
            for sp in SPLITS4:
                for name, curve in cols[sp].items():
                    for l, v in curve.items():
                        layer_means[sp].setdefault(l, []).append(v)
            means = {sp: {l: sum(v) / len(v)
                          for l, v in layer_means[sp].items()}
                     for sp in SPLITS4}
            best = {sp: max(means[sp], key=means[sp].get)
                    for sp in SPLITS4}
            lab = "tab:S2" + model.replace(" ", "").replace("(", "") \
                .replace(")", "") + arm
            parts += [r"\begin{table}[H]\centering\small",
                      "\\caption{" + model + ", " + ARM_NAMES[arm] +
                      " arm: per-layer mean $\\rho$ by split (best layer "
                      "per column in bold). $n=63$ (contiguous 57).}",
                      "\\label{" + lab + "}",
                      r"\begin{tabular}{rcccc}", r"\toprule",
                      "Layer & Random & Modulo & Contiguous & "
                      "Pos.-grouped \\\\", r"\midrule"]
            for l in range(33):
                cells = []
                for sp in SPLITS4:
                    v = means[sp].get(l)
                    cells.append("\\textbf{%.3f}" % v if best[sp] == l
                                 else "%.3f" % v)
                parts.append(str(l) + " & " + " & ".join(cells) + " \\\\")
            parts += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n\n".join(parts)


# ── S3: per-dataset mutant/modulo ─────────────────────────────────
def table_s3():
    cols = {"ESM-2": _load(PROBE_DIRS["ESM-2"], "mutant", "modulo"),
            "SaProt (3Di)": _load(PROBE_DIRS["SaProt (3Di)"], "mutant",
                                  "modulo")}
    bk = json.load(open(SR / "probing_v4_esm2" /
                        "v4__mutant__combined__modulo.json"))
    names = sorted(cols["ESM-2"].keys())
    parts = [r"\begin{table}[H]\centering\scriptsize",
             r"\caption{Per-dataset mutant-arm probing (modulo split, "
             r"combined features): best layer (and $\rho$) and "
             r"final-layer (L32) $\rho$.}",
             r"\label{tab:S3perdataset}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Assay & $n$ & \multicolumn{3}{c}{ESM-2} & "
             r"\multicolumn{3}{c}{SaProt (3Di)} \\",
             r" & rows & best L & best $\rho$ & L32 $\rho$ & best L & "
             r"best $\rho$ & L32 $\rho$ \\", r"\midrule"]
    for name in names:
        e, s3 = cols["ESM-2"][name], cols["SaProt (3Di)"][name]
        eb, sb = max(e, key=e.get), max(s3, key=s3.get)
        n = bk.get(name, {}).get("n_rows", "--")
        parts.append(
            f"{esc(name)} & {n} & {eb} & {e[eb]:.3f} & {e[32]:.3f} & "
            f"{sb} & {s3[sb]:.3f} & {s3[32]:.3f} \\\\")
    parts += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(parts)


# ── S4: TOST ──────────────────────────────────────────────────────
def table_s4():
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{Best-versus-final-layer equivalence tests under "
             r"the leakage-free splits (margin $\pm0.01$). Verdict rule: "
             r"\emph{Equivalent} when the bootstrap 90\% CI of the paired "
             r"mean difference lies entirely inside the margin, "
             r"\emph{Difference} when entirely outside, \emph{Inconclusive}"
             r" otherwise. The best layer is the argmax of the same "
             r"per-dataset layer-mean curve (selection favours detecting "
             r"a difference, so equivalence verdicts are conservative). "
             r"In 7 further model--arm--split combinations the best layer "
             r"is layer 32 itself (no test defined): ESM-2 masked\_wt "
             r"and unmasked\_wt (all three splits each) and SaProt "
             r"(mask) mutant (contiguous). Result: 8 equivalent, 12 "
             r"inconclusive, 0 differences.}",
             r"\label{tab:S4tost}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lllrrllll}", r"\toprule",
             r"Model & Arm & Split & Best & $n$ & Mean $\Delta$ & "
             r"90\% CI & TOST $p$ & Verdict \\", r"\midrule"]
    for r in a1["rows"]:
        b = r["boot_ci90"]
        md = r.get("mean_delta", r.get("mean_diff"))
        lines.append(
            f"{r['model']} & {ARM_NAMES[r['arm']]} & "
            f"{SPLIT_NAMES[r['split']]} & L{r['best_layer']} & {r['n']} & "
            f"{fmt(md, 4)} & "
            f"[{b[0]:+.4f}, {b[1]:+.4f}] & {r['tost_p']:.3f} & "
            f"{r['verdict']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


# ── S5: 3Di disclosure (carried over from v2) ─────────────────────
def table_s5():
    src = (ROOT / "supplementary_v2.tex").read_text(encoding="utf-8")
    i = src.find(r"\begin{table}[H]\centering\scriptsize")
    j = src.find(r"\end{table}", i) + len(r"\end{table}")
    block = src[i:j]
    block = block.replace(
        "\\begin{tabular}{lrrllrr}",
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrllrr}", 1)
    block = block.replace("\\end{tabular}", "\\end{tabular}}", 1)
    return block


# ── S6: fusion-vs-last full grid ──────────────────────────────────
def table_s6():
    lines = [r"\begin{table}[H]\centering\scriptsize",
             r"\caption{Paired per-dataset differences (strategy $-$ "
             r"last layer; mutant arm, combined features, 20 datasets "
             r"per model) for every split and readout. Cells: mean "
             r"$\Delta\rho$ with bootstrap 90\% CI; \dag{} marks "
             r"intervals excluding zero.}",
             r"\label{tab:S6fusion}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llllrrr}", r"\toprule",
             r"Model & Split & Readout & Strategy & $\Delta\rho$ & "
             r"90\% CI & Wilcoxon $p$ \\", r"\midrule"]
    for key in sorted(pool["fusion_vs_last"].keys()):
        tbl = pool["fusion_vs_last"][key]
        model, split = key.split("|")
        for sk in sorted(tbl.keys()):
            strat, ro = sk.split("|")
            e = tbl[sk]
            b = e["boot_ci90"]
            dag = r"\dag" if (b[0] > 0 or b[1] < 0) else ""
            p = e["wilcoxon_p"]
            pstr = "%.3g" % p if p is not None else "--"
            lines.append(
                f"{model} & {SPLIT_NAMES.get(split, split)} & {ro} & "
                f"{esc(strat)} & {fmt(e['mean_delta'], 4)}{dag} & "
                f"[{b[0]:+.4f}, {b[1]:+.4f}] & {pstr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


# ── S7: LightGBM vs Ridge ─────────────────────────────────────────
def table_s7():
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{Paired per-dataset differences (LightGBM $-$ "
             r"Ridge) on the last layer (mutant arm, combined features, "
             r"20 datasets per model).}",
             r"\label{tab:S7readout}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Model & Split & $n$ & Mean $\Delta$ & 90\% CI & "
             r"Wilcoxon $p$ \\", r"\midrule"]
    for key in sorted(pool["lgbm_vs_ridge"].keys()):
        e = pool["lgbm_vs_ridge"][key]
        model, split = key.split("|")
        b = e["boot_ci90"]
        lines.append(
            f"{model} & {SPLIT_NAMES.get(split, split)} & {e['n']} & "
            f"{fmt(e['mean_delta'], 4)} & [{b[0]:+.4f}, {b[1]:+.4f}] & "
            f"{e['wilcoxon_p']:.3g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── S8: pooling means all splits ──────────────────────────────────
def table_s8():
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{Pooling means (mutant arm, combined features; "
             r"mean Spearman $\rho$ over 20 datasets per model) for all "
             r"splits, strategies, and readouts.}",
             r"\label{tab:S8poolmeans}",
             r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"Model & Split & Strategy & Ridge & RF & MLP & LightGBM \\",
             r"\midrule"]
    ros = ["ridge", "rf", "mlp", "lightgbm"]
    for model in ["ESM-2", "SaProt"]:
        for split in ["random", "modulo", "contiguous",
                      "position_groupkfold"]:
            block = pool["means"].get(model, {}).get(split, {})
            for strat in ["last_layer", "mean_20_33", "concat_5L",
                          "attention"]:
                vals = [block.get(f"{strat}__{ro}", {}).get("mean")
                        for ro in ros]
                if all(v is None for v in vals):
                    continue
                cells = " & ".join("%.3f" % v if v is not None else "--"
                                   for v in vals)
                lines.append(f"{model} & {SPLIT_NAMES.get(split, split)} "
                             f"& {esc(strat)} & {cells} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── S9: CKA 50 vs 200 ─────────────────────────────────────────────
def table_s9():
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{Layer-similarity summaries at 50 and 200 sampled "
             r"positions (10 datasets each). The debiased "
             r"(unbiased-HSIC) estimator is unstable at $n<d$ (estimates "
             r"can exceed 1), so the 200-position standard estimator is "
             r"the primary estimate.}",
             r"\label{tab:S9cka}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{Adjacent} & "
             r"\multicolumn{2}{c}{L0 vs L32} & "
             r"\multicolumn{2}{c}{Pearson $r$ vs distance} \\",
             r"Model & cos & CKA & cos & CKA & cos & CKA \\", r"\midrule"]
    for m in ["ESM-2", "SaProt"]:
        c5, c2 = cka50[m], cka[m]
        lines.append(
            f"{m} (50 pos.) & {c5['adjacent_cos']:.3f} & "
            f"{c5['adjacent_cka']:.3f} & {c5['L0_vs_L32_cos']:+.3f} & "
            f"{c5['L0_vs_L32_cka']:.3f} & "
            f"{c5['pearson_r_cos_vs_distance_unique_pairs']:+.2f} & "
            f"{c5['pearson_r_cka_vs_distance_unique_pairs']:+.2f} \\\\")
        lines.append(
            f"{m} (200 pos.) & {c2['adjacent_cos']:.3f} & "
            f"{c2['adjacent_cka']:.3f} & {c2['L0_vs_L32_cos']:+.3f} & "
            f"{c2['L0_vs_L32_cka']:.3f} & "
            f"{c2['pearson_r_cos_vs_distance_unique_pairs']:+.2f} & "
            f"{c2['pearson_r_cka_vs_distance_unique_pairs']:+.2f} \\\\")
        lines.append(
            f"\\quad debiased (200) & -- & "
            f"{c2['adjacent_cka_debiased']:.3f} & -- & "
            f"{c2['L0_vs_L32_cka_debiased']:.3f} & -- & -- \\\\")
        if m == "ESM-2":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)






# ── S10: published anchors ────────────────────────────────────────
def table_s10():
    zs = json.load(open(R2 / "r2_anchors_zeroshot.json"))
    sup = json.load(open(R2 / "r2_anchors_supervised.json"))
    zs_means = zs.get("summary", {})
    sup_mod = sup["summary"]["fold_modulo_5"]
    sup_con = sup["summary"]["fold_contiguous_5"]
    order = [("ESM-2 650M (published)", "ESM2_650M"),
             ("GEMME", "GEMME"),
             ("EVE (single)", "EVE_single"),
             ("DeepSequence (single)", "DeepSequence_single"),
             ("MSA Transformer (single)", "MSA_Transformer_single"),
             ("ESM-1v (single)", "ESM1v_single"),
             ("ESM-2 3B", "ESM2_3B"),
             ("ESM-2 150M", "ESM2_150M")]
    sup_sel = [("Kermut", "Kermut"),
               ("ProteinNPT", "ProteinNPT"),
               ("Emb.-aug. MSA Transformer",
                "Embeddings - Augmented - MSA Transformer"),
               ("OHE, not augmented", "OHE - Not augmented")]
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{Published ProteinGym v1.3 results recomputed on "
             r"the same assays as this work (46 of 63 covered by the "
             r"release). Zero-shot column: mean per-assay Spearman "
             r"$\rho$ over the 46 assays; our lens and reproduction rows "
             r"are matched to the same assays. Supervised columns: "
             r"official out-of-fold predictions under the release fold "
             r"schemes (random-fold values: Kermut 0.711, ProteinNPT "
             r"0.697, Emb.-aug. MSA Transformer 0.641, OHE 0.570; "
             r"our Envision-style baseline scores 0.634 under random).}",
             r"\label{tab:S10anchors}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Zero-shot method & $\rho$ & Supervised method & Modulo & "
             r"Contiguous \\", r"\midrule"]
    for i in range(max(len(order), len(sup_sel))):
        left = ""
        if i < len(order):
            lbl, key = order[i]
            m = zs_means.get(key, {}).get("mean")
            left = f"{lbl} & {m:.3f}" if m is not None else lbl + " & --"
        right = ""
        if i < len(sup_sel):
            lbl, key = sup_sel[i]
            mm = sup_mod.get(key, {}).get("mean")
            mc = sup_con.get(key, {}).get("mean")
            right = f"{lbl} & {mm:+.3f} & {mc:+.3f}"
        lines.append(left + (" & " + right if right else " & & ") + " \\\\")
    lines += [r"\midrule",
              r"Our lens: ESM-2 final & 0.438 & Our leakage-free probing & "
              r"\multicolumn{2}{c}{0.32--0.53} \\",
              r"Our lens: SaProt final & 0.440 & & & \\",
              r"Our SaProt reproduction & 0.484 & & & \\",
              r"\midrule",
              r"Envision-style supervised (ours) & -- & "
              r"Envision-style (Section 3.7) & 0.410 & 0.302 \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── S11: single-substitution sensitivity ───────────────────────────
def table_s11():
    ss = json.load(open(R2 / "single_site_summary.json"))
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{Single-substitution-only sensitivity analysis "
             r"(mutant arm, combined features). Each variant contributes "
             r"one row, so no variant-level label sharing across folds "
             r"remains. Best layer per model--split with the mean paired "
             r"best-minus-final-layer difference and its bootstrap 90\% "
             r"CI. The best layer is L30--L31 in every leakage-free "
             r"column. Panel B: real-3Di versus structure-masked SaProt "
             r"at the final layer on the same subset.}",
             r"\label{tab:S11singlesite}",
             r"\resizebox{478pt}{!}{%",
             r"\begin{tabular}{llrrrrl}", r"\toprule",
             r"Model & Split & Best & Best $\rho$ & L32 $\rho$ & "
             r"$\Delta$ & 90\% CI \\",
             r"\midrule"]
    ORDER = [("ESM-2", "modulo"), ("ESM-2", "contiguous"),
             ("ESM-2", "position_groupkfold"),
             ("SaProt (mask)", "modulo"), ("SaProt (mask)", "contiguous"),
             ("SaProt (mask)", "position_groupkfold"),
             ("SaProt (3Di)", "modulo"), ("SaProt (3Di)", "contiguous"),
             ("SaProt (3Di)", "position_groupkfold")]
    for model, split in ORDER:
        e = ss["configs"].get(f"{model}|mutant|{split}")
        if not e:
            continue
        ci = e.get("ci90_best_minus_L32")
        cistr = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "--"
        lines.append(
            f"{model} & {SPLIT_NAMES[split]} & L{e['best_layer']} & "
            f"{e['best_rho']:.3f} & {e['L32_rho']:.3f} & "
            f"{e['mean_best_minus_L32']:+.4f} & {cistr} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{7}{l}{\emph{B. Real-3Di vs structure-masked "
              r"at L32 (paired, single-site subset)}} \\",
              r"Split & $n$ & Mean $\Delta$ & Wilcoxon $p$ & & & \\",
              r"\midrule"]
    for sp in ["random", "modulo", "contiguous", "position_groupkfold"]:
        e = (ss.get("threedee_single_site") or {}).get(sp)
        if not e:
            continue
        p = e.get("wilcoxon_p")
        pstr = "%.2e" % p if p is not None else "--"
        lines.append(f"{SPLIT_NAMES[sp]} & {e['n']} & "
                     f"{e['mean_delta']:+.4f} & {pstr} & & & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parts = []
    parts.append(
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{float}\n"
        "\\usepackage[margin=2cm]{geometry}\n"
        "\\usepackage[hidelinks]{hyperref}\n"
        "\\usepackage{caption}\n"
        "\\usepackage{xcolor}\n"
        "\\captionsetup{font=small}\n"
        "\\title{Supplementary Information for\\\\ ``Depth Matters but the "
        "Final Layer Suffices: Leakage-Free Evaluation of Layer Choice in "
        "Protein Language Models for Mutation Effect Prediction''}\n"
        "\\date{}\n\\begin{document}\n\\maketitle\n")
    parts.append("\n\\section*{S1. Assay composition of the benchmark}\n")
    parts.append(table_s1())
    parts.append("\n\\section*{S2. Full 33-layer mean Spearman $\\rho$ "
                 "(combined features)}\n")
    parts.append(table_s2())
    parts.append("\n\\section*{S3. Per-dataset results (mutant arm, "
                 "modulo split)}\n")
    parts.append(table_s3())
    parts.append("\n\\section*{S4. Best-versus-final equivalence tests}\n")
    parts.append(table_s4())
    parts.append("\n\\section*{S5. Foldseek 3Di token preparation "
                 "disclosure}\n")
    parts.append(table_s5())
    parts.append("\n\\section*{S6. Fusion versus last layer: all paired "
                 "comparisons}\n")
    parts.append(table_s6())
    parts.append("\n\\section*{S7. Readout contrast on the last layer "
                 "(LightGBM vs Ridge)}\n")
    parts.append(table_s7())
    parts.append("\n\\section*{S8. Pooling means under all splits}\n")
    parts.append(table_s8())
    parts.append("\n\\section*{S9. Layer-similarity analysis: 50 vs 200 "
                 "positions, debiased CKA}\n")
    parts.append(table_s9())
    parts.append("\n\\section*{S10. Published ProteinGym anchors}\n")
    parts.append(table_s10())
    parts.append("\n\\section*{S11. Single-substitution sensitivity "
                 "analysis}\n")
    parts.append(table_s11())
    parts.append("\n\\end{document}\n")
    out = ROOT / "supplementary_v3.tex"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
