"""Final in-paper presence audit: every key result of this revision round
must appear verbatim in the compiled manuscript or supplementary."""
import re

man = re.sub(r'\s+', ' ', open('manuscript_revised_v3.tex', encoding='utf-8').read())
sup = re.sub(r'\s+', ' ', open('supplementary_v3.tex', encoding='utf-8').read())

CHECKS = [
    ("TOST 统一口径 8/20 (正文)", man, "8 comparisons were equivalent"),
    ("TOST 8/12/0 (S4表题)", sup, "8 equivalent, 12"),
    ("摘要单点限定句", man, "for SaProt and within $0.02$ for ESM-2"),
    ("结论单点限定句", man, "single-substitution-only subset for ESM-2"),
    ("引言单点限定句", man, "single-substitution-only subset for ESM-2"),
    ("官方 ESM2_650M 锚 0.472", man, "0.472"),
    ("我方 lens@46: 0.438/0.440", man, "0.438"),
    ("SaProt 独立复现 0.484", man, "0.484"),
    ("监督锚 Kermut/ProteinNPT", man, "Kermut 0.605"),
    ("OHE 崩塌 -0.003", man, "0.570 under the random"),
    ("单点 3Di contiguous +0.132", man, "+0.132"),
    ("单点 ESM-2 +0.016 CI", man, "90\\% CI $+0.006$ to"),
    ("Envision 0.634/0.410/0.302", man, "0.634"),
    ("CKA@200: SaProt 0.78", man, "0.78 (0.81"),
    ("CKA@200: ESM-2 0.41", man, "CKA 0.41 with cosine 0.54"),
    ("去偏CKA不稳定披露", man, "can exceed 1"),
    ("多位点 84.6%/95.8%", man, "84.6\\% of variants and contribute 95.8\\%"),
    ("上限 11/26", man, "(26 of 63 datasets"),
    ("跨fold 1.10M", man, "1.10M"),
    ("随机划分融合增益 +0.009~0.050", man, "$+0.009$ to $+0.050$"),
    ("modulo下融合显著为负", man, "$-0.050$"),
    ("lgbm>ridge ESM-2 +0.045 p=.011", man, "$p = 0.011$"),
    ("SaProt lgbm n.s. +0.016", man, "$+0.016$, 90\\% CI $-0.016$"),
    ("masked_wt 单位置遮蔽", man, "single-position masking"),
    ("pooling 5外折统一", man, "five outer"),
    ("3Di随机mutant +0.032 显著", man, "$+0.032$"),
    ("lens 单调末层最优", man, "$\\rho = 0.443$"),
    ("ProteinGym v1.3/63来源声明", man, "v1.3"),
    ("S1引用", man, "Supplementary Table S1"),
    ("S2引用", man, "Supplementary Table S2"),
    ("S3引用", man, "Supplementary Table S3"),
    ("S4引用", man, "Supplementary Table S4"),
    ("S5引用", man, "Supplementary Table S5"),
    ("S6引用", man, "Supplementary Table S6"),
    ("S7引用", man, "Supplementary Table S7"),
    ("S8引用", man, "Supplementary Table S8"),
    ("S9引用", man, "Supplementary Table S9"),
    ("S10引用", man, "Supplementary Table S10"),
    ("S11引用", man, "Supplementary Table S11"),
    ("融合表 SaProt attn lgbm -0.014", man, "$-0.014$ ($-0.027$"),
    ("fig5 范围+最小值注", man, "minimum 0.274"),
    ("S8 含 random 列均值", sup, "Random & last"),
    ("S11 单点表在", sup, "tab:S11singlesite"),
    ("S10 Envision 行", sup, "Envision-style (Section 3.7)"),
]

ok = 0
for name, where, pat in CHECKS:
    hit = pat in where
    print(("OK   " if hit else "MISS ") + name)
    ok += hit
print(f"--- {ok}/{len(CHECKS)} passed")
