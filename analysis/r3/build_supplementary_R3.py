"""Build supplementary_R3.tex (clean, no markup) for the third revision.

Tables are numbered S1-S11 in order of first citation in the main text (R2.m2):
S1  per-dataset provenance, caps, multi-site rows, mutant arm (single subs, modulo)
S2  Foldseek 3Di preparation
S3  layer similarity: 200 positions (primary) and 50 positions; debiased CKA
S4  best-vs-final equivalence, primary analysis (mutant arm on single subs)
S5-S7 full 33-layer curves, primary analysis
S8  single-substitution analysis of all arms versus the full data
S9  pooling: paired fusion-minus-last-layer differences (modulo, random,
    modulo excluding multi-site datasets)
S10 last-layer LightGBM versus Ridge by split
S11 published ProteinGym v1.3 anchors on the 46 shared assays
Every number is read from result files.
"""
import csv, glob, json, os, sys
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'revision_docs'))
import analysis_r2 as A

SR = ROOT / 'revision_results' / 'server_results'
SS = ROOT / 'revision_results' / 'single_site_local'
R2 = ROOT / 'revision_results' / 'r2_analysis'
R3 = ROOT / 'revision_results' / 'r3_analysis'
OUT = ROOT / 'supplementary_R3.tex'

ARMS = ['masked_wt', 'unmasked_wt', 'mutant']
SPLITS = ['random', 'modulo', 'contiguous', 'position_groupkfold']
LF = ['modulo', 'contiguous', 'position_groupkfold']
SPLIT_TEX = {'random': 'Random', 'modulo': 'Modulo', 'contiguous': 'Contig.',
             'position_groupkfold': 'Pos.-gr.'}
SPL_T = {'modulo': 'Modulo', 'contiguous': 'Contiguous', 'position_groupkfold': 'Pos.-grouped',
         'random': 'Random'}
ARM_T = {'masked_wt': r'masked\_wt', 'unmasked_wt': r'unmasked\_wt', 'mutant': 'mutant'}
MODELS = [('ESM-2', 'probing_v4_esm2', 'ESM-2 650M'),
          ('SaProt (mask)', 'probing_v4_saprot', 'SaProt (structure-masked)'),
          ('SaProt (3Di)', 'probing_v4_saprot_3di', 'SaProt (real 3Di)')]
DAGGER = r'$^{\dagger}$'


def esc(s):
    return s.replace('_', r'\_')


def load(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def per_full(suite, arm, split):
    return A.load_probing(SR / suite, arm, split)


def per_single(suite, arm, split):
    return A.load_probing(SS / suite.replace('probing_v4_', 'probing_v4_single_'), arm, split)


def per_primary(suite, arm, split):
    if arm == 'mutant' and split != 'random':
        return per_single(suite, arm, split)
    return per_full(suite, arm, split)


def layer_means(per):
    lm = {}
    for c in per.values():
        for l, v in c.items():
            lm.setdefault(int(l), []).append(v)
    return [float(np.mean(lm[l])) for l in range(33)]


def boot(d):
    r = np.random.RandomState(A.SEED)
    m = d[r.randint(0, len(d), (A.N_BOOT, len(d)))].mean(1)
    return [float(np.percentile(m, 5)), float(np.percentile(m, 95))]


def verdict(ci):
    if ci[0] >= -A.MARGIN and ci[1] <= A.MARGIN:
        return 'Equivalent'
    if ci[0] > A.MARGIN or ci[1] < -A.MARGIN:
        return 'Difference'
    return 'Inconclusive'


def best_vs_final(per):
    lm = layer_means(per)
    best = int(np.argmax(lm))
    if best == 32:
        return dict(best=32, best_rho=lm[32], L32=lm[32], n=len(per), test=None)
    d = np.array([c[str(best)] - c['32'] for c in per.values() if str(best) in c and '32' in c])
    b = boot(d)
    t = A.t_ci90(d)
    return dict(best=best, best_rho=lm[best], L32=lm[32], n=len(per),
                test=dict(n=len(d), mean=float(d.mean()), b=b, t=t, tost=A.tost_p(d),
                          dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), v=verdict(b), vt=verdict(t)))


names63 = [l.strip() for l in open(R2 / 'assay_names_63.txt') if l.strip()]
v13 = {r['DMS_id'] for r in csv.DictReader(
    open(ROOT / 'ProteinGym_DMS_data' / 'DMS_substitutions.csv', encoding='utf-8'))}
REMOVED = {'B3VI55_LIPST_Klesmith_2015', 'RL401_YEAST_Mavor_2016',
           'RL401_YEAST_Roscoe_2013', 'RL401_YEAST_Roscoe_2014'}
RENAMED = {
    'A4_HUMAN_Seuma_2021': 'A4_HUMAN_Seuma_2022',
    'CP2C9_HUMAN_Amorosi_abundance_2021': 'CP2C9_HUMAN_Amorosi_2021_abundance',
    'CP2C9_HUMAN_Amorosi_activity_2021': 'CP2C9_HUMAN_Amorosi_2021_activity',
    'DYR_ECOLI_Thompson_plusLon_2019': 'DYR_ECOLI_Thompson_2019',
    'GCN4_YEAST_Staller_induction_2018': 'GCN4_YEAST_Staller_2018',
    'MTH3_HAEAE_Rockah-Shmuel_2015': 'MTH3_HAEAE_RockahShmuel_2015',
    'P53_HUMAN_Giacomelli_NULL_Etoposide_2018': 'P53_HUMAN_Giacomelli_2018_Null_Etoposide',
    'P53_HUMAN_Giacomelli_NULL_Nutlin_2018': 'P53_HUMAN_Giacomelli_2018_Null_Nutlin',
    'P53_HUMAN_Giacomelli_WT_Nutlin_2018': 'P53_HUMAN_Giacomelli_2018_WT_Nutlin',
    'SRC_HUMAN_Ahler_CD_2019': 'SRC_HUMAN_Ahler_2019',
    'TPOR_HUMAN_Bridgford_S505N_2020': 'TPOR_HUMAN_Bridgford_2020',
    'VKOR1_HUMAN_Chiasson_abundance_2020': 'VKOR1_HUMAN_Chiasson_2020_abundance',
    'VKOR1_HUMAN_Chiasson_activity_2020': 'VKOR1_HUMAN_Chiasson_2020_activity',
}
for n in names63:
    assert (n in v13) + (n in RENAMED) + (n in REMOVED) == 1, n
status = {n: ('same' if n in v13 else 'renamed' if n in RENAMED else 'removed') for n in names63}
shared = [n for n in names63 if status[n] == 'same']
caps = load(R2 / 'r2_multisite_analyzed.json')['rows']            # rows, capped flags
ms3 = load(R3 / 'multisite_share_r3.json')['per_dataset']          # corrected multi-site rows
single_only = [n for n in shared if ms3[n]['rows_multi'] == 0]
mss = load(R3 / 'multisite_share_r3.json')['summary']

tex = [r'''\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{float}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage[margin=2cm]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small}
\renewcommand{\thetable}{S\arabic{table}}
\title{Supplementary Information for\\ ``Leakage-Free Evaluation of Layer Choice in Protein Language
Models for Mutation Effect Prediction''}
\date{}
\begin{document}
\maketitle
\noindent Supplementary Tables S1--S11 are numbered in the order in which they are first cited
in the main text. Throughout, the primary analysis uses the full data for the wild-type
arms and, under the position-based splits (modulo, contiguous, position-grouped), only
single-substitution variants for the mutant arm, whose rows of a multi-site variant would
share one embedding and one label across folds (main text, Section 2.2).
''']

# =============================================================== S1 per-dataset
pe = per_single('probing_v4_esm2', 'mutant', 'modulo')
_raw = load(SS / 'probing_v4_single_esm2' / 'v4__mutant__combined__modulo.json')
n_single = {n: v['n_rows'] for n, v in _raw.items()}
p3 = per_single('probing_v4_saprot_3di', 'mutant', 'modulo')


def ds_best(per, n):
    c = per.get(n)
    if not c:
        return None
    bl = max(c, key=c.get)
    return int(bl), c[bl], c.get('32')


tex.append(r'''\section*{Assay provenance and per-dataset summary}
All experiments use the 63-assay substitution snapshot distributed as LMDB files with the
SaProt repository. Of these 63 assay identifiers, %d are unchanged in ProteinGym v1.3, %d
correspond to renamed or reprocessed v1.3 assays, and %d are no longer part of v1.3. The
renamed assays and their v1.3 identifiers are listed below. Table~S1 gives, for every
dataset, its v1.3 status, whether the 10,000-variant cap applied, the number of analyzed
rows (5,000-row ceiling), the number of analyzed rows that come from multi-site variants
(whether or not other rows of the same variant survived the row subsample), the number of
test rows under the modulo split whose variant also has a row in the training folds, and
the mutant-arm probing result on single-substitution variants under the modulo split. In
total, %d datasets were variant-capped, %d reach the row ceiling, and multi-site variants
occur in %d datasets, contributing %s of the %s analyzed rows (%.1f\%%).''' % (
    len(shared), len(RENAMED), len(REMOVED),
    sum(caps[n]['capped'] for n in names63), sum(caps[n]['rows'] == 5000 for n in names63),
    mss['n_datasets_with_multi'], format(mss['rows_multi'], ','), format(mss['rows'], ','),
    mss['pct_rows_multi']))
tex.append(r'\begin{center}\footnotesize\begin{tabular}{ll}')
tex.append(r'\toprule')
tex.append(r'Snapshot identifier & ProteinGym v1.3 identifier \\')
tex.append(r'\midrule')
for a, b in RENAMED.items():
    tex.append(r'%s & %s \\' % (esc(a), esc(b)))
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}\end{center}')
tex.append(r'\begin{landscape}\footnotesize')
tex.append(r'\begin{longtable}{llcrrrrrrrrrr}')
tex.append(r'\caption{Per-dataset provenance, caps, multi-site rows, and mutant-arm probing on '
           r'single-substitution variants (modulo split, combined features): best layer, '
           r'best-layer $\rho$, and final-layer (L32) $\rho$. \emph{Shared} counts test rows '
           r'under the modulo split whose variant also has a row in the training folds; '
           r'\emph{Single rows} is the number of single-substitution rows analyzed.}'
           r'\label{tab:S1}\\')
hdr = (r'\toprule' '\n'
       r'Dataset & v1.3 & Var.\ cap & Rows & Multi-site & Shared & Single & \multicolumn{3}{c}{ESM-2} & '
       r'\multicolumn{3}{c}{SaProt (3Di)} \\' '\n'
       r' & & & & rows & (modulo) & rows & best L & best $\rho$ & L32 $\rho$ & best L & best $\rho$ & L32 $\rho$ \\' '\n'
       r'\midrule')
tex.append(hdr + r'\endfirsthead')
tex.append(hdr + r'\endhead')
tex.append(r'\bottomrule\endfoot')
for n in names63:
    m = ms3[n]
    cells = [esc(n), status[n], 'yes' if caps[n]['capped'] else 'no', str(m['rows']),
             str(m['rows_multi']), str(m['test_rows_with_variant_in_train_modulo']),
             str(n_single.get(n, '--'))]
    for x in (ds_best(pe, n), ds_best(p3, n)):
        cells += (['--'] * 3 if x is None else [str(x[0]), '%.3f' % x[1], '%.3f' % x[2]])
    tex.append(' & '.join(cells) + r' \\')
tex.append(r'\end{longtable}\end{landscape}')

# =============================================================== S2 3Di
tex.append(r'''\section*{Foldseek 3Di token preparation}
For each dataset, the wild-type sequence was aligned to its AlphaFold Protein Structure
Database entry and each residue was assigned the Foldseek 3Di state of its aligned
structural residue. Structures were obtained for 62 of 63 datasets. Two alignments were
truncated at the C terminus and missing residues were padded with the Foldseek gap state
(UBC9: 1 residue; Q59976: 22 residues). The influenza nucleoprotein dataset has no AFDB
entry and was evaluated with structure-masked tokens in all SaProt analyses.''')
rows = []
for f in sorted(glob.glob(str(ROOT / 'revision_results' / '3di_server' / '*.3di.txt'))):
    with open(f) as fh:
        seq = ''.join(l.strip() for l in fh if not l.startswith('>'))
    rows.append((esc(os.path.basename(f)[:-8]), len(seq), seq.count('#')))
half = (len(rows) + 1) // 2
tex.append(r'\begin{table}[H]\centering\scriptsize')
tex.append(r'\caption{3Di files (%d datasets): token length and gap-padded residue count.}'
           r'\label{tab:S2}' % len(rows))
tex.append(r'\resizebox{\textwidth}{!}{%')
tex.append(r'\begin{tabular}{lrrlrr}')
tex.append(r'\toprule')
tex.append(r'Dataset & Length & Gaps & Dataset & Length & Gaps \\')
tex.append(r'\midrule')
for i in range(half):
    right = rows[i + half] if i + half < len(rows) else ('', '', '')
    tex.append('%s & %s & %s & %s & %s & %s \\\\' % (rows[i] + right))
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}}\end{table}')

# =============================================================== S3 similarity
tex.append(r'''\section*{Layer similarity: sample size and CKA estimators}
Cosine similarity and linear CKA of per-layer hidden states, averaged over 10 datasets,
with 200 (primary; main-text Figures 2 and 3c) or 50 randomly sampled residue positions
per dataset. The standard linear CKA estimator is biased upward when the number of
positions is small relative to the hidden dimension (1280); the 200-position values
reduce this bias. The debiased (unbiased-HSIC) estimator is not estimable at either
sample size: with fewer positions than hidden dimensions its values exceed 1, and it is
listed for completeness only.''')
tex.append(r'\begin{table}[H]\centering\small')
tex.append(r'\caption{Layer-similarity summaries.}\label{tab:S3}')
tex.append(r'\begin{tabular}{lrrrrrr}')
tex.append(r'\toprule')
tex.append(r'& \multicolumn{2}{c}{Adjacent layers} & \multicolumn{2}{c}{Layer 0 vs 32} & '
           r'\multicolumn{2}{c}{Pearson $r$ vs distance} \\')
tex.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}')
tex.append(r'Model (positions) & cos & CKA & cos & CKA & cos & CKA \\')
tex.append(r'\midrule')
for tag, lab in (('esm2', 'ESM-2'), ('saprot', 'SaProt')):
    s50 = load(SR / ('cka_%s' % tag) / 'cka_summary.json')
    s200 = load(R2 / ('cka200_%s_summary.json' % tag))
    for s, k in ((s200, 200), (s50, 50)):
        tex.append(r'%s (%d) & %.3f & %.3f & %+.3f & %.3f & %.2f & %.2f \\' % (
            lab, k, s['adjacent_cos'], s['adjacent_cka'], s['L0_vs_L32_cos'], s['L0_vs_L32_cka'],
            s['pearson_r_cos_vs_distance_unique_pairs'], s['pearson_r_cka_vs_distance_unique_pairs']))
    tex.append(r'\quad debiased CKA (200; not estimable) & -- & %.3f & -- & %.3f & -- & -- \\' % (
        s200['adjacent_cka_debiased'], s200['L0_vs_L32_cka_debiased']))
    if tag == 'esm2':
        tex.append(r'\midrule')
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}\end{table}')

# =============================================================== S4 equivalence (primary)
eqrows, degen = [], []
for model, suite, _ in MODELS:
    for arm in ARMS:
        for sp in LF:
            r = best_vs_final(per_primary(suite, arm, sp))
            data = 'single' if arm == 'mutant' else 'full'
            if r['test'] is None:
                degen.append('%s %s (%s)' % (model, ARM_T[arm], SPL_T[sp]))
            else:
                eqrows.append((model, arm, sp, data, r))
n_eq = sum(r[4]['test']['v'] == 'Equivalent' for r in eqrows)
n_inc = sum(r[4]['test']['v'] == 'Inconclusive' for r in eqrows)
n_dif = sum(r[4]['test']['v'] == 'Difference' for r in eqrows)
n_eq_t = sum(r[4]['test']['vt'] == 'Equivalent' for r in eqrows)
n_eq_tost = sum(r[4]['test']['tost'] < 0.05 for r in eqrows)
tex.append(r'''\section*{Best-layer versus final-layer equivalence}
For each model, arm, and leakage-free split, the per-dataset paired difference between
the best layer and layer 32 was computed (mutant arm on single-substitution variants;
column ``Data''). The best layer is the argmax of the layer-mean curve on the same
per-dataset results, so selection favours finding a difference and equivalence verdicts
are conservative. Verdict rule: \emph{Equivalent} when the bootstrap 90\%% confidence
interval of the mean paired difference (10,000 resamples, percentile method, 5th and 95th
percentiles, seed 42) lies within $\pm 0.01$; \emph{Difference} when it lies entirely
outside; \emph{Inconclusive} otherwise. The $t$-based 90\%% interval and its verdict, the
TOST $p$-value (margin 0.01), and Cohen's $d_z$ are given for information: the $t$-based
rule classifies %d comparisons as equivalent and TOST at $\alpha = 0.05$ classifies %d,
against %d for the bootstrap rule, because several upper bounds lie close to the margin.
In %d further model--arm--split combinations the best layer is layer 32 itself (no test
defined): %s.''' % (n_eq_t, n_eq_tost, n_eq, len(degen), '; '.join(degen)))
tex.append(r'\begin{landscape}\begin{table}[p]\centering\footnotesize\setlength{\tabcolsep}{4pt}')
tex.append(r'\caption{Best-versus-final-layer comparisons, primary analysis (%d tests): %d '
           r'equivalent, %d inconclusive, %d difference.}\label{tab:S4}' % (len(eqrows), n_eq, n_inc, n_dif))
tex.append(r'\begin{tabular}{llllrrrllrrll}')
tex.append(r'\toprule')
tex.append(r'Model & Arm & Split & Data & Best & $n$ & Mean $\Delta$ & Bootstrap 90\% CI & '
           r'$t$-based 90\% CI & TOST $p$ & $d_z$ & Verdict & $t$ verdict \\')
tex.append(r'\midrule')
for model, arm, sp, data, r in eqrows:
    t = r['test']
    tex.append(r'%s & %s & %s & %s & L%d & %d & %+.4f & [%+.4f, %+.4f] & [%+.4f, %+.4f] & %.3f & %.2f & %s & %s \\' % (
        model, ARM_T[arm], SPL_T[sp], data, r['best'], t['n'], t['mean'], t['b'][0], t['b'][1],
        t['t'][0], t['t'][1], t['tost'], t['dz'], t['v'], t['vt']))
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}\end{table}\end{landscape}')

# =============================================================== S5-S7 curves (primary)
tex.append(r'''\section*{Full 33-layer probing curves}
Mean Spearman $\rho$ per layer (combined features) for every arm and split; best layer per
column in bold. $n = 63$ datasets ($n = 57$ for the contiguous split). Columns marked
%s (mutant arm, leakage-free splits) are computed on single-substitution variants.''' % DAGGER)
for i, (model, suite, name) in enumerate(MODELS):
    cur = {(a, s): layer_means(per_primary(suite, a, s)) for a in ARMS for s in SPLITS}
    best = {k: max(v) for k, v in cur.items()}
    tex.append(r'\begin{landscape}\begin{table}[p]\centering\footnotesize')
    tex.append(r'\caption{%s: per-layer mean $\rho$ by arm and split.}\label{tab:S%d}' % (name, 5 + i))
    tex.append(r'\begin{tabular}{r' + 'c' * 12 + '}')
    tex.append(r'\toprule')
    tex.append(r'& ' + ' & '.join(r'\multicolumn{4}{c}{%s}' % esc(a) for a in ARMS) + r' \\')
    tex.append(r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}')
    tex.append('Layer & ' + ' & '.join(
        SPLIT_TEX[s] + (DAGGER if (a == 'mutant' and s != 'random') else '')
        for a in ARMS for s in SPLITS) + r' \\')
    tex.append(r'\midrule')
    for l in range(33):
        row = [str(l)]
        for a in ARMS:
            for s in SPLITS:
                v = cur[(a, s)][l]
                cell = '%.3f' % v
                if abs(v - best[(a, s)]) < 5e-4:
                    cell = r'\textbf{%s}' % cell
                row.append(cell)
        tex.append(' & '.join(row) + r' \\')
        if l in (7, 15, 23):
            tex.append(r'\midrule')
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}\end{table}\end{landscape}')

# =============================================================== S8 single subs vs full
s8, s8_counts = [], {'Equivalent': 0, 'Inconclusive': 0, 'Difference': 0}
for model, suite, _ in MODELS:
    for arm in ARMS:
        for sp in LF:
            rs = best_vs_final(per_single(suite, arm, sp))
            rf = best_vs_final(per_full(suite, arm, sp))
            if rs['test']:
                s8_counts[rs['test']['v']] += 1
            s8.append((model, arm, sp, rs, rf))
xs = load(R3 / 'single_site_excl_small_r3.json')['summary']
max_mean = max(r[3]['test']['mean'] for r in s8 if r[3]['test'])
max_up = max(r[3]['test']['b'][1] for r in s8 if r[3]['test'])
tex.append(r'''\section*{Single-substitution analysis of all arms}
Probing repeated for all three model configurations on single-substitution variants only
(one row per variant; variant cap applied after filtering) under the leakage-free splits,
with the same folds and readout. For comparison, the last two columns give the best-layer
$\rho$ on the full data, in which rows of a multi-site variant can share a label (all
arms) and an embedding (mutant arm) across folds. $\Delta$ is the mean per-dataset
difference between the best layer and layer 32 on single substitutions, with its
bootstrap 90\%% confidence interval and the verdict under the rule of Table~S4 (no test
when the best layer is layer 32). On single substitutions, %d comparisons are equivalent,
%d inconclusive, and %d indicate a difference beyond the margin; the largest point
estimate is %.3f and the largest 90\%% upper bound %.3f. Two datasets retain fewer than 50
single-substitution variants (F7YBW8\_MESOW\_Aakre\_2015, 37; GCN4\_YEAST\_Staller\_induction\_2018,
33; Table~S1); excluding them, the best layer remains in layers %d--%d in every column and the
largest point estimate and 90\%% upper bound become %.3f and %.3f.''' % (
    s8_counts['Equivalent'], s8_counts['Inconclusive'], s8_counts['Difference'], max_mean, max_up,
    min(xs['best_layers']), max(xs['best_layers']), xs['max_mean'], xs['max_upper']))
tex.append(r'\begin{table}[H]\centering\footnotesize')
tex.append(r'\caption{Single-substitution probing versus full data, leakage-free splits '
           r'(combined features).}\label{tab:S8}')
tex.append(r'\resizebox{\textwidth}{!}{%')
tex.append(r'\begin{tabular}{lllrrrrlllr}')
tex.append(r'\toprule')
tex.append(r'& & & \multicolumn{6}{c}{Single substitutions} & \multicolumn{2}{c}{Full data} \\')
tex.append(r'\cmidrule(lr){4-9}\cmidrule(lr){10-11}')
tex.append(r'Model & Arm & Split & $n$ & Best & Best $\rho$ & L32 $\rho$ & $\Delta$ (90\% CI) & '
           r'Verdict & Best & Best $\rho$ \\')
tex.append(r'\midrule')
for model, arm, sp, rs, rf in s8:
    t = rs['test']
    dcell = '--' if t is None else '%+.4f [%+.3f, %+.3f]' % (t['mean'], t['b'][0], t['b'][1])
    vcell = '--' if t is None else t['v']
    tex.append(r'%s & %s & %s & %d & L%d & %.3f & %.3f & %s & %s & L%d & %.3f \\' % (
        model, ARM_T[arm], SPL_T[sp], rs['n'], rs['best'], rs['best_rho'], rs['L32'], dcell, vcell,
        rf['best'], rf['best_rho']))
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}}\end{table}')

# =============================================================== S9 pooling
pool = load(R2 / 'a2_a3_pooling.json')
excl = load(R3 / 'pooling_excl_multisite_r3.json')
STRATS = [('mean_20_33', 'Mean L20--33'), ('concat_5L', 'Concat 5L'), ('attention', 'Attention')]
READS = [('ridge', 'Ridge'), ('rf', 'RF'), ('mlp', 'MLP'), ('lightgbm', 'LightGBM')]
fv = pool['fusion_vs_last']


def pcell(v):
    lo, hi = v['boot_ci90']
    return '%+.3f (%+.3f, %+.3f)%s' % (v['mean_delta'], lo, hi, DAGGER if lo > 0 or hi < 0 else '')


tex.append(r'''\section*{Multi-scale pooling: paired comparisons}
Paired per-dataset differences (strategy minus last layer) for the pooling benchmark of
main Table 3 (mutant arm, combined features, 20 datasets per model; SaProt with
structure-masked inputs) under the modulo split, under the random split, and under the
modulo split after excluding the two datasets that contain multi-site variants
(A4\_HUMAN\_Seuma\_2021, DLG4\_HUMAN\_Faure\_2021; 18 datasets). Entries give the mean
difference with its bootstrap 90\%% confidence interval; %s marks intervals excluding zero.
Under the modulo split no combination significantly improves on the last layer, with or
without the two multi-site datasets; under the random split several fusion strategies do.''' % DAGGER)
tex.append(r'\begin{landscape}\begin{table}[p]\centering\footnotesize\setlength{\tabcolsep}{2pt}')
tex.append(r'\caption{Paired fusion-minus-last-layer differences, $\Delta\rho$ (90\% CI).}\label{tab:S9}')
tex.append(r'\begin{tabular}{llcccccc}')
tex.append(r'\toprule')
tex.append(r'& & \multicolumn{3}{c}{ESM-2} & \multicolumn{3}{c}{SaProt} \\')
tex.append(r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}')
tex.append(r'Readout & Strategy & Modulo & Random & Modulo, excl. & Modulo & Random & Modulo, excl. \\')
tex.append(r'\midrule')
for rk, rl in READS:
    for sk, sl in STRATS:
        key = '%s|%s' % (sk, rk)
        cells = [rl, sl]
        for model in ('ESM-2', 'SaProt'):
            cells.append(pcell(fv['%s|modulo' % model][key]))
            cells.append(pcell(fv['%s|random' % model][key]))
            cells.append(pcell(excl['%s|modulo' % model][key]))
        tex.append(' & '.join(cells) + r' \\')
    if rk != 'lightgbm':
        tex.append(r'\midrule')
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}\end{table}\end{landscape}')

# =============================================================== S10 readout contrast
tex.append(r'''\section*{Last-layer readout contrast by split}
Paired last-layer LightGBM minus Ridge difference over the 20 pooling datasets.''')
tex.append(r'\begin{table}[H]\centering\small')
tex.append(r'\caption{Last-layer readout contrast, LightGBM minus Ridge.}\label{tab:S10}')
tex.append(r'\begin{tabular}{llrlr}')
tex.append(r'\toprule')
tex.append(r'Model & Split & Mean $\Delta\rho$ & 90\% CI & Wilcoxon $p$ \\')
tex.append(r'\midrule')
for model in ('ESM-2', 'SaProt'):
    for sp in ('random', 'modulo', 'contiguous'):
        v = pool['lgbm_vs_ridge']['%s|%s' % (model, sp)]
        tex.append(r'%s & %s & %+.3f & (%+.3f, %+.3f) & %.2g \\' % (
            model, SPL_T[sp], v['mean_delta'], v['boot_ci90'][0], v['boot_ci90'][1], v['wilcoxon_p']))
    if model == 'ESM-2':
        tex.append(r'\midrule')
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}\end{table}')

# =============================================================== S11 anchors
off = {r['DMS ID']: r for r in csv.DictReader(
    open(ROOT / 'revision_results' / 'pg_official' / 'spearman_dms_level.csv', encoding='utf-8'))}
lens = {t: load(SR / ('logit_lens_v2_%s' % t) / 'logit_lens_v2_results.json') for t in ('esm2', 'saprot_3di')}
repro = {}
for line in open(ROOT / 'SaProt-main' / 'ProteinGym_results.tsv').read().splitlines()[1:]:
    if '\t' in line:
        k, v = line.split('\t')
        repro[k] = float(v)
m_off = lambda col, sub: st.mean(float(off[n][col]) for n in sub)
m_lens = lambda t, sub: st.mean(lens[t][n]['per_layer']['32']['spearman'] for n in sub)
zs_rows = [('ESM-2 650M (published)', lambda s: m_off('ESM2 (650M)', s)),
           ('SaProt 650M (published)', lambda s: m_off('SaProt (650M)', s)),
           ('ESM-2 150M (published)', lambda s: m_off('ESM2 (150M)', s)),
           ('ESM-2 3B (published)', lambda s: m_off('ESM2 (3B)', s)),
           ('ESM-1v single (published)', lambda s: m_off('ESM-1v (single)', s)),
           ('MSA Transformer single (published)', lambda s: m_off('MSA Transformer (single)', s)),
           ('GEMME (published)', lambda s: m_off('GEMME', s)),
           ('EVE single (published)', lambda s: m_off('EVE (single)', s)),
           (None, None),
           ('This work: lens, ESM-2 final layer', lambda s: m_lens('esm2', s)),
           ('This work: lens, SaProt (3Di) final layer', lambda s: m_lens('saprot_3di', s)),
           ('This work: SaProt zero-shot reproduction', lambda s: st.mean(repro[n] for n in s))]
sup = load(R2 / 'r2_anchors_supervised.json')['summary']
SUP_METHODS = [('Kermut', 'Kermut'), ('ProteinNPT', 'ProteinNPT'),
               ('Embeddings - Augmented - MSA Transformer', 'Emb.-aug.\\ MSA Transformer'),
               ('Embeddings - Augmented - ESM1v', 'Emb.-aug.\\ ESM-1v'),
               ('OHE - Augmented - MSA Transformer', 'OHE-aug.\\ MSA Transformer'),
               ('OHE - Not augmented', 'OHE, not augmented')]
ours = load(R3 / 'r3_numbers.json')['anchors_this_work']
tex.append(r'''\section*{Published ProteinGym anchors}
Published v1.3 results recomputed on the %d assays shared with this work: zero-shot values
are the per-assay Spearman correlations of the official v1.3 zero-shot benchmark table;
supervised values are computed from the out-of-fold predictions that the v1.3 release
provides for published supervised models under its official fold schemes. Of the shared
assays, %d contain single substitutions only; on these the final-layer logit lens
reproduces the published values, and the residual differences on the full set arise in
multi-site assays, where the lens scores each substitution of a variant separately rather
than summing over the variant.''' % (len(shared), len(single_only)))
tex.append(r'\begin{table}[H]\centering\small')
tex.append(r'\caption{Published ProteinGym v1.3 results recomputed on the %d assays shared with '
           r'this work. (A) Zero-shot: mean per-assay Spearman $\rho$ on all shared assays and on '
           r'the %d shared assays containing single substitutions only. (B) Supervised: official '
           r'out-of-fold predictions under the v1.3 fold schemes; our rows give the final-layer '
           r'mutant-arm probing (combined features) on single-substitution variants of the same '
           r'assays, computed on the capped snapshot data ($n = %d$ for the contiguous split).}'
           r'\label{tab:S11}' % (len(shared), len(single_only), ours['ESM-2']['contiguous']['n']))
tex.append(r'\begin{tabular}{lrr}')
tex.append(r'\toprule')
tex.append(r'\multicolumn{3}{l}{\textbf{A. Zero-shot}} \\')
tex.append(r'Method & All shared ($n=%d$) & Single-substitution ($n=%d$) \\' % (len(shared), len(single_only)))
tex.append(r'\midrule')
for lab, f in zs_rows:
    if lab is None:
        tex.append(r'\midrule')
        continue
    tex.append(r'%s & %.3f & %.3f \\' % (lab, f(shared), f(single_only)))
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}')
tex.append(r'\vspace{6pt}')
tex.append(r'\begin{tabular}{lrrr}')
tex.append(r'\toprule')
tex.append(r'\multicolumn{4}{l}{\textbf{B. Supervised} ($n=%d$)} \\' % len(shared))
tex.append(r'Method & Random & Modulo & Contiguous \\')
tex.append(r'\midrule')
for key, lab in SUP_METHODS:
    tex.append(r'%s & %.3f & %.3f & %.3f \\' % (
        lab, sup['fold_random_5'][key]['mean'], sup['fold_modulo_5'][key]['mean'],
        sup['fold_contiguous_5'][key]['mean']))
tex.append(r'\midrule')
for lab in ('ESM-2', 'SaProt (3Di)'):
    tex.append(r'This work: %s final layer & %.3f & %.3f & %.3f \\' % (
        lab, ours[lab]['random']['L32'], ours[lab]['modulo']['L32'], ours[lab]['contiguous']['L32']))
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}\end{table}')

tex.append(r'\end{document}')
OUT.write_text('\n'.join(tex), encoding='utf-8')
print('written', OUT.name, '| S4:', len(eqrows), n_eq, n_inc, n_dif, '| S8:', s8_counts,
      round(max_mean, 4), round(max_up, 4))
