"""Build manuscript_revised_R3.tex from manuscript_revised_R2.tex (the R2
submission, BIOINF-2026-2348.R2) with minimal, auditable edits addressing the
01-Nov-2026 reviewer report. Every edit is an exact, unique string
replacement; the script aborts if any anchor is missing or ambiguous.

New text is wrapped in \\rthree{} (red). Earlier \\revised{} and \\rtwo{}
markup is rendered black. Pass --clean to render \\rthree black as well.

Numbers come from revision_results/r3_analysis/ (compute_r3_numbers.py,
multisite_share_r3.py, pooling_excl_multisite_r3.py, equivalence_check_r3.py)
and revision_results/r2_analysis/ (a2_a3_pooling.json, cka200_*_summary.json).
Plan A: under the position-based splits the mutant arm is analyzed on
single-substitution variants only (reviewer R2.1).

Supplementary tables are renumbered in order of first citation (R2.m2):
S1 per-dataset, S2 3Di, S3 similarity, S4 equivalence, S5-S7 curves,
S8 single-substitution vs full data, S9 pooling paired, S10 readout contrast,
S11 published anchors.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "manuscript_revised_R2.tex"
CLEAN = "--clean" in sys.argv
OUT = ROOT / ("manuscript_revised_R3_clean.tex" if CLEAN else "manuscript_revised_R3.tex")

E = []


def ed(tag, old, new):
    E.append((tag, old, new))


# ------------------------------------------------------------------ preamble
ed("preamble",
   "% First-round (R1) changes are no longer highlighted; second-round (R2)\n"
   "% changes are shown in red via \\rtwo{}.\n"
   r"\newcommand{\revised}[1]{#1}" "\n"
   r"\newcommand{\rtwo}[1]{\textcolor{red}{#1}}",
   "% Changes from earlier rounds are no longer highlighted; third-round (R3)\n"
   "% changes are shown in red via \\rthree{}.\n"
   r"\newcommand{\revised}[1]{#1}" "\n"
   r"\newcommand{\rtwo}[1]{#1}" "\n"
   + (r"\newcommand{\rthree}[1]{#1}" if CLEAN else r"\newcommand{\rthree}[1]{\textcolor{red}{#1}}"))

# ------------------------------------------------------------------ abstract (R2.2, R2.m7)
ed("abstract_bestlayer",
   r"""the best layer (layers 27--32) exceeded the final layer by at most
$\Delta\rho = 0.009$ (within $0.02$ on single substitutions). No pooling strategy beat
the final layer under leakage-free evaluation. Real 3Di structure tokens improved
SaProt slightly under the random split and substantially under leakage-free splits
($+0.06$ to $+0.12$).""",
   r"""the best layer (layers 27--32) exceeded the final layer by at most
\rthree{$\Delta\rho = 0.019$ (90\% upper bound $0.036$). This deficit is small relative
to the cross-dataset standard deviation of $0.17$. Multi-scale pooling did not reliably
improve on the final layer under leakage-free evaluation (1 of 96 comparisons
significantly better, 55 worse), whereas fusion gains appeared under the random
split.} Real 3Di structure tokens improved
SaProt slightly under the random split and substantially under leakage-free splits
(\rthree{$+0.06$ to $+0.13$}).""")

# ------------------------------------------------------------------ 2.1 (R2.m2 renumbering)
ed("s5_status",
   r"no longer included (per-assay status in Supplementary Table S5).",
   r"no longer included (per-assay status in Supplementary Table \rthree{S1}).")
ed("s7_3di",
   r"3Di token lengths and gap counts in Supplementary Table S7)}.}",
   r"3Di token lengths and gap counts in Supplementary Table \rthree{S2})}.}")

# ------------------------------------------------------------------ 2.2 mutant arm (R2.1)
ed("mutant_arm_def",
   r"""multi-site variants)}; it is the only arm whose representation differs between
two mutations at the same position, and therefore the only one that can carry
mutation-specific information beyond position identity.""",
   r"""multi-site variants)}; it is the only arm whose representation differs between
two mutations at the same position, and therefore the only one that can carry
mutation-specific information beyond position identity\rthree{ and the auxiliary
substitution features}.""")
ed("position_split_multisite",
   r"""In these position-based splits a row's
fold is determined by its own substitution position, so rows of the same multi-site
variant can fall into different folds and share a fitness label across folds; we
therefore also report a single-substitution-only sensitivity analysis
(Section~\ref{sec:boundary}).}""",
   r"""In these position-based splits a row's
fold is determined by its own substitution position, so rows of the same multi-site
variant can fall into different folds and share a fitness label across folds\rthree{.
In the mutant arm, such rows also share the variant's averaged embedding. Under the
position-based splits the mutant arm is therefore analyzed on single-substitution
variants only (one row per variant, with the variant cap applied after filtering),
and this is the primary mutant-arm analysis throughout. The wild-type arms, which share
only the label, are reported on the full data, and all three arms are additionally
reported on single substitutions (Section~\ref{sec:boundary}).}}""")
ed("multisite_share",
   r"""The analysis comprises 213,393 rows from 203,056
variants; multi-site variants occur in 10 of the 63 datasets and account for 3.8\% of
analyzed variants and 8.4\% of analyzed rows (per-dataset counts in Supplementary
Table S5).}""",
   r"""The analysis comprises 213,393 rows from 203,056
variants; multi-site variants occur in 10 of the 63 datasets\rthree{, where they make
up most variants, and 49,131 analyzed rows (23.0\%) come from multi-site variants
(per-dataset counts in Supplementary Table S1). Because the 5,000-row subsample is
drawn over rows, most of these rows appear without the other rows of their variant
but carry the fitness of the full multi-mutant.}}""")

# ------------------------------------------------------------------ 2.3 pooling (R2.m3, R2.1)
ed("pooling_protocol",
   r"""evaluated on the \textbf{mutant} arm under the modulo split as the primary
leakage-free protocol:""",
   r"""evaluated on the \textbf{mutant} arm under the modulo split as the primary
leakage-free protocol\rthree{, and additionally under the contiguous and random splits
and on the masked\_wt arm with the same pipeline}:""")
ed("pooling_folds",
   r"""using the same five cross-validation folds as the probing
experiment for every dataset}.""",
   r"""using the same five cross-validation folds as the probing
experiment for every dataset}.\rthree{ Two of the 20 datasets (A4\_HUMAN\_Seuma\_2021
and DLG4\_HUMAN\_Faure\_2021) contain multi-site variants; because strategies are
compared within the same arm and rows, we report a sensitivity analysis excluding
these two datasets rather than restricting the benchmark to single substitutions.}""")

# ------------------------------------------------------------------ 2.5 similarity (R2.m4)
ed("similarity_methods",
   r"""\revised{We computed the $33 \times 33$ cosine similarity matrix of per-layer hidden states,
averaged over 50 randomly selected residue positions across 10 datasets.""",
   r"""\revised{We computed the $33 \times 33$ cosine similarity matrix of per-layer hidden states,
averaged over \rthree{200 randomly selected residue positions in each of} 10 datasets.""")
ed("similarity_bias",
   r"""\rtwo{Because the standard CKA estimator is biased upward when the number of positions
is small relative to the hidden dimension, we repeated the analysis with 200 positions
per dataset (Supplementary Table S10).}""",
   r"""\rthree{The standard CKA estimator is biased upward when the number of positions is
small relative to the hidden dimension, so we report it at 200 positions per dataset,
with a 50-position analysis for comparison. With fewer positions than hidden dimensions
the debiased estimator is not estimable (its values exceed 1), and it is listed for
completeness only (Supplementary Table S3).}""")

# ------------------------------------------------------------------ 2.6 equivalence (R2.2)
ed("equivalence_count",
   r"""In 7 of the 27
model--arm--split combinations the best layer was the final layer itself, leaving 20
comparisons (Supplementary Table S6);}""",
   r"""\rthree{The mutant arm enters these
comparisons on single substitutions (Section 2.2). In 6 of the 27
model--arm--split combinations the best layer was the final layer itself, leaving 21
comparisons (Supplementary Table S4). Because several intervals end close to the
margin, the verdicts obtained with the $t$-based interval and with TOST are reported
alongside;}}""")

# ------------------------------------------------------------------ 3.1 similarity results (R2.m4)
ed("r31_positions",
   r"""cosine similarity matrix of per-layer hidden states, averaged over 50 randomly selected
residue positions across 10 datasets (Figure~\ref{fig:similarity}).""",
   r"""cosine similarity matrix of per-layer hidden states, averaged over \rthree{200} randomly selected
residue positions \rthree{in each of} 10 datasets (Figure~\ref{fig:similarity}).""")
ed("fig2_caption",
   r"""Averaged over 10 datasets and 50 random residue positions per dataset.
(\textbf{a}) Cosine: adjacent layers are highly similar ($\cos=0.98$); mean similarity
decays monotonically with distance and collapses to $-0.01$ only between the most
distant layers (Layer~0 vs.~32). (\textbf{b}) Linear CKA over the same hidden states:
the most distant layers retain CKA 0.81, indicating a rotation that preserves linearly
decodable information rather than information loss. \rtwo{Both panels share a
colour scale from 0.5 to 1 with contour lines at 0.7--0.9; cells below 0.5 (the
final-layer row and column in panel a) are white.}}}""",
   r"""Averaged over 10 datasets and \rthree{200} random residue positions per dataset.
(\textbf{a}) Cosine: adjacent layers are highly similar ($\cos=0.98$); mean similarity
decays monotonically with distance and collapses to $-0.01$ only between the most
distant layers (Layer~0 vs.~32). (\textbf{b}) Linear CKA over the same hidden states:
the most distant layers retain CKA \rthree{0.78}, indicating a rotation that preserves linearly
decodable information rather than information loss. \rtwo{Both panels share a
colour scale from 0.5 to 1 with contour lines at 0.7--0.9; cells below 0.5 (the
final-layer row and column in panel a\rthree{, and parts of the layer-0 and layer-1 rows
and columns in panel b}) are white.}}}""")
ed("r31_body",
   r"""Adjacent layers
were highly similar ($\cos = 0.979 \pm 0.011$), and similarity \rtwo{decreased with
layer distance (Pearson $r$ over unique layer pairs: $-0.50$ for SaProt, $-0.93$ for
ESM-2)}. \revised{The decay was smooth and
monotonic, with no intermediate trough: for SaProt the mean cosine by distance fell
from 0.97 ($d=2$) through 0.80 ($d=20$) to 0.50 ($d=30$), collapsing to $-0.009$
between the most distant layers (Layer~0 vs.\ Layer~32). ESM-2 decayed more gently and
remained positive throughout (0.93 at $d=2$, 0.56 at $d=32$). Crucially, near-zero raw
cosine does not imply lost information: the linear CKA between SaProt layers 0 and 32
was 0.808 while their cosine was $-0.009$ (ESM-2: CKA 0.554, cosine 0.558; adjacent-layer
CKA 0.97 for both models). \rtwo{With 200 positions per dataset, which reduces the
upward small-sample bias of the standard estimator, the distant-layer CKA was lower
(SaProt 0.78, ESM-2 0.41) but remained high for SaProt while its cosine stayed at
$-0.008$; the debiased estimator is unstable at this sample size and gave higher, not
lower, values (Supplementary Table S10).} The depth-wise transformation is therefore
well described as a rotation of the representation that \rtwo{retains much} linearly
decodable information (Figure~\ref{fig:probing}c).}""",
   r"""Adjacent layers
were highly similar (\rthree{mean adjacent-layer cosine 0.979 for SaProt and 0.962 for
ESM-2}), and similarity \rtwo{decreased with
layer distance (Pearson $r$ over unique layer pairs: $-0.50$ for SaProt, $-0.93$ for
ESM-2)}. \revised{The decay was smooth and
monotonic, with no intermediate trough: for SaProt the mean cosine by distance fell
from 0.97 ($d=2$) through \rthree{0.81} ($d=20$) to 0.50 ($d=30$), collapsing to \rthree{$-0.008$}
between the most distant layers (Layer~0 vs.\ Layer~32). ESM-2 decayed more gently and
remained positive throughout (0.93 at $d=2$, \rthree{0.54} at $d=32$). Crucially, near-zero raw
cosine does not imply lost information: the linear CKA between SaProt layers 0 and 32
was \rthree{0.78} while their cosine was \rthree{$-0.008$ (ESM-2: CKA 0.41, cosine 0.54; adjacent-layer
CKA 0.97 for SaProt and 0.95 for ESM-2). With 50 positions per dataset the standard
estimator, which is biased upward when positions are fewer than hidden dimensions, gave
higher distant-layer values (SaProt 0.81, ESM-2 0.55); the debiased estimator is not
estimable at either sample size (Supplementary Table S3). For SaProt,} the depth-wise
transformation is therefore
well described as a rotation of the representation that \rtwo{retains much} linearly
decodable information (Figure~\ref{fig:probing}c)\rthree{; for ESM-2, the most distant
layers share less linear structure (CKA 0.41) despite a positive raw cosine}.}""")

# ------------------------------------------------------------------ 3.2 (R2.1, R2.2)
ed("r32_curves_ref",
   r"""with depth (Supplementary Tables S2--S4; Figure~\ref{fig:position_level}b)""",
   r"""with depth (Supplementary Tables \rthree{S5--S7}; Figure~\ref{fig:position_level}b)""")
ed("r32_third",
   r"""\rtwo{Third, the best layer is at most marginally better than the final layer: the mean
per-dataset difference never exceeded $\Delta\rho = 0.009$; of the 20 combinations in
which the best layer was not the final layer, 8 were equivalent within the $\pm 0.01$
margin, 12 were inconclusive, and none indicated a difference beyond the margin
(Supplementary Table S6).""",
   r"""\rtwo{Third, the best layer is at most marginally better than the final layer: \rthree{the mean
per-dataset difference was at most $\Delta\rho = 0.009$ for the wild-type arms and at most
$0.019$ for the mutant arm (ESM-2, position-grouped split; 90\% upper bound $0.036$). Of
the 21 combinations in which the best layer was not the final layer, 7 were equivalent
within the $\pm 0.01$ margin, 14 were inconclusive, and none indicated a difference
beyond the margin (Supplementary Table S4). Three of the seven equivalence verdicts have
upper bounds within $0.0005$ of the margin, and the $t$-based interval and TOST each
classify one fewer comparison as equivalent, so these verdicts depend on the interval
method. Deficits of this size are small compared with the cross-dataset standard
deviation of about $0.17$.}""")
ed("r32_mutant",
   r"""The mutant arm exceeded the masked arms in every split (e.g., 0.681 versus 0.606
under random, 0.483 versus 0.459 under modulo for ESM-2), confirming that
mutation-specific signal exists and is recoverable once the representation actually
depends on the mutation.}""",
   r"""\rthree{The mutant arm exceeded the wild-type arms under the random split (ESM-2:
0.681 versus 0.606) but not under leakage-free evaluation on single substitutions. There
it was below both wild-type arms for ESM-2 (modulo, best layer: 0.443 versus 0.476 and
0.492) and level with them for SaProt with real 3Di tokens (0.492 versus 0.484 and
0.500; Supplementary Table S8). Under leakage-free evaluation, the mutant-sequence
embedding therefore did not outperform the wild-type context embedding combined with
the auxiliary substitution features.}}""")

# Table 2: mutant rows under leakage-free splits now on single substitutions
ed("tab2_caption",
   r"""layer is not layer 32). $n = 63$; contiguous $n = 57$. Full 33-layer curves in
Supplementary \rtwo{Tables S2--S4}; per-dataset results in Supplementary Table \rtwo{S5}.}}""",
   r"""layer is not layer 32). $n = 63$; contiguous $n = 57$. \rthree{Mutant-arm entries under
the leakage-free splits ($^{\dagger}$) are computed on single-substitution variants
(Section 2.2).} Full 33-layer curves in
Supplementary \rthree{Tables S5--S7}; per-dataset results in Supplementary Table \rthree{S1}.}}""")
ed("tab2_esm2_mutant",
   r"ESM-2 & mutant       & 0.681 (32) & 0.483 (31) $\rightarrow$ 0.479 & 0.365 (31) $\rightarrow$ 0.360 & 0.477 (31) $\rightarrow$ 0.475 \\",
   r"ESM-2 & mutant       & 0.681 (32) & \rthree{0.443 (30) $\rightarrow$ 0.427$^{\dagger}$} & \rthree{0.353 (31) $\rightarrow$ 0.346$^{\dagger}$} & \rthree{0.444 (30) $\rightarrow$ 0.425$^{\dagger}$} \\")
ed("tab2_sapm_mutant",
   r"SaProt (mask) & mutant       & 0.652 (23) $\rightarrow$ 0.651 & 0.465 (30) $\rightarrow$ 0.460 & 0.323 (32) & 0.461 (30) $\rightarrow$ 0.458 \\",
   r"SaProt (mask) & mutant       & 0.652 (23) $\rightarrow$ 0.651 & \rthree{0.434 (30) $\rightarrow$ 0.428$^{\dagger}$} & \rthree{0.310 (30) $\rightarrow$ 0.307$^{\dagger}$} & \rthree{0.429 (30) $\rightarrow$ 0.423$^{\dagger}$} \\")
ed("tab2_sap3_mutant",
   r"SaProt (3Di) & mutant       & 0.684 (26) $\rightarrow$ 0.683 & 0.531 (31) $\rightarrow$ 0.525 & 0.451 (31) $\rightarrow$ 0.445 & 0.532 (31) $\rightarrow$ 0.524 \\",
   r"SaProt (3Di) & mutant       & 0.684 (26) $\rightarrow$ 0.683 & \rthree{0.492 (30) $\rightarrow$ 0.486$^{\dagger}$} & \rthree{0.449 (31) $\rightarrow$ 0.439$^{\dagger}$} & \rthree{0.490 (31) $\rightarrow$ 0.481$^{\dagger}$} \\")

# ------------------------------------------------------------------ 3.3 pooling (R2.1, R2.m3, R2.m7)
ed("r33_paired",
   r"""\rtwo{Paired per-dataset comparisons confirm
this (Supplementary Table S8): no strategy--readout combination
exceeded the last layer (largest mean difference $+0.018$, Concat~5L with Ridge, 90\%
CI $-0.008$ to $+0.044$), and 14 of 24 combinations, all with nonlinear readouts, were
significantly worse (90\% CI below zero).}}""",
   r"""\rtwo{Paired per-dataset comparisons confirm
this (Supplementary Table \rthree{S9}): no strategy--readout combination
\rthree{significantly improved on} the last layer (largest mean difference $+0.018$, Concat~5L with Ridge, 90\%
CI $-0.008$ to $+0.044$\rthree{, which does not exclude a gain of that size}), and 14 of 24 combinations, all with nonlinear readouts, were
significantly worse (90\% CI below zero).} \rthree{Excluding the two datasets that
contain multi-site variants left this unchanged (no combination significantly better;
17 of 24 significantly worse). Under the random split, by contrast, some fusion
strategies did exceed the last layer: Concat~5L for ESM-2 with Ridge, Random Forest,
and LightGBM ($+0.009$ to $+0.012$, 90\% CIs excluding zero), and 10 of 12
strategy--readout combinations for SaProt ($+0.010$ to $+0.064$).} \rthree{The same
benchmark on the masked\_wt arm gave a similar picture (Supplementary Table S9). Under
the modulo and contiguous splits no combination significantly improved on the last
layer for ESM-2, and one did for SaProt (Concat~5L with Ridge, modulo split, $+0.020$,
90\% CI $+0.006$ to $+0.034$). Across both arms, both models, and both position-based
fold schemes, 1 of 96 strategy--readout comparisons favoured fusion and 55 favoured
the last layer; one gain among 96 intervals is fewer than expected by chance at this
interval level.}}""")
ed("tab3_caption",
   r"""Spreads across strategies: ESM-2 Ridge 0.041, RF 0.057, MLP 0.045, LightGBM 0.050;
SaProt Ridge 0.005, RF 0.010, MLP 0.039, LightGBM 0.014.}}""",
   r"""Spreads across strategies: ESM-2 Ridge 0.041, RF 0.057, MLP 0.045, LightGBM 0.050;
SaProt Ridge 0.005, RF 0.010, MLP 0.039, LightGBM 0.014.\rthree{ Two of the 20 datasets
contain multi-site variants; excluding them does not change the paired comparisons
(Supplementary Table S9).}}}""")
ed("r331_title",
   r"\subsubsection{\rtwo{Readout Choice Matters More Than Layer Fusion for ESM-2}}",
   r"\subsubsection{\rtwo{Readout Choice Matters More Than Layer Fusion for ESM-2}\rthree{ Under Leakage-Free Evaluation}}")
ed("r331_body",
   r"""($+0.016$, CI $-0.016$ to $+0.048$, $p = 0.57$; Supplementary Table S9).}""",
   r"""($+0.016$, CI $-0.016$ to $+0.048$, $p = 0.57$; Supplementary Table \rthree{S10}).}
\rthree{The readout ranking is itself protocol-dependent: under the random split,
last-layer LightGBM was below Ridge for both models (ESM-2 $-0.035$, SaProt $-0.065$;
90\% CIs excluding zero).}""")
ed("fig3a_caption",
   r"""under leakage-free splits (modulo,
mutant arm) performance is lower and the best layers are consistently late (layers 30--32,
$\Delta\rho \leq 0.009$ versus the final layer).}""",
   r"""under leakage-free splits (modulo,
mutant arm\rthree{ on single substitutions}) performance is lower and the best layers are
\rthree{late (layer 30 for both models).}}""")
ed("fig3c_caption",
   r"""ESM-2 remains positive; linear CKA between the most distant SaProt layers remains 0.81
(Figure~\ref{fig:similarity}), indicating rotation with preserved linear information
rather than loss.}}""",
   r"""ESM-2 remains positive; linear CKA between the most distant SaProt layers remains \rthree{0.78}
(Figure~\ref{fig:similarity}), indicating rotation with preserved linear information
rather than loss\rthree{ (200 positions per dataset)}.}}""")

# ------------------------------------------------------------------ 3.4 lens and anchors (R2.1, R2.m7)
ed("r34_zeroshot_vs_sup",
   r"""Zero-shot absolute
levels remain below supervised probing (0.44 versus up to 0.53),""",
   r"""Zero-shot absolute
levels remain below supervised probing (0.44 versus up to \rthree{0.49}),""")
ed("r34_anchor_ref",
   r"""on the 46 assays shared with our snapshot (Supplementary Table S1).""",
   r"""on the 46 assays shared with our snapshot (Supplementary Table \rthree{S11}).""")
ed("r34_supervised",
   r"""On the same 46 assays, Kermut reaches 0.605 and ProteinNPT 0.566 under
the modulo split (0.561 and 0.525 under the contiguous split), above our leakage-free
probing range (0.27--0.53), whereas""",
   r"""On the same 46 assays, Kermut reaches 0.605 and ProteinNPT 0.566 under
the modulo split (0.561 and 0.525 under the contiguous split), above our leakage-free
probing \rthree{on the same assays (0.24--0.47 across models and arms)}, whereas""")
ed("r34_supervised_ours",
   r"""split, an external counterpart of our position-lookup control.}""",
   r"""split, an external counterpart of our position-lookup control.} \rthree{On these
assays, our final-layer mutant arm on single substitutions reaches 0.407 (ESM-2) and
0.469 (SaProt with real 3Di) under the modulo split (Supplementary Table S11).}""")

# ------------------------------------------------------------------ 3.5 boundary conditions
ed("r351_levels",
   r"""modulo and position-grouped give similar levels (e.g., SaProt real-3Di
mutant arm: $0.531$ and $0.532$) while contiguous is uniformly hardest ($0.451$),""",
   r"""modulo and position-grouped give similar levels (e.g., SaProt real-3Di
\rthree{unmasked\_wt arm: $0.473$ and $0.467$}) while contiguous is uniformly hardest (\rthree{$0.415$}),""")
ed("r351_inflation",
   r"""therefore inflates absolute performance estimates by roughly
$0.15$--$0.20$ $\rho$ in these arms""",
   r"""therefore inflates \rthree{final-layer} performance estimates by
\rthree{$0.14$--$0.26$ $\rho$ relative to the modulo and position-grouped splits and by
$0.20$--$0.34$ relative to the contiguous split}""")
ed("r352_title",
   r"\subsubsection{\rtwo{Multi-Site Variants: Single-Substitution Sensitivity Analysis}}",
   r"\subsubsection{\rtwo{Multi-Site Variants\rthree{ and Single-Substitution Analysis}}}")
ed("r352_body",
   r"""\rtwo{Because the position-based splits assign each row by its own substitution
position, rows of the same multi-site variant can share a fitness label across folds
(6,481 of the 7,663 analyzed multi-site variants straddle modulo folds and 2,022
straddle contiguous folds). We therefore re-ran the probing experiment for all three
model configurations restricted to single-substitution variants (Supplementary Table
S11). The depth trend and the late optimum were unchanged: in every leakage-free
column the best layer lay in layers 27--32. The best layer exceeded the final layer by
at most 0.012 for SaProt (both input modes) and at most 0.019 for ESM-2 (mutant arm,
position-grouped split; 90\% CI $+0.004$ to $+0.036$), so on single-substitution data
the final layer remains within 0.02 of the best layer.}""",
   r"""\rtwo{Because the position-based splits assign each row by its own substitution
position, rows of the same multi-site variant can share a fitness label across folds\rthree{.
In the mutant arm they also share the variant's averaged embedding. Under the modulo
split, 15,624 of the 213,393 analyzed rows (7.3\%) are test rows whose variant also has
a row in the training folds (15,960 under the position-grouped split and 6,158 under
the contiguous split). Re-running all three arms on single-substitution variants only
(Supplementary Table S8) shows that this inflated the full-data mutant arm (ESM-2,
modulo, best layer: 0.483 on the full data versus 0.443 on single substitutions;
SaProt with real 3Di: 0.531 versus 0.492). It did not inflate the wild-type arms (ESM-2
masked\_wt: 0.459 versus 0.476), so only the mutant arm is analyzed on single
substitutions under these splits. On single substitutions the depth trend and the late
optimum were unchanged for all arms, with the best layer in layers 27--32 in every
leakage-free column. The best layer exceeded the final layer by at most 0.019 (point
estimates; 90\% upper bounds up to 0.036; 3 of 23 comparisons equivalent, 20
inconclusive, none beyond the margin). Two datasets retain fewer than 50 single-substitution variants;
excluding them leaves these conclusions unchanged (largest point estimate 0.014, 90\%
upper bound 0.025).}}""")
ed("fig5_caption",
   r"""\caption{\textbf{\revised{Random versus leakage-free split probing (SaProt, 63 datasets).}}""",
   r"""\caption{\textbf{\revised{Random versus leakage-free split probing (\rthree{ESM-2 and SaProt}, 63 datasets).}}""")
ed("fig5b_caption",
   r"""\revised{Leakage-free splits (modulo, position-grouped, contiguous): performance
drops to $0.27$--$0.53$ depending on arm and model, the invariance score rises to
$4$--$7 \times 10^{-2}$, and the best layers shift consistently to layers 30--32.}}""",
   r"""\revised{Leakage-free splits (modulo, position-grouped, contiguous\rthree{; mutant arm on
single substitutions}): performance
drops to \rthree{$0.35$--$0.49$}, the invariance score rises to
$4$--$7 \times 10^{-2}$, and the best layers shift consistently to layers \rthree{30--31}.}}""")

# ------------------------------------------------------------------ 3.6 structure tokens (R2.1)
ed("r36_leakfree",
   r"""Under every
leakage-free split the real-3Di inputs were consistently superior, with mean paired
advantages of $+0.059$ to $+0.122$ across arms and splits, all with Wilcoxon
$p \leq 10^{-4}$.""",
   r"""Under every
leakage-free split the real-3Di inputs were consistently superior, with mean paired
advantages of \rthree{$+0.057$ to $+0.132$} across arms and splits, all with Wilcoxon
\rthree{$p \leq 0.0011$ (mutant arm on single substitutions)}.""")
ed("tab4_caption",
   r"""and mutant arms)} and large and highly significant under all leakage-free splits.}}""",
   r"""and mutant arms)} and large and highly significant under all leakage-free splits.\rthree{
Mutant-arm entries under the leakage-free splits ($^{\dagger}$) are computed on
single-substitution variants.}}}""")
ed("tab4_mutant",
   r"mutant       & $+0.032$ & $2.5 \times 10^{-6}$ & $+0.064$ ($6.9 \times 10^{-5}$) & $+0.122$ ($6.5 \times 10^{-7}$) & $+0.067$ ($4.6 \times 10^{-5}$) \\",
   r"mutant       & $+0.032$ & $2.5 \times 10^{-6}$ & \rthree{$+0.057$ ($1.0 \times 10^{-3}$)$^{\dagger}$} & \rthree{$+0.132$ ($4.8 \times 10^{-7}$)$^{\dagger}$} & \rthree{$+0.058$ ($1.1 \times 10^{-3}$)$^{\dagger}$} \\")

# ------------------------------------------------------------------ Discussion
ed("disc_open",
   r"""the best layers are the last few (layers 27--32), the final layer is within
$\Delta\rho = 0.009$ of the best layer on the full benchmark (within 0.02 on single
substitutions), and structure tokens deliver a measurable benefit ($+0.06$ to $+0.12$).
No multi-scale pooling strategy improved on the final layer under leakage-free
evaluation.""",
   r"""the best layers are the last few (layers 27--32), the final layer is within
\rthree{$\Delta\rho = 0.019$ of the best layer (point estimates; 90\% upper bounds up to
0.036)}, and structure tokens deliver a measurable benefit (\rthree{$+0.06$ to $+0.13$}).
\rthree{Multi-scale pooling did not reliably improve on the final layer under leakage-free
evaluation (1 of 96 comparisons significantly better).}""")
ed("disc_cka",
   r"""while the linear CKA between the same layers remains 0.81.""",
   r"""while the linear CKA between the same layers remains \rthree{0.78}.""")
ed("disc_cka2",
   r"""layers ($\cos \approx -0.01$) \revised{alongside their high linear CKA (0.81)}""",
   r"""layers ($\cos \approx -0.01$) \revised{alongside their high linear CKA (\rthree{0.78})}""")
ed("disc_nlp",
   r"""Notably, this pattern contrasts with findings from NLP transformers, where
probing classifiers consistently recover different linguistic features from
different layer depths~\cite{tenney2019bert,rogers2021bertology}. In BERT, for example,
surface syntactic features peak in early layers while semantic features dominate
in middle-to-late layers. \rtwo{The single-peaked, monotonic profile of the PLM
curves, with no intermediate optimum, may} reflect the fundamentally different nature of the learning signal in protein
versus natural language: whereas language contains hierarchical structure at
multiple timescales (morphology, syntax, semantics, discourse), the MLM task on
protein sequences operates on a single level of abstraction (amino acid identity),
potentially eliminating the pressure for layer-wise specialization.""",
   r"""\rthree{In NLP transformers, by contrast, probing classifiers recover different
linguistic features at different depths~\cite{tenney2019bert,rogers2021bertology}. The
comparison is limited: we probe a single prediction target, which would be expected to
give one peaked curve, whereas NLP studies probe many distinct features, and early PLM
layers are known to encode local physicochemical and secondary-structure
properties~\cite{detlefsen2022}. Whether PLM layers specialize less than NLP layers is
therefore a hypothesis that would require probing several distinct protein properties
under leakage-free splits.}""")
ed("disc_saprot_esm2",
   r"""ways: how much their final representations retain early-layer geometry (ESM-2
distant-layer cosine 0.56 with CKA 0.55; SaProt cosine $-0.01$ with CKA 0.81), and the benefit conferred by structure tokens. The directly measured
difference is the structure-token benefit: real 3Di inputs improve leakage-free
probing by $+0.06$ to $+0.12$""",
   r"""ways: how much their final representations retain early-layer geometry (ESM-2
distant-layer cosine \rthree{0.54 with CKA 0.41}; SaProt cosine $-0.01$ with CKA \rthree{0.78}), and the benefit conferred by structure tokens. The directly measured
difference is the structure-token benefit: real 3Di inputs improve leakage-free
probing by \rthree{$+0.06$ to $+0.13$}""")
ed("disc_kumar",
   r"""This is compatible with our narrower conclusion:
different downstream tasks may induce different optimal readout directions, while
mutation effect prediction under random-split probing shows little layer dependence.
The key distinction is between \emph{task-level} specialization and
\emph{within-task} layer complementarity under a specified readout and evaluation
protocol.""",
   r"""\rthree{Our leakage-free results are broadly in line with this observation: for
mutation effect prediction the best layers lie in layers 27--32 and the late layers are
close to the final layer, consistent with the finding that layers 20--33 perform well.
The difference is one of degree: for kinase function prediction, intermediate layers
outperformed the final layer, whereas for mutation effect prediction the final layer is
within a small margin of the best layer. The flat random-split profile is a property of
the evaluation and is not informative for such comparisons.}""")
ed("disc_practical",
   r"""Multi-scale fusion provided no benefit over the final layer for either
model under leakage-free evaluation,""",
   r"""Multi-scale fusion \rthree{did not reliably improve on} the final layer for either
model under leakage-free evaluation\rthree{ (one significant gain in 96 comparisons)},""")
ed("disc_readout",
   r"""for ESM-2, last-layer LightGBM exceeded last-layer Ridge by
$+0.045$ (paired, $p = 0.011$) and reached $\rho = 0.508$, the best pooling
configuration overall, whereas for SaProt the same contrast was not significant.}""",
   r"""for ESM-2, last-layer LightGBM exceeded last-layer Ridge by
$+0.045$ (paired, $p = 0.011$) and reached $\rho = 0.508$, the best pooling
configuration overall, whereas for SaProt the same contrast was not significant\rthree{;
this ranking holds under the modulo split and reverses under the random split}.}""")

ed("r33_heading",
   r"\subsection{\revised{Multi-Scale Pooling Shows No Practical Gain Under a Leakage-Free Protocol}}",
   r"\subsection{\revised{Multi-Scale Pooling \rthree{Does Not Reliably Improve on the Final Layer} Under a Leakage-Free Protocol}}")
ed("disc_concat_nobenefit",
   "Concat~5L\noffered no benefit on\n\\revised{either model} in our benchmark",
   "Concat~5L\noffered no \\rthree{significant} benefit on\n\\revised{either model} in our \\rthree{mutant-arm} benchmark")

# ------------------------------------------------------------------ alt text under each legend (editor)
ALT = {
    "fig:concept": "Four-panel schematic of the study design: per-layer probing under three "
                   "feature arms and four train--test splits with a position-lookup control (a), "
                   "multi-scale pooling with four fusion strategies and four readouts (b), the "
                   "logit lens (c), and layer-wise similarity analysis (d).",
    "fig:similarity": "Two 33-by-33 heat maps of similarity between SaProt layers. Cosine "
                      "similarity (a) stays high except between the final and first layers "
                      "(about $-0.01$); linear CKA (b) remains moderately high throughout, 0.78 "
                      "between layers 0 and 32.",
    "fig:probing": "Three panels. (a) Under the random split, per-layer probing is flat at "
                   "about 0.60 and matches the position-lookup control; under the modulo split, "
                   "single-substitution mutant-arm curves are lower and peak at layer 30. "
                   "(b) Pooling results by readout for both models. (c) Cosine similarity "
                   "declines with layer distance.",
    "fig:logit_lens": "Line plot of zero-shot logit-lens Spearman correlation against layer for "
                      "ESM-2 and SaProt; both curves rise with depth and peak at the final layer "
                      "at about 0.44.",
    "fig:position_level": "Two panels of per-layer probing for ESM-2 and SaProt. (a) Under the "
                          "random split, curves are flat at about 0.60 and match the "
                          "position-lookup control. (b) Under leakage-free splits, "
                          "single-substitution mutant-arm curves are lower (0.35 to 0.49) and "
                          "peak at layers 30 to 31.",
}
for _lab, _txt in ALT.items():
    ed("alt_" + _lab, r"\label{%s}" % _lab + "\n" + r"\end{figure}",
       r"\label{%s}" % _lab + "\n" + r"\smallskip{\footnotesize\rthree{\textit{Alt text:} %s}}" % _txt
       + "\n" + r"\end{figure}")

# ------------------------------------------------------------------ Limitations
ed("lim_final",
   r"""\rtwo{. Finally, our
equivalence margin ($\Delta\rho = 0.01$) was fixed before the equivalence analysis as
a practical convention rather than a task-derived threshold, and because the best
layer is selected on the same data, the equivalence verdicts are conservative.}""",
   r"""\rtwo{\rthree{. Ninth, under the position-based splits the mutant arm is analyzed on
single substitutions, which excludes most variants of the ten datasets that contain
multi-site variants; the wild-type arms are reported on the full data, in which rows of
a multi-site variant share labels across folds, although their single-substitution
results are similar (Supplementary Table S8)}. Finally, our
equivalence margin ($\Delta\rho = 0.01$) was fixed before the equivalence analysis as
a practical convention rather than a task-derived threshold, and because the best
layer is selected on the same data, the equivalence verdicts are conservative\rthree{;
three of the seven equivalence verdicts nevertheless lie at the margin and depend on
the interval method}.}""")

# ------------------------------------------------------------------ Conclusion
ed("conclusion",
   r"""The final layer is a sufficient choice---within
$\Delta\rho = 0.009$ of the best layer on the full benchmark and within 0.02 on single
substitutions---and no multi-scale pooling strategy improved on it under leakage-free
evaluation.""",
   r"""The final layer is a sufficient choice---within
\rthree{$\Delta\rho = 0.019$ of the best layer (point estimates; 90\% upper bounds up to
0.036), a small deficit relative to the variation across datasets}---and \rthree{multi-scale
pooling did not reliably improve on it} under leakage-free
evaluation.""")

# ------------------------------------------------------------------ Data availability (R2.m1)
ed("zenodo_version",
   r"archived version \rtwo{v3.0.0, DOI 10.5281/zenodo.23264208}).}",
   r"archived version \rthree{v3.1.1, DOI 10.5281/zenodo.23878920}).}")


def main():
    s = SRC.read_text(encoding="utf-8")
    for tag, old, new in E:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"[{tag}] anchor found {n} times")
        s = s.replace(old, new)
    # \textcolor drops a space at the start of its argument: move it outside
    n_sp = s.count("\\rthree{ ") + s.count("\\rthree{\n")
    s = s.replace("\\rthree{ ", " \\rthree{").replace("\\rthree{\n", "\n\\rthree{")
    print(f"moved {n_sp} leading spaces out of \\rthree{{}}")
    n_fig = s.count("figures/output_v2/")
    s = s.replace("figures/output_v2/", "figures/output_r3/")
    OUT.write_text(s, encoding="utf-8")
    print(f"written {OUT.name}: {len(E)} edits, {n_fig} figure paths")


if __name__ == "__main__":
    main()
