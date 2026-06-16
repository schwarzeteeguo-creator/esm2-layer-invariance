# Layer Invariance in Protein Language Models: Multi-Scale Embeddings Do Not Improve Mutation Effect Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code, results, and the manuscript for our paper submitted to **Bioinformatics** (Oxford).

> **Abstract.** Protein language models such as ESM-2 encode biochemical information across 33 transformer layers. It is widely assumed that different layers capture complementary features and that fusing multi-scale embeddings should improve mutation effect prediction. We systematically test this assumption on ProteinGym's DMS benchmarks (62 datasets) using two 650M-parameter models and three independent experimental paradigms. Our results show that all 33 layers achieve nearly identical predictive performance, and multi-scale fusion provides no benefit over single-layer embeddings — especially for structure-aware models like SaProt. For standard ESM-2, switching to tree-based nonlinear readouts eliminates the apparent layer dependence, achieving the benefits of multi-scale fusion without the dimensionality cost.

## Repository Structure

```
esm2-layer-invariance/
├── README.md                          # This file
├── manuscript/                        # LaTeX source & compiled PDF
│   ├── manuscript.tex
│   └── manuscript.pdf
├── figures/                           # Figure generation & output
│   ├── generate_figures.py            # Python script to generate all 5 figures
│   └── output/                        # PNG + PDF for all figures
│       ├── fig1_logit_lens.png/pdf
│       ├── fig2_similarity_heatmap.png/pdf
│       ├── fig3_probing_pooling.png/pdf
│       ├── fig4_summary.png/pdf
│       └── fig5_position_level.png/pdf
├── results/                           # All experimental results (JSON)
│   ├── multiscale_results_v2.json     # SaProt multi-scale pooling (20 datasets)
│   ├── multiscale_results_v2_esm2.json # ESM-2 multi-scale pooling (20 datasets)
│   ├── saprot_v2_full.json            # SaProt per-layer probing (62 datasets)
│   ├── poslevel_saprot_combined.json   # Position-level split probing
│   ├── logit_lens_*_summary.json       # Logit lens zero-shot results
│   ├── pooling_v1_full.json            # Pooling benchmark v1
│   └── final_summary.txt              # Pipeline completion summary
├── code/                              # Core experiment pipeline
│   ├── benchmark_v2.py                # Multi-scale pooling benchmark (main)
│   ├── layer_probing_v3.py            # Per-layer Ridge probing (latest)
│   ├── layer_probing_v2.py            # Per-layer probing v2
│   ├── logit_lens.py                  # Logit lens for SaProt
│   ├── logit_lens_esm2.py             # Logit lens for ESM-2
│   ├── layer_similarity.py            # Layer-wise cosine similarity
│   ├── extract.py                     # Masked-position feature extraction
│   ├── model.py                       # Model loading utilities
│   ├── pooling.py                     # Pooling strategy implementations
│   ├── stats_analysis.py              # Statistical analysis & invariance scores
│   ├── build_esm2_from_saprot.py      # SaProt → ESM-2 model conversion
│   ├── final_chain.py                 # End-to-end automated pipeline
│   └── __init__.py
├── analysis/                          # Result analysis scripts
│   ├── final_analysis.py              # Comprehensive analysis
│   ├── analyze_esm2.py                # ESM-2 result analysis
│   ├── analyze_probe.py               # Probing result analysis
│   └── download_and_analyze.py         # AutoDL download & analysis
```

## Key Findings

1. **Complete layer equivalence.** All 33 transformer layers achieve near-identical Spearman ρ across 62 DMS datasets (SaProt: μ(ρ) ∈ [0.6054, 0.6071], σ = 2.96×10⁻⁴; ESM-2: μ(ρ) ∈ [0.596, 0.603], σ = 8.29×10⁻⁴). Zero of 528 pairwise layer comparisons survive FDR correction.

2. **Structure-aware training compresses layer-wise information.** SaProt shows invariance across ALL readout types (Ridge σ = 1.89×10⁻⁴, RF σ = 2.70×10⁻³, LGBM σ = 1.29×10⁻³). Standard ESM-2 shows large layer differences under Ridge (Last Layer ρ = 0.386 vs. Concat 5L ρ = 0.639, +66%) — but tree-based readouts (RF ρ ≈ 0.63, LightGBM ρ ≈ 0.66) eliminate the gap.

3. **37% position-leakage inflation.** Random mutation splitting inflates performance by 37–43% compared to position-level cross-validation. Invariance score increases 150-fold under position-level evaluation.

4. **Representations change dramatically while information is preserved.** Layer 0 vs. Layer 32 cosine similarity ≈ −0.01 (near orthogonal), yet predictive information is invariant across all layers.

## Models & Data

- **SaProt_650M_AF2**: Structure-aware PLM with 446-token vocabulary (21 aa × 21 Foldseek 3Di states)
- **ESM-2 650M**: Standard sequence-only PLM, 33 layers, 1280-dim hidden states
- **ProteinGym DMS benchmark**: 62 substitution datasets spanning diverse protein families

## Quick Start

### Requirements

```bash
pip install torch transformers scikit-learn lightgbm numpy scipy
```

### Reproduce Per-Layer Probing

```bash
python code/layer_probing_v3.py
```

### Reproduce Multi-Scale Pooling Benchmark

```bash
python code/benchmark_v2.py
```

### Generate Figures

```bash
python figures/generate_figures.py
```

### Compile Manuscript

```bash
cd manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## Computational Resources

All experiments were conducted on a single NVIDIA RTX 5090 32GB GPU via AutoDL. Total wall-clock time for the complete benchmark (per-layer probing + multi-scale pooling + logit lens) was approximately 12 hours.

## Citation

If you use this work, please cite:

```bibtex
@article{guo2026layer,
  title={Layer Invariance in Protein Language Models: Multi-Scale Embeddings Do Not Improve Mutation Effect Prediction},
  author={Guo, Yutao and Zhang, Zihan and Zhao, Xuezhou and Chen, Mengxi and Wu, Dan},
  journal={Bioinformatics},
  year={2026},
  publisher={Oxford University Press}
}
```

## License

MIT License. See the manuscript for data availability statements regarding ProteinGym.
