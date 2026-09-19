# Code and per-dataset results (BIOINF-2026-2348 revision deposit)

This deposit accompanies the revised manuscript "Conditional Layer Invariance in
Protein Language Models: Position Leakage and Readout Choice Determine the Apparent
Benefit of Multi-Scale Embeddings" (Bioinformatics, manuscript BIOINF-2026-2348,
revision of 2026). It contains the unified analysis pipeline and the complete
per-dataset results for every experiment reported in the revision.

## Contents

```
esm_embedding/                  analysis pipeline (all Python modules)
  probing_v4.py                 three-arm x four-split per-layer probing + random-position control
  pg_splits.py                  ProteinGym split schemes (random / modulo / contiguous / position-grouped)
  benchmark_v3.py               multi-scale pooling benchmark (RidgeCV, RF, MLP, LightGBM)
  logit_lens_v2.py              fp32 logit lens with per-dataset validation gate
  logit_lens_validate.py        official-logits validation utilities
  cka.py                        cosine + linear CKA similarity analysis
  stats_tost.py                 paired TOST, bootstrap CIs, Cohen's d_z
  aggregate_results.py          rebuilds the master results table
  merge_shards.py               merges sharded outputs
  fix_3di_gaps.py               3Di alignment / gap-padding checks
  (earlier *_v2/v3, layer_probing, extract, model, pooling files document
   the submitted-version pipeline retained for the withdrawal record)
pipelines/run_chain8.sh         server orchestration used for the rerun
results/
  server_results/
    probing_v4_esm2/            ESM-2 probing (63 datasets x 3 arms x 4 splits x 2 feature sets)
    probing_v4_saprot/          SaProt probing, structure-masked inputs
    probing_v4_saprot_3di/      SaProt probing, real Foldseek 3Di inputs
    pooling_v3_esm2/            pooling benchmark, ESM-2 (20 datasets)
    pooling_v3_saprot/          pooling benchmark, SaProt structure-masked (20 datasets)
    logit_lens_v2_esm2/         logit lens summary, ESM-2 (63 datasets)
    logit_lens_v2_saprot_3di/   logit lens summary, SaProt real 3Di (63 datasets)
    cka_esm2/, cka_saprot/      cosine and CKA matrices + summaries (10 datasets x 50 positions)
  probing_v4_gcv/               random-position control rerun under GCV protocol + CONTROL_SUMMARY.md
  3di_server/                   per-dataset Foldseek 3Di token files (62 datasets)
  old_ll_fp16_repro/            reproduction record of the withdrawn fp16 lens curves
  MAIN_TABLE.md                 master results table with all numbers and ns
```

## Environment

- Python 3.10+ (tested on 3.13); PyTorch (CUDA build for feature extraction);
  transformers (SaProt_650M_AF2 and ESM-2 650M weights from HuggingFace);
  scikit-learn, scipy, numpy, lightgbm, lmdb.
- Feature extraction ran on a single NVIDIA RTX 5090 (24 GB); supervised probing,
  statistics, and aggregation are CPU-only.
- All subsampling is seeded (default seed 42) and identical across arms, splits,
  and layers. The random-position control uses 33 independent draws (n_random_draws=33).

## Data

ProteinGym substitution benchmark (Notin et al., 2024), 63 DMS datasets, read from
the official LMDB. 3Di tokens were assigned by aligning each wild-type sequence to
its AlphaFold Protein Structure Database entry (62 of 63 datasets; two structures
C-terminally truncated and gap-padded; influenza nucleoprotein has no AFDB entry
and is structure-masked). Foldseek 3Di files are included in results/3di_server/.

## Reproduction

```bash
# per-layer probing, e.g. SaProt with real 3Di inputs
python -m esm_embedding.probing_v4 --model_key saprot_650m \
    --struct_dir results/3di_server --splits random,modulo,contiguous,position_groupkfold \
    --arms masked_wt,unmasked_wt,mutant --output_dir results/probing_v4_saprot_3di

# pooling benchmark (mutant arm, leakage-free split)
python -m esm_embedding.benchmark_v3 --model_key esm2_650m \
    --arms mutant --splits modulo --output_dir results/pooling_v3_esm2

# logit lens (fp32, validation gate |delta logP| < 1e-3 per dataset)
python -m esm_embedding.logit_lens_v2 --model esm2_650m --output_dir results/logit_lens_v2_esm2

# similarity analysis (cosine + linear CKA)
python -m esm_embedding.cka --output_dir results/cka_esm2

# master table
python -m esm_embedding.aggregate_results results
```

JSON schema (probing): `{dataset: {n_rows, n_variants, capped, result:
{layer: {spearman, n, alphas}}}}`. Pooling adds an `arms` level and nests
attention cells under `rho` with per-fold layer weights. Contiguous-split files
contain null Spearman values for the six datasets whose mutated positions form a
single contiguous block (reported on the remaining 57).

## License

Code: MIT. Results and 3Di token files: CC-BY-4.0.
