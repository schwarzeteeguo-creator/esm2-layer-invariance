"""ESM Multi-Scale Embedding Extraction Framework.

Phase A (M1): Extract and pool embeddings from multiple ESM/SaProt layers
for protein variant effect prediction.

Architecture:
    1. Frozen ESM/SaProt backbone (33 layers, 1280-dim)
    2. Multi-layer hidden state extraction
    3. Configurable pooling strategies:
       - last_layer: single last layer (baseline, replicates standard approach)
       - mean: mean-pool across selected layers
       - concat: concatenate fixed layer set
       - attention: learned attention-weighted pooling
    4. Per-residue embedding output for mutation site scoring
"""

from .model import load_model, get_available_models
from .extract import (
    extract_embeddings,
    extract_multi_scale,
    extract_mutation_features,
    extract_mutation_features_multi_scale,
)
from .pooling import (
    pool_last_layer,
    pool_mean,
    pool_concat,
    pool_attention_weighted,
    POOLING_METHODS,
)

__all__ = [
    "load_model",
    "get_available_models",
    "extract_embeddings",
    "extract_multi_scale",
    "extract_mutation_features",
    "extract_mutation_features_multi_scale",
    "pool_last_layer",
    "pool_mean",
    "pool_concat",
    "pool_attention_weighted",
    "POOLING_METHODS",
]
