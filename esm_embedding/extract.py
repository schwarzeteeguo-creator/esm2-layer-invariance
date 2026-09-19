"""Multi-scale embedding extraction for protein variant effect prediction.

Core pipeline:
    1. Load frozen ESM/SaProt model
    2. Run forward pass with output_hidden_states=True
    3. Extract per-residue hidden states from all layers
    4. Apply pooling strategy (last-layer, mean, concat, attention-weighted)
    5. Extract mutation-site-specific embeddings for downstream scoring

Supports both wild-type embedding extraction and mutation-specific feature vectors,
enabling the full Phase A ablation pipeline.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .model import load_model, tokenize_sequence
from .pooling import (
    POOLING_METHODS,
    pool_last_layer,
    pool_mean,
    pool_concat,
    pool_attention_weighted,
)


@dataclass
class EmbeddingResult:
    """Container for extracted embeddings."""
    residue_embeddings: torch.Tensor  # (L, D) — per-position pooled embeddings
    layer_hidden_states: List[torch.Tensor]  # All raw layer outputs
    pooling_method: str
    sequence: str
    model_key: str

    def __repr__(self):
        return (f"EmbeddingResult({self.sequence[:10]}..., "
                f"{self.residue_embeddings.shape}, "
                f"pool={self.pooling_method})")


def extract_embeddings(
    sequence: str,
    model_key: str = "saprot_650m",
    pooling: str = "last_layer",
    structure_tokens: Optional[str] = None,
    layer_indices: Optional[List[int]] = None,
    device: Optional[str] = None,
    return_all_layers: bool = True,
) -> EmbeddingResult:
    """Extract residue-level embeddings from a frozen ESM/SaProt model.

    This is the main entry point for Phase A (M1) experiments.

    Args:
        sequence: Amino acid sequence (e.g., "MADE")
        model_key: Which model to use (see model._MODEL_PATHS)
        pooling: Pooling strategy — "last_layer", "mean", "concat", "attention"
        structure_tokens: 3Di tokens for SaProt. If None, uses dummy 'p' tokens.
        layer_indices: For "mean" and "concat" — which layers to use.
        device: 'cuda', 'cpu', or None (auto)
        return_all_layers: If True, include all raw layer hidden states

    Returns:
        EmbeddingResult with pooled embeddings and metadata
    """
    # Load model
    model_info = load_model(model_key, device=device)
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    dev = model_info["device"]

    # Tokenize
    inputs = tokenize_sequence(tokenizer, sequence, structure_tokens)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states (all layers including embedding)
    # ESMForMaskedLM returns hidden_states as tuple:
    # (embedding_output, layer_0, layer_1, ..., layer_N)
    # We want only the transformer layer outputs (skip embedding layer)
    all_hidden = list(outputs.hidden_states)
    transformer_hidden = all_hidden[1:]  # skip token embedding layer

    # Each is (1, seq_len, hidden_dim) — squeeze batch dim
    # Strip <cls> (pos 0) and <eos> (pos -1) special tokens
    layer_hidden = [h.squeeze(0)[1:-1].cpu() for h in transformer_hidden]

    # Apply pooling
    if pooling == "last_layer":
        pooled = pool_last_layer(layer_hidden)
    elif pooling == "mean":
        pooled = pool_mean(layer_hidden, layer_indices=layer_indices)
    elif pooling == "concat":
        pooled = pool_concat(layer_hidden, layer_indices=layer_indices)
    elif pooling == "attention":
        pooled = pool_attention_weighted(layer_hidden)
        if isinstance(pooled, torch.Tensor):
            pooled = pooled.detach()
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    # Ensure all returned embeddings are detached
    if isinstance(pooled, torch.Tensor):
        pooled = pooled.detach()

    return EmbeddingResult(
        residue_embeddings=pooled,
        layer_hidden_states=layer_hidden if return_all_layers else [],
        pooling_method=pooling,
        sequence=sequence,
        model_key=model_key,
    )


def extract_multi_scale(
    sequence: str,
    model_key: str = "saprot_650m",
    structure_tokens: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, EmbeddingResult]:
    """Extract embeddings with ALL pooling strategies for ablation comparison.

    Runs the model ONCE and applies different pooling strategies to the
    same forward pass, enabling direct comparison without re-running.

    Args:
        sequence: Amino acid sequence
        model_key: Model to use
        structure_tokens: Optional 3Di tokens for SaProt
        device: 'cuda', 'cpu', or None

    Returns:
        Dict mapping method name → EmbeddingResult
    """
    model_info = load_model(model_key, device=device)
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    dev = model_info["device"]
    num_layers = model_info["num_layers"]

    inputs = tokenize_sequence(tokenizer, sequence, structure_tokens)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    all_hidden = list(outputs.hidden_states)
    transformer_hidden = all_hidden[1:]
    # Strip <cls> and <eos> special tokens
    layer_hidden = [h.squeeze(0)[1:-1].cpu() for h in transformer_hidden]

    results = {}

    # 1. Last-layer baseline (replicates standard ESM approach)
    results["last_layer"] = EmbeddingResult(
        residue_embeddings=pool_last_layer(layer_hidden),
        layer_hidden_states=[],
        pooling_method="last_layer",
        sequence=sequence,
        model_key=model_key,
    )

    # 2. Mean pool layers 20-33 (function/motif focused)
    func_layers = list(range(20, 34))
    func_layers = [i for i in func_layers if i < num_layers]
    results["mean_20_33"] = EmbeddingResult(
        residue_embeddings=pool_mean(layer_hidden, layer_indices=func_layers),
        layer_hidden_states=[],
        pooling_method="mean_20_33",
        sequence=sequence,
        model_key=model_key,
    )

    # 3. Concat multi-depth: early + middle + late
    concat_layers = [6, 14, 20, 26, 33]
    concat_layers = [i for i in concat_layers if i < num_layers]
    results["concat_6_14_20_26_33"] = EmbeddingResult(
        residue_embeddings=pool_concat(layer_hidden, layer_indices=concat_layers),
        layer_hidden_states=[],
        pooling_method="concat_6_14_20_26_33",
        sequence=sequence,
        model_key=model_key,
    )

    # 4. Attention-weighted (detach to remove gradient tracking from nn.Parameter)
    attn_emb = pool_attention_weighted(layer_hidden)
    if isinstance(attn_emb, torch.Tensor):
        attn_emb = attn_emb.detach()
    results["attention"] = EmbeddingResult(
        residue_embeddings=attn_emb,
        layer_hidden_states=[],
        pooling_method="attention",
        sequence=sequence,
        model_key=model_key,
    )

    # 5. Full layer range for analysis
    results["_raw_layers"] = EmbeddingResult(
        residue_embeddings=torch.zeros(0),  # placeholder
        layer_hidden_states=layer_hidden,
        pooling_method="raw",
        sequence=sequence,
        model_key=model_key,
    )

    return results


def extract_mutation_features(
    sequence: str,
    mutations: List[Tuple[int, str, str]],  # (position, from_aa, to_aa)
    model_key: str = "saprot_650m",
    pooling: str = "last_layer",
    structure_tokens: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[Tuple[int, str, str], torch.Tensor]:
    """Extract per-mutation embedding features.

    This is the key function for mutation effect prediction:
    given a wild-type sequence and a list of mutations, returns the
    pooled embedding at each mutated position.

    Args:
        sequence: Wild-type amino acid sequence
        mutations: List of (position, from_aa, to_aa), 1-indexed positions
        model_key: Model to use
        pooling: Pooling strategy
        structure_tokens: Optional 3Di tokens
        device: Device

    Returns:
        Dict mapping (pos, from_aa, to_aa) → embedding tensor of shape (D,)
    """
    result = extract_embeddings(
        sequence=sequence,
        model_key=model_key,
        pooling=pooling,
        structure_tokens=structure_tokens,
        device=device,
        return_all_layers=False,
    )

    embeddings = result.residue_embeddings  # (L, D)

    mutation_features = {}
    for pos, from_aa, to_aa in mutations:
        if pos < 1 or pos > embeddings.shape[0]:
            raise ValueError(f"Position {pos} out of range [1, {embeddings.shape[0]}]")
        mutation_features[(pos, from_aa, to_aa)] = embeddings[pos - 1]  # 0-index

    return mutation_features


def extract_mutation_features_multi_scale(
    sequence: str,
    mutations: List[Tuple[int, str, str]],
    model_key: str = "saprot_650m",
    structure_tokens: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Dict[Tuple[int, str, str], torch.Tensor]]:
    """Extract per-mutation features with ALL pooling strategies.

    Runs model ONCE, applies all pooling strategies for fair comparison.

    Args:
        sequence: Wild-type amino acid sequence
        mutations: List of (position, from_aa, to_aa), 1-indexed
        model_key: Model to use
        structure_tokens: Optional 3Di tokens
        device: Device

    Returns:
        Dict: {pooling_method: {(pos, from, to): embedding}}
    """
    multi_results = extract_multi_scale(
        sequence=sequence,
        model_key=model_key,
        structure_tokens=structure_tokens,
        device=device,
    )

    all_features = {}
    for method, result in multi_results.items():
        if method == "_raw_layers":
            continue  # skip raw layer storage

        embeddings = result.residue_embeddings
        features = {}
        for pos, from_aa, to_aa in mutations:
            if pos < 1 or pos > embeddings.shape[0]:
                raise ValueError(
                    f"Position {pos} out of range [1, {embeddings.shape[0]}]"
                )
            features[(pos, from_aa, to_aa)] = embeddings[pos - 1]
        all_features[method] = features

    return all_features
