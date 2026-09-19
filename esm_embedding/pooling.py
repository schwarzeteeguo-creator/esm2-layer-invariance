"""Pooling strategies for multi-scale ESM hidden states.

Each strategy takes a list of hidden state tensors from different layers
and produces a single per-residue embedding.

Literature grounding:
    - Last-layer (baseline): Standard approach used by ESM-1v, ESM-Effect
    - Layers 20-33: Best for function/motif prediction (Kumar et al. 2025)
    - Layers 6-14: Capture local physicochemical features (bioRxiv 2024)
    - Attention-weighted pooling: +12% Spearman over concat (Elshaffei et al. 2025)
"""

import torch
import torch.nn as nn
from typing import List, Optional


def pool_last_layer(
    hidden_states: List[torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    """Last-layer pooling — the standard baseline.

    Args:
        hidden_states: List [layer_0, layer_1, ..., layer_N] of shape (L, D)

    Returns:
        Tensor of shape (L, D)
    """
    return hidden_states[-1]


def pool_mean(
    hidden_states: List[torch.Tensor],
    layer_indices: Optional[List[int]] = None,
    **kwargs,
) -> torch.Tensor:
    """Mean-pool across selected layers.

    Args:
        hidden_states: List of per-layer hidden states
        layer_indices: Which layers to pool. Default: 20-33 (function/motif focused)

    Returns:
        Tensor of shape (L, D)
    """
    if layer_indices is None:
        layer_indices = list(range(20, 34))  # layers 20-33 (0-indexed)
    selected = [hidden_states[i] for i in layer_indices if i < len(hidden_states)]
    return torch.stack(selected).mean(dim=0)


def pool_concat(
    hidden_states: List[torch.Tensor],
    layer_indices: Optional[List[int]] = None,
    **kwargs,
) -> torch.Tensor:
    """Concatenate embeddings from multiple representational depths.

    Strategy: sample layers from early (local features), middle (convergence),
    and late (global function) stages.

    Default: layers 6, 14, 20, 26, 33 — spanning the full depth.
    Output dim = D * len(layer_indices).

    Args:
        hidden_states: List of per-layer hidden states
        layer_indices: Which layers to concatenate

    Returns:
        Tensor of shape (L, D * len(layer_indices))
    """
    if layer_indices is None:
        layer_indices = [6, 14, 20, 26, 33]
    selected = [hidden_states[i] for i in layer_indices if i < len(hidden_states)]
    return torch.cat(selected, dim=-1)


class AttentionWeightedPool(nn.Module):
    """Learned attention-weighted pooling across layers.

    Each layer gets a learned scalar weight, softmax-normalized.
    Implemented per Elshaffei et al. (arXiv 2505.20036, 2025):
    attention-based pooling improved Spearman by +12% over naive concatenation.

    Args:
        num_layers: Total number of layers to pool over
        hidden_dim: Dimension of each layer's hidden state
    """

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        # Optional: per-dimension gating for finer control
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        use_gate: bool = False,
    ) -> torch.Tensor:
        """Attention-weighted pooling.

        Args:
            hidden_states: List of per-layer hidden states
            use_gate: If True, apply per-dimension gating (experimental)

        Returns:
            Tensor of shape (L, D)
        """
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states)  # (N_layers, L, D)

        if use_gate:
            gate_values = torch.sigmoid(self.gate(stacked))  # (N_layers, L, 1)
            weighted = (stacked * weights[:, None, None] * gate_values).sum(dim=0)
        else:
            weighted = (stacked * weights[:, None, None]).sum(dim=0)

        return weighted


def pool_attention_weighted(
    hidden_states: List[torch.Tensor],
    attention_pool: Optional[AttentionWeightedPool] = None,
    **kwargs,
) -> torch.Tensor:
    """Apply learned attention-weighted pooling.

    Args:
        hidden_states: List of per-layer hidden states
        attention_pool: Pre-initialized AttentionWeightedPool module.
                        If None, creates one with random init.

    Returns:
        Tensor of shape (L, D)
    """
    if attention_pool is None:
        num_layers = len(hidden_states)
        hidden_dim = hidden_states[0].shape[-1]
        attention_pool = AttentionWeightedPool(num_layers, hidden_dim)

    return attention_pool(hidden_states)


# Registry
POOLING_METHODS = {
    "last_layer": pool_last_layer,
    "mean": pool_mean,
    "concat": pool_concat,
    "attention": pool_attention_weighted,
}
