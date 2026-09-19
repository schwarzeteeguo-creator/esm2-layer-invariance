"""Model loading for ESM/SaProt variants.

Supports:
    - SaProt 650M (our primary baseline)
    - ESM-2 variants (via HuggingFace)
    - ESM-1v (via fair-esm or HuggingFace)

All models exposed through a unified interface that supports
output_hidden_states for multi-scale extraction.
"""

import os
import torch
from pathlib import Path
from transformers import (
    EsmConfig,
    EsmForMaskedLM,
    EsmTokenizer,
    EsmModel,
    AutoTokenizer,
    AutoModel,
)

# Model registry — paths resolved relative to project root or ESM_MODEL_ROOT env var
_ESM_ROOT = Path(os.environ.get("ESM_MODEL_ROOT", Path(__file__).parent.parent))

_MODEL_PATHS = {
    "saprot_650m": _ESM_ROOT / "SaProt/weights/PLMs/SaProt_650M_AF2_hf",
    "saprot_650m_raw": _ESM_ROOT / "SaProt/weights/PLMs/SaProt_650M_AF2",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_3b": "facebook/esm2_t36_3B_UR50D",
}

_MODEL_CACHE = {}


def get_available_models():
    """Return list of model keys that can be loaded."""
    available = []
    for key, path in _MODEL_PATHS.items():
        path_obj = Path(path) if not str(path).startswith("facebook/") else None
        if path_obj is None:
            available.append(key)  # HuggingFace model
        elif path_obj.exists():
            available.append(key)
    return available


def load_model(
    model_key: str = "saprot_650m",
    device: str = None,
) -> dict:
    """Load model, tokenizer, and config. Cached after first load.

    Args:
        model_key: Key from _MODEL_PATHS
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        dict with keys: model, tokenizer, config, model_key, num_layers, hidden_dim
    """
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    path = _MODEL_PATHS[model_key]

    print(f"Loading {model_key}...")

    if model_key == "saprot_650m":
        # Load our converted HF SaProt
        tokenizer = EsmTokenizer.from_pretrained(str(path))
        model = EsmForMaskedLM.from_pretrained(str(path))
        config = model.config
        num_layers = config.num_hidden_layers
        hidden_dim = config.hidden_size

    elif model_key == "saprot_650m_raw":
        # Load raw SaProt tokenizer (same as converted)
        raw_path = _MODEL_PATHS["saprot_650m_raw"]
        tokenizer = EsmTokenizer.from_pretrained(str(raw_path))
        # Use converted model
        hf_path = _MODEL_PATHS["saprot_650m"]
        model = EsmForMaskedLM.from_pretrained(str(hf_path))
        config = model.config
        num_layers = config.num_hidden_layers
        hidden_dim = config.hidden_size

    elif model_key.startswith("esm2_"):
        # HuggingFace ESM-2
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = EsmForMaskedLM.from_pretrained(path)
        config = model.config
        num_layers = config.num_hidden_layers
        hidden_dim = config.hidden_size

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    result = {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "model_key": model_key,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "device": device,
    }

    _MODEL_CACHE[model_key] = result
    print(f"  Loaded: {num_layers} layers, {hidden_dim}-dim, on {device}")
    return result


def tokenize_sequence(tokenizer, sequence: str, structure_tokens: str = None):
    """Tokenize a protein sequence with optional SaProt 3Di structure tokens.

    Args:
        tokenizer: ESM/SaProt tokenizer
        sequence: Amino acid sequence (e.g., "MADE")
        structure_tokens: 3Di tokens (e.g., "pppp"). If None, uses 'p' for all.

    Returns:
        dict with input_ids, attention_mask
    """
    # Determine if this is a SaProt tokenizer (446 vocab) or standard ESM (33 vocab)
    is_saprot = tokenizer.vocab_size > 100

    if is_saprot:
        if structure_tokens is None:
            structure_tokens = "p" * len(sequence)

        if len(structure_tokens) != len(sequence):
            raise ValueError(
                f"structure_tokens length ({len(structure_tokens)}) != "
                f"sequence length ({len(sequence)})"
            )

        # Build SaProt tokens: AA + 3Di (e.g., "Mp Ap Dp Ep")
        tokens = []
        for aa, st in zip(sequence, structure_tokens):
            token = aa.upper() + st.lower()
            if token not in tokenizer.get_vocab():
                # Fallback: try first available structure token for this AA
                found = False
                for fallback_st in "pynwrqhgdlavtmfsekyic":
                    fb_token = aa.upper() + fallback_st
                    if fb_token in tokenizer.get_vocab():
                        token = fb_token
                        found = True
                        break
                if not found:
                    token = aa.upper() + "p"  # ultimate fallback
            tokens.append(token)

        token_str = " ".join(tokens)
    else:
        # Standard ESM: space-separated amino acids
        token_str = " ".join(list(sequence.upper()))

    return tokenizer(token_str, return_tensors="pt")
