"""Build ESM-2 equivalent model from SaProt weights — FIXED version.

SaProt's structure-aware vocabulary includes "AA#" tokens (e.g., "M#", "A#")
where "#" masks the structure. The embedding and LM head rows for these tokens
semantically correspond to ESM-2's single-letter AA tokens.

Architecture notes:
- ESM LM head: dense(1280→1280) → layer_norm → decoder(1280→vocab)
- Embeddings: only word_embeddings (no separate position_embeddings)
- No embedding LayerNorm in this HF version
"""

import os
import sys
import json
import torch
import copy
from pathlib import Path
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig


def build_esm2_model(saprot_model, saprot_tokenizer, esm2_tokenizer):
    """Build an ESM-2 model by extracting weights from SaProt."""

    # Build ID mapping
    esm2_vocab = esm2_tokenizer.get_vocab()
    saprot_vocab = saprot_tokenizer.get_vocab()

    esm2_to_saprot = {}
    unk_id = saprot_tokenizer.unk_token_id

    for token, esm2_id in esm2_vocab.items():
        if token in saprot_vocab:
            esm2_to_saprot[esm2_id] = saprot_vocab[token]
        elif len(token) == 1 and token in "ACDEFGHIKLMNPQRSTVWY":
            sa_token = f"{token}#"
            esm2_to_saprot[esm2_id] = saprot_vocab.get(sa_token, unk_id)
        elif token == "<mask>":
            esm2_to_saprot[esm2_id] = saprot_tokenizer.mask_token_id
        else:
            esm2_to_saprot[esm2_id] = unk_id

    print(f"  Mapped {len(esm2_to_saprot)}/{len(esm2_vocab)} tokens")
    for aa in "MADE":
        print(f"    '{aa}': ESM-2 id={esm2_vocab[aa]} → SaProt id={esm2_to_saprot[esm2_vocab[aa]]}")

    # ── Create ESM-2 config ─────────────────────────────────────
    saprot_cfg = saprot_model.config
    esm2_config = EsmConfig(
        vocab_size=len(esm2_vocab),
        hidden_size=saprot_cfg.hidden_size,
        num_hidden_layers=saprot_cfg.num_hidden_layers,
        num_attention_heads=saprot_cfg.num_attention_heads,
        intermediate_size=saprot_cfg.intermediate_size,
        hidden_act=saprot_cfg.hidden_act,
        hidden_dropout_prob=saprot_cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=saprot_cfg.attention_probs_dropout_prob,
        max_position_embeddings=saprot_cfg.max_position_embeddings,
        mask_token_id=esm2_vocab.get("<mask>", 32),
        pad_token_id=esm2_vocab.get("<pad>", 1),
        bos_token_id=esm2_vocab.get("<cls>", 0),
        eos_token_id=esm2_vocab.get("<eos>", 2),
        position_embedding_type="absolute",
    )

    # Create empty ESM-2 model
    esm2_model = EsmForMaskedLM(esm2_config)
    esm2_model.eval()
    NEW_V = len(esm2_vocab)
    OLD_V = len(saprot_vocab)
    H = saprot_cfg.hidden_size

    # ── 1. Word embeddings ─────────────────────────────────────
    old_emb = saprot_model.esm.embeddings.word_embeddings.weight.data  # (446, 1280)
    new_emb = esm2_model.esm.embeddings.word_embeddings.weight.data    # (33, 1280)

    for esm2_id in range(NEW_V):
        saprot_id = esm2_to_saprot.get(esm2_id)
        if saprot_id is not None and 0 <= saprot_id < OLD_V:
            new_emb[esm2_id] = old_emb[saprot_id].clone()
        else:
            # Use mean of known embeddings
            known = [i for i in range(NEW_V) if i != esm2_id and esm2_to_saprot.get(i) is not None and 0 <= esm2_to_saprot[i] < OLD_V]
            if known:
                new_emb[esm2_id] = torch.stack([new_emb[i] for i in known]).mean(dim=0)

    print(f"  Embeddings: {list(new_emb.shape)}")

    # ── 2. Transformer layers (exact copy) ────────────────────
    for l_idx in range(saprot_cfg.num_hidden_layers):
        src = saprot_model.esm.encoder.layer[l_idx]
        dst = esm2_model.esm.encoder.layer[l_idx]

        # Pre-attention LayerNorm
        dst.attention.LayerNorm.weight.data.copy_(src.attention.LayerNorm.weight.data)
        dst.attention.LayerNorm.bias.data.copy_(src.attention.LayerNorm.bias.data)

        # Self-attention QKV
        dst.attention.self.query.weight.data.copy_(src.attention.self.query.weight.data)
        dst.attention.self.query.bias.data.copy_(src.attention.self.query.bias.data)
        dst.attention.self.key.weight.data.copy_(src.attention.self.key.weight.data)
        dst.attention.self.key.bias.data.copy_(src.attention.self.key.bias.data)
        dst.attention.self.value.weight.data.copy_(src.attention.self.value.weight.data)
        dst.attention.self.value.bias.data.copy_(src.attention.self.value.bias.data)

        # Attention output
        dst.attention.output.dense.weight.data.copy_(src.attention.output.dense.weight.data)
        dst.attention.output.dense.bias.data.copy_(src.attention.output.dense.bias.data)
        dst.attention.output.LayerNorm.weight.data.copy_(src.attention.output.LayerNorm.weight.data)
        dst.attention.output.LayerNorm.bias.data.copy_(src.attention.output.LayerNorm.bias.data)

        # FFN
        dst.intermediate.dense.weight.data.copy_(src.intermediate.dense.weight.data)
        dst.intermediate.dense.bias.data.copy_(src.intermediate.dense.bias.data)
        dst.output.dense.weight.data.copy_(src.output.dense.weight.data)
        dst.output.dense.bias.data.copy_(src.output.dense.bias.data)

        # Post-FFN LayerNorm
        dst.output.LayerNorm.weight.data.copy_(src.output.LayerNorm.weight.data)
        dst.output.LayerNorm.bias.data.copy_(src.output.LayerNorm.bias.data)

        # Pre-attention LayerNorm (newer ESM versions have this)
        # SaProt has: layer.LayerNorm (before attention) - checked, it's dst.LayerNorm
        # The SaProt implementation has attention.LayerNorm which is used for Q/K/V LayerNorm?
        # Actually, ESM implements attention with separate LayerNorm for QKV components
        # But from the parameter list, the structure looks standard
        # Let me also check if there's a layer.LayerNorm
        if hasattr(src, 'LayerNorm'):
            dst.LayerNorm.weight.data.copy_(src.LayerNorm.weight.data)
            dst.LayerNorm.bias.data.copy_(src.LayerNorm.bias.data)

    print(f"  Copied {saprot_cfg.num_hidden_layers} transformer layers")

    # ── 3. Final encoder LayerNorm ────────────────────────────
    src_ln = saprot_model.esm.encoder.emb_layer_norm_after
    dst_ln = esm2_model.esm.encoder.emb_layer_norm_after
    dst_ln.weight.data.copy_(src_ln.weight.data)
    dst_ln.bias.data.copy_(src_ln.bias.data)
    print(f"  Final LayerNorm: ok")

    # ── 4. LM head ───────────────────────────────────────────
    # SaProt LM head: dense(1280→1280) → LayerNorm → decoder(1280→446)
    # ESM-2 LM head:  dense(1280→1280) → LayerNorm → decoder(1280→33)

    # Copy dense and layer_norm (input-side transformations, token-independent)
    esm2_model.lm_head.dense.weight.data.copy_(saprot_model.lm_head.dense.weight.data)
    esm2_model.lm_head.dense.bias.data.copy_(saprot_model.lm_head.dense.bias.data)
    esm2_model.lm_head.layer_norm.weight.data.copy_(saprot_model.lm_head.layer_norm.weight.data)
    esm2_model.lm_head.layer_norm.bias.data.copy_(saprot_model.lm_head.layer_norm.bias.data)

    # Extract decoder rows for ESM-2 tokens
    old_decoder = saprot_model.lm_head.decoder.weight.data  # (446, 1280)
    new_decoder = esm2_model.lm_head.decoder.weight.data    # (33, 1280)

    for esm2_id in range(NEW_V):
        saprot_id = esm2_to_saprot.get(esm2_id)
        if saprot_id is not None and 0 <= saprot_id < OLD_V:
            new_decoder[esm2_id] = old_decoder[saprot_id].clone()
        else:
            known = [i for i in range(NEW_V) if i != esm2_id and esm2_to_saprot.get(i) is not None and 0 <= esm2_to_saprot[i] < OLD_V]
            if known:
                new_decoder[esm2_id] = torch.stack([new_decoder[i] for i in known]).mean(dim=0)

    # Also handle decoder bias
    if hasattr(saprot_model.lm_head.decoder, 'bias') and saprot_model.lm_head.decoder.bias is not None:
        old_bias = saprot_model.lm_head.decoder.bias.data  # (446,)
        new_bias = esm2_model.lm_head.decoder.bias.data    # (33,)
        for esm2_id in range(NEW_V):
            saprot_id = esm2_to_saprot.get(esm2_id)
            if saprot_id is not None and 0 <= saprot_id < OLD_V:
                new_bias[esm2_id] = old_bias[saprot_id].clone()

    # Also copy the final bias (lm_head.bias)
    if hasattr(saprot_model.lm_head, 'bias') and saprot_model.lm_head.bias is not None:
        old_final_bias = saprot_model.lm_head.bias.data  # (446,)
        new_final_bias = esm2_model.lm_head.bias.data    # (33,)
        for esm2_id in range(NEW_V):
            saprot_id = esm2_to_saprot.get(esm2_id)
            if saprot_id is not None and 0 <= saprot_id < OLD_V:
                new_final_bias[esm2_id] = old_final_bias[saprot_id].clone()

    print(f"  LM head decoder: {list(new_decoder.shape)}")
    print("  ESM-2 model built successfully!")

    return esm2_model


def main():
    sys.path.insert(0, "/root/autodl-tmp")
    from esm_embedding.model import load_model

    print("Loading SaProt model...")
    saprot_info = load_model("saprot_650m", device="cpu")
    saprot_model = saprot_info["model"]
    saprot_tokenizer = saprot_info["tokenizer"]

    # Create ESM-2 tokenizer files
    esm2_dir = "/root/autodl-tmp/esm2_model_built"
    os.makedirs(esm2_dir, exist_ok=True)

    esm2_vocab_tokens = [
        "<cls>", "<pad>", "<eos>", "<unk>",
        "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
        "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
        "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"
    ]

    with open(f"{esm2_dir}/vocab.txt", "w") as f:
        for tok in esm2_vocab_tokens:
            f.write(tok + "\n")

    with open(f"{esm2_dir}/tokenizer_config.json", "w") as f:
        json.dump({"tokenizer_class": "EsmTokenizer"}, f)

    with open(f"{esm2_dir}/special_tokens_map.json", "w") as f:
        json.dump({
            "cls_token": "<cls>", "pad_token": "<pad>",
            "eos_token": "<eos>", "unk_token": "<unk>",
            "mask_token": "<mask>",
        }, f)

    esm2_tokenizer = EsmTokenizer.from_pretrained(esm2_dir)
    print(f"ESM-2 tokenizer: {esm2_tokenizer.vocab_size} tokens")

    # Build model
    print("\nBuilding ESM-2 from SaProt...")
    esm2_model = build_esm2_model(saprot_model, saprot_tokenizer, esm2_tokenizer)

    # ── Test forward pass ─────────────────────────────────────
    print("\n=== Testing forward pass ===")
    test_seq = "M A <mask> D E"
    inputs = esm2_tokenizer(test_seq, return_tensors="pt")
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Decoded: {esm2_tokenizer.decode(inputs['input_ids'][0])}")

    with torch.no_grad():
        outputs = esm2_model(**inputs, output_hidden_states=True)

    print(f"Hidden states: {len(outputs.hidden_states)} (embedding + {len(outputs.hidden_states)-1} layers)")
    logits = outputs.logits[0, 2, :]  # Masked position
    probs = torch.softmax(logits, dim=-1)
    top5 = torch.topk(probs, 5)
    print("Top 5 at masked position:")
    for i in range(5):
        tid = top5.indices[i].item()
        print(f"  {esm2_vocab_tokens[tid]}: prob={top5.values[i].item():.4f}")

    # Verify hidden states per layer
    all_hidden = list(outputs.hidden_states)
    print(f"\nLayer shapes: embed={all_hidden[0].shape}, layer_32={all_hidden[-1].shape}")

    # Save model
    print(f"\nSaving model to {esm2_dir}...")
    esm2_model.save_pretrained(esm2_dir)
    esm2_model.config.save_pretrained(esm2_dir)
    print("Done! Model saved.")

    return esm2_model, esm2_tokenizer


if __name__ == "__main__":
    main()
