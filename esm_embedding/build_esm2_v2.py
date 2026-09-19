"""Build ESM-2 from SaProt via state_dict mapping (robust approach).

Instead of copying modules one-by-one (fragile), we:
1. Create an empty ESM-2 model
2. Get its state_dict keys
3. Map SaProt's state_dict keys to match
4. Load mapped state_dict
"""

import os, sys, json, torch, copy
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig


def map_state_dict(saprot_sd, saprot_tokenizer, esm2_tokenizer):
    """Map SaProt state_dict → ESM-2 state_dict."""
    esm2_vocab = esm2_tokenizer.get_vocab()
    saprot_vocab = saprot_tokenizer.get_vocab()
    NEW_V = len(esm2_vocab)
    OLD_V = len(saprot_vocab)

    # Build token ID mapping
    esm2_to_saprot = {}
    for token, esm2_id in esm2_vocab.items():
        if token in saprot_vocab:
            esm2_to_saprot[esm2_id] = saprot_vocab[token]
        elif len(token) == 1 and token in "ACDEFGHIKLMNPQRSTVWY":
            sa_token = f"{token}#"
            esm2_to_saprot[esm2_id] = saprot_vocab.get(sa_token, saprot_tokenizer.unk_token_id)
        elif token == "<mask>":
            esm2_to_saprot[esm2_id] = saprot_tokenizer.mask_token_id
        else:
            esm2_to_saprot[esm2_id] = saprot_tokenizer.unk_token_id

    # Check which source keys exist
    src_keys = set(saprot_sd.keys())

    # Key mapping rules:
    # 1. Most keys are identical: esm.encoder.layer.X.* → esm.encoder.layer.X.*
    # 2. Embedding: esm.embeddings.word_embeddings.weight (446,1280) → (33,1280)
    # 3. LM head: lm_head.decoder.weight (446,1280) → (33,1280)
    # 4. LM head decoder bias: lm_head.decoder.bias (446,) → (33,)
    # 5. LM head final bias: lm_head.bias (446,) → (33,)

    new_sd = {}

    # Copy non-vocab-dependent weights directly
    for key, value in saprot_sd.items():
        if 'word_embeddings' not in key and 'lm_head' not in key:
            new_sd[key] = value.clone()
        elif key == 'esm.embeddings.word_embeddings.weight':
            # Map embeddings
            new_w = torch.zeros(NEW_V, value.shape[1])
            for esm2_id in range(NEW_V):
                saprot_id = esm2_to_saprot.get(esm2_id)
                if saprot_id is not None and 0 <= saprot_id < OLD_V:
                    new_w[esm2_id] = value[saprot_id].clone()
                else:
                    # Fallback: mean of mapped embeddings
                    valid_ids = [esm2_to_saprot[i] for i in range(NEW_V) if esm2_to_saprot.get(i) is not None and 0 <= esm2_to_saprot[i] < OLD_V]
                    if valid_ids:
                        new_w[esm2_id] = value[valid_ids].mean(dim=0)
            new_sd[key] = new_w
        elif key == 'lm_head.decoder.weight':
            # Map LM head decoder
            new_w = torch.zeros(NEW_V, value.shape[1])
            for esm2_id in range(NEW_V):
                saprot_id = esm2_to_saprot.get(esm2_id)
                if saprot_id is not None and 0 <= saprot_id < OLD_V:
                    new_w[esm2_id] = value[saprot_id].clone()
                else:
                    valid_ids = [esm2_to_saprot[i] for i in range(NEW_V) if esm2_to_saprot.get(i) is not None and 0 <= esm2_to_saprot[i] < OLD_V]
                    if valid_ids:
                        new_w[esm2_id] = value[valid_ids].mean(dim=0)
            new_sd[key] = new_w
        elif key == 'lm_head.decoder.bias':
            # Map LM head decoder bias
            new_b = torch.zeros(NEW_V)
            for esm2_id in range(NEW_V):
                saprot_id = esm2_to_saprot.get(esm2_id)
                if saprot_id is not None and 0 <= saprot_id < OLD_V:
                    new_b[esm2_id] = value[saprot_id].clone()
            new_sd[key] = new_b
        elif key == 'lm_head.bias':
            # Map LM head final bias
            new_b = torch.zeros(NEW_V)
            for esm2_id in range(NEW_V):
                saprot_id = esm2_to_saprot.get(esm2_id)
                if saprot_id is not None and 0 <= saprot_id < OLD_V:
                    new_b[esm2_id] = value[saprot_id].clone()
            new_sd[key] = new_b
        elif key == 'lm_head.dense.weight' or key == 'lm_head.dense.bias' or key == 'lm_head.layer_norm.weight' or key == 'lm_head.layer_norm.bias':
            # These are token-independent transformations, copy directly
            new_sd[key] = value.clone()

    return new_sd


def main():
    sys.path.insert(0, "/root/autodl-tmp")
    from esm_embedding.model import load_model

    print("Loading SaProt model...")
    saprot_info = load_model("saprot_650m", device="cpu")
    saprot_model = saprot_info["model"]
    saprot_tokenizer = saprot_info["tokenizer"]

    # Create ESM-2 tokenizer
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

    # Create ESM-2 config
    saprot_cfg = saprot_model.config
    esm2_config = EsmConfig(
        vocab_size=len(esm2_tokenizer),
        hidden_size=saprot_cfg.hidden_size,
        num_hidden_layers=saprot_cfg.num_hidden_layers,
        num_attention_heads=saprot_cfg.num_attention_heads,
        intermediate_size=saprot_cfg.intermediate_size,
        hidden_act=saprot_cfg.hidden_act,
        max_position_embeddings=saprot_cfg.max_position_embeddings,
        mask_token_id=esm2_tokenizer.mask_token_id,
        pad_token_id=esm2_tokenizer.pad_token_id,
        bos_token_id=esm2_tokenizer.cls_token_id,
        eos_token_id=esm2_tokenizer.eos_token_id,
    )

    # Create empty model
    esm2_model = EsmForMaskedLM(esm2_config)
    print(f"ESM-2 model created: {esm2_config.num_hidden_layers} layers, {esm2_config.hidden_size}-dim")

    # Get source state_dict
    saprot_sd = saprot_model.state_dict()

    # Map to ESM-2 state_dict
    print("\nMapping state_dict...")
    new_sd = map_state_dict(saprot_sd, saprot_tokenizer, esm2_tokenizer)

    # Check for missing/unexpected keys
    dst_keys = set(esm2_model.state_dict().keys())
    mapped_keys = set(new_sd.keys())

    missing = dst_keys - mapped_keys
    unexpected = mapped_keys - dst_keys

    if missing:
        print(f"\nWARNING: Missing keys ({len(missing)}):")
        for k in sorted(missing):
            print(f"  {k}: {esm2_model.state_dict()[k].shape}")

    if unexpected:
        print(f"\nNOTE: Unexpected keys ({len(unexpected)}):")
        for k in sorted(unexpected):
            print(f"  {k}: {new_sd[k].shape}")

    # Load mapped state_dict
    # First handle missing keys
    for k in missing:
        if 'word_embeddings' in k or 'lm_head' in k:
            # Already handled in mapping
            new_sd[k] = esm2_model.state_dict()[k]
        else:
            print(f"  WARNING: cannot fill {k}")

    # Filter to only expected keys
    filtered_sd = {k: v for k, v in new_sd.items() if k in dst_keys}

    # Load
    esm2_model.load_state_dict(filtered_sd, strict=False)
    esm2_model.eval()
    print("Weights loaded.")

    # ── Test ───────────────────────────────────────────────────
    print("\n=== Testing ===")
    test_seq = "M A <mask> D E"
    inputs = esm2_tokenizer(test_seq, return_tensors="pt")
    print(f"Input: {esm2_tokenizer.decode(inputs['input_ids'][0])}")

    with torch.no_grad():
        outputs = esm2_model(**inputs, output_hidden_states=True)

    n_hidden = len(outputs.hidden_states)
    print(f"Hidden states: {n_hidden} (embedding + {n_hidden-1} layers)")
    print(f"Layer 0 shape: {outputs.hidden_states[0].shape}")
    print(f"Layer {n_hidden-1} shape: {outputs.hidden_states[-1].shape}")

    logits = outputs.logits[0, 2, :]
    probs = torch.softmax(logits, dim=-1)
    top5 = torch.topk(probs, 5)
    print("Top 5 at masked position:")
    for i in range(5):
        tid = top5.indices[i].item()
        tok = esm2_tokenizer.convert_ids_to_tokens(tid)
        print(f"  {tok}: prob={top5.values[i].item():.4f}")

    # ── Save ───────────────────────────────────────────────────
    print(f"\nSaving to {esm2_dir}...")
    esm2_model.save_pretrained(esm2_dir)
    esm2_model.config.save_pretrained(esm2_dir)
    print("Done!")


if __name__ == "__main__":
    main()
