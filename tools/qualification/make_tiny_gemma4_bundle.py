"""Builds a tiny, randomly initialised Gemma 4 text checkpoint in the exact
Hugging Face bundle layout of `google/gemma-4-31B-it` (nested `text_config`,
`model.language_model.*` tensor names, F32 safetensors, a word-level
`tokenizer.json`), small enough to run anywhere.

The weights are random but *scaled* so every mechanism is exercised with
non-trivial values: norm weights away from 1, a `layer_scalar` away from 1,
logits large enough for the soft-cap to bite, a sliding window shorter than
the sequences used. The same directory feeds both the `transformers`
reference (`gemma4_reference_logits.py`) and Magnetar's loader.

Usage: python3 make_tiny_gemma4_bundle.py <out_dir> [--variant dense|moe|ple] [--seed N]
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

VOCAB = 64
HIDDEN = 64
LAYERS = 6
HEADS = 4


def text_config(variant):
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    cfg = dict(
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        bos_token_id=2,
        enable_moe_block=False,
        eos_token_id=1,
        final_logit_softcapping=30.0,
        global_head_dim=32,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=HIDDEN,
        hidden_size_per_layer_input=0,
        initializer_range=0.02,
        intermediate_size=128,
        layer_types=layer_types,
        max_position_embeddings=256,
        model_type="gemma4_text",
        num_attention_heads=HEADS,
        num_global_key_value_heads=1,
        num_hidden_layers=LAYERS,
        num_key_value_heads=2,
        num_kv_shared_layers=0,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        sliding_window=6,
        tie_word_embeddings=True,
        use_bidirectional_attention=None,
        use_cache=True,
        use_double_wide_mlp=False,
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
    )
    if variant == "dense":
        pass
    elif variant == "unified":
        # `gemma-4-12B-it`-style bundle: the same dense text decoder under the
        # `gemma4_unified` model type, with image patches embedded by
        # `model.vision_embedder.*` (not part of the text decoder).
        cfg.update(use_bidirectional_attention="vision")
    elif variant == "shared":
        # Gemma 4 E2B/E4B style KV sharing without per-layer inputs: the last
        # 3 layers reuse K/V of the last earlier layer of their attention
        # type, and have a feed-forward twice as wide.
        cfg.update(num_kv_shared_layers=3, use_double_wide_mlp=True, attention_k_eq_v=False)
    elif variant == "ple":
        # E2B-like: per-layer inputs on top of shared KV.
        cfg.update(
            num_kv_shared_layers=3,
            use_double_wide_mlp=True,
            attention_k_eq_v=False,
            # A multiple of the 32-element quantization block, so a packed table can be cut per layer.
            hidden_size_per_layer_input=32,
        )
    elif variant == "moe":
        # 26B-A4B-like: a dense MLP and routed experts side by side.
        cfg.update(
            enable_moe_block=True,
            num_experts=8,
            top_k_experts=2,
            moe_intermediate_size=32,
        )
    else:
        raise SystemExit(f"unknown variant {variant!r}")
    return cfg


def scaled_init(model, seed):
    """Deterministic, well-scaled random weights."""
    g = torch.Generator().manual_seed(seed)

    def normal(shape, std):
        return torch.randn(shape, generator=g) * std

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and "embed_tokens" in name and "per_layer" not in name:
                # Tied LM head. Small enough that the layers, not the last
                # token's own embedding, decide the next token.
                p.copy_(normal(p.shape, 0.1))
            elif p.ndim == 2:
                p.copy_(normal(p.shape, p.shape[1] ** -0.5))
            elif name.endswith(("router.scale", "router.per_expert_scale")):
                p.copy_(1.0 + normal(p.shape, 0.2))
            elif name.endswith(("experts.gate_up_proj", "experts.down_proj")):
                p.copy_(normal(p.shape, p.shape[-1] ** -0.5))
            elif "embed_tokens_per_layer" in name:
                p.copy_(normal(p.shape, 0.5))
            elif name.endswith(("post_attention_layernorm.weight", "post_feedforward_layernorm.weight")):
                # Let every block move the residual stream noticeably.
                p.copy_(2.0 + normal(p.shape, 0.3))
            elif name == "model.norm.weight":
                # Final norm gain: logits large enough for the soft-cap to bite.
                p.copy_(6.0 + normal(p.shape, 0.5))
            elif p.ndim == 1:
                p.copy_(1.0 + normal(p.shape, 0.2))
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.6 + torch.rand(b.shape, generator=g) * 0.8)


def scaled_vision(seed):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randn((16, 8), generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--variant", default="dense")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = text_config(args.variant)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**cfg)).float().eval()
    scaled_init(model, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    tensors = {}
    for name, p in model.state_dict().items():
        if name.startswith("lm_head."):
            continue  # tied to the embedding table, absent from the bundle
        # `Gemma4ForCausalLM` names the decoder `model.*`; the published
        # multimodal bundle nests it under `model.language_model.*`.
        assert name.startswith("model."), name
        tensors["model.language_model." + name[len("model.") :]] = p.detach().float().contiguous()
    unified = args.variant == "unified"
    if unified:
        tensors["model.vision_embedder.patch_dense.weight"] = scaled_vision(args.seed)
    bundle_config = {
        "architectures": ["Gemma4UnifiedForConditionalGeneration" if unified else "Gemma4ForConditionalGeneration"],
        "model_type": "gemma4_unified" if unified else "gemma4",
        "tie_word_embeddings": True,
        "dtype": "float32",
        "eos_token_id": [1, 106],
        "text_config": {k: v for k, v in cfg.items() if k != "model_type"}
        | {"model_type": "gemma4_unified_text" if unified else "gemma4_text", "dtype": "float32"},
    }
    save_file(tensors, os.path.join(args.out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(bundle_config, f, indent=2)
        f.write("\n")

    vocab = {f"t{i}": i for i in range(VOCAB)}
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"},
    }
    with open(os.path.join(args.out_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)
    with open(os.path.join(args.out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({}, f)
    print("wrote", sorted(tensors)[:6], "...", len(tensors), "tensors")


if __name__ == "__main__":
    sys.exit(main())
