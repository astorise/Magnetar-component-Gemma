"""Turns a tiny Gemma 4 bundle (`make_tiny_gemma4_bundle.py`) into a quantized
`gemma4` GGUF file, laid out like llama.cpp's own (metadata keys and tensor
names of `MODEL_ARCH.GEMMA4`), plus a reference bundle whose weights are the
*dequantized* values, so `transformers` computes on exactly the numbers the
quantized file holds. Comparing Magnetar's run of the GGUF with that reference
therefore isolates Magnetar's handling of the quantized format, not the
quantization error itself.

Usage: python3 make_tiny_gemma4_gguf.py <bundle_dir> <out_dir> [--proj q4_0|q8_0|f32] [--embd q8_0|q4_0|f32]
  <out_dir>/gguf/model.gguf   the quantized file
  <out_dir>/ref/              config.json + model.safetensors (dequantized), for gemma4_reference_logits.py
"""
import argparse
import json
import os
import shutil

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import GGUFWriter
from gguf.quants import dequantize, quantize
from safetensors.torch import load_file, save_file
import torch

PREFIX = "model.language_model."
QUANT = {"q4_0": T.Q4_0, "q8_0": T.Q8_0, "f32": T.F32}
PROJ = {"per_layer_input_gate": "inp_gate", "per_layer_projection": "proj",
        "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down"}
NORM = {"input_layernorm": "attn_norm", "post_attention_layernorm": "post_attention_norm",
        "pre_feedforward_layernorm": "ffn_norm", "post_feedforward_layernorm": "post_ffw_norm",
        "pre_feedforward_layernorm_2": "pre_ffw_norm_2", "post_feedforward_layernorm_1": "post_ffw_norm_1",
        "post_feedforward_layernorm_2": "post_ffw_norm_2", "post_per_layer_input_norm": "post_norm",
        "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
        "layer_scalar": "layer_output_scale"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("out")
    ap.add_argument("--proj", default="q4_0")
    ap.add_argument("--embd", default="q8_0")
    args = ap.parse_args()

    cfg = json.load(open(f"{args.bundle}/config.json"))
    t = cfg["text_config"]
    n = t["num_hidden_layers"]
    tensors = load_file(f"{args.bundle}/model.safetensors")

    shared = t.get("num_kv_shared_layers", 0)
    first_shared = n - shared
    kinds = t["layer_types"]
    k_eq_v = t.get("attention_k_eq_v", False)
    kv_full = t["num_global_key_value_heads"] if k_eq_v else t["num_key_value_heads"]
    wide = t.get("use_double_wide_mlp", False)
    ffn = [t["intermediate_size"] * (2 if wide and shared and i >= first_shared else 1) for i in range(n)]

    os.makedirs(f"{args.out}/gguf", exist_ok=True)
    os.makedirs(f"{args.out}/ref", exist_ok=True)
    w = GGUFWriter(f"{args.out}/gguf/model.gguf", "gemma4")
    w.add_string("general.name", "tiny-gemma4")
    a = "gemma4."
    w.add_uint32(a + "embedding_length", t["hidden_size"])
    w.add_array(a + "feed_forward_length", ffn)
    w.add_uint32(a + "block_count", n)
    w.add_uint32(a + "attention.head_count", t["num_attention_heads"])
    w.add_array(a + "attention.head_count_kv", [t["num_key_value_heads"] if k == "sliding_attention" else kv_full for k in kinds])
    w.add_float32(a + "attention.layer_norm_rms_epsilon", t["rms_norm_eps"])
    rp = t["rope_parameters"]
    w.add_float32(a + "rope.freq_base", rp["full_attention"]["rope_theta"])
    w.add_float32(a + "rope.freq_base_swa", rp["sliding_attention"]["rope_theta"])
    w.add_uint32(a + "attention.key_length", t["global_head_dim"])
    w.add_uint32(a + "attention.value_length", t["global_head_dim"])
    w.add_uint32(a + "attention.key_length_swa", t["head_dim"])
    w.add_uint32(a + "attention.value_length_swa", t["head_dim"])
    w.add_uint32(a + "attention.sliding_window", t["sliding_window"])
    w.add_uint32(a + "attention.shared_kv_layers", shared)
    w.add_uint32(a + "embedding_length_per_layer_input", t.get("hidden_size_per_layer_input", 0))
    if t.get("enable_moe_block"):
        w.add_uint32(a + "expert_count", t["num_experts"])
        w.add_uint32(a + "expert_used_count", t["top_k_experts"])
        w.add_uint32(a + "expert_feed_forward_length", t["moe_intermediate_size"])
    if t.get("final_logit_softcapping"):
        w.add_float32(a + "final_logit_softcapping", t["final_logit_softcapping"])
    w.add_array(a + "attention.sliding_window_pattern", [k == "sliding_attention" for k in kinds])
    vocab = t["vocab_size"]
    w.add_uint32("tokenizer.ggml.bos_token_id", 0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "ab"] + [f"t{i}" for i in range(3, vocab)])
    # One merge that yields a vocabulary token; the tokenizer is never used to encode text here.
    w.add_array("tokenizer.ggml.merges", ["a b"])

    ref = {}

    def put(hf_name, gguf_name, kind):
        x = tensors[hf_name].float().numpy()
        if kind == "f32" or x.ndim < 2:
            w.add_tensor(gguf_name, x.astype(np.float32))
            ref[hf_name] = torch.from_numpy(x.astype(np.float32))
            return
        q = quantize(x, QUANT[kind])
        w.add_tensor(gguf_name, q, raw_dtype=QUANT[kind])
        ref[hf_name] = torch.from_numpy(dequantize(q, QUANT[kind]).astype(np.float32))

    put(PREFIX + "embed_tokens.weight", "token_embd.weight", args.embd)
    put(PREFIX + "norm.weight", "output_norm.weight", "f32")
    full_dim = t["global_head_dim"]
    rotated = int(full_dim * rp["full_attention"].get("partial_rotary_factor", 1.0) // 2)
    freqs = np.array([1.0] * rotated + [1e30] * (full_dim // 2 - rotated), dtype=np.float32)
    w.add_tensor("rope_freqs.weight", freqs)
    if t.get("hidden_size_per_layer_input"):
        put(PREFIX + "embed_tokens_per_layer.weight", "per_layer_token_embd.weight", args.embd)
        put(PREFIX + "per_layer_model_projection.weight", "per_layer_model_proj.weight", args.proj)
        put(PREFIX + "per_layer_projection_norm.weight", "per_layer_proj_norm.weight", "f32")
    for name in tensors:
        if not name.startswith(PREFIX + "layers."):
            continue
        layer, rest = name[len(PREFIX + "layers."):].split(".", 1)
        base = rest[:-len(".weight")] if rest.endswith(".weight") else rest
        if base == "experts.gate_up_proj":  # fused: each expert's gate rows, then its up rows
            put(name, f"blk.{layer}.ffn_gate_up_exps.weight", args.proj)
        elif base == "experts.down_proj":
            put(name, f"blk.{layer}.ffn_down_exps.weight", args.proj)
        elif base == "router.proj":
            put(name, f"blk.{layer}.ffn_gate_inp.weight", "f32")
        elif base == "router.scale":
            put(name, f"blk.{layer}.ffn_gate_inp.scale", "f32")
        elif base == "router.per_expert_scale":
            put(name, f"blk.{layer}.ffn_down_exps.scale", "f32")
        elif base in PROJ:
            put(name, f"blk.{layer}.{PROJ[base]}.weight", args.proj)
        elif base in NORM:
            put(name, f"blk.{layer}.{NORM[base]}.weight", "f32")
        else:
            raise SystemExit(f"unmapped tensor {name}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # Reference bundle: same config, every weight at the value the GGUF holds.
    full = {k: v.float() for k, v in tensors.items() if k.startswith(PREFIX)}
    full.update(ref)
    save_file({k: v.contiguous() for k, v in full.items()}, f"{args.out}/ref/model.safetensors", metadata={"format": "pt"})
    shutil.copy(f"{args.bundle}/config.json", f"{args.out}/ref/config.json")
    print("wrote", f"{args.out}/gguf/model.gguf", os.path.getsize(f"{args.out}/gguf/model.gguf"), "bytes")


if __name__ == "__main__":
    main()
