# magnetar-component-gemma

## Purpose

A WebAssembly Component implementing the Google Gemma 4 model family for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It
supplies prefill/decode graph semantics for Gemma-compatible Model Artifacts
through the Runtime's Model Component contract, without ever selecting a
Provider, Device, or receiving native Runtime handles -- per Magnetar's
architectural invariants (Components request Capabilities, not Providers
or Devices).

## Status

**Gemma 4 dense text: graph production implemented; Runtime support pending.**
The Component builds prefill/decode graphs (and layer segments for
multi-Device pipelines) for dense Gemma 4 text checkpoints
(`google/gemma-4-31B-it`, `gemma-4-12B-it`). The graph logic is in
`src/plan.rs`, independent of any host, and covered by unit tests against a
recording `graph-builder` (including the real 31B shape). `src/component.rs`
only adapts the WIT bindings to it.

It does **not** run end to end yet: it needs the Runtime changes listed under
[Runtime prerequisites](#runtime-prerequisites).

### What a Gemma 4 layer is

Verified against `transformers`' `modeling_gemma4.py` and the checkpoints'
`config.json`; it differs from Gemma 1-3:

- RMSNorm scale is the plain `weight` (not `1 + weight`); token embeddings are
  scaled by `sqrt(hidden_size)`; the output projection is tied to the
  embedding table.
- Sandwich normalization: `input_norm` -> attention -> `post_attn_norm` ->
  residual, then `pre_ffn_norm` -> GeGLU MLP (`gelu_pytorch_tanh`) ->
  `post_ffn_norm` -> residual, then the result is multiplied by a per-layer
  `layer_scalar`.
- Queries and keys get a per-head RMSNorm, values a weightless one
  (`v_norm`); attention uses `scale = 1.0`.
- `layer_types` alternates `sliding_attention` layers (window 1024, 256-wide
  heads, RoPE base 10 000) with `full_attention` layers (512-wide global heads
  with their own KV head count, RoPE base 1 000 000, "proportional" partial
  rotary factor 0.25). With `attention_k_eq_v`, full layers have no V
  projection: V is K's raw projection.
- Logits are soft-capped: `30 * tanh(logits / 30)`.

### Not implemented (refused, never built wrong)

Building a graph traps with an explicit reason for: mixture-of-experts
(26B-A4B), per-layer input embeddings and cross-layer KV sharing (E2B, E4B),
RoPE frequency scaling, and attention bias. Gemma 1-3 are not served by this
Component (they need `1 + weight` norms and, for Gemma 2/3, different
attention rules). Vision/audio towers are out of scope; this is the text
decoder only.

## Governing contract

Targets `magnetar:model-component-graph`, world `gemma-model-component-producer`
(`wit/model-component-graph.wit`, version **1.6.0**): the `1.5.0`
`graph-builder` and `model-config` interfaces unchanged, plus one new additive
interface, `gemma-config`, carrying what `architecture-config` cannot (per-layer
attention kind, sliding window, global head width and KV heads, per-flavour
RoPE bases, partial rotary factor, K=V, logit soft-capping, and the fields that
let the Component detect the variants it refuses). Component exports are the
same four as `components/qwen` and `components/llama`.

Weight logical names follow `components/llama` (`layers.N.self_attn.q_proj`,
`layers.N.mlp.gate_proj`, ...) plus `q_norm`, `k_norm`, `v_norm`,
`pre_ffn_norm`, `post_ffn_norm`, `layer_scalar` per layer.

## Runtime prerequisites

None of these exist in Magnetar `main` today; each is rejected fail-closed
until added, so this Component cannot silently misbehave against an older
Runtime.

1. **WIT `1.6.0`** with `gemma-config`, populated from the checkpoint's text
   config (`layer_types`, `sliding_window`, `global_head_dim`,
   `num_global_key_value_heads`, `rope_parameters`, `attention_k_eq_v`,
   `final_logit_softcapping`, `hidden_size_per_layer_input`,
   `num_kv_shared_layers`).
2. **Operator attributes** the portable Operators do not yet accept:
   `embedding.scale`; `attention.scale`; `rope.partial_rotary_factor`;
   `gelu.approximate = "tanh"`; per-head `rmsnorm` (`head_count`,
   `head_dimension`); `mul` with `broadcast = "scalar"` (for `layer_scalar`);
   and an `activation` of `kind = "softcap"` with `cap`. `attention.window_size`
   and `gelu` already exist.
3. **Loader**: map the checkpoint's `model.language_model.*` tensors to the
   logical names above, and resolve `layers.N.self_attn.v_norm` (absent from
   the checkpoint) to a unit-scale vector.

## Verification

```bash
cargo test --features bindings          # graph logic + WIT bridge
cargo build --release --target wasm32-wasip2
```

`rust-version = "1.98"` is Magnetar's pinned toolchain. A newer-than-installed
check can be bypassed locally with `--ignore-rust-version`.

Not yet verified: running the built Component under Magnetar's Wasmtime
Engine and comparing logits with `transformers` on a real checkpoint, which
requires the Runtime prerequisites above and the checkpoint weights, which
were not downloaded here. Only the checkpoints' `config.json` and tensor index
were read.

## Relationship to magnetar-runtime

This Component will be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/gemma`.
