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

**Gemma 4 text decoder: graph production implemented; Runtime support pending.**
The Component builds prefill/decode graphs (and layer segments for
multi-Device pipelines) for every Gemma 4 text checkpoint family: dense
(`google/gemma-4-31B-it`, `gemma-4-12B-it`), mixture-of-experts
(`gemma-4-26B-A4B-it`) and the per-layer-input / shared-KV E2B and E4B. The graph logic is in
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

### Variants

- **MoE (26B-A4B)**: each layer runs the dense GeGLU MLP *and* a routed
  expert branch side by side (`post_ffn_norm_1`, router + `moe-experts`,
  `pre_ffn_norm_2`/`post_ffn_norm_2`), sums them, then applies the layer's
  `post_ffn_norm`. The router scores the un-normalized residual stream.
  Uses Magnetar's existing `moe-experts` Operator.
- **Per-layer inputs (E2B/E4B)**: every layer receives a slice of a
  token-identity embedding mixed with a projection of the scaled embeddings,
  gates the hidden state with it and adds a normalized residual before the
  `layer_scalar`. A segment not starting at layer 0 re-declares `token_ids`
  and recomputes the embeddings it needs.
- **Shared KV (E2B/E4B)**: the last `num_kv_shared_layers` layers have no
  K/V weights and attend over the K/V of the last earlier layer of the same
  attention kind, using the producer's edges (so they allocate no KV
  resource). Their feed-forward is twice as wide when `use_double_wide_mlp`.
  A segment must contain the producers of its shared layers.

### Not implemented

- **Vision and audio towers.** Gemma 4's image/audio inputs need Operators and
  input types (patch embedding, 2-D RoPE, bidirectional vision attention,
  conformer audio blocks, embedding injection at image/audio token positions)
  and a way to pass non-token inputs, none of which `graph-builder` models
  today (inputs are token ids and `float32` tensors only). This belongs in the
  multimodal work tracked by `explore-multimodal-and-embedding-components`,
  as its own Components; this one is the text decoder only.
- Building a graph traps with an explicit reason for RoPE frequency scaling,
  attention bias, a shared expert, a router that does not renormalize, and any
  inconsistent configuration.
- Gemma 1-3 are not served by this Component (they need `1 + weight` norms
  and, for Gemma 2/3, different attention rules).

## Governing contract

Targets `magnetar:model-component-graph`, world `gemma-model-component-producer`
(`wit/model-component-graph.wit`, version **1.6.0**): the `1.5.0`
`graph-builder` and `model-config` interfaces unchanged, plus one new additive
interface, `gemma-config`, carrying what `architecture-config` cannot (per-layer
attention kind, sliding window, global head width and KV heads, per-flavour
RoPE bases, partial rotary factor, K=V, logit soft-capping, and the fields that
whether per-layer inputs, shared KV and its wider MLP apply). Component exports are the
same four as `components/qwen` and `components/llama`.

Weight logical names follow `components/llama` (`layers.N.self_attn.q_proj`,
`layers.N.mlp.gate_proj`, ...) plus `q_norm`, `k_norm`, `v_norm`,
`pre_ffn_norm`, `post_ffn_norm`, `layer_scalar` per layer.

## Runtime prerequisites

None of these exist in Magnetar `main` today; each is rejected fail-closed
until added, so this Component cannot silently misbehave against an older
Runtime. They are tracked in
[astorise/Magnetar#187](https://github.com/astorise/Magnetar/issues/187) and
detailed in [`docs/magnetar-runtime-requirements.md`](docs/magnetar-runtime-requirements.md).

1. **WIT `1.6.0`** with `gemma-config`, populated from the checkpoint's text
   config.
2. **Operator attributes** the portable Operators do not yet accept:
   `embedding.scale`; `attention.scale`; `rope.partial_rotary_factor`;
   `gelu.approximate = "tanh"`; per-head `rmsnorm`; `mul` with
   `broadcast = "scalar"`; `matmul`/`add` `output_scale`;
   `moe-experts.activation = "gelu_tanh"`; and an `activation` of
   `kind = "softcap"`. `attention.window_size`, `gelu` and `moe-experts`
   already exist.
3. **Loader**: map the checkpoint's `model.language_model.*` tensors to the
   logical names, and *derive* the ones the checkpoint does not contain as
   such (unit-scale `v_norm`/`router_norm`; router weight with
   `scale * hidden^-0.5` folded in; experts split from `gate_up_proj` with
   `per_expert_scale` folded into `down`; per-layer slices of the packed
   per-layer embedding and projection).

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
