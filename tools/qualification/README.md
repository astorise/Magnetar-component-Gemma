# Logit comparison against `transformers`

Checks that the Gemma 4 Component, run by Magnetar, computes the same logits as
the canonical `Gemma4ForCausalLM` in `transformers` (float32).

Nothing here needs the real checkpoints. `make_tiny_gemma4_bundle.py` builds a
tiny **randomly initialised** Gemma 4 text checkpoint in the exact layout of
`google/gemma-4-*-it` (nested `text_config`, `model.language_model.*` tensors,
safetensors, a tokenizer). Its weights are random but scaled so that every
mechanism matters: norm gains away from 1, a `layer_scalar` away from 1, logits
large enough for the soft-cap to bite, a sliding window (6 tokens) shorter than
the sequences, and close margins between the top tokens. The same directory
feeds both sides.

| File | Role |
|---|---|
| `make_tiny_gemma4_bundle.py` | tiny bundle, variants `dense`, `shared`, `ple`, `moe`, `unified` (a `gemma-4-12B-it`-style bundle: `gemma4_unified` model type plus `model.vision_embedder.*` tensors the text decoder must ignore) |
| `make_tiny_gemma4_gguf.py` | turns such a bundle into a quantized `gemma4` GGUF (llama.cpp's metadata keys and tensor names): `Q4_0` projections, `Q8_0` embedding and per-layer tables. It also writes a reference bundle holding the *dequantized* weights, so the reference computes on exactly the numbers the file holds and the comparison isolates Magnetar's handling of the format, not the quantization error |
| `gemma4_reference_logits.py` | `transformers` reference, greedy, whole sequence recomputed at every step (no KV cache), written as a Magnetar `logit-reference/v1` JSON |
| `magnetar/gemma4_logits.rs` | Magnetar test: registers the Component, ingests the bundle (Hugging Face loader, or GGUF with `GEMMA_FORMAT=gguf`), generates greedily on the Reference CPU Provider, compares with Magnetar's own `compare_logit_references` |
| `magnetar/Cargo.toml` | the test crate (standalone, outside Magnetar's workspace) |
| `magnetar/magnetar-runtime-gemma4-local.patch` | the local Runtime changes that were needed on `84093cb`; obsolete since #197 |
| `run_logit_comparison.sh` | runs all of it; a variant named `gguf-<variant>` runs its quantized GGUF |

```bash
SKIP_RUNTIME_PATCH=1 MAGNETAR_DIR=/path/to/magnetar CARGO_FLAGS=--ignore-rust-version \
  tools/qualification/run_logit_comparison.sh dense shared ple moe unified gguf-dense gguf-shared gguf-ple gguf-moe
```

It needs a Magnetar checkout (`1f21bd7` or later, or `84093cb` with the patch) whose `loaders/huggingface` and
`formats/safetensors` submodules are populated, `wasm-tools`, and Python with
torch (CPU), `transformers` 5.x, `safetensors`, `numpy` and `gguf`.

## Results

Magnetar `main` at `dbe2f1c` (#200), **no local patch**, with `loaders/huggingface`
at `3c9a930`, `loaders/gguf` at `7bd7a23`, `formats/gguf` at `ef6378d` and
`formats/safetensors` at `f4b10dc`.

Prompt `[5, 17, 33, 8, 41, 2, 60, 12]`, 12 greedy tokens (one prefill, eleven
decode steps with a KV cache), Reference CPU, f32, tolerance 1e-3 on the top 8
logits and identical tokens required. Tokens are identical in every row.

Hugging Face bundles (f32 weights):

| Variant | What it exercises | Max abs logit difference |
|---|---|---|
| `dense` | sliding and full layers, 16- and 32-wide heads, K=V, "proportional" partial RoPE, q/k/v norms, sandwich norms, `layer_scalar`, GeGLU, soft-cap | 3.5e-5 |
| `shared` | the above, plus the last 3 layers reusing K/V and a twice-as-wide MLP | 5.5e-5 |
| `ple` | shared KV plus per-layer inputs (32-wide, tables sliced per layer) | 3.3e-5 |
| `moe` | dense MLP and 8 routed experts (top-2) side by side, router, `per_expert_scale` | 2.0e-5 |
| `unified` | `dense` as a `gemma4_unified` bundle (the 12B's model type), with a vision tensor to ignore | 3.5e-5 |

Quantized GGUF files (`Q4_0` projections, `Q8_0` embedding and per-layer tables;
the reference is computed on the dequantized values):

| Variant | What it exercises | Max abs logit difference |
|---|---|---|
| `gguf-dense` | layer layout derived from the GGUF metadata (`sliding_window_pattern`, per-layer `head_count_kv`, `rope_freqs` giving the rotated fraction), `Q4_0` projections, `Q8_0` tied embedding | 3.3e-5 |
| `gguf-shared` | plus shared KV and the wider MLP | 1.1e-4 |
| `gguf-ple` | plus `Q8_0` per-layer tables cut per layer on block boundaries | 2.7e-5 |
| `gguf-moe` | plus the fused `Q4_0` `ffn_gate_up_exps` restacked, the router fold and `ffn_down_exps.scale` | 3.0e-5 |

Logits are of order 10, so these are float32 rounding differences.

**The comparison can fail.** Changing one mechanism in the *reference's*
`config.json` only (weights untouched) makes it fail by a wide margin:

| Change | Max abs logit difference |
|---|---|
| sliding window 6 to 64 | 14.0 |
| RoPE base of full layers 1e6 to 1e4 | 10.9 |
| partial rotary factor 0.25 to 1.0 | 1.8 |
| soft-cap removed (tokens unchanged) | 1.1 |
| quantized GGUF compared with the *unquantized* reference | 11.5 |

## Not run

- The real checkpoints. Only the tiny random ones above were run; the 31B needs
  about 60 GB of weights and even the E2B does not fit this machine in float32.
  A comparison on real weights, and on Google's QAT GGUFs, is still to do.
- Sizes: the checks are at 6 layers, 64 hidden, 8 experts. The 128-expert
  router, 512-wide heads, `Q6_K` tables and the 256-wide per-layer tables were
  only checked for graph acceptance (`docs/engine-smoke-test`), not numerically.
- CUDA and the optimized CPU Provider (only the Reference CPU Provider was
  used), multi-device segment placement, and the decode-step replay path.

## The local Runtime patch (no longer needed)

The Runtime fixed these (merged as #197); run the script with
`SKIP_RUNTIME_PATCH=1`. Without the patch or that fix, Magnetar `84093cb` cannot run these models through its production
path at all (the bundle ingests, then loading fails). Three places still assume
Qwen's tensor inventory; the patch changes only them, and it is a diagnostic, not
a proposal for how Magnetar should fix them:

1. `validate_first_native_tensor_inventory` requires `self_attn.k_proj` and
   `self_attn.v_proj` in every layer. A layer with no value projection
   (`value_from_key`) or reusing another layer's K/V does not have them
   (*"qwen tensor inventory missing: layers.2.self_attn.v_proj"*).
2. `first_native_expected_tensor_shape` sizes every layer with the global head
   width and feed-forward width (*"expected shape [64, 64], got [128, 64]"* on a
   full layer).
3. `weight_shapes_for_architecture_config`, which tells the graph-builder which
   weights exist, has no `q_norm`, `k_norm`, `pre_ffn_norm`, `post_ffn_norm` or
   `layer_scalar` (*"no weight resource bound for logical name
   'layers.0.self_attn.q_norm'"*), although the weights themselves are
   materialized from the manifest.
