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
| `make_tiny_gemma4_bundle.py` | tiny bundle, variants `dense`, `shared`, `ple`, `moe` |
| `gemma4_reference_logits.py` | `transformers` reference, greedy, whole sequence recomputed at every step (no KV cache), written as a Magnetar `logit-reference/v1` JSON |
| `magnetar/gemma4_logits.rs` | Magnetar test: registers the Component, ingests the bundle with the real Hugging Face loader, generates greedily on the Reference CPU Provider, compares with Magnetar's own `compare_logit_references` |
| `magnetar/Cargo.toml` | the test crate (standalone, outside Magnetar's workspace) |
| `magnetar/magnetar-runtime-gemma4-local.patch` | the **local** Runtime changes that were needed, see below |
| `run_logit_comparison.sh` | runs all of it |

```bash
MAGNETAR_DIR=/path/to/magnetar CARGO_FLAGS=--ignore-rust-version \
  tools/qualification/run_logit_comparison.sh dense shared
```

It needs a Magnetar checkout at `84093cb` whose `loaders/huggingface` and
`formats/safetensors` submodules are populated, `wasm-tools`, and Python with
torch (CPU), `transformers` 5.x, `safetensors` and `numpy`.

## Results (Magnetar `84093cb` plus the local patch)

Prompt `[5, 17, 33, 8, 41, 2, 60, 12]`, 12 greedy tokens (one prefill, eleven
decode steps with a KV cache), Reference CPU, f32, tolerance 1e-3 on the top 8
logits and identical tokens required:

| Variant | What it exercises | Tokens | Max abs logit difference |
|---|---|---|---|
| `dense` | sliding and full layers, 16- and 32-wide heads, K=V, "proportional" partial RoPE, q/k/v norms, sandwich norms, `layer_scalar`, GeGLU, soft-cap | identical | 3.5e-5 |
| `shared` | the above, plus the last 3 layers reusing K/V and a twice-as-wide MLP | identical | 5.5e-5 |

Logits are of order 10, so these are float32 rounding differences.

**The comparison can fail.** Changing one mechanism in the *reference's*
`config.json` only (weights untouched) makes it fail by a wide margin:

| Change | Max abs logit difference |
|---|---|
| sliding window 6 to 64 | 14.0 |
| RoPE base of full layers 1e6 to 1e4 | 10.9 |
| partial rotary factor 0.25 to 1.0 | 1.8 |
| soft-cap removed (tokens unchanged) | 1.1 |

## Not run

- `ple` (per-layer inputs) and `moe` (routed experts): the references are
  produced, but Magnetar's Hugging Face loader refuses these bundles for now
  (*"packs every layer's per-layer input weights into one table ... does not
  slice per layer yet"*, *"routed experts ... does not restack yet"*). The same
  commands run them once the loader transforms exist.
- The real checkpoints. Only the tiny random ones above were run; 31B needs
  about 60 GB of weights. The comparison on real weights (and on a quantized
  GGUF) is still to do.
- CUDA, other Providers, and the decode-step replay path.

## The local Runtime patch

Without it, Magnetar `84093cb` cannot run these models through its production
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
