# magnetar-component-gemma

## Purpose

A WebAssembly Component implementing the Google Gemma model families for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It will
supply prefill/decode graph semantics for Gemma-compatible Model Artifacts
through the Runtime's Model Component contract, without ever selecting a
Provider, Device, or receiving native Runtime handles -- per Magnetar's
architectural invariants (Components request Capabilities, not Providers
or Devices).

## Status

**Empty template.** This crate is a scaffold: no Gemma graph logic, no WIT
bindings and no Component Model exports exist here yet, and the Runtime has
no Gemma implementation of its own either. It is pinned into Magnetar as a
placeholder so the work has a home; it is not yet scheduled against an
OpenSpec change of its own.

## What this family needs

What sets this family apart from the generic decoder block
`components/qwen` and `components/llama` already implement, each to be
checked against the exact pinned checkpoint's `config.json` before it is
implemented:

- A GeGLU MLP (tanh-approximated GELU gate) instead of SwiGLU, an RMSNorm whose scale is `1 + weight`, and token embeddings scaled by `sqrt(hidden_size)`, with the output projection tied to the embedding table.
- Gemma 2: logit and attention soft-capping, sandwich normalization (a norm both before and after each attention and MLP block), and attention layers alternating between a sliding local window and global attention.
- Gemma 3: query/key normalization, mostly local sliding-window layers with periodic global ones, and a different RoPE base for local and global layers; its larger checkpoints are multimodal (vision input), which `explore-multimodal-and-embedding-components` covers.
- A 256k-token vocabulary, so the LM head is the largest single matmul of a decode step.

## Governing contract

None dedicated yet. A real implementation will target the same
`model-component-graph-contract` WIT interface (`magnetar:model-component-graph`)
`components/qwen` and `components/llama` implement, extended additively
where this family needs Operators or state the contract does not model yet.
Family-specific requirements will be scoped in the main repository's
`openspec/` (the `qualify-additional-model-and-state-families` change tracks
qualifying new families).

## Relationship to magnetar-runtime

This Component will be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/gemma`.
