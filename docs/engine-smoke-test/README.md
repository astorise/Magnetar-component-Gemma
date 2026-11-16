# Smoke test: the Component inside Magnetar's Wasmtime engine

`magnetar-engine-gemma-smoke.patch` adds scratch tests to
`magnetar-engine/src/component_wasmtime/tests.rs` of Magnetar (applied to
`84093cb`, "Gemma 4 runtime steps ... #196"). They load this Component into
the real `WasmtimeComponentEngine`, link the real `GraphBuilderCapability` and
`ModelConfigCapability`, bind a layered `ModelArchitectureConfig` and invoke
the exports. Only **shapes** are supplied (no weight data), so they check that
the graph the Component builds is accepted by the Runtime's graph-builder
validation, not that its numbers are right.

Reproduce:

```bash
# 1. Build the Component as the Runtime's own fixtures are built.
cargo build --release --target wasm32-unknown-unknown
wasm-tools component new target/wasm32-unknown-unknown/release/magnetar_component_gemma.wasm \
  -o /tmp/claude-0/e2e/gemma.component.wasm
# 2. In a Magnetar checkout at 84093cb:
git apply docs/engine-smoke-test/magnetar-engine-gemma-smoke.patch
cargo test -p magnetar-engine --lib gemma_e2e -- --nocapture --test-threads=1
```

The patch reads the component from `/tmp/claude-0/e2e/gemma.component.wasm`; edit that path in
`gemma_graph` to use another location.

Results at `84093cb`: all 7 tests pass.

| Test | Result |
|---|---|
| Dense, sliding + full layers, K=V, partial RoPE, shared KV, softcap | 81 nodes |
| MoE (2 layers) | 61 nodes, 2 `moe-experts` |
| Per-layer inputs + shared KV with a wider MLP | 101 nodes |
| Decode graph; segment `[0, 2)` (44 nodes) | built |
| Segment `[2, 4)` that splits shared-KV layers from their producers | refused (trap) |
| Real 31B shapes (60 layers) | 1314 nodes |
| Real 26B-A4B shapes (30 layers, 128 experts) | 869 nodes |
| Real E2B shapes (35 layers, 20 shared, per-layer inputs) | 1094 nodes |
