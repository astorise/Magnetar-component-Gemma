//! Google Gemma 4 (dense text) Model Component for the Magnetar local AI
//! Runtime (`magnetar:model-component-graph@1.6.0`).
//!
//! Builds prefill and decode Execution Graphs through the Runtime-owned
//! `graph-builder`, `model-config` and `gemma-config` Capabilities. It never
//! selects a Provider or Device and never sees a native handle.
//!
//! The graph logic lives in [`plan`], which is host-independent and unit
//! tested; this file only adapts the generated WIT bindings to it.
//!
//! Supported: dense Gemma 4 text checkpoints (31B, 12B). Refused with a trap
//! rather than built incorrectly: mixture-of-experts (26B-A4B) and the
//! per-layer-input / shared-KV E2B and E4B variants.

pub mod plan;

/// The architecture family this Component serves.
pub const ARCHITECTURE_FAMILY: &str = "gemma";

#[cfg(any(target_arch = "wasm32", feature = "bindings"))]
mod component;
