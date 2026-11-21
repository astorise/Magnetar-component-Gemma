//! Google Gemma 4 (text decoder) Model Component for the Magnetar local AI
//! Runtime (`magnetar:model-component-graph@1.8.0`).
//!
//! Builds prefill and decode Execution Graphs through the Runtime-owned
//! `graph-builder` and `model-config` Capabilities; the additive per-layer
//! layout `model-config` gained in 1.6.0 is family-neutral. It never selects a
//! Provider or Device and never sees a native handle.
//!
//! The graph logic lives in [`plan`], which is host-independent and unit
//! tested; `component` (built for `wasm32` or with the `bindings` feature)
//! only adapts the generated WIT bindings to it.
//!
//! Supported: every Gemma 4 text checkpoint family: dense (31B),
//! mixture-of-experts (26B-A4B) and the per-layer-input / shared-KV E2B and
//! E4B. Refused with a trap rather than built incorrectly: RoPE frequency
//! scaling, attention bias, a shared expert, a non-renormalizing router and
//! any inconsistent per-layer layout. Vision and audio are out of scope.

pub mod plan;

/// The architecture family this Component serves.
pub const ARCHITECTURE_FAMILY: &str = "gemma";

#[cfg(any(target_arch = "wasm32", feature = "bindings"))]
mod component;
