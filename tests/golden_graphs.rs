//! Golden graphs: every call the builder makes, recorded as text and compared
//! with the files in `tests/golden/`.
//!
//! These files were recorded from the builder before it moved onto
//! `magnetar-component-sdk` (the SDK's graph IR replaced the copy this crate
//! kept in `plan.rs`) and `magnetar-component-api` (the contract replaced the
//! WIT copy and the bindings generator this crate carried). They pin the graph
//! a Model Instance receives: a change that alters one node, attribute, input
//! order, shape or KV involvement fails here, whoever made it. Regenerate them
//! deliberately with `UPDATE_GOLDEN=1 cargo test --test golden_graphs`.
//!
//! The test uses only the crate's public `plan` API, so it holds across the
//! change of the types behind it. Its configurations are the ones the unit
//! tests in `src/plan/tests.rs` model: a miniature of each variant, and the
//! real geometry of the checkpoints they name.

use magnetar_component_gemma::plan::{
    Architecture, Config, GraphSink, KvRole, LayerSpec, Moe, Node, Options, Phase, Segment, Value,
    build_graph_segment,
};
use std::{fmt::Write as _, path::PathBuf};

/// Records every call, one line each, in call order.
#[derive(Default)]
struct TextSink {
    text: String,
}

impl TextSink {
    fn line(&mut self, line: std::fmt::Arguments<'_>) {
        writeln!(self.text, "{line}").unwrap();
    }
}

impl GraphSink for TextSink {
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64) {
        self.line(format_args!(
            "begin {phase:?} sequence={sequence_length} offset={position_offset}"
        ));
    }
    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String {
        self.line(format_args!("input {name} {shape:?}"));
        format!("input:{name}")
    }
    fn weight_edge(&mut self, name: &str) -> String {
        self.line(format_args!("weight {name}"));
        format!("weight:{name}")
    }
    fn alias_weight_edge(&mut self, edge: &str, target: &str) {
        self.line(format_args!("alias {edge} -> {target}"));
    }
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        self.line(format_args!("kv {layer} {role:?}"));
        format!("kv:{layer}:{role:?}")
    }
    fn add_node(&mut self, node: Node) -> String {
        let mut line = format!(
            "node {} op={} family={} inputs={:?} shape={:?}",
            node.id, node.operator, node.family, node.inputs, node.shape
        );
        for attr in &node.attrs {
            let value = match &attr.value {
                Value::Bool(v) => format!("bool:{v}"),
                Value::Int(v) => format!("int:{v}"),
                Value::Float(v) => format!("float:{v:?}"),
                Value::Text(v) => format!("text:{v}"),
            };
            write!(line, " {}={value}", attr.name).unwrap();
        }
        if let Some(kv) = &node.kv {
            write!(
                line,
                " kv=({}, {:?}, append={})",
                kv.resource, kv.role, kv.append
            )
            .unwrap();
        }
        self.line(format_args!("{line}"));
        format!("out:{}", node.id)
    }
    fn finish_graph(&mut self, output: &str) -> String {
        self.line(format_args!("finish {output}"));
        output.into()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Attention {
    Sliding,
    Full,
}
use Attention::{Full, Sliding};

/// A checkpoint's `config.json` values, as the Runtime derives `model-config`'s
/// layer layout from them.
#[derive(Clone)]
struct Checkpoint {
    arch: Architecture,
    layers: Vec<Attention>,
    sliding_window: u32,
    global_head_dimension: u64,
    num_global_key_value_heads: Option<u32>,
    sliding_rope_theta: f64,
    full_rope_theta: f64,
    full_partial_rotary_factor: f64,
    attention_k_eq_v: bool,
    final_logit_softcapping: Option<f64>,
    per_layer_input_size: u32,
    kv_shared_layers: u32,
    double_wide_mlp_on_shared_layers: bool,
}

impl Checkpoint {
    fn config(&self) -> Config {
        let a = &self.arch;
        let n = self.layers.len();
        let first_shared = n.saturating_sub(self.kv_shared_layers as usize);
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                let consumer = self.kv_shared_layers != 0 && i >= first_shared;
                let kv_source_layer = consumer
                    .then(|| (0..first_shared).rev().find(|j| self.layers[*j] == *kind))
                    .flatten()
                    .map(|j| j as u32);
                let intermediate_size = a.intermediate_size
                    * if consumer && self.double_wide_mlp_on_shared_layers {
                        2
                    } else {
                        1
                    };
                match kind {
                    Sliding => LayerSpec {
                        sliding_window: Some(self.sliding_window),
                        head_dimension: a.head_dimension,
                        num_key_value_heads: a.num_key_value_heads,
                        rope_theta: self.sliding_rope_theta,
                        partial_rotary_factor: None,
                        value_from_key: false,
                        kv_source_layer,
                        intermediate_size,
                    },
                    Full => LayerSpec {
                        sliding_window: None,
                        head_dimension: self.global_head_dimension,
                        num_key_value_heads: match (
                            self.attention_k_eq_v,
                            self.num_global_key_value_heads,
                        ) {
                            (true, Some(global)) => global,
                            _ => a.num_key_value_heads,
                        },
                        rope_theta: self.full_rope_theta,
                        partial_rotary_factor: Some(self.full_partial_rotary_factor),
                        value_from_key: self.attention_k_eq_v,
                        kv_source_layer,
                        intermediate_size,
                    },
                }
            })
            .collect();
        Config {
            arch: a.clone(),
            layers,
            options: Options {
                final_logit_softcapping: self.final_logit_softcapping,
                per_layer_input_size: (self.per_layer_input_size != 0)
                    .then_some(self.per_layer_input_size),
            },
        }
    }
}

/// A miniature Gemma 4 dense shape: six layers, `s s f s s f`.
fn tiny() -> Checkpoint {
    Checkpoint {
        arch: Architecture {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dimension: 8,
            vocabulary_size: 100,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_bias: false,
            has_rope_scaling: false,
            moe: None,
        },
        layers: vec![Sliding, Sliding, Full, Sliding, Sliding, Full],
        sliding_window: 16,
        global_head_dimension: 16,
        num_global_key_value_heads: Some(1),
        sliding_rope_theta: 10_000.0,
        full_rope_theta: 1_000_000.0,
        full_partial_rotary_factor: 0.25,
        attention_k_eq_v: true,
        final_logit_softcapping: Some(30.0),
        per_layer_input_size: 0,
        kv_shared_layers: 0,
        double_wide_mlp_on_shared_layers: false,
    }
}

/// `google/gemma-4-31B-it`'s text config.
fn gemma4_31b() -> Checkpoint {
    Checkpoint {
        arch: Architecture {
            hidden_size: 5376,
            intermediate_size: 21504,
            num_hidden_layers: 60,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            head_dimension: 256,
            vocabulary_size: 262_144,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_bias: false,
            has_rope_scaling: false,
            moe: None,
        },
        layers: (0..60)
            .map(|i| if i % 6 == 5 { Full } else { Sliding })
            .collect(),
        sliding_window: 1024,
        global_head_dimension: 512,
        num_global_key_value_heads: Some(4),
        ..tiny()
    }
}

/// The E2B shape: 35 layers, 20 of them sharing K/V, a double-wide MLP on
/// those, per-layer inputs, K and V projected separately.
fn gemma4_e2b() -> Checkpoint {
    Checkpoint {
        arch: Architecture {
            hidden_size: 1536,
            intermediate_size: 6144,
            num_hidden_layers: 35,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dimension: 256,
            vocabulary_size: 262_144,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_bias: false,
            has_rope_scaling: false,
            moe: None,
        },
        layers: (0..35)
            .map(|i| if i % 5 == 4 { Full } else { Sliding })
            .collect(),
        sliding_window: 512,
        global_head_dimension: 512,
        num_global_key_value_heads: None,
        attention_k_eq_v: false,
        per_layer_input_size: 256,
        kv_shared_layers: 20,
        double_wide_mlp_on_shared_layers: true,
        ..tiny()
    }
}

/// The 26B-A4B shape: a dense MLP beside 128 routed experts on every layer.
fn gemma4_26b_a4b() -> Checkpoint {
    Checkpoint {
        arch: Architecture {
            hidden_size: 2816,
            intermediate_size: 2112,
            num_hidden_layers: 30,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dimension: 256,
            vocabulary_size: 262_144,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_bias: false,
            has_rope_scaling: false,
            moe: Some(Moe {
                experts: 128,
                experts_per_token: 8,
                expert_intermediate_size: 704,
                normalize_top_k: true,
                has_shared_expert: false,
            }),
        },
        layers: (0..30)
            .map(|i| if i % 6 == 5 { Full } else { Sliding })
            .collect(),
        sliding_window: 1024,
        global_head_dimension: 512,
        num_global_key_value_heads: Some(2),
        ..tiny()
    }
}

fn segment(phase: Phase, sequence_length: u64, offset: u64, start: u32, end: u32) -> Segment {
    Segment {
        phase,
        sequence_length,
        position_offset: offset,
        kv_append: phase == Phase::Decode,
        start_layer: start,
        end_layer: end,
    }
}

fn check(name: &str, checkpoint: &Checkpoint, segment: Segment) {
    let mut sink = TextSink::default();
    build_graph_segment(&mut sink, &checkpoint.config(), segment);
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::write(&path, &sink.text).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("golden graph {}: {error}", path.display()));
    assert_eq!(
        sink.text, expected,
        "graph '{name}' differs from tests/golden/{name}.txt"
    );
}

#[test]
fn dense_prefill_whole_stack() {
    check(
        "dense_tiny_prefill",
        &tiny(),
        segment(Phase::Prefill, 5, 0, 0, 6),
    );
}

#[test]
fn dense_decode_appends_kv() {
    check(
        "dense_tiny_decode",
        &tiny(),
        segment(Phase::Decode, 1, 9, 0, 6),
    );
}

#[test]
fn middle_segment_takes_hidden_states() {
    check(
        "dense_tiny_middle_segment",
        &tiny(),
        segment(Phase::Prefill, 5, 0, 2, 4),
    );
}

#[test]
fn last_segment_ends_with_softcap_and_alias() {
    check(
        "dense_tiny_last_segment",
        &tiny(),
        segment(Phase::Decode, 1, 9, 4, 6),
    );
}

#[test]
fn untied_embeddings_without_softcap() {
    let mut checkpoint = tiny();
    checkpoint.arch.tie_word_embeddings = false;
    checkpoint.final_logit_softcapping = None;
    check(
        "dense_tiny_untied_no_softcap",
        &checkpoint,
        segment(Phase::Prefill, 3, 0, 4, 6),
    );
}

#[test]
fn shared_kv_layers_reuse_their_producers() {
    let mut checkpoint = tiny();
    checkpoint.kv_shared_layers = 3;
    check(
        "shared_kv_tiny_prefill",
        &checkpoint,
        segment(Phase::Prefill, 5, 0, 0, 6),
    );
}

#[test]
fn per_layer_inputs_in_the_whole_stack_and_a_later_segment() {
    let mut checkpoint = tiny();
    checkpoint.per_layer_input_size = 8;
    check(
        "per_layer_input_tiny_prefill",
        &checkpoint,
        segment(Phase::Prefill, 4, 0, 0, 6),
    );
    check(
        "per_layer_input_tiny_later_segment",
        &checkpoint,
        segment(Phase::Prefill, 4, 0, 3, 6),
    );
}

#[test]
fn routed_experts_beside_the_dense_mlp() {
    let mut checkpoint = tiny();
    checkpoint.arch.moe = Some(Moe {
        experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 24,
        normalize_top_k: true,
        has_shared_expert: false,
    });
    check(
        "moe_tiny_prefill",
        &checkpoint,
        segment(Phase::Prefill, 4, 0, 0, 6),
    );
}

#[test]
fn gemma4_31b_first_full_attention_period() {
    check(
        "gemma4_31b_prefill_layers_0_6",
        &gemma4_31b(),
        segment(Phase::Prefill, 8, 0, 0, 6),
    );
}

#[test]
fn gemma4_31b_last_layers_end_the_graph() {
    check(
        "gemma4_31b_decode_layers_58_60",
        &gemma4_31b(),
        segment(Phase::Decode, 1, 1024, 58, 60),
    );
}

#[test]
fn gemma4_e2b_across_the_producer_consumer_boundary() {
    // Layers 13 and 14 produce the K/V that layers 15 and 16 reuse.
    check(
        "gemma4_e2b_prefill_layers_13_17",
        &gemma4_e2b(),
        segment(Phase::Prefill, 8, 0, 13, 17),
    );
}

#[test]
fn gemma4_26b_a4b_first_two_layers() {
    check(
        "gemma4_26b_a4b_prefill_layers_0_2",
        &gemma4_26b_a4b(),
        segment(Phase::Prefill, 4, 0, 0, 2),
    );
}
