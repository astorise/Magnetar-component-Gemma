//! WIT adapter: `graph-builder` -> [`GraphSink`], `model-config` ->
//! [`Config`], and the Component exports.

use crate::plan::{self, Architecture, Config, LayerSpec, Moe, Options, Phase, Segment};
use magnetar_component_api::v1_8::{
    Guest, export,
    magnetar::model_component_graph::{graph_builder as gb, model_config},
};

struct Host;

// The graph-builder adapter is the SDK's, written once for every Component.
magnetar_component_sdk::wit_graph_sink!(Host, gb);

/// Reads the active Model Instance's configuration. Each call re-queries the
/// Capabilities: no dimension is cached or compiled in.
fn load_config() -> Config {
    match model_config::hidden_activation().as_deref() {
        Some("gelu_pytorch_tanh") => {}
        Some(other) => panic!("gemma: unsupported checkpoint activation {other:?}"),
        None => panic!("gemma: checkpoint activation is unknown"),
    }
    let a = model_config::model_config();
    // A Gemma graph needs the per-layer layout; a Runtime that has none for
    // the active Model Instance cannot be served by this Component.
    let layout = model_config::layer_layout()
        .expect("gemma: the Runtime supplied no per-layer layout for this Model Instance");
    let options = model_config::options();
    Config {
        arch: Architecture {
            hidden_size: a.hidden_size,
            intermediate_size: a.intermediate_size,
            num_hidden_layers: a.num_hidden_layers,
            num_attention_heads: a.num_attention_heads,
            num_key_value_heads: a.num_key_value_heads,
            head_dimension: a.head_dimension,
            vocabulary_size: a.vocabulary_size,
            rms_norm_eps: a.rms_norm_eps,
            tie_word_embeddings: a.tie_word_embeddings,
            attention_bias: a.attention_bias,
            has_rope_scaling: model_config::rope_scaling().is_some(),
            moe: model_config::mixture_of_experts().map(|m| Moe {
                experts: m.experts,
                experts_per_token: m.experts_per_token,
                expert_intermediate_size: m.expert_intermediate_size,
                normalize_top_k: m.normalize_top_k,
                has_shared_expert: m.shared_expert_intermediate_size.is_some(),
            }),
        },
        layers: layout
            .into_iter()
            .map(|l| LayerSpec {
                sliding_window: l.sliding_window,
                head_dimension: l.head_dimension,
                num_key_value_heads: l.num_key_value_heads,
                rope_theta: l.rope_theta,
                partial_rotary_factor: l.partial_rotary_factor,
                value_from_key: l.value_from_key,
                kv_source_layer: l.kv_source_layer,
                intermediate_size: l.intermediate_size,
            })
            .collect(),
        options: Options {
            final_logit_softcapping: options.final_logit_softcapping,
            per_layer_input_size: options.per_layer_input_size,
        },
    }
}

fn build(segment: impl FnOnce(u32) -> Segment) -> String {
    let cfg = load_config();
    let segment = segment(cfg.arch.num_hidden_layers);
    plan::build_graph_segment(&mut Host, &cfg, segment)
}

struct Component;

impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        Self::build_prefill_graph_segment_with(sequence_length, |n| (0, n))
    }

    fn build_decode_graph(cached_token_count: u64) -> String {
        Self::build_decode_graph_segment_with(cached_token_count, |n| (0, n))
    }

    fn build_prefill_graph_segment(
        sequence_length: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        Self::build_prefill_graph_segment_with(sequence_length, |_| (start_layer, end_layer))
    }

    fn build_decode_graph_segment(
        cached_token_count: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        Self::build_decode_graph_segment_with(cached_token_count, |_| (start_layer, end_layer))
    }
}

impl Component {
    fn build_prefill_graph_segment_with(
        sequence_length: u64,
        range: impl FnOnce(u32) -> (u32, u32),
    ) -> String {
        build(|layers| {
            let (start_layer, end_layer) = range(layers);
            Segment {
                phase: Phase::Prefill,
                sequence_length,
                position_offset: 0,
                kv_append: false,
                start_layer,
                end_layer,
            }
        })
    }

    fn build_decode_graph_segment_with(
        cached_token_count: u64,
        range: impl FnOnce(u32) -> (u32, u32),
    ) -> String {
        build(|layers| {
            let (start_layer, end_layer) = range(layers);
            Segment {
                phase: Phase::Decode,
                sequence_length: 1,
                position_offset: cached_token_count,
                kv_append: true,
                start_layer,
                end_layer,
            }
        })
    }
}

export!(Component);
