//! WIT adapter: `graph-builder` -> [`GraphSink`], `model-config` and
//! `gemma-config` -> [`Config`], and the Component exports.

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "gemma-model-component-producer",
});

use crate::plan::{
    self, Architecture, Attr, Config, GemmaArchitecture, GraphSink, KvRole, LayerAttention, Moe,
    Node, Phase, Segment, Value,
};
use magnetar::model_component_graph::gemma_config::{self, LayerAttention as WitLayer};
use magnetar::model_component_graph::graph_builder as gb;
use magnetar::model_component_graph::model_config;

struct Host;

impl GraphSink for Host {
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64) {
        let phase = match phase {
            Phase::Prefill => gb::GraphPhase::Prefill,
            Phase::Decode => gb::GraphPhase::Decode,
        };
        gb::begin_graph(phase, sequence_length, position_offset);
    }

    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String {
        gb::declare_input(
            name,
            &gb::TensorShape {
                dimensions: shape.to_vec(),
            },
        )
    }

    fn weight_edge(&mut self, logical_name: &str) -> String {
        gb::weight_edge(logical_name)
    }

    fn alias_weight_edge(&mut self, edge: &str, target_logical_name: &str) {
        gb::alias_weight_edge(edge, target_logical_name);
    }

    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        gb::kv_resource(layer, wit_role(role))
    }

    fn add_node(&mut self, node: Node) -> String {
        let attrs: Vec<gb::Attribute> = node.attrs.iter().map(wit_attr).collect();
        let kv = node.kv.as_ref().map(|kv| gb::KvInvolvement {
            resource_id: kv.resource.clone(),
            role: wit_role(kv.role),
            append: kv.append,
        });
        gb::add_node(
            &node.id,
            node.operator,
            node.family,
            &attrs,
            &node.inputs,
            &gb::TensorShape {
                dimensions: node.shape,
            },
            kv.as_ref(),
        )
    }

    fn finish_graph(&mut self, output_edge: &str) -> String {
        gb::finish_graph(output_edge)
    }
}

fn wit_role(role: KvRole) -> gb::KvCacheRole {
    match role {
        KvRole::K => gb::KvCacheRole::K,
        KvRole::V => gb::KvCacheRole::V,
    }
}

fn wit_attr(attr: &Attr) -> gb::Attribute {
    gb::Attribute {
        name: attr.name.to_string(),
        value: match &attr.value {
            Value::Bool(b) => gb::AttributeValue::Boolean(*b),
            Value::Int(i) => gb::AttributeValue::Integer(*i),
            Value::Float(f) => gb::AttributeValue::FloatValue(*f),
            Value::Text(t) => gb::AttributeValue::Text(t.clone()),
        },
    }
}

/// Reads the active Model Instance's configuration. Each call re-queries the
/// Capabilities: no dimension is cached or compiled in.
fn load_config() -> Config {
    let a = model_config::model_config();
    let g = gemma_config::architecture();
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
        gemma: GemmaArchitecture {
            layer_attention: g
                .layer_attention
                .iter()
                .map(|l| match l {
                    WitLayer::Sliding => LayerAttention::Sliding,
                    WitLayer::Full => LayerAttention::Full,
                })
                .collect(),
            sliding_window: g.sliding_window,
            global_head_dimension: g.global_head_dimension,
            num_global_key_value_heads: g.num_global_key_value_heads,
            sliding_rope_theta: g.sliding_rope_theta,
            full_rope_theta: g.full_rope_theta,
            full_partial_rotary_factor: g.full_partial_rotary_factor,
            attention_k_eq_v: g.attention_k_eq_v,
            final_logit_softcapping: g.final_logit_softcapping,
            per_layer_input_size: g.per_layer_input_size,
            kv_shared_layers: g.kv_shared_layers,
            double_wide_mlp_on_shared_layers: g.double_wide_mlp_on_shared_layers,
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
