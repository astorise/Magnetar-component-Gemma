use super::*;

/// Records every `graph-builder` call.
#[derive(Default)]
struct Recorder {
    began: Vec<(Phase, u64, u64)>,
    inputs: Vec<(String, Vec<u64>)>,
    weights: Vec<String>,
    aliases: Vec<(String, String)>,
    nodes: Vec<Node>,
    edges: u32,
}

impl Recorder {
    fn edge(&mut self, kind: &str) -> String {
        self.edges += 1;
        format!("{kind}{}", self.edges)
    }
    fn node(&self, id: &str) -> &Node {
        self.nodes
            .iter()
            .find(|n| n.id == id)
            .unwrap_or_else(|| panic!("no node {id}"))
    }
    fn has(&self, id: &str) -> bool {
        self.nodes.iter().any(|n| n.id == id)
    }
}

impl GraphSink for Recorder {
    fn begin_graph(&mut self, phase: Phase, s: u64, o: u64) {
        self.began.push((phase, s, o));
    }
    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String {
        self.inputs.push((name.into(), shape.to_vec()));
        self.edge("in")
    }
    fn weight_edge(&mut self, name: &str) -> String {
        self.weights.push(name.into());
        // Stable per name, like the Runtime: `w:<name>`.
        format!("w:{name}")
    }
    fn alias_weight_edge(&mut self, edge: &str, target: &str) {
        self.aliases.push((edge.into(), target.into()));
    }
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        format!("kv:{layer}:{role:?}")
    }
    fn add_node(&mut self, node: Node) -> String {
        let out = format!("out:{}", node.id);
        self.nodes.push(node);
        out
    }
    fn finish_graph(&mut self, output: &str) -> String {
        format!("graph({output})")
    }
}

/// A miniature Gemma 4 dense shape: 6 layers, `sssssf` pattern is shortened
/// to `s s f s s f` so both flavours repeat.
fn tiny() -> Config {
    use LayerAttention::*;
    Config {
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
            has_mixture_of_experts: false,
        },
        gemma: GemmaArchitecture {
            layer_attention: vec![Sliding, Sliding, Full, Sliding, Sliding, Full],
            sliding_window: 16,
            global_head_dimension: 16,
            num_global_key_value_heads: 1,
            sliding_rope_theta: 10_000.0,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            final_logit_softcapping: Some(30.0),
            per_layer_input_size: 0,
            kv_shared_layers: 0,
        },
    }
}

fn prefill(cfg: &Config, seq: u64, start: u32, end: u32) -> (Recorder, String) {
    let mut r = Recorder::default();
    let g = build_graph_segment(
        &mut r,
        cfg,
        Segment {
            phase: Phase::Prefill,
            sequence_length: seq,
            position_offset: 0,
            kv_append: false,
            start_layer: start,
            end_layer: end,
        },
    );
    (r, g)
}

fn attr_of<'a>(n: &'a Node, name: &str) -> Option<&'a Value> {
    n.attrs.iter().find(|a| a.name == name).map(|a| &a.value)
}

#[test]
fn full_graph_shape_and_output() {
    let (r, g) = prefill(&tiny(), 5, 0, 6);
    assert_eq!(g, "graph(out:logit_softcap)");
    assert_eq!(r.began, vec![(Phase::Prefill, 5, 0)]);
    assert_eq!(r.inputs, vec![("token_ids".to_string(), vec![5])]);
    // embedding + 6 layers + final_norm + lm_head + softcap.
    let per_layer_sliding = 22; // see layer_node_counts
    let per_layer_full = 21; // no v_proj
    assert_eq!(
        r.nodes.len(),
        1 + 4 * per_layer_sliding + 2 * per_layer_full + 3
    );
    assert_eq!(r.node("lm_head").shape, vec![5, 100]);
    assert_eq!(
        r.aliases,
        vec![("w:lm_head".into(), "token_embedding".into())]
    );
}

#[test]
fn layer_node_counts() {
    let (r, _) = prefill(&tiny(), 3, 0, 1);
    let count = |p: &str| r.nodes.iter().filter(|n| n.id.starts_with(p)).count();
    assert_eq!(count("layer0."), 22);
    let (r, _) = prefill(&tiny(), 3, 2, 3);
    assert_eq!(
        r.nodes
            .iter()
            .filter(|n| n.id.starts_with("layer2."))
            .count(),
        21
    );
}

#[test]
fn embedding_is_scaled_by_sqrt_hidden() {
    let (r, _) = prefill(&tiny(), 2, 0, 1);
    let e = r.node("embedding");
    assert_eq!(attr_of(e, "scale"), Some(&Value::Float(32f64.sqrt())));
}

#[test]
fn rmsnorm_is_plain_weight_not_one_plus_weight() {
    // Gemma 4 dropped the Gemma 1-3 `1 + weight`: a norm node carries no
    // offset attribute at all.
    let (r, _) = prefill(&tiny(), 2, 0, 1);
    for n in r.nodes.iter().filter(|n| n.operator == "rmsnorm") {
        assert!(attr_of(n, "epsilon").is_some(), "{}", n.id);
        assert!(attr_of(n, "weight_offset").is_none(), "{}", n.id);
    }
}

#[test]
fn sliding_layer_geometry() {
    let (r, _) = prefill(&tiny(), 4, 0, 1);
    assert_eq!(r.node("layer0.q_proj").shape, vec![4, 4 * 8]);
    assert_eq!(r.node("layer0.k_proj").shape, vec![4, 2 * 8]);
    assert!(r.has("layer0.v_proj"));
    let att = r.node("layer0.attention");
    assert_eq!(attr_of(att, "window_size"), Some(&Value::Int(16)));
    assert_eq!(attr_of(att, "scale"), Some(&Value::Float(1.0)));
    assert_eq!(attr_of(att, "head_dimension"), Some(&Value::Int(8)));
    let rope = r.node("layer0.rope_q");
    assert_eq!(attr_of(rope, "base"), Some(&Value::Float(10_000.0)));
    assert!(attr_of(rope, "partial_rotary_factor").is_none());
}

#[test]
fn full_layer_geometry_and_k_eq_v() {
    let (r, _) = prefill(&tiny(), 4, 2, 3);
    // 16-wide global heads, 1 KV head.
    assert_eq!(r.node("layer2.q_proj").shape, vec![4, 4 * 16]);
    assert_eq!(r.node("layer2.k_proj").shape, vec![4, 16]);
    assert!(
        !r.has("layer2.v_proj"),
        "k_eq_v layers have no V projection"
    );
    // V is K's raw projection (pre-norm, pre-RoPE).
    let v = r.node("layer2.v_norm");
    assert_eq!(v.inputs[0], "out:layer2.k_proj");
    assert_eq!(v.kv.as_ref().unwrap().role, KvRole::V);
    let k = r.node("layer2.rope_k");
    assert_eq!(k.inputs[0], "out:layer2.k_norm");
    let att = r.node("layer2.attention");
    assert!(attr_of(att, "window_size").is_none());
    assert_eq!(attr_of(att, "kv_head_count"), Some(&Value::Int(1)));
    let rope = r.node("layer2.rope_q");
    assert_eq!(attr_of(rope, "base"), Some(&Value::Float(1_000_000.0)));
    assert_eq!(
        attr_of(rope, "partial_rotary_factor"),
        Some(&Value::Float(0.25))
    );
    assert_eq!(attr_of(rope, "dimension"), Some(&Value::Int(16)));
}

#[test]
fn full_layer_keeps_v_projection_when_not_k_eq_v() {
    let mut cfg = tiny();
    cfg.gemma.attention_k_eq_v = false;
    let (r, _) = prefill(&cfg, 2, 2, 3);
    assert!(r.has("layer2.v_proj"));
    assert_eq!(r.node("layer2.v_norm").inputs[0], "out:layer2.v_proj");
}

#[test]
fn sandwich_norms_and_layer_scalar_order() {
    let (r, _) = prefill(&tiny(), 2, 0, 1);
    let ids: Vec<&str> = r.nodes.iter().map(|n| n.id.as_str()).collect();
    let pos = |id: &str| ids.iter().position(|i| *i == id).unwrap();
    for pair in [
        ["layer0.input_norm", "layer0.q_proj"],
        ["layer0.o_proj", "layer0.post_attn_norm"],
        ["layer0.post_attn_norm", "layer0.residual1"],
        ["layer0.residual1", "layer0.pre_ffn_norm"],
        ["layer0.down_proj", "layer0.post_ffn_norm"],
        ["layer0.post_ffn_norm", "layer0.residual2"],
        ["layer0.residual2", "layer0.layer_scalar"],
    ] {
        assert!(pos(pair[0]) < pos(pair[1]), "{pair:?}");
    }
    // The residual stream is the *un-normalized* input.
    assert_eq!(r.node("layer0.residual1").inputs[0], "out:embedding");
    assert_eq!(r.node("layer0.residual2").inputs[0], "out:layer0.residual1");
    assert_eq!(
        r.node("layer0.layer_scalar").inputs[0],
        "out:layer0.residual2"
    );
}

#[test]
fn mlp_is_geglu_with_tanh_gelu() {
    let (r, _) = prefill(&tiny(), 2, 0, 1);
    let g = r.node("layer0.gelu");
    assert_eq!(g.operator, "gelu");
    assert_eq!(attr_of(g, "approximate"), Some(&Value::Text("tanh".into())));
    assert_eq!(g.inputs, vec!["out:layer0.gate_proj".to_string()]);
    assert_eq!(
        r.node("layer0.mul").inputs,
        vec![
            "out:layer0.gelu".to_string(),
            "out:layer0.up_proj".to_string()
        ]
    );
}

#[test]
fn qk_norms_are_per_head() {
    let (r, _) = prefill(&tiny(), 2, 2, 3);
    let q = r.node("layer2.q_norm");
    assert_eq!(attr_of(q, "head_count"), Some(&Value::Int(4)));
    assert_eq!(attr_of(q, "head_dimension"), Some(&Value::Int(16)));
    let k = r.node("layer2.k_norm");
    assert_eq!(attr_of(k, "head_count"), Some(&Value::Int(1)));
}

#[test]
fn kv_involvement_follows_cache_semantics() {
    let mut r = Recorder::default();
    build_graph_segment(
        &mut r,
        &tiny(),
        Segment {
            phase: Phase::Decode,
            sequence_length: 1,
            position_offset: 7,
            kv_append: true,
            start_layer: 0,
            end_layer: 1,
        },
    );
    assert_eq!(r.began, vec![(Phase::Decode, 1, 7)]);
    let k = r.node("layer0.rope_k").kv.clone().unwrap();
    assert_eq!(
        (k.resource.as_str(), k.role, k.append),
        ("kv:0:K", KvRole::K, true)
    );
    let v = r.node("layer0.v_norm").kv.clone().unwrap();
    assert_eq!(
        (v.resource.as_str(), v.role, v.append),
        ("kv:0:V", KvRole::V, true)
    );
    let (r, _) = prefill(&tiny(), 3, 0, 1);
    assert!(!r.node("layer0.rope_k").kv.as_ref().unwrap().append);
}

#[test]
fn softcap_is_last_and_optional() {
    let mut cfg = tiny();
    let (r, g) = prefill(&cfg, 2, 0, 6);
    let cap = r.node("logit_softcap");
    assert_eq!(attr_of(cap, "cap"), Some(&Value::Float(30.0)));
    assert_eq!(cap.inputs, vec!["out:lm_head".to_string()]);
    assert_eq!(g, "graph(out:logit_softcap)");
    cfg.gemma.final_logit_softcapping = None;
    let (r, g) = prefill(&cfg, 2, 0, 6);
    assert!(!r.has("logit_softcap"));
    assert_eq!(g, "graph(out:lm_head)");
}

#[test]
fn untied_embeddings_skip_the_alias() {
    let mut cfg = tiny();
    cfg.arch.tie_word_embeddings = false;
    let (r, _) = prefill(&cfg, 2, 0, 6);
    assert!(r.aliases.is_empty());
}

#[test]
fn segments_compose_the_full_stack() {
    let cfg = tiny();
    let (first, g1) = prefill(&cfg, 4, 0, 3);
    assert_eq!(g1, "graph(out:layer2.layer_scalar)");
    assert!(first.has("embedding") && !first.has("lm_head"));

    let (second, g2) = prefill(&cfg, 4, 3, 6);
    assert_eq!(
        second.inputs,
        vec![("hidden_states_in".to_string(), vec![4, 32])]
    );
    assert!(!second.has("embedding") && second.has("lm_head"));
    assert_eq!(g2, "graph(out:logit_softcap)");
    // Tied aliasing needs the embedding edge registered in this segment.
    assert!(second.weights.contains(&"token_embedding".to_string()));

    let (full, _) = prefill(&cfg, 4, 0, 6);
    assert_eq!(first.nodes.len() + second.nodes.len(), full.nodes.len());
}

#[test]
fn empty_and_invalid_segments() {
    let (r, g) = prefill(&tiny(), 2, 3, 3);
    assert_eq!(r.nodes.len(), 0);
    assert_eq!(g, "graph(in1)");
    let bad = std::panic::catch_unwind(|| prefill(&tiny(), 2, 4, 3));
    assert!(bad.is_err());
    let bad = std::panic::catch_unwind(|| prefill(&tiny(), 2, 0, 7));
    assert!(bad.is_err());
}

#[test]
fn weight_names_match_the_checkpoint_layout() {
    let (r, _) = prefill(&tiny(), 2, 0, 6);
    for w in [
        "token_embedding",
        "final_norm",
        "layers.0.input_norm",
        "layers.0.self_attn.q_norm",
        "layers.0.self_attn.k_norm",
        "layers.0.self_attn.v_proj",
        "layers.0.pre_ffn_norm",
        "layers.0.post_ffn_norm",
        "layers.0.layer_scalar",
        "layers.2.self_attn.k_proj",
    ] {
        assert!(r.weights.iter().any(|x| x == w), "missing weight {w}");
    }
    assert!(!r.weights.iter().any(|w| w == "layers.2.self_attn.v_proj"));
}

#[test]
fn unsupported_gemma4_variants_are_refused() {
    let mut moe = tiny();
    moe.arch.has_mixture_of_experts = true;
    assert!(moe.validate().unwrap_err().contains("mixture-of-experts"));

    let mut ple = tiny();
    ple.gemma.per_layer_input_size = 256;
    assert!(ple.validate().unwrap_err().contains("per-layer"));

    let mut shared = tiny();
    shared.gemma.kv_shared_layers = 20;
    assert!(shared.validate().unwrap_err().contains("KV sharing"));

    let mut scaled = tiny();
    scaled.arch.has_rope_scaling = true;
    assert!(scaled.validate().is_err());

    // Building must trap, not emit a wrong dense graph.
    assert!(std::panic::catch_unwind(|| prefill(&moe, 2, 0, 6)).is_err());
}

#[test]
fn inconsistent_configs_are_refused() {
    let mut c = tiny();
    c.gemma.layer_attention.pop();
    assert!(c.validate().unwrap_err().contains("layer_types"));

    let mut c = tiny();
    c.gemma.sliding_window = 0;
    assert!(c.validate().is_err());

    let mut c = tiny();
    c.gemma.num_global_key_value_heads = 3;
    assert!(c.validate().is_err());

    let mut c = tiny();
    c.gemma.full_partial_rotary_factor = 0.0;
    assert!(c.validate().is_err());
    c.gemma.full_partial_rotary_factor = 1.5;
    assert!(c.validate().is_err());
}

#[test]
fn full_rotary_factor_one_declares_no_partial_attribute() {
    let mut c = tiny();
    c.gemma.full_partial_rotary_factor = 1.0;
    let (r, _) = prefill(&c, 2, 2, 3);
    assert!(attr_of(r.node("layer2.rope_q"), "partial_rotary_factor").is_none());
}

#[test]
fn real_gemma4_31b_shape_builds() {
    // google/gemma-4-31B-it text_config.
    use LayerAttention::*;
    let layers: Vec<_> = (0..60)
        .map(|i| if i % 6 == 5 { Full } else { Sliding })
        .collect();
    let cfg = Config {
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
            has_mixture_of_experts: false,
        },
        gemma: GemmaArchitecture {
            layer_attention: layers,
            sliding_window: 1024,
            global_head_dimension: 512,
            num_global_key_value_heads: 4,
            sliding_rope_theta: 10_000.0,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            final_logit_softcapping: Some(30.0),
            per_layer_input_size: 0,
            kv_shared_layers: 0,
        },
    };
    cfg.validate().unwrap();
    let (r, g) = prefill(&cfg, 8, 0, 60);
    assert_eq!(g, "graph(out:logit_softcap)");
    assert_eq!(r.node("layer5.q_proj").shape, vec![8, 32 * 512]);
    assert_eq!(r.node("layer5.k_proj").shape, vec![8, 4 * 512]);
    assert_eq!(r.node("layer0.k_proj").shape, vec![8, 16 * 256]);
    assert_eq!(r.nodes.len(), 1 + 50 * 22 + 10 * 21 + 3);
}
