use super::*;

/// Attention flavour as a checkpoint's `layer_types` states it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LayerAttention {
    Sliding,
    Full,
}

/// A checkpoint's `config.json` values. The Component never sees these: the
/// Runtime derives `model-config.layer-layout` from them (transformers'
/// `per_layer_config` and `kv_sharing_modes`), which `Case::config` mirrors so
/// the tests keep modelling real checkpoints.
#[derive(Clone)]
struct HfGemma {
    layer_attention: Vec<LayerAttention>,
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

#[derive(Clone)]
struct Case {
    arch: Architecture,
    gemma: HfGemma,
}

impl Case {
    fn config(&self) -> Config {
        let (a, g) = (&self.arch, &self.gemma);
        let n = g.layer_attention.len();
        let first_shared = n.saturating_sub(g.kv_shared_layers as usize);
        let layers = g
            .layer_attention
            .iter()
            .enumerate()
            .take(a.num_hidden_layers as usize)
            .map(|(i, kind)| {
                let consumer = g.kv_shared_layers != 0 && i >= first_shared;
                let kv_source_layer = consumer
                    .then(|| {
                        (0..first_shared)
                            .rev()
                            .find(|j| g.layer_attention[*j] == *kind)
                    })
                    .flatten()
                    .map(|j| j as u32);
                let intermediate_size = a.intermediate_size
                    * if consumer && g.double_wide_mlp_on_shared_layers {
                        2
                    } else {
                        1
                    };
                match kind {
                    LayerAttention::Sliding => LayerSpec {
                        sliding_window: Some(g.sliding_window),
                        head_dimension: a.head_dimension,
                        num_key_value_heads: a.num_key_value_heads,
                        rope_theta: g.sliding_rope_theta,
                        partial_rotary_factor: None,
                        value_from_key: false,
                        kv_source_layer,
                        intermediate_size,
                    },
                    LayerAttention::Full => LayerSpec {
                        sliding_window: None,
                        head_dimension: g.global_head_dimension,
                        // `per_layer_config` only overrides the KV head count
                        // of full layers when `attention_k_eq_v`.
                        num_key_value_heads: match (
                            g.attention_k_eq_v,
                            g.num_global_key_value_heads,
                        ) {
                            (true, Some(global)) => global,
                            _ => a.num_key_value_heads,
                        },
                        rope_theta: g.full_rope_theta,
                        partial_rotary_factor: Some(g.full_partial_rotary_factor),
                        value_from_key: g.attention_k_eq_v,
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
                final_logit_softcapping: g.final_logit_softcapping,
                per_layer_input_size: g.per_layer_input_size,
            },
        }
    }

    fn validate(&self) -> Result<(), String> {
        self.config().validate()
    }

    fn kv_producer(&self, layer: usize) -> Option<u32> {
        self.config().layers[layer].kv_source_layer
    }
}

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
fn tiny() -> Case {
    use LayerAttention::*;
    Case {
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
        gemma: HfGemma {
            layer_attention: vec![Sliding, Sliding, Full, Sliding, Sliding, Full],
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
        },
    }
}

fn prefill(cfg: &Case, seq: u64, start: u32, end: u32) -> (Recorder, String) {
    let mut r = Recorder::default();
    let g = build_graph_segment(
        &mut r,
        &cfg.config(),
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
        &tiny().config(),
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
fn inconsistent_configs_are_refused() {
    let mut c = tiny();
    c.gemma.layer_attention.pop();
    assert!(c.validate().unwrap_err().contains("layer layout"));

    let mut c = tiny();
    c.gemma.sliding_window = 0;
    assert!(c.validate().is_err());

    let mut c = tiny();
    c.gemma.num_global_key_value_heads = Some(3);
    assert!(c.validate().is_err());

    let mut c = tiny();
    c.gemma.full_partial_rotary_factor = 0.0;
    assert!(c.validate().is_err());
    c.gemma.full_partial_rotary_factor = 1.5;
    assert!(c.validate().is_err());
}

#[test]
fn rope_scaling_and_attention_bias_are_refused() {
    let mut c = tiny();
    c.arch.has_rope_scaling = true;
    assert!(c.validate().is_err());
    assert!(std::panic::catch_unwind(|| prefill(&c, 2, 0, 6)).is_err());
    let mut c = tiny();
    c.arch.attention_bias = true;
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
    let cfg = Case {
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
        gemma: HfGemma {
            layer_attention: layers,
            sliding_window: 1024,
            global_head_dimension: 512,
            num_global_key_value_heads: Some(4),
            sliding_rope_theta: 10_000.0,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            final_logit_softcapping: Some(30.0),
            per_layer_input_size: 0,
            kv_shared_layers: 0,
            double_wide_mlp_on_shared_layers: false,
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

// ---------------------------------------------------------------------------
// Shared KV (E2B / E4B)
// ---------------------------------------------------------------------------

/// `tiny()` with the last 3 of 6 layers (`s s f | s s f` -> layers 3, 4, 5)
/// reusing K/V.
fn shared_kv() -> Case {
    let mut c = tiny();
    c.gemma.kv_shared_layers = 3;
    c
}

#[test]
fn shared_layers_reuse_the_last_producer_of_their_kind() {
    let c = shared_kv();
    // Producers are layers 0..3 = s s f. Layer 3 (s) and 4 (s) reuse layer 1;
    // layer 5 (f) reuses layer 2.
    assert_eq!(c.kv_producer(3), Some(1));
    assert_eq!(c.kv_producer(4), Some(1));
    assert_eq!(c.kv_producer(5), Some(2));
    assert_eq!(c.kv_producer(2), None);

    let (r, _) = prefill(&c, 4, 0, 6);
    let att = r.node("layer3.attention");
    assert_eq!(
        att.inputs,
        vec![
            "out:layer3.rope_q".to_string(),
            "out:layer1.rope_k".to_string(),
            "out:layer1.v_norm".to_string()
        ]
    );
    let att = r.node("layer5.attention");
    assert_eq!(att.inputs[1], "out:layer2.rope_k");
    assert_eq!(att.inputs[2], "out:layer2.v_norm");
}

#[test]
fn shared_layers_have_no_kv_weights_nodes_or_cache() {
    let (r, _) = prefill(&shared_kv(), 4, 0, 6);
    for layer in [3, 4, 5] {
        for gone in ["k_proj", "v_proj", "k_norm", "rope_k", "v_norm"] {
            assert!(
                !r.has(&format!("layer{layer}.{gone}")),
                "layer{layer}.{gone}"
            );
        }
        assert!(r.has(&format!("layer{layer}.q_proj")));
        assert!(
            !r.weights
                .iter()
                .any(|w| w.starts_with(&format!("layers.{layer}.self_attn.k")))
        );
    }
    // Only producers allocate KV resources: none of the 3 consumers.
    assert!(r.nodes.iter().filter(|n| n.kv.is_some()).all(|n| {
        let layer: u64 = n.id[5..6].parse().unwrap();
        layer < 3
    }));
}

#[test]
fn shared_layers_keep_their_own_geometry() {
    let (r, _) = prefill(&shared_kv(), 4, 1, 6);
    // Layer 5 is a full layer: 16-wide heads, partial RoPE, no window.
    assert_eq!(r.node("layer5.q_proj").shape, vec![4, 4 * 16]);
    let rope = r.node("layer5.rope_q");
    assert_eq!(
        attr_of(rope, "partial_rotary_factor"),
        Some(&Value::Float(0.25))
    );
    // Layer 4 is sliding: windowed.
    assert_eq!(
        attr_of(r.node("layer4.attention"), "window_size"),
        Some(&Value::Int(16))
    );
}

#[test]
fn double_wide_mlp_applies_to_shared_layers_only() {
    let mut c = shared_kv();
    c.gemma.double_wide_mlp_on_shared_layers = true;
    let (r, _) = prefill(&c, 2, 0, 6);
    assert_eq!(r.node("layer2.gate_proj").shape, vec![2, 64]);
    assert_eq!(r.node("layer3.gate_proj").shape, vec![2, 128]);
    assert_eq!(r.node("layer5.down_proj").shape, vec![2, 32]);
    // Without KV sharing the flag is inert (E4B-like).
    let mut c = tiny();
    c.gemma.double_wide_mlp_on_shared_layers = true;
    let (r, _) = prefill(&c, 2, 0, 6);
    assert_eq!(r.node("layer5.gate_proj").shape, vec![2, 64]);
}

#[test]
fn segments_must_contain_the_producers_of_shared_layers() {
    let c = shared_kv();
    assert!(prefill_ok(&c, 0, 6));
    assert!(prefill_ok(&c, 0, 3));
    assert!(prefill_ok(&c, 1, 6));
    // Layer 3 needs layer 1.
    assert!(!prefill_ok(&c, 2, 6));
    assert!(!prefill_ok(&c, 3, 4));
}

fn prefill_ok(c: &Case, start: u32, end: u32) -> bool {
    let c = c.clone();
    std::panic::catch_unwind(move || prefill(&c, 2, start, end)).is_ok()
}

#[test]
fn full_layers_without_k_eq_v_keep_the_sliding_kv_head_count() {
    // E2B: attention_k_eq_v = false, num_global_key_value_heads = null.
    let mut c = tiny();
    c.gemma.attention_k_eq_v = false;
    c.gemma.num_global_key_value_heads = None;
    c.validate().unwrap();
    let (r, _) = prefill(&c, 3, 2, 3);
    assert_eq!(r.node("layer2.k_proj").shape, vec![3, 2 * 16]);
    assert!(r.has("layer2.v_proj"));
    assert_eq!(
        attr_of(r.node("layer2.attention"), "kv_head_count"),
        Some(&Value::Int(2))
    );
}

#[test]
fn layer_layout_is_validated_not_derived() {
    // The Runtime supplies each layer's K/V source; the Component only checks
    // the layout is self-consistent.
    let mut c = shared_kv().config();
    c.validate().unwrap();

    // A source that is not an earlier layer.
    let mut bad = c.clone();
    bad.layers[3].kv_source_layer = Some(4);
    assert!(bad.validate().unwrap_err().contains("not an earlier layer"));
    let mut bad = c.clone();
    bad.layers[3].kv_source_layer = Some(3);
    assert!(bad.validate().unwrap_err().contains("not an earlier layer"));

    // A source that is itself a consumer.
    let mut bad = c.clone();
    bad.layers[5].kv_source_layer = Some(4);
    assert!(bad.validate().unwrap_err().contains("same shape"));

    // A source with another head width (sliding layer 3 over full layer 2).
    let mut bad = c.clone();
    bad.layers[3].kv_source_layer = Some(2);
    assert!(bad.validate().unwrap_err().contains("same shape"));

    // A layout of the wrong length.
    c.layers.pop();
    assert!(c.validate().unwrap_err().contains("layer layout"));
}

// ---------------------------------------------------------------------------
// Per-layer inputs (E2B / E4B)
// ---------------------------------------------------------------------------

fn ple() -> Case {
    let mut c = tiny();
    c.gemma.per_layer_input_size = 8;
    c
}

#[test]
fn per_layer_input_is_built_from_tokens_and_embeddings() {
    let (r, _) = prefill(&ple(), 4, 0, 6);
    let tok = r.node("layer1.ple_token");
    assert_eq!(tok.operator, "embedding");
    assert_eq!(
        tok.inputs,
        vec![
            "in1".to_string(),
            "w:layers.1.per_layer_embedding".to_string()
        ]
    );
    assert_eq!(attr_of(tok, "scale"), Some(&Value::Float(8f64.sqrt())));
    assert_eq!(tok.shape, vec![4, 8]);

    let ctx = r.node("layer1.ple_context");
    assert_eq!(
        ctx.inputs[0], "out:embedding",
        "projects the scaled embeddings"
    );
    assert_eq!(ctx.shape, vec![4, 8]);
    let scaled = r.node("layer1.ple_context_scale");
    assert_eq!(attr_of(scaled, "kind"), Some(&Value::Text("scale".into())));
    assert_eq!(
        attr_of(scaled, "factor"),
        Some(&Value::Float(32f64.powf(-0.5)))
    );
    assert_eq!(scaled.inputs, vec!["out:layer1.ple_context".to_string()]);

    let norm = r.node("layer1.ple_context_norm");
    assert_eq!(norm.inputs[0], "out:layer1.ple_context_scale");
    assert_eq!(norm.inputs[1], "w:per_layer_projection_norm");

    let sum = r.node("layer1.ple_sum");
    assert_eq!(
        sum.inputs,
        vec![
            "out:layer1.ple_context_norm".to_string(),
            "out:layer1.ple_token".to_string()
        ]
    );
    let mix = r.node("layer1.ple_mix");
    assert_eq!(mix.inputs, vec!["out:layer1.ple_sum".to_string()]);
    assert_eq!(attr_of(mix, "kind"), Some(&Value::Text("scale".into())));
    assert_eq!(attr_of(mix, "factor"), Some(&Value::Float(0.5f64.sqrt())));
}

#[test]
fn per_layer_tail_sits_between_the_mlp_and_the_layer_scalar() {
    let (r, _) = prefill(&ple(), 2, 0, 1);
    let ids: Vec<&str> = r.nodes.iter().map(|n| n.id.as_str()).collect();
    let pos = |id: &str| ids.iter().position(|i| *i == id).unwrap();
    for pair in [
        ["layer0.residual2", "layer0.ple_gate"],
        ["layer0.ple_gate", "layer0.ple_gelu"],
        ["layer0.ple_gelu", "layer0.ple_mul"],
        ["layer0.ple_mul", "layer0.ple_projection"],
        ["layer0.ple_projection", "layer0.ple_post_norm"],
        ["layer0.ple_post_norm", "layer0.ple_residual"],
        ["layer0.ple_residual", "layer0.layer_scalar"],
    ] {
        assert!(pos(pair[0]) < pos(pair[1]), "{pair:?}");
    }
    assert_eq!(r.node("layer0.ple_gate").shape, vec![2, 8]);
    assert_eq!(r.node("layer0.ple_projection").shape, vec![2, 32]);
    assert_eq!(
        r.node("layer0.ple_residual").inputs[0],
        "out:layer0.residual2"
    );
    assert_eq!(
        r.node("layer0.layer_scalar").inputs[0],
        "out:layer0.ple_residual"
    );
    assert_eq!(
        attr_of(r.node("layer0.ple_gelu"), "approximate"),
        Some(&Value::Text("tanh".into()))
    );
}

#[test]
fn per_layer_inputs_are_absent_by_default() {
    let (r, _) = prefill(&tiny(), 2, 0, 6);
    assert!(!r.nodes.iter().any(|n| n.id.contains(".ple_")));
}

#[test]
fn later_segments_recompute_embeddings_for_per_layer_inputs() {
    let (r, _) = prefill(&ple(), 4, 3, 6);
    assert_eq!(
        r.inputs,
        vec![
            ("hidden_states_in".to_string(), vec![4, 32]),
            ("token_ids".to_string(), vec![4])
        ]
    );
    assert!(r.has("embedding_per_layer_inputs"));
    assert_eq!(
        r.node("layer3.ple_context").inputs[0],
        "out:embedding_per_layer_inputs"
    );
    // The residual stream still starts from the incoming hidden states.
    assert_eq!(r.node("layer3.residual1").inputs[0], "in1");
    // Without per-layer inputs a later segment needs neither.
    let (r, _) = prefill(&tiny(), 4, 3, 6);
    assert_eq!(r.inputs.len(), 1);
}

#[test]
fn an_empty_later_segment_declares_no_token_ids() {
    let (r, _) = prefill(&ple(), 4, 3, 3);
    assert_eq!(r.inputs.len(), 1);
}

#[test]
fn e2b_style_config_builds_end_to_end() {
    // 35 layers, 20 shared, double-wide MLP, per-layer inputs, k != v.
    use LayerAttention::*;
    let layers: Vec<_> = (0..35)
        .map(|i| if i % 5 == 4 { Full } else { Sliding })
        .collect();
    let c = Case {
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
        gemma: HfGemma {
            layer_attention: layers,
            sliding_window: 512,
            global_head_dimension: 512,
            num_global_key_value_heads: None,
            sliding_rope_theta: 10_000.0,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: false,
            final_logit_softcapping: Some(30.0),
            per_layer_input_size: 256,
            kv_shared_layers: 20,
            double_wide_mlp_on_shared_layers: true,
        },
    };
    c.validate().unwrap();
    let (r, g) = prefill(&c, 8, 0, 35);
    assert_eq!(g, "graph(out:logit_softcap)");
    // Producers 0..15; layer 15 is the first consumer (sliding) -> layer 13.
    assert_eq!(c.kv_producer(15), Some(13));
    assert_eq!(c.kv_producer(19), Some(14)); // full -> last full producer
    assert_eq!(r.node("layer14.gate_proj").shape, vec![8, 6144]);
    assert_eq!(r.node("layer15.gate_proj").shape, vec![8, 12288]);
    assert_eq!(r.node("layer4.q_proj").shape, vec![8, 8 * 512]);
    assert!(r.has("layer34.ple_residual"));
    let kv_nodes = r.nodes.iter().filter(|n| n.kv.is_some()).count();
    assert_eq!(kv_nodes, 2 * 15, "only the 15 producing layers cache K/V");
}

// ---------------------------------------------------------------------------
// Mixture of experts (26B-A4B)
// ---------------------------------------------------------------------------

fn moe() -> Case {
    let mut c = tiny();
    c.arch.moe = Some(Moe {
        experts: 8,
        experts_per_token: 2,
        expert_intermediate_size: 24,
        normalize_top_k: true,
        has_shared_expert: false,
    });
    c
}

#[test]
fn moe_layer_runs_dense_mlp_and_routed_experts_side_by_side() {
    let (r, _) = prefill(&moe(), 3, 0, 1);
    let ids: Vec<&str> = r.nodes.iter().map(|n| n.id.as_str()).collect();
    let pos = |id: &str| ids.iter().position(|i| *i == id).unwrap();

    // Dense path.
    assert_eq!(
        r.node("layer0.pre_ffn_norm").inputs[0],
        "out:layer0.residual1"
    );
    assert_eq!(
        r.node("layer0.post_ffn_norm_1").inputs[0],
        "out:layer0.down_proj"
    );
    // Router sees the residual stream, not the normed MLP input.
    assert_eq!(
        r.node("layer0.router_norm").inputs[0],
        "out:layer0.residual1"
    );
    assert_eq!(r.node("layer0.router").inputs[0], "out:layer0.router_norm");
    assert_eq!(r.node("layer0.router").shape, vec![3, 8]);
    assert_eq!(
        r.node("layer0.pre_ffn_norm_2").inputs[0],
        "out:layer0.residual1"
    );

    let e = r.node("layer0.experts");
    assert_eq!(e.operator, "moe-experts");
    assert_eq!(
        e.inputs,
        vec![
            "out:layer0.pre_ffn_norm_2".to_string(),
            "out:layer0.router".to_string(),
            "w:layers.0.experts.gate_proj".to_string(),
            "w:layers.0.experts.up_proj".to_string(),
            "w:layers.0.experts.down_proj".to_string(),
            // Applied by the Operator, never folded into (possibly
            // quantized) expert weights.
            "w:layers.0.experts.per_expert_scale".to_string(),
        ]
    );
    assert_eq!(attr_of(e, "experts"), Some(&Value::Int(8)));
    assert_eq!(attr_of(e, "top_k"), Some(&Value::Int(2)));
    assert_eq!(attr_of(e, "normalize_top_k"), Some(&Value::Bool(true)));
    assert_eq!(
        attr_of(e, "activation"),
        Some(&Value::Text("gelu_tanh".into()))
    );
    assert_eq!(e.shape, vec![3, 32]);

    assert_eq!(
        r.node("layer0.post_ffn_norm_2").inputs[0],
        "out:layer0.experts"
    );
    assert_eq!(
        r.node("layer0.moe_sum").inputs,
        vec![
            "out:layer0.post_ffn_norm_1".to_string(),
            "out:layer0.post_ffn_norm_2".to_string()
        ]
    );
    // The final sandwich norm covers the sum, then the residual closes it.
    assert_eq!(
        r.node("layer0.post_ffn_norm").inputs[0],
        "out:layer0.moe_sum"
    );
    assert_eq!(
        r.node("layer0.residual2").inputs,
        vec![
            "out:layer0.residual1".to_string(),
            "out:layer0.post_ffn_norm".to_string()
        ]
    );
    for pair in [
        ["layer0.down_proj", "layer0.post_ffn_norm_1"],
        ["layer0.router", "layer0.experts"],
        ["layer0.experts", "layer0.post_ffn_norm_2"],
        ["layer0.moe_sum", "layer0.post_ffn_norm"],
    ] {
        assert!(pos(pair[0]) < pos(pair[1]), "{pair:?}");
    }
}

#[test]
fn moe_weight_names_follow_the_loader_contract() {
    let (r, _) = prefill(&moe(), 2, 0, 1);
    for w in [
        "layers.0.router_proj",
        "layers.0.experts.per_expert_scale",
        "layers.0.pre_ffn_norm_2",
        "layers.0.post_ffn_norm_1",
        "layers.0.post_ffn_norm_2",
        "layers.0.experts.gate_proj",
        "layers.0.experts.up_proj",
        "layers.0.experts.down_proj",
        "layers.0.mlp.gate_proj",
    ] {
        assert!(r.weights.iter().any(|x| x == w), "missing {w}");
    }
}

#[test]
fn dense_graphs_have_no_moe_nodes() {
    let (r, _) = prefill(&tiny(), 2, 0, 6);
    assert!(!r.nodes.iter().any(|n| n.operator == "moe-experts"));
    assert!(!r.has("layer0.router"));
}

#[test]
fn moe_and_dense_node_counts_differ_by_the_expert_branch() {
    let (dense, _) = prefill(&tiny(), 2, 0, 1);
    let (sparse, _) = prefill(&moe(), 2, 0, 1);
    // post_ffn_norm_1, router_norm, router, pre_ffn_norm_2, experts,
    // post_ffn_norm_2, moe_sum.
    assert_eq!(sparse.nodes.len(), dense.nodes.len() + 7);
}

#[test]
fn unsupported_moe_shapes_are_refused() {
    let mut c = moe();
    c.arch.moe.as_mut().unwrap().has_shared_expert = true;
    assert!(c.validate().unwrap_err().contains("shared expert"));
    let mut c = moe();
    c.arch.moe.as_mut().unwrap().normalize_top_k = false;
    assert!(c.validate().unwrap_err().contains("renormalizes"));
    let mut c = moe();
    c.arch.moe.as_mut().unwrap().experts_per_token = 9;
    assert!(c.validate().unwrap_err().contains("inconsistent"));
    assert!(std::panic::catch_unwind(|| prefill(&c, 2, 0, 6)).is_err());
}

#[test]
fn real_gemma4_26b_a4b_shape_builds() {
    use LayerAttention::*;
    let layers: Vec<_> = (0..30)
        .map(|i| if i % 6 == 5 { Full } else { Sliding })
        .collect();
    let c = Case {
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
        gemma: HfGemma {
            layer_attention: layers,
            sliding_window: 1024,
            global_head_dimension: 512,
            num_global_key_value_heads: Some(2),
            sliding_rope_theta: 10_000.0,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            final_logit_softcapping: Some(30.0),
            per_layer_input_size: 0,
            kv_shared_layers: 0,
            double_wide_mlp_on_shared_layers: false,
        },
    };
    c.validate().unwrap();
    let (r, g) = prefill(&c, 4, 0, 30);
    assert_eq!(g, "graph(out:logit_softcap)");
    assert_eq!(r.node("layer0.router").shape, vec![4, 128]);
    assert_eq!(r.node("layer5.k_proj").shape, vec![4, 2 * 512]);
    assert_eq!(
        r.nodes
            .iter()
            .filter(|n| n.operator == "moe-experts")
            .count(),
        30
    );
}

// ---------------------------------------------------------------------------
// Runtime review: one scaling mechanism, no fabricated weights, no folding
// of scales into possibly quantized weights
// ---------------------------------------------------------------------------

/// Every node of a full build of each variant.
fn every_variant() -> Vec<Recorder> {
    [tiny(), shared_kv(), ple(), moe()]
        .iter()
        .map(|c| prefill(c, 3, 0, 6).0)
        .collect()
}

#[test]
fn constants_are_applied_by_the_one_scale_operation() {
    for r in every_variant() {
        for n in &r.nodes {
            assert!(attr_of(n, "output_scale").is_none(), "{}", n.id);
            assert!(attr_of(n, "broadcast").is_none(), "{}", n.id);
            if n.operator == "mul" {
                assert_eq!(n.inputs.len(), 2);
                assert!(!n.id.ends_with("layer_scalar"), "{}", n.id);
            }
        }
    }
    let (r, _) = prefill(&ple(), 3, 0, 6);
    for id in ["layer0.ple_context_scale", "layer0.ple_mix"] {
        let n = r.node(id);
        assert_eq!((n.operator, n.family), ("activation", "Activation"), "{id}");
        assert_eq!(
            attr_of(n, "kind"),
            Some(&Value::Text("scale".into())),
            "{id}"
        );
        assert!(
            matches!(attr_of(n, "factor"), Some(Value::Float(_))),
            "{id}"
        );
        assert_eq!(n.inputs.len(), 1, "{id}");
    }
}

#[test]
fn layer_scalar_is_a_scale_with_a_weight_input() {
    let (r, _) = prefill(&tiny(), 3, 0, 1);
    let n = r.node("layer0.layer_scalar");
    assert_eq!(n.operator, "activation");
    assert_eq!(attr_of(n, "kind"), Some(&Value::Text("scale".into())));
    assert!(attr_of(n, "factor").is_none(), "the factor is the input");
    assert_eq!(
        n.inputs,
        vec![
            "out:layer0.residual2".to_string(),
            "w:layers.0.layer_scalar".to_string()
        ]
    );
}

#[test]
fn scale_less_norms_take_one_input_and_request_no_weight() {
    let (r, _) = prefill(&moe(), 3, 2, 3);
    for id in ["layer2.v_norm", "layer2.router_norm"] {
        let n = r.node(id);
        assert_eq!(n.operator, "rmsnorm", "{id}");
        assert_eq!(n.inputs.len(), 1, "{id}");
        assert_eq!(attr_of(n, "with_scale"), Some(&Value::Bool(false)), "{id}");
    }
    // No unit vector is ever requested from the loader.
    for w in &r.weights {
        assert!(!w.ends_with("v_norm") && !w.ends_with("router_norm"), "{w}");
    }
    // Scaled norms keep their weight and no `with_scale` attribute.
    assert_eq!(r.node("layer2.input_norm").inputs.len(), 2);
    assert!(attr_of(r.node("layer2.input_norm"), "with_scale").is_none());
    // The value norm is still per head.
    assert_eq!(
        attr_of(r.node("layer2.v_norm"), "head_dimension"),
        Some(&Value::Int(16))
    );
}

#[test]
fn expert_scales_are_an_operator_input_not_folded_into_weights() {
    let (r, _) = prefill(&moe(), 3, 0, 1);
    let e = r.node("layer0.experts");
    assert_eq!(e.inputs.len(), 6);
    assert_eq!(e.inputs[4], "w:layers.0.experts.down_proj");
    assert_eq!(e.inputs[5], "w:layers.0.experts.per_expert_scale");
}

#[test]
fn the_component_names_no_model_family_in_its_layout_types() {
    // `LayerSpec`/`Options` are what `model-config` carries; they must stay
    // expressible without Gemma vocabulary (other families need them too).
    let spec = LayerSpec {
        sliding_window: None,
        head_dimension: 64,
        num_key_value_heads: 4,
        rope_theta: 1e4,
        partial_rotary_factor: None,
        value_from_key: false,
        kv_source_layer: None,
        intermediate_size: 256,
    };
    assert_eq!(spec.clone(), spec);
    assert!(Options::default().final_logit_softcapping.is_none());
}
