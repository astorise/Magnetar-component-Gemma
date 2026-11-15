//! Gemma 4 dense decoder graph, expressed against a [`GraphSink`].
//!
//! This module has no WIT or host dependency: `lib.rs` adapts the
//! Runtime-owned `graph-builder` Capability to [`GraphSink`], and the tests
//! below drive the same code with a recording sink. Everything here is
//! derived from the checkpoint's `config.json` values carried by
//! `model-config` and `gemma-config`; nothing is a compiled-in dimension.
//!
//! Reference: Hugging Face `transformers` `modeling_gemma4.py`
//! (`Gemma4TextDecoderLayer`, `Gemma4TextAttention`, `Gemma4ForCausalLM`).
//!
//! # What one Gemma 4 decoder layer is
//!
//! ```text
//! x ─ input_norm ─ q/k/v proj ─ q_norm, k_norm, v_norm ─ RoPE(q,k) ─ attention ─ o_proj
//!   ─ post_attn_norm ─ (+ x) ─ pre_ffn_norm ─ GeGLU MLP ─ post_ffn_norm ─ (+) ─ × layer_scalar
//! ```
//!
//! Unlike Gemma 1-3 the RMSNorm scale is the plain `weight` (not `1 + weight`),
//! and attention uses `scale = 1.0` because queries and keys are normalized.
//! Sliding layers and full layers differ in head width, KV head count, RoPE
//! base and partial rotation, mask window, and (full layers, when
//! `attention_k_eq_v`) in having no V projection.

/// Which phase a graph is built for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvRole {
    K,
    V,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attr {
    pub name: &'static str,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Kv {
    pub resource: String,
    pub role: KvRole,
    pub append: bool,
}

/// One graph node, as `graph-builder.add-node` takes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: String,
    pub operator: &'static str,
    pub family: &'static str,
    pub attrs: Vec<Attr>,
    pub inputs: Vec<String>,
    pub shape: Vec<u64>,
    pub kv: Option<Kv>,
}

/// The `graph-builder` Capability, as this module uses it.
pub trait GraphSink {
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64);
    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String;
    fn weight_edge(&mut self, logical_name: &str) -> String;
    fn alias_weight_edge(&mut self, edge: &str, target_logical_name: &str);
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String;
    fn add_node(&mut self, node: Node) -> String;
    fn finish_graph(&mut self, output_edge: &str) -> String;
}

/// `model-config.architecture-config`, the fields this family reads.
#[derive(Clone, Debug)]
pub struct Architecture {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    /// KV heads of sliding layers (`num_key_value_heads`).
    pub num_key_value_heads: u32,
    /// Head width of sliding layers (`head_dim`).
    pub head_dimension: u64,
    pub vocabulary_size: u64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    /// `model-config.rope-scaling` is `some`.
    pub has_rope_scaling: bool,
    /// `model-config.mixture-of-experts` is `some`.
    pub has_mixture_of_experts: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerAttention {
    Sliding,
    Full,
}

/// `gemma-config.gemma-architecture`.
#[derive(Clone, Debug)]
pub struct GemmaArchitecture {
    pub layer_attention: Vec<LayerAttention>,
    pub sliding_window: u32,
    pub global_head_dimension: u64,
    pub num_global_key_value_heads: u32,
    pub sliding_rope_theta: f64,
    pub full_rope_theta: f64,
    pub full_partial_rotary_factor: f64,
    pub attention_k_eq_v: bool,
    pub final_logit_softcapping: Option<f64>,
    pub per_layer_input_size: u32,
    pub kv_shared_layers: u32,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: Architecture,
    pub gemma: GemmaArchitecture,
}

/// Geometry of one decoder layer, resolved from its attention flavour.
#[derive(Clone, Debug, PartialEq)]
struct Layer {
    index: u64,
    head_dimension: u64,
    kv_heads: u64,
    /// `Some(window)` for sliding layers.
    window: Option<u64>,
    rope_theta: f64,
    /// `Some(f)` when only a fraction `f < 1` of each head is rotated.
    partial_rotary_factor: Option<f64>,
    /// No V projection: V is derived from K's raw projection.
    k_eq_v: bool,
}

impl Config {
    /// Rejects every configuration this Component cannot build a *correct*
    /// graph for. The caller traps on `Err`: building a dense Gemma graph for
    /// a checkpoint that is not one would silently produce wrong logits.
    pub fn validate(&self) -> Result<(), String> {
        let (a, g) = (&self.arch, &self.gemma);
        if a.has_mixture_of_experts {
            return Err("mixture-of-experts Gemma 4 (e.g. 26B-A4B) is not implemented".into());
        }
        if g.per_layer_input_size != 0 {
            return Err(
                "per-layer input embeddings (Gemma 4 E2B/E4B, hidden_size_per_layer_input > 0) \
                 are not implemented"
                    .into(),
            );
        }
        if g.kv_shared_layers != 0 {
            return Err(
                "cross-layer KV sharing (Gemma 4 E2B/E4B, num_kv_shared_layers > 0) \
                 is not implemented"
                    .into(),
            );
        }
        if a.has_rope_scaling {
            return Err(
                "checkpoint declares RoPE frequency scaling, which Gemma 4 does not use".into(),
            );
        }
        if a.attention_bias {
            return Err("attention_bias = true is not a Gemma 4 configuration".into());
        }
        if g.layer_attention.len() != a.num_hidden_layers as usize {
            return Err(format!(
                "layer_types has {} entries for {} decoder layers",
                g.layer_attention.len(),
                a.num_hidden_layers
            ));
        }
        let any = |kind| g.layer_attention.contains(&kind);
        if any(LayerAttention::Sliding) && g.sliding_window == 0 {
            return Err("sliding layers declared with sliding_window = 0".into());
        }
        for (what, heads) in [
            ("num_key_value_heads", a.num_key_value_heads),
            ("num_global_key_value_heads", g.num_global_key_value_heads),
        ] {
            if heads == 0 || a.num_attention_heads % heads != 0 {
                return Err(format!(
                    "{what} = {heads} does not divide num_attention_heads = {}",
                    a.num_attention_heads
                ));
            }
        }
        if a.head_dimension == 0 || g.global_head_dimension == 0 {
            return Err("head dimensions must be non-zero".into());
        }
        if !(g.full_partial_rotary_factor > 0.0 && g.full_partial_rotary_factor <= 1.0) {
            return Err(format!(
                "partial_rotary_factor {} is outside (0, 1]",
                g.full_partial_rotary_factor
            ));
        }
        Ok(())
    }

    fn layer(&self, index: u64) -> Layer {
        let (a, g) = (&self.arch, &self.gemma);
        match g.layer_attention[index as usize] {
            LayerAttention::Sliding => Layer {
                index,
                head_dimension: a.head_dimension,
                kv_heads: a.num_key_value_heads as u64,
                window: Some(g.sliding_window as u64),
                rope_theta: g.sliding_rope_theta,
                partial_rotary_factor: None,
                k_eq_v: false,
            },
            LayerAttention::Full => Layer {
                index,
                head_dimension: g.global_head_dimension,
                kv_heads: g.num_global_key_value_heads as u64,
                window: None,
                rope_theta: g.full_rope_theta,
                partial_rotary_factor: (g.full_partial_rotary_factor < 1.0)
                    .then_some(g.full_partial_rotary_factor),
                k_eq_v: g.attention_k_eq_v,
            },
        }
    }
}

fn attr(name: &'static str, value: Value) -> Attr {
    Attr { name, value }
}

fn tensor_parallel(kind: &str) -> Attr {
    attr("tensor_parallel", Value::Text(kind.to_string()))
}

/// Canonical Model Artifact tensor name for a per-layer weight.
///
/// `v_norm` has no checkpoint tensor (`Gemma4RMSNorm(with_scale=False)`): the
/// Model Artifact loader must resolve it to a unit-scale vector so the
/// portable two-input `rmsnorm` Operator stays valid.
pub fn weight_name(layer: u64, suffix: &str) -> String {
    match suffix {
        "input_norm" | "post_attn_norm" | "pre_ffn_norm" | "post_ffn_norm" | "layer_scalar" => {
            format!("layers.{layer}.{suffix}")
        }
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "q_norm" | "k_norm" | "v_norm" => {
            format!("layers.{layer}.self_attn.{suffix}")
        }
        "gate_proj" | "up_proj" | "down_proj" => format!("layers.{layer}.mlp.{suffix}"),
        other => unreachable!("weight_name: unknown weight suffix '{other}'"),
    }
}

struct Builder<'a, S: GraphSink> {
    sink: &'a mut S,
    cfg: &'a Config,
    seq: u64,
    kv_append: bool,
}

impl<S: GraphSink> Builder<'_, S> {
    // Mirrors `graph-builder.add-node`'s own seven parameters.
    #[allow(clippy::too_many_arguments)]
    fn node(
        &mut self,
        id: String,
        operator: &'static str,
        family: &'static str,
        attrs: Vec<Attr>,
        inputs: Vec<String>,
        shape: Vec<u64>,
        kv: Option<Kv>,
    ) -> String {
        self.sink.add_node(Node {
            id,
            operator,
            family,
            attrs,
            inputs,
            shape,
            kv,
        })
    }

    fn eps(&self) -> Attr {
        attr("epsilon", Value::Float(self.cfg.arch.rms_norm_eps))
    }

    /// RMSNorm over the hidden dimension with a checkpoint weight.
    fn hidden_norm(&mut self, id: String, x: &str, weight: &str) -> String {
        let w = self.sink.weight_edge(weight);
        let shape = vec![self.seq, self.cfg.arch.hidden_size];
        let attrs = vec![self.eps()];
        self.node(
            id,
            "rmsnorm",
            "Normalization",
            attrs,
            vec![x.into(), w],
            shape,
            None,
        )
    }

    /// RMSNorm applied independently to each `head_dimension`-wide head.
    fn head_norm(&mut self, id: String, x: &str, weight: &str, heads: u64, dim: u64) -> String {
        let w = self.sink.weight_edge(weight);
        let attrs = vec![
            self.eps(),
            attr("head_count", Value::Int(heads as i64)),
            attr("head_dimension", Value::Int(dim as i64)),
        ];
        self.node(
            id,
            "rmsnorm",
            "Normalization",
            attrs,
            vec![x.into(), w],
            vec![self.seq, heads * dim],
            None,
        )
    }

    fn matmul(
        &mut self,
        id: String,
        x: &str,
        weight: &str,
        out_dim: u64,
        tp: Option<&str>,
    ) -> String {
        let w = self.sink.weight_edge(weight);
        let attrs = tp.map(tensor_parallel).into_iter().collect();
        self.node(
            id,
            "matmul",
            "LinearAlgebra",
            attrs,
            vec![x.into(), w],
            vec![self.seq, out_dim],
            None,
        )
    }

    fn rope(&mut self, id: String, x: &str, layer: &Layer, heads: u64, kv: Option<Kv>) -> String {
        let mut attrs = vec![
            attr("base", Value::Float(layer.rope_theta)),
            attr("dimension", Value::Int(layer.head_dimension as i64)),
            attr("position_mode", Value::Text("sequential".into())),
            // Hugging Face rotates (i, i + dimension/2) (`rotate_half`).
            attr("rotation_pairing", Value::Text("half_split".into())),
            attr("position_offset", Value::Int(0)),
            attr("head_count", Value::Int(heads as i64)),
        ];
        if let Some(factor) = layer.partial_rotary_factor {
            // "proportional" RoPE: frequencies use the full head width as
            // their denominator, only the first `factor * dimension / 2`
            // pairs rotate and the remaining pairs have frequency zero.
            attrs.push(attr("partial_rotary_factor", Value::Float(factor)));
        }
        self.node(
            id,
            "rope",
            "PositionEncoding",
            attrs,
            vec![x.into()],
            vec![self.seq, heads * layer.head_dimension],
            kv,
        )
    }

    fn residual(&mut self, id: String, a: &str, b: &str) -> String {
        let shape = vec![self.seq, self.cfg.arch.hidden_size];
        self.node(
            id,
            "residual-add",
            "Tensor",
            vec![],
            vec![a.into(), b.into()],
            shape,
            None,
        )
    }

    fn kv(&mut self, layer: u64, role: KvRole) -> Kv {
        Kv {
            resource: self.sink.kv_resource(layer, role),
            role,
            append: self.kv_append,
        }
    }

    fn layer(&mut self, layer: &Layer, x: &str) -> String {
        let a = self.cfg.arch.clone();
        let p = format!("layer{}", layer.index);
        let heads = a.num_attention_heads as u64;
        let hd = layer.head_dimension;
        let (q_dim, kv_dim) = (heads * hd, layer.kv_heads * hd);

        // --- attention -----------------------------------------------------
        let normed = self.hidden_norm(
            format!("{p}.input_norm"),
            x,
            &weight_name(layer.index, "input_norm"),
        );

        let q = self.matmul(
            format!("{p}.q_proj"),
            &normed,
            &weight_name(layer.index, "q_proj"),
            q_dim,
            None,
        );
        let q = self.head_norm(
            format!("{p}.q_norm"),
            &q,
            &weight_name(layer.index, "q_norm"),
            heads,
            hd,
        );
        let q = self.rope(format!("{p}.rope_q"), &q, layer, heads, None);

        let k_raw = self.matmul(
            format!("{p}.k_proj"),
            &normed,
            &weight_name(layer.index, "k_proj"),
            kv_dim,
            None,
        );
        // With `attention_k_eq_v` there is no V projection: V is the raw K
        // projection, taken before K's own norm and RoPE.
        let v_raw = if layer.k_eq_v {
            k_raw.clone()
        } else {
            self.matmul(
                format!("{p}.v_proj"),
                &normed,
                &weight_name(layer.index, "v_proj"),
                kv_dim,
                None,
            )
        };
        let k = self.head_norm(
            format!("{p}.k_norm"),
            &k_raw,
            &weight_name(layer.index, "k_norm"),
            layer.kv_heads,
            hd,
        );
        let k_kv = self.kv(layer.index, KvRole::K);
        let k = self.rope(format!("{p}.rope_k"), &k, layer, layer.kv_heads, Some(k_kv));
        let v_kv = self.kv(layer.index, KvRole::V);
        let v_scale = self.sink.weight_edge(&weight_name(layer.index, "v_norm"));
        let v = self.node(
            format!("{p}.v_norm"),
            "rmsnorm",
            "Normalization",
            vec![
                self.eps(),
                attr("head_count", Value::Int(layer.kv_heads as i64)),
                attr("head_dimension", Value::Int(hd as i64)),
            ],
            vec![v_raw, v_scale],
            vec![self.seq, kv_dim],
            Some(v_kv),
        );

        let mut attn_attrs = vec![
            attr("causal", Value::Bool(true)),
            attr("head_count", Value::Int(heads as i64)),
            attr("kv_head_count", Value::Int(layer.kv_heads as i64)),
            attr("head_dimension", Value::Int(hd as i64)),
            attr("attention_mask_kind", Value::Text("causal".into())),
            // Queries and keys are normalized; Gemma 4 does not rescale the
            // scores by 1/sqrt(head_dimension).
            attr("scale", Value::Float(1.0)),
        ];
        if let Some(window) = layer.window {
            attn_attrs.push(attr("window_size", Value::Int(window as i64)));
        }
        let attn = self.node(
            format!("{p}.attention"),
            "attention",
            "Attention",
            attn_attrs,
            vec![q, k, v],
            vec![self.seq, q_dim],
            None,
        );
        let o = self.matmul(
            format!("{p}.o_proj"),
            &attn,
            &weight_name(layer.index, "o_proj"),
            a.hidden_size,
            None,
        );
        let o = self.hidden_norm(
            format!("{p}.post_attn_norm"),
            &o,
            &weight_name(layer.index, "post_attn_norm"),
        );
        let h = self.residual(format!("{p}.residual1"), x, &o);

        // --- GeGLU MLP -----------------------------------------------------
        let m = self.hidden_norm(
            format!("{p}.pre_ffn_norm"),
            &h,
            &weight_name(layer.index, "pre_ffn_norm"),
        );
        let inter = a.intermediate_size;
        let gate = self.matmul(
            format!("{p}.gate_proj"),
            &m,
            &weight_name(layer.index, "gate_proj"),
            inter,
            Some("column"),
        );
        let up = self.matmul(
            format!("{p}.up_proj"),
            &m,
            &weight_name(layer.index, "up_proj"),
            inter,
            Some("column"),
        );
        let act = self.node(
            format!("{p}.gelu"),
            "gelu",
            "Activation",
            // `hidden_activation = "gelu_pytorch_tanh"`.
            vec![
                attr("approximate", Value::Text("tanh".into())),
                tensor_parallel("local"),
            ],
            vec![gate],
            vec![self.seq, inter],
            None,
        );
        let mixed = self.node(
            format!("{p}.mul"),
            "mul",
            "Tensor",
            vec![tensor_parallel("local")],
            vec![act, up],
            vec![self.seq, inter],
            None,
        );
        let down = self.matmul(
            format!("{p}.down_proj"),
            &mixed,
            &weight_name(layer.index, "down_proj"),
            a.hidden_size,
            Some("row"),
        );
        let down = self.hidden_norm(
            format!("{p}.post_ffn_norm"),
            &down,
            &weight_name(layer.index, "post_ffn_norm"),
        );
        let out = self.residual(format!("{p}.residual2"), &h, &down);

        // --- per-layer output scalar (`hidden_states *= layer_scalar`) -----
        let scalar = self
            .sink
            .weight_edge(&weight_name(layer.index, "layer_scalar"));
        self.node(
            format!("{p}.layer_scalar"),
            "mul",
            "Tensor",
            vec![attr("broadcast", Value::Text("scalar".into()))],
            vec![out, scalar],
            vec![self.seq, a.hidden_size],
            None,
        )
    }
}

/// One graph to build: its phase and the decoder layers `[start_layer,
/// end_layer)` it covers (`build-*-graph-segment` in the WIT).
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub phase: Phase,
    pub sequence_length: u64,
    pub position_offset: u64,
    /// Whether K/V outputs are appended to the KV cache (decode) or not.
    pub kv_append: bool,
    pub start_layer: u32,
    pub end_layer: u32,
}

/// Builds one [`Segment`]. Panics (a trapped host call) on an unsupported
/// configuration or an invalid layer range.
pub fn build_graph_segment<S: GraphSink>(sink: &mut S, cfg: &Config, segment: Segment) -> String {
    let Segment {
        phase,
        sequence_length,
        position_offset,
        kv_append,
        start_layer,
        end_layer,
    } = segment;
    if let Err(reason) = cfg.validate() {
        panic!("gemma: unsupported configuration: {reason}");
    }
    let layers = cfg.arch.num_hidden_layers;
    assert!(
        start_layer <= end_layer && end_layer <= layers,
        "gemma: invalid layer segment [{start_layer}, {end_layer}) of {layers}"
    );
    let a = &cfg.arch;
    sink.begin_graph(phase, sequence_length, position_offset);

    let mut b = Builder {
        sink,
        cfg,
        seq: sequence_length,
        kv_append,
    };

    let mut hidden = if start_layer == 0 {
        let ids = b.sink.declare_input("token_ids", &[sequence_length]);
        let table = b.sink.weight_edge("token_embedding");
        b.node(
            "embedding".into(),
            "embedding",
            "Tensor",
            // Gemma multiplies token embeddings by sqrt(hidden_size).
            vec![attr("scale", Value::Float((a.hidden_size as f64).sqrt()))],
            vec![ids, table],
            vec![sequence_length, a.hidden_size],
            None,
        )
    } else {
        b.sink
            .declare_input("hidden_states_in", &[sequence_length, a.hidden_size])
    };

    for index in start_layer as u64..end_layer as u64 {
        let layer = cfg.layer(index);
        hidden = b.layer(&layer, &hidden);
    }

    if end_layer != layers {
        return b.sink.finish_graph(&hidden);
    }

    let hidden = b.hidden_norm("final_norm".into(), &hidden, "final_norm");
    let lm_head = b.sink.weight_edge("lm_head");
    if a.tie_word_embeddings {
        b.sink.weight_edge("token_embedding");
        b.sink.alias_weight_edge(&lm_head, "token_embedding");
    }
    let mut logits = b.node(
        "lm_head".into(),
        "matmul",
        "LinearAlgebra",
        vec![],
        vec![hidden, lm_head],
        vec![sequence_length, a.vocabulary_size],
        None,
    );
    if let Some(cap) = cfg.gemma.final_logit_softcapping {
        // logits = cap * tanh(logits / cap)
        logits = b.node(
            "logit_softcap".into(),
            "activation",
            "Activation",
            vec![
                attr("kind", Value::Text("softcap".into())),
                attr("cap", Value::Float(cap)),
            ],
            vec![logits],
            vec![sequence_length, a.vocabulary_size],
            None,
        );
    }
    b.sink.finish_graph(&logits)
}

#[cfg(test)]
mod tests;
