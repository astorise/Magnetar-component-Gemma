//! Gemma 4 dense decoder graph, expressed against a [`GraphSink`].
//!
//! This module has no WIT or host dependency: `lib.rs` adapts the
//! Runtime-owned `graph-builder` Capability to [`GraphSink`], and the tests
//! below drive the same code with a recording sink. Everything here is
//! derived from the checkpoint's `config.json` values carried by
//! `model-config`; nothing is a compiled-in dimension.
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
    /// `model-config.mixture-of-experts`; `Some` for Gemma 4 26B-A4B.
    pub moe: Option<Moe>,
}

/// `model-config.moe-config`, the fields this family reads.
#[derive(Clone, Debug)]
pub struct Moe {
    pub experts: u32,
    pub experts_per_token: u32,
    pub expert_intermediate_size: u64,
    pub normalize_top_k: bool,
    /// An always-active shared expert is declared (Gemma 4 has none: its
    /// dense MLP runs *beside* the routed experts instead).
    pub has_shared_expert: bool,
}

/// `model-config.layer-spec`: one decoder layer's attention geometry. The
/// Runtime derives it from the checkpoint; nothing here is Gemma-specific.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerSpec {
    /// Look-back window in tokens; `None` attends over the whole context.
    pub sliding_window: Option<u32>,
    pub head_dimension: u64,
    pub num_key_value_heads: u32,
    pub rope_theta: f64,
    /// Fraction of each head that rotates; `None` rotates all of it.
    pub partial_rotary_factor: Option<f64>,
    /// No V projection: V is K's raw projection.
    pub value_from_key: bool,
    /// Reuse the K/V of this earlier layer instead of computing them.
    pub kv_source_layer: Option<u32>,
    /// Dense feed-forward width of this layer.
    pub intermediate_size: u64,
}

/// `model-config.architecture-options`.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// `cap * tanh(logits / cap)` on the logits.
    pub final_logit_softcapping: Option<f64>,
    /// Width of the per-layer inputs, `None` for none.
    pub per_layer_input_size: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: Architecture,
    /// One entry per decoder layer.
    pub layers: Vec<LayerSpec>,
    pub options: Options,
}

/// Where a layer's K/V come from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KvSource {
    /// The layer projects, normalizes and caches its own K/V.
    Own,
    /// The layer attends over the K/V of an earlier layer and has no K/V
    /// weights.
    Reuse { producer: u64 },
}

/// Geometry of one decoder layer.
#[derive(Clone, Debug, PartialEq)]
struct Layer {
    index: u64,
    kv_source: KvSource,
    intermediate_size: u64,
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
    /// graph for. The caller traps on `Err`: building a wrong graph would
    /// silently produce wrong logits.
    pub fn validate(&self) -> Result<(), String> {
        let a = &self.arch;
        if a.has_rope_scaling {
            return Err(
                "checkpoint declares RoPE frequency scaling, which Gemma 4 does not use".into(),
            );
        }
        if a.attention_bias {
            return Err("attention_bias = true is not a Gemma 4 configuration".into());
        }
        if self.layers.len() != a.num_hidden_layers as usize {
            return Err(format!(
                "layer layout has {} entries for {} decoder layers",
                self.layers.len(),
                a.num_hidden_layers
            ));
        }
        for (index, l) in self.layers.iter().enumerate() {
            if l.sliding_window == Some(0) {
                return Err(format!("layer {index}: sliding window of 0 tokens"));
            }
            if l.head_dimension == 0 || l.intermediate_size == 0 {
                return Err(format!("layer {index}: zero head or feed-forward width"));
            }
            let kv = l.num_key_value_heads;
            if kv == 0 || !a.num_attention_heads.is_multiple_of(kv) {
                return Err(format!(
                    "layer {index}: {kv} KV heads do not divide {} attention heads",
                    a.num_attention_heads
                ));
            }
            if let Some(f) = l.partial_rotary_factor
                && !(f > 0.0 && f <= 1.0)
            {
                return Err(format!(
                    "layer {index}: partial rotary factor {f} is outside (0, 1]"
                ));
            }
            if let Some(source) = l.kv_source_layer {
                let source = source as usize;
                let Some(producer) = self.layers.get(source).filter(|_| source < index) else {
                    return Err(format!(
                        "layer {index} reuses K/V of layer {source}, which is not an earlier layer"
                    ));
                };
                if producer.kv_source_layer.is_some()
                    || producer.head_dimension != l.head_dimension
                    || producer.num_key_value_heads != l.num_key_value_heads
                {
                    return Err(format!(
                        "layer {index} reuses K/V of layer {source}, which does not produce \
                         K/V of the same shape"
                    ));
                }
            }
        }
        if self.options.per_layer_input_size == Some(0) {
            return Err("per-layer input width of 0 must be expressed as none".into());
        }
        if let Some(moe) = &a.moe {
            if moe.has_shared_expert {
                return Err("a shared expert is not part of Gemma 4's MoE layer".into());
            }
            if !moe.normalize_top_k {
                return Err("Gemma 4 renormalizes the chosen experts' probabilities; \
                     normalize-top-k = false is not a Gemma 4 router"
                    .into());
            }
            if moe.experts == 0
                || moe.experts_per_token == 0
                || moe.experts_per_token > moe.experts
                || moe.expert_intermediate_size == 0
            {
                return Err(format!(
                    "inconsistent mixture of experts: {} experts, top-{}, intermediate {}",
                    moe.experts, moe.experts_per_token, moe.expert_intermediate_size
                ));
            }
        }
        Ok(())
    }

    fn layer(&self, index: u64) -> Layer {
        let l = &self.layers[index as usize];
        Layer {
            index,
            kv_source: match l.kv_source_layer {
                Some(producer) => KvSource::Reuse {
                    producer: producer as u64,
                },
                None => KvSource::Own,
            },
            intermediate_size: l.intermediate_size,
            head_dimension: l.head_dimension,
            kv_heads: l.num_key_value_heads as u64,
            window: l.sliding_window.map(u64::from),
            rope_theta: l.rope_theta,
            partial_rotary_factor: l.partial_rotary_factor.filter(|f| *f < 1.0),
            k_eq_v: l.value_from_key,
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
/// Every name is a checkpoint tensor, possibly re-laid-out by the loader
/// (stacked experts, per-layer slices, `router_proj` with the router
/// `scale * hidden_size^-0.5` folded in, as routers stay in float). The two
/// scale-less norms (`v_norm`, the router's) have no tensor at all and are
/// expressed as `rmsnorm` with `with_scale = false`; `per_expert_scale` stays
/// a separate input of `moe-experts`, so quantized expert weights are never
/// rewritten.
pub fn weight_name(layer: u64, suffix: &str) -> String {
    match suffix {
        "input_norm"
        | "post_attn_norm"
        | "pre_ffn_norm"
        | "post_ffn_norm"
        | "layer_scalar"
        // MoE layers (26B-A4B).
        | "pre_ffn_norm_2"
        | "post_ffn_norm_1"
        | "post_ffn_norm_2"
        | "router_proj"
        // Per-layer inputs (E2B/E4B).
        | "per_layer_input_gate"
        | "per_layer_projection"
        | "post_per_layer_input_norm"
        | "per_layer_embedding"
        | "per_layer_model_projection" => format!("layers.{layer}.{suffix}"),
        "experts_gate_proj"
        | "experts_up_proj"
        | "experts_down_proj"
        | "experts_per_expert_scale" => {
            format!("layers.{layer}.experts.{}", &suffix["experts_".len()..])
        }
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "q_norm" | "k_norm" => {
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
    /// K (after RoPE) and V (after its norm) edges of each layer built so
    /// far, for later layers that reuse them.
    produced_kv: std::collections::HashMap<u64, (String, String)>,
    /// `token_ids` input and scaled token embeddings, kept when per-layer
    /// inputs need them.
    token_ids: Option<String>,
    embeddings: Option<String>,
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

    /// The single constant-scaling mechanism: `activation{kind = "scale",
    /// factor}`. (`layer_scalar` is a weight, so it is a plain `mul`.)
    fn scale(&mut self, id: String, x: &str, factor: f64, width: u64) -> String {
        self.node(
            id,
            "activation",
            "Activation",
            vec![
                attr("kind", Value::Text("scale".into())),
                attr("factor", Value::Float(factor)),
            ],
            vec![x.into()],
            vec![self.seq, width],
            None,
        )
    }

    fn gelu(&mut self, id: String, x: &str, width: u64) -> String {
        self.node(
            id,
            "gelu",
            "Activation",
            // `hidden_activation = "gelu_pytorch_tanh"`.
            vec![
                attr("approximate", Value::Text("tanh".into())),
                tensor_parallel("local"),
            ],
            vec![x.into()],
            vec![self.seq, width],
            None,
        )
    }

    /// The K and V edges a layer attends over: its own, built here, or those
    /// of the earlier layer it shares them with.
    fn key_values(&mut self, layer: &Layer, normed: &str) -> (String, String) {
        let p = format!("layer{}", layer.index);
        let hd = layer.head_dimension;
        let kv_dim = layer.kv_heads * hd;
        if let KvSource::Reuse { producer } = layer.kv_source {
            return self.produced_kv.get(&producer).cloned().unwrap_or_else(|| {
                panic!(
                    "gemma: layer {} reuses the K/V of layer {producer}, which is not in this \
                     graph segment",
                    layer.index
                )
            });
        }
        let k_raw = self.matmul(
            format!("{p}.k_proj"),
            normed,
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
                normed,
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
        let v = self.node(
            format!("{p}.v_norm"),
            "rmsnorm",
            "Normalization",
            vec![
                self.eps(),
                attr("with_scale", Value::Bool(false)),
                attr("head_count", Value::Int(layer.kv_heads as i64)),
                attr("head_dimension", Value::Int(hd as i64)),
            ],
            vec![v_raw],
            vec![self.seq, kv_dim],
            Some(v_kv),
        );
        self.produced_kv.insert(layer.index, (k.clone(), v.clone()));
        (k, v)
    }

    /// The dense GeGLU MLP over `m`, ending in its `down_proj` output.
    fn dense_mlp(&mut self, layer: &Layer, m: &str) -> String {
        let p = format!("layer{}", layer.index);
        let (inter, hidden) = (layer.intermediate_size, self.cfg.arch.hidden_size);
        let gate = self.matmul(
            format!("{p}.gate_proj"),
            m,
            &weight_name(layer.index, "gate_proj"),
            inter,
            Some("column"),
        );
        let up = self.matmul(
            format!("{p}.up_proj"),
            m,
            &weight_name(layer.index, "up_proj"),
            inter,
            Some("column"),
        );
        let act = self.gelu(format!("{p}.gelu"), &gate, inter);
        let mixed = self.node(
            format!("{p}.mul"),
            "mul",
            "Tensor",
            vec![tensor_parallel("local")],
            vec![act, up],
            vec![self.seq, inter],
            None,
        );
        self.matmul(
            format!("{p}.down_proj"),
            &mixed,
            &weight_name(layer.index, "down_proj"),
            hidden,
            Some("row"),
        )
    }

    /// The feed-forward block's normalized output for residual stream `h`:
    /// dense (`post_ffn_norm(mlp(pre_ffn_norm(h)))`) or, for Gemma 4
    /// mixture-of-experts layers, the dense MLP and the routed experts side
    /// by side, each normalized, summed and normalized again.
    fn feed_forward(&mut self, layer: &Layer, h: &str) -> String {
        let p = format!("layer{}", layer.index);
        let m = self.hidden_norm(
            format!("{p}.pre_ffn_norm"),
            h,
            &weight_name(layer.index, "pre_ffn_norm"),
        );
        let down = self.dense_mlp(layer, &m);
        let Some(moe) = self.cfg.arch.moe.clone() else {
            return self.hidden_norm(
                format!("{p}.post_ffn_norm"),
                &down,
                &weight_name(layer.index, "post_ffn_norm"),
            );
        };
        let hidden = self.cfg.arch.hidden_size;
        let dense = self.hidden_norm(
            format!("{p}.post_ffn_norm_1"),
            &down,
            &weight_name(layer.index, "post_ffn_norm_1"),
        );
        // Router: score the *un-normalized* residual stream through a
        // scale-less RMSNorm and a projection into which the loader folds
        // `router.scale * hidden_size^-0.5`.
        let routed_in = self.node(
            format!("{p}.router_norm"),
            "rmsnorm",
            "Normalization",
            vec![self.eps(), attr("with_scale", Value::Bool(false))],
            vec![h.into()],
            vec![self.seq, hidden],
            None,
        );
        let scores = self.matmul(
            format!("{p}.router"),
            &routed_in,
            &weight_name(layer.index, "router_proj"),
            moe.experts as u64,
            None,
        );
        let experts_in = self.hidden_norm(
            format!("{p}.pre_ffn_norm_2"),
            h,
            &weight_name(layer.index, "pre_ffn_norm_2"),
        );
        let gate = self
            .sink
            .weight_edge(&weight_name(layer.index, "experts_gate_proj"));
        let up = self
            .sink
            .weight_edge(&weight_name(layer.index, "experts_up_proj"));
        let down_w = self
            .sink
            .weight_edge(&weight_name(layer.index, "experts_down_proj"));
        let per_expert = self
            .sink
            .weight_edge(&weight_name(layer.index, "experts_per_expert_scale"));
        let routed = self.node(
            format!("{p}.experts"),
            "moe-experts",
            "LinearAlgebra",
            vec![
                attr("experts", Value::Int(moe.experts as i64)),
                attr("top_k", Value::Int(moe.experts_per_token as i64)),
                attr("normalize_top_k", Value::Bool(true)),
                attr("activation", Value::Text("gelu_tanh".into())),
                tensor_parallel("experts"),
            ],
            vec![experts_in, scores, gate, up, down_w, per_expert],
            vec![self.seq, hidden],
            None,
        );
        let routed = self.hidden_norm(
            format!("{p}.post_ffn_norm_2"),
            &routed,
            &weight_name(layer.index, "post_ffn_norm_2"),
        );
        let both = self.node(
            format!("{p}.moe_sum"),
            "add",
            "Tensor",
            vec![],
            vec![dense, routed],
            vec![self.seq, hidden],
            None,
        );
        self.hidden_norm(
            format!("{p}.post_ffn_norm"),
            &both,
            &weight_name(layer.index, "post_ffn_norm"),
        )
    }

    /// This layer's slice of the per-layer inputs:
    /// `(norm(embeddings @ projection_i * hidden^-0.5) + embed_i(token_ids) * sqrt(P)) * 2^-0.5`.
    fn per_layer_input(&mut self, layer: u64) -> String {
        let p = format!("layer{layer}");
        let width = self.cfg.options.per_layer_input_size.unwrap_or_default() as u64;
        let hidden = self.cfg.arch.hidden_size as f64;
        let token_ids = self
            .token_ids
            .clone()
            .expect("per-layer inputs need token_ids");
        let embeddings = self
            .embeddings
            .clone()
            .expect("per-layer inputs need embeddings");
        let table = self
            .sink
            .weight_edge(&weight_name(layer, "per_layer_embedding"));
        let token_part = self.node(
            format!("{p}.ple_token"),
            "embedding",
            "Tensor",
            vec![attr("scale", Value::Float((width as f64).sqrt()))],
            vec![token_ids, table],
            vec![self.seq, width],
            None,
        );
        let context = self.matmul(
            format!("{p}.ple_context"),
            &embeddings,
            &weight_name(layer, "per_layer_model_projection"),
            width,
            None,
        );
        let context = self.scale(
            format!("{p}.ple_context_scale"),
            &context,
            hidden.powf(-0.5),
            width,
        );
        let norm_w = self.sink.weight_edge("per_layer_projection_norm");
        let context = self.node(
            format!("{p}.ple_context_norm"),
            "rmsnorm",
            "Normalization",
            vec![self.eps()],
            vec![context, norm_w],
            vec![self.seq, width],
            None,
        );
        let sum = self.node(
            format!("{p}.ple_sum"),
            "add",
            "Tensor",
            vec![],
            vec![context, token_part],
            vec![self.seq, width],
            None,
        );
        self.scale(format!("{p}.ple_mix"), &sum, 0.5f64.sqrt(), width)
    }

    /// Gemma 4 E2B/E4B: gate the hidden state, modulate it with the layer's
    /// per-layer input, project back and add it as a normalized residual.
    fn per_layer_tail(&mut self, layer: u64, x: &str) -> String {
        let p = format!("layer{layer}");
        let width = self.cfg.options.per_layer_input_size.unwrap_or_default() as u64;
        let hidden = self.cfg.arch.hidden_size;
        let input = self.per_layer_input(layer);
        let gated = self.matmul(
            format!("{p}.ple_gate"),
            x,
            &weight_name(layer, "per_layer_input_gate"),
            width,
            None,
        );
        let gated = self.gelu(format!("{p}.ple_gelu"), &gated, width);
        let modulated = self.node(
            format!("{p}.ple_mul"),
            "mul",
            "Tensor",
            vec![],
            vec![gated, input],
            vec![self.seq, width],
            None,
        );
        let projected = self.matmul(
            format!("{p}.ple_projection"),
            &modulated,
            &weight_name(layer, "per_layer_projection"),
            hidden,
            None,
        );
        let projected = self.hidden_norm(
            format!("{p}.ple_post_norm"),
            &projected,
            &weight_name(layer, "post_per_layer_input_norm"),
        );
        self.residual(format!("{p}.ple_residual"), x, &projected)
    }

    fn layer(&mut self, layer: &Layer, x: &str) -> String {
        let a = self.cfg.arch.clone();
        let p = format!("layer{}", layer.index);
        let heads = a.num_attention_heads as u64;
        let hd = layer.head_dimension;
        let q_dim = heads * hd;

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
        let (k, v) = self.key_values(layer, &normed);

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

        // --- feed-forward --------------------------------------------------
        let ffn = self.feed_forward(layer, &h);
        let mut out = self.residual(format!("{p}.residual2"), &h, &ffn);

        // --- per-layer input (E2B/E4B) -------------------------------------
        if self.cfg.options.per_layer_input_size.is_some() {
            out = self.per_layer_tail(layer.index, &out);
        }

        // --- per-layer output scalar (`hidden_states *= layer_scalar`) -----
        let scalar = self
            .sink
            .weight_edge(&weight_name(layer.index, "layer_scalar"));
        // The `[1]`-shaped weight broadcasts from its shape alone.
        self.node(
            format!("{p}.layer_scalar"),
            "mul",
            "Tensor",
            vec![],
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
        produced_kv: Default::default(),
        token_ids: None,
        embeddings: None,
    };
    let per_layer_inputs = cfg.options.per_layer_input_size.is_some();
    for index in start_layer as u64..end_layer as u64 {
        if let KvSource::Reuse { producer } = cfg.layer(index).kv_source {
            assert!(
                producer >= start_layer as u64,
                "gemma: layer {index} reuses the K/V of layer {producer}, which is before the \
                 segment start {start_layer}; a segment must contain the producers of its \
                 shared-KV layers"
            );
        }
    }

    // Token embeddings, scaled by sqrt(hidden_size). A segment that does not
    // start at layer 0 only needs them again for per-layer inputs.
    let embedding = |b: &mut Builder<'_, S>, ids: String, id: &str| {
        let table = b.sink.weight_edge("token_embedding");
        b.node(
            id.into(),
            "embedding",
            "Tensor",
            vec![attr("scale", Value::Float((a.hidden_size as f64).sqrt()))],
            vec![ids, table],
            vec![sequence_length, a.hidden_size],
            None,
        )
    };
    let mut hidden = if start_layer == 0 {
        let ids = b.sink.declare_input("token_ids", &[sequence_length]);
        let embedded = embedding(&mut b, ids.clone(), "embedding");
        b.token_ids = Some(ids);
        b.embeddings = Some(embedded.clone());
        embedded
    } else {
        let incoming = b
            .sink
            .declare_input("hidden_states_in", &[sequence_length, a.hidden_size]);
        if per_layer_inputs && start_layer < end_layer {
            let ids = b.sink.declare_input("token_ids", &[sequence_length]);
            let embedded = embedding(&mut b, ids.clone(), "embedding_per_layer_inputs");
            b.token_ids = Some(ids);
            b.embeddings = Some(embedded);
        }
        incoming
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
    if let Some(cap) = cfg.options.final_logit_softcapping {
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
