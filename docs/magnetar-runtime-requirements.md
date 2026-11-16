# Gemma 4 Component — attentes envers le Runtime Magnetar

Destinataires : équipe Magnetar · Suivi : [astorise/Magnetar#187](https://github.com/astorise/Magnetar/issues/187)
Source : `astorise/Magnetar-component-Gemma`, branche `claude/charming-gates-7orqd1`
Références : `modeling_gemma4.py` de `transformers`, `config.json` des checkpoints `google/gemma-4-*`, convertisseur et `gguf-py` de llama.cpp.

Cette version intègre la revue du Runtime sur #187 : plus d'interface ni de monde nommés Gemma, un seul mécanisme de mise à l'échelle, aucun poids fabriqué par le loader, aucun facteur replié dans un poids potentiellement quantifié.

Le Component construit les graphes de tous les décodeurs texte Gemma 4 : dense (31B ; le 12B déclare `gemma4_unified`, non mappé par les loaders), MoE (26B-A4B), E2B/E4B (per-layer inputs, KV partagé). Tant que les points ci-dessous manquent, il est rejeté explicitement (échec de link, d'attribut ou de validation), sans repli silencieux.

## 1. WIT `1.6.0` : extension neutre de `model-config`

Additive, sans nouvelle interface ni nouveau monde : le Component utilise le monde existant `model-component-graph-producer`. Fichier : `wit/model-component-graph.wit`.

```wit
record layer-spec {
  sliding-window: option<u32>,        // none = tout le contexte
  head-dimension: u64,
  num-key-value-heads: u32,
  rope-theta: f64,
  partial-rotary-factor: option<f64>, // none = toute la tête tourne
  value-from-key: bool,               // pas de v_proj : V dérive de la projection K brute
  kv-source-layer: option<u32>,       // réutilise le K/V de cette couche antérieure
  intermediate-size: u64,
}
layer-layout: func() -> option<list<layer-spec>>;   // une entrée par couche

record architecture-options {
  final-logit-softcapping: option<f64>,
  per-layer-input-size: option<u32>,                 // none = aucun
}
options: func() -> architecture-options;
```

Le **Runtime dérive** chaque champ depuis le checkpoint ; le Component ne recalcule aucune règle par couche. Il valide seulement la cohérence : une entrée par couche, `kv-source-layer` strictement antérieure, elle-même productrice, avec la même largeur de tête et le même nombre de têtes KV, fenêtre non nulle, facteur de rotation dans `(0, 1]`.

Dérivations attendues pour Gemma 4 (`layer_types`, `per_layer_config`, `kv_sharing_modes`) :

| Champ | Source `config.json` |
|---|---|
| `sliding-window` | `sliding_window` pour `sliding_attention`, `none` pour `full_attention` |
| `head-dimension` | `head_dim` (sliding), `global_head_dim` (full) |
| `num-key-value-heads` | `num_key_value_heads` ; pour `full_attention`, `num_global_key_value_heads` **seulement si** `attention_k_eq_v` |
| `rope-theta`, `partial-rotary-factor` | `rope_parameters.<layer_type>` (`partial_rotary_factor` 0.25 en full) |
| `value-from-key` | `attention_k_eq_v` et couche full |
| `kv-source-layer` | pour les `num_kv_shared_layers` dernières couches : la dernière couche non partagée du même `layer_type` |
| `intermediate-size` | `intermediate_size`, doublé sur les couches partagées si `use_double_wide_mlp` |
| `final-logit-softcapping`, `per-layer-input-size` | `final_logit_softcapping`, `hidden_size_per_layer_input` |

`per-layer-input-size` est une `option<u32>` : `some(0)` est refusé, l'absence s'écrit `none`.

`architecture-config` : `head-dimension`, `num-key-value-heads`, `intermediate-size` ne servent plus qu'à titre indicatif ; `tie-word-embeddings = true`, `attention-bias = false`. `rope-scaling()` doit renvoyer `none`. `mixture-of-experts()` pour le 26B-A4B : `experts = 128`, `experts-per-token = 8`, `expert-intermediate-size = 704`, `normalize-top-k = true`, pas d'expert partagé (refusés sinon).

`layer-layout` à `none` : le Component refuse de construire (trap).

## 2. Opérateurs

Un seul opérateur nouveau (`activation`) ; le reste sont des attributs.

| Opérateur | Attribut / entrée | Sémantique |
|---|---|---|
| `embedding` | `scale` (float) | Sortie × `scale` : `sqrt(hidden_size)`, et `sqrt(per_layer_input_size)` pour les per-layer inputs. |
| `attention` | `scale` (float) | Facteur sur `QKᵀ` **à la place** de `1/sqrt(head_dim)` ; Gemma 4 passe `1.0`. |
| `attention` | `window_size` (existe) | Fenêtre glissante causale, à vérifier en prefill et en decode avec historique KV. |
| `rope` | `partial_rotary_factor` | RoPE « proportional », voir ci-dessous. |
| `gelu` | `approximate = "tanh"` | `gelu_pytorch_tanh`. |
| `rmsnorm` | `head_count`, `head_dimension` | Normalisation par tête, poids `[head_dimension]` partagé. |
| `rmsnorm` | `with_scale = false`, **une seule entrée** | Sans poids (`Gemma4RMSNorm(with_scale=False)`) : `v_norm` et la norme du routeur. Aucun vecteur de 1 n'est demandé au loader. |
| `moe-experts` | `activation = "gelu_tanh"` | Défaut : SiLU. |
| `moe-experts` | 6ᵉ entrée `per_expert_scale` `[E]` | Multiplie la sortie de chaque expert (équivalent à `router.per_expert_scale[idx]` après renormalisation). |
| `activation` | `kind = "softcap"`, `cap` | `cap * tanh(x / cap)` sur les logits (`cap = 30`). |
| `activation` | `kind = "scale"`, `factor` | Multiplication par une constante. `activation` garde **une seule entrée** pour tous les `kind`. |
| `mul` | aucun attribut | Le second opérande de forme `[1]` se diffuse d'après sa seule forme (règle à ajouter à la validation et aux noyaux) : `layer_scalar`. |

Remplacent la première version : `matmul.output_scale`, `add.output_scale` et `mul.broadcast = "scalar"` n'existent plus. `scale` sert à `hidden_size^-0.5` (projection de contexte) et `2^-0.5` (mélange token/contexte) des per-layer inputs ; `layer_scalar` est un `mul(h, layer_scalar)` (réponse de l'équipe Runtime sur #187).

### RoPE « proportional » (couches `full`)

Reproduit `_compute_proportional_rope_parameters`. Pour `head_dim = 512`, `partial_rotary_factor = 0.25` :
- `rope_angles = int(factor * head_dim // 2)` = 64 paires tournent ;
- `inv_freq[i] = 1 / base^(2i / head_dim)` avec la largeur **complète** comme dénominateur ;
- les autres paires ont une fréquence nulle (identité) ; appariement `half_split` sur la largeur complète ;
- absent ou `1.0` : RoPE standard.

(llama.cpp encode la même chose avec un tenseur `rope_freqs` : 1.0 pour les paires qui tournent, 1e30 pour les autres.)

### Graphe par couche (ordre)

`input_norm → q/k(/v) → q_norm, k_norm, v_norm(sans échelle) → RoPE → attention → o_proj → post_attn_norm → +x → [pre_ffn_norm → GeGLU] (+ branche experts en MoE) → post_ffn_norm → + → [per-layer input] → mul(layer_scalar)`.

MoE : la MLP dense et la branche experts tournent en parallèle sur le flux résiduel `h`. Branche experts : `router_norm(h)` (sans échelle) → `router_proj` → scores ; `pre_ffn_norm_2(h)` → `moe-experts(…, per_expert_scale)` → `post_ffn_norm_2`. Dense : `pre_ffn_norm → MLP → post_ffn_norm_1`. Somme, puis `post_ffn_norm`.

## 3. Loader

Aucun poids n'est fabriqué ; aucun facteur n'est replié dans un poids pouvant être quantifié. Seule dérivation : le facteur du routeur, replié dans `router_proj` (le routeur reste en flottant).

| Nom logique | HF `model.language_model.*` | GGUF `gemma4` |
|---|---|---|
| `token_embedding` ; `lm_head` (aliasé) | `embed_tokens.weight` | `token_embd.weight` |
| `final_norm` | `norm.weight` | `output_norm.weight` |
| `layers.N.input_norm` | `layers.N.input_layernorm` | `blk.N.attn_norm` |
| `layers.N.post_attn_norm` | `post_attention_layernorm` | `blk.N.post_attention_norm` |
| `layers.N.pre_ffn_norm` | `pre_feedforward_layernorm` | `blk.N.ffn_norm` |
| `layers.N.post_ffn_norm` | `post_feedforward_layernorm` | `blk.N.post_ffw_norm` |
| `layers.N.self_attn.{q,k,v,o}_proj` | `self_attn.{q,k,v,o}_proj` | `blk.N.attn_{q,k,v,output}` |
| `layers.N.self_attn.{q,k}_norm` | `self_attn.{q,k}_norm` | `blk.N.attn_{q,k}_norm` |
| `layers.N.mlp.{gate,up,down}_proj` | `mlp.{gate,up,down}_proj` | `blk.N.ffn_{gate,up,down}` |
| `layers.N.layer_scalar` | `layer_scalar` | `blk.N.layer_output_scale` |
| `layers.N.pre_ffn_norm_2` | `pre_feedforward_layernorm_2` | `blk.N.pre_ffw_norm_2` |
| `layers.N.post_ffn_norm_1`, `_2` | `post_feedforward_layernorm_1`, `_2` | `blk.N.post_ffw_norm_1`, `_2` |
| `layers.N.moe.router_proj` | `router.proj` × `router.scale` × `hidden_size^-0.5` | `blk.N.ffn_gate_inp` × `blk.N.ffn_gate_inp.scale` × `hidden_size^-0.5` |
| `layers.N.moe.experts_{gate,up}_proj` | moitiés de `experts.gate_up_proj` `[E, 2I, H]`, empilées `[H, E*I]` | `blk.N.ffn_gate_exps`, `ffn_up_exps` (ou `ffn_gate_up_exps` fusionné) |
| `layers.N.moe.experts_down_proj` | `experts.down_proj`, empilé `[I, E*H]` | `blk.N.ffn_down_exps` |
| `layers.N.moe.per_expert_scale` | `router.per_expert_scale` | `blk.N.ffn_down_exps.scale` |
| `layers.N.per_layer_embedding` | colonnes `[N*P, (N+1)*P)` de `embed_tokens_per_layer` | `per_layer_token_embd` (même tranche) |
| `layers.N.per_layer_model_projection` | tranche `[N*P, (N+1)*P)` de `per_layer_model_projection`, en `[H, P]` | `per_layer_model_proj` (même tranche) |
| `per_layer_projection_norm` | `per_layer_projection_norm` | `per_layer_proj_norm` |
| `layers.N.per_layer_input_gate` | `per_layer_input_gate` | `blk.N.inp_gate` |
| `layers.N.per_layer_projection` | `per_layer_projection` | `blk.N.proj` |
| `layers.N.post_per_layer_input_norm` | `post_per_layer_input_norm` | `blk.N.post_norm` |

Notes :
- Les noms GGUF viennent de `MODEL_ARCH.GEMMA4` dans `gguf-py/gguf/constants.py` et du convertisseur de llama.cpp, qui garde `router.scale` et `per_expert_scale` comme tenseurs **séparés** (`ffn_gate_inp.scale`, `ffn_down_exps.scale`) plutôt que de les replier. **Non vérifié** : les formes exactes et l'ordre des dimensions des tenseurs GGUF.
- Découpage des tenseurs per-layer : avec `P = 256`, les coupes tombent sur des multiples de 256 le long de l'axe contigu des lignes, donc alignées sur les blocs de quantification usuels (≤ 256 éléments, ex. Q4_K) ; les tranches de `per_layer_model_projection` sont des tranches de lignes. À confirmer pour chaque type de quantification supporté.
- Les couches qui réutilisent le KV n'ont ni `k_proj`, ni `v_proj`, ni `k_norm` : le Component ne les demande pas. Les couches full avec `value-from-key` n'ont pas de `v_proj`.

## 4. Travail Runtime identifié par la revue

- Géométrie KV par couche (le moteur suppose un `kv_head_count` et un `head_dimension` globaux).
- Couches à KV partagé : historique, commit et replay de decode suivent les écritures de la couche productrice. Un nœud `attention` consommateur lit les arêtes K/V du producteur et doit voir son historique complet ; seules les couches productrices allouent des ressources KV.
- Attention à fenêtre glissante dans les noyaux CUDA (Reference CPU seulement aujourd'hui).
- Sélection de lignes de logits quand un opérateur élémentaire (`softcap`) suit la tête LM.

## 5. Contraintes de segments (multi-Device)

Un segment doit contenir les producteurs de ses couches partagées (trap sinon). Avec per-layer inputs, un segment qui ne commence pas à la couche 0 déclare en plus l'entrée `token_ids` et recalcule les embeddings.

## 6. Critères d'acceptation proposés

1. Le Component link et s'instancie sous Wasmtime avec `graph-builder` et `model-config` en `1.6.0`. **Vérifié** sur `84093cb` (voir `docs/engine-smoke-test`).
2. Les graphes du 31B, du 26B-A4B et d'un E2B passent la validation Runtime sans attribut rejeté. **Vérifié** aux formes réelles (formes seules, sans poids) : 1314, 869 et 1094 nœuds.
3. Conformité Reference CPU, avec fixtures, de : `softcap`, `scale`, `rmsnorm` par tête et sans échelle, `rope` partiel, `attention` avec `scale` et `window_size`, `gelu` tanh, `moe-experts` avec `per_expert_scale` et `gelu_tanh`.
4. Logits comparés à `transformers`. **Fait sur de petits checkpoints aléatoires** de même format (`tools/qualification`), sur Magnetar `1f21bd7` sans patch : dense, KV partagé, per-layer inputs, MoE et « unifié » (12B), écart ≤ 5,5e-5, tokens identiques. **Non fait** : les checkpoints réels (31B, 26B-A4B, E2B).
5. Variantes quantifiées : même comparaison sur un export GGUF. **Fait sur de petits GGUF** (`Q4_0` pour les projections et les experts, `Q8_0` pour l'embedding et les tables per-layer), `main` à `dbe2f1c` : dense, KV partagé, per-layer inputs et MoE, écart ≤ 1,1e-4. **Non fait** : les GGUF QAT réels (`Q4_0`/`Q6_K`) et CUDA.

## 6 bis. Écarts du chemin « production » trouvés par la comparaison (corrigés)

Sur `84093cb`, un bundle Gemma 4 s'ingérait mais ne se chargeait pas : trois endroits supposaient l'inventaire de tenseurs de Qwen (`validate_first_native_tensor_inventory`, `first_native_expected_tensor_shape`, `weight_shapes_for_architecture_config`). Le Runtime les a corrigés (OpenSpec `validate-layered-artifacts-in-production`, branche `claude/charming-lamport-1b9m74`, pas encore fusionnée) : l'inventaire et les formes suivent le `LayerSpec` de chaque couche, et les formes données au graph-builder viennent de l'inventaire de l'artefact. La comparaison de `tools/qualification` a été rejouée sur cette branche **sans** le patch local, avec les mêmes résultats (3,481e-5 et 5,484e-5).

Convention MoE retenue par le Runtime, que ce Component suit : `layers.N.moe.router_proj`, `layers.N.moe.experts_{gate,up,down}_proj` et `layers.N.moe.per_expert_scale`. Les normes `pre_ffn_norm_2`, `post_ffn_norm_1` et `post_ffn_norm_2` gardent leurs noms.

## 7. Hors périmètre

- **Vision et audio** : opérateurs (patch embedding, RoPE 2-D, attention bidirectionnelle sur les tokens image, blocs conformer), injection d'embeddings aux positions image/audio et entrées non-token absents de `graph-builder`. À traiter dans `explore-multimodal-and-embedding-components`, en Components séparés.
- Gemma 1 à 3.

## 8. Livré côté Component

`src/plan.rs` : logique de graphe indépendante de l'hôte, 45 tests unitaires contre un `graph-builder` simulé (formes réelles du 31B, du 26B-A4B et d'un E2B). `src/component.rs` : pont WIT ; le binaire `wasm32-wasip2` n'importe que `graph-builder` et `model-config` (plus l'E/S standard wasip2 utilisée par les panics).
