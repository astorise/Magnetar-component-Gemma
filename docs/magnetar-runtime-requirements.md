# Gemma 4 Component — attentes envers le Runtime Magnetar

Destinataires : équipe Magnetar
Source : `astorise/Magnetar-component-Gemma`, branche `claude/charming-gates-7orqd1`
Référence d'architecture : `transformers` `modeling_gemma4.py` et `config.json` de `google/gemma-4-31B-it` / `gemma-4-12B-it`

## Contexte

Le Component Gemma 4 (texte dense) produit les graphes prefill/decode via `graph-builder`, `model-config` et un nouveau `gemma-config`. Le contrat WIT 1.5.0 et le vocabulaire d'opérateurs actuels ne permettent pas d'exprimer plusieurs particularités de Gemma 4. Le Component ne fait **aucun repli silencieux** : tant que les points ci-dessous manquent, il est rejeté explicitement (échec de link, d'attribut ou de validation).

Variantes ciblées : Gemma 4 dense (31B, 12B).
Variantes également construites (voir §5) : 26B-A4B (MoE), E2B/E4B (embeddings par couche, KV partagé).

## 1. WIT : nouvelle interface `gemma-config` (version 1.6.0)

Évolution **additive** : `graph-builder` et `model-config` sont inchangés. Seul le monde `gemma-model-component-producer` importe la nouvelle interface, donc les autres Components (Qwen, Llama) ne sont pas affectés. Le package passe en `1.6.0`, ce qui implique de ré-enregistrer les interfaces sous ce numéro côté hôte, comme pour les bumps précédents.

Fichier de référence : `wit/model-component-graph.wit` du dépôt Gemma.

Fonction : `gemma-config.architecture() -> gemma-architecture`. Le Runtime doit la remplir depuis le `text_config` du checkpoint, et **trapper plutôt qu'inventer** une valeur manquante.

| Champ WIT | Clé `config.json` | Valeur 31B |
|---|---|---|
| `layer-attention` (liste, `num-hidden-layers` entrées) | `layer_types` | 50 `sliding`, 10 `full` |
| `sliding-window` | `sliding_window` | 1024 |
| `global-head-dimension` | `global_head_dim` | 512 |
| `num-global-key-value-heads` | `num_global_key_value_heads` | 4 |
| `sliding-rope-theta` | `rope_parameters.sliding_attention.rope_theta` | 10 000 |
| `full-rope-theta` | `rope_parameters.full_attention.rope_theta` | 1 000 000 |
| `full-partial-rotary-factor` | `rope_parameters.full_attention.partial_rotary_factor` (1.0 si absent) | 0.25 |
| `attention-k-eq-v` | `attention_k_eq_v` | true |
| `final-logit-softcapping` | `final_logit_softcapping` | 30.0 |
| `per-layer-input-size` | `hidden_size_per_layer_input` | 0 |
| `kv-shared-layers` | `num_kv_shared_layers` | 0 |

Valeurs déjà portées par `architecture-config` (à renseigner avec la sémantique Gemma) :

- `head-dimension` = `head_dim` (256, couches sliding) ;
- `num-key-value-heads` = `num_key_value_heads` (couches sliding) ;
- `rope-theta` : non utilisé par ce Component, qui lit les bases dans `gemma-config` ;
- `tie-word-embeddings` = true, `attention-bias` = false ;
- `rope-scaling()` doit renvoyer `none`, et `mixture-of-experts()` doit renvoyer `none` pour le dense.

## 2. Opérateurs : attributs à accepter

Le Component n'invente aucun nouvel opérateur, sauf `activation` de type `softcap`. Il utilise des attributs que les schémas actuels (`magnetar-core/src/operator.rs`) n'acceptent pas encore. **Sémantique attendue** :

| Opérateur | Attribut | Sémantique |
|---|---|---|
| `embedding` | `scale` (float) | Sortie multipliée par `scale`. Le Component passe `sqrt(hidden_size)`. |
| `attention` | `scale` (float) | Facteur appliqué à `QKᵀ` **à la place** de `1/sqrt(head_dim)`. Gemma 4 passe `1.0` (Q et K sont normalisés). |
| `attention` | `window_size` (int, existe déjà) | Fenêtre glissante causale : une position voit les `window_size` dernières positions. Utilisé sur les couches `sliding` uniquement. À vérifier en prefill **et** en decode avec l'historique KV. |
| `rope` | `partial_rotary_factor` (float, optionnel) | Voir détail ci-dessous. |
| `gelu` | `approximate = "tanh"` (string) | GELU approximé par tanh (`gelu_pytorch_tanh`). |
| `rmsnorm` | `head_count`, `head_dimension` (int) | Normalisation **par tête** : le dernier axe `[seq, head_count * head_dimension]` est découpé en groupes de `head_dimension`, chacun normalisé indépendamment, le poids `[head_dimension]` est partagé entre têtes. Sans ces attributs : comportement actuel. |
| `mul` | `broadcast = "scalar"` (string) | Le second opérande a un seul élément, multiplié sur tout le premier. Sert au `layer_scalar` (poids de forme `[1]`). |
| `activation` | `kind = "softcap"`, `cap` (float) | `cap * tanh(x / cap)` élément par élément. Appliqué aux logits (`cap = 30`). `activation` exige déjà `kind`. |

### Détail : RoPE « proportional » (couches `full`)

Reproduit `_compute_proportional_rope_parameters` de transformers. Pour `head_dim = 512` et `partial_rotary_factor = 0.25` :

- `rope_angles = int(factor * head_dim // 2)` = 64 paires tournent ;
- fréquences : `inv_freq[i] = 1 / base^(2i / head_dim)` avec `head_dim` = **largeur complète** (512) comme dénominateur, pour `i < rope_angles` ;
- les `head_dim/2 - rope_angles` paires restantes ont une fréquence **nulle** (rotation identité) ;
- l'appariement reste `half_split` : `(i, i + head_dim/2)` sur la largeur complète ;
- absent ou `1.0` : RoPE standard sur toute la tête.

Les couches `sliding` utilisent un RoPE standard (base 10 000, tête de 256).

### Contraintes de forme à autoriser

- Largeurs de tête **variables selon la couche** (256 en sliding, 512 en full) et nombre de têtes KV variable (16 vs 4 pour le 31B) : les nœuds `attention`/`rope`/`matmul` portent déjà leurs dimensions en attributs et en `output-shape`, aucune hypothèse globale de `head_dimension` ne doit être appliquée par la validation du Runtime.
- Le KV cache doit allouer **par couche** avec la largeur et le nombre de têtes de cette couche (`kv-resource(layer, role)`). Il ne peut pas supposer une géométrie uniforme.
- Dans les couches `full` avec `attention-k-eq-v`, il n'y a pas de `v_proj` : V est la sortie brute de `k_proj` (avant norme et RoPE de K), passée dans un `rmsnorm` sans échelle. Le nœud V porte la `kv-involvement` ; le nœud K roped porte celle de K.

## 3. Loader / Model Artifact

Le Component demande des **noms logiques** via `weight-edge`, calqués sur Llama. Le loader doit les résoudre depuis les noms HF `model.language_model.*`.

| Nom logique | Tenseur du checkpoint |
|---|---|
| `token_embedding` | `model.language_model.embed_tokens.weight` |
| `final_norm` | `model.language_model.norm.weight` |
| `lm_head` | aliasé à `token_embedding` (`tie-word-embeddings`) |
| `layers.N.input_norm` | `...layers.N.input_layernorm.weight` |
| `layers.N.post_attn_norm` | `...layers.N.post_attention_layernorm.weight` |
| `layers.N.pre_ffn_norm` | `...layers.N.pre_feedforward_layernorm.weight` |
| `layers.N.post_ffn_norm` | `...layers.N.post_feedforward_layernorm.weight` |
| `layers.N.layer_scalar` | `...layers.N.layer_scalar` (forme `[1]`) |
| `layers.N.self_attn.{q,k,o}_proj` | `...self_attn.{q,k,o}_proj.weight` |
| `layers.N.self_attn.v_proj` | idem, **absent sur les couches full quand `attention_k_eq_v`** |
| `layers.N.self_attn.{q,k}_norm` | `...self_attn.{q,k}_norm.weight` |
| `layers.N.self_attn.v_norm` | **n'existe pas dans le checkpoint** |
| `layers.N.mlp.{gate,up,down}_proj` | `...mlp.{gate,up,down}_proj.weight` |

Point d'attention : `v_norm` correspond à `Gemma4RMSNorm(with_scale=False)`. Pour garder `rmsnorm` à deux entrées, le loader doit fournir un vecteur de 1 de taille `head_dimension` de la couche (256 ou 512) sous ce nom logique. Alternative acceptable : nous le signaler pour que le Component passe à `rmsnorm` à une entrée avec un attribut `with_scale = false`.

Note de normalisation : contrairement à Gemma 1-3, le poids RMSNorm de Gemma 4 est utilisé tel quel (`x * rsqrt(mean(x²)+eps) * weight`, pas `1 + weight`). Le loader ne doit pas appliquer de décalage.

## 4. Critères d'acceptation proposés

1. Le Component `gemma-model-component-producer` link et s'instancie sous Wasmtime avec `graph-builder`, `model-config` et `gemma-config` en 1.6.0.
2. Le graphe prefill du 31B passe la validation Runtime sans attribut rejeté, avec 60 couches : 50 sliding et 10 full.
3. Chaque opérateur du §2 a une conformité Reference CPU avec jeu de test (au minimum : `softcap`, `rmsnorm` par tête, `rope` partiel, `attention` avec `scale` et `window_size`).
4. Comparaison des logits avec `transformers` sur un prompt court (prefill puis quelques tokens de decode), sur un checkpoint réel, avec tolérance à définir. Ce test n'a **pas** été fait de notre côté : aucun poids n'a été téléchargé.
5. Mêmes comparaisons de logits pour un E2B/E4B (per-layer inputs, KV partagé) et pour le 26B-A4B (MoE).

## 5. Variantes MoE, per-layer inputs et KV partagé (ajout)

Le Component construit désormais aussi ces variantes. Prérequis supplémentaires, en plus des §1 à §3.

### WIT (`gemma-config`)

- `num-global-key-value-heads` devient `option<u32>` : `none` pour E2B/E4B (`num_global_key_value_heads: null`). Dans ce cas les couches `full` gardent `num-key-value-heads` (`per_layer_config` ne remplace le nombre de têtes KV que si `attention_k_eq_v`).
- Nouveau champ `double-wide-mlp-on-shared-layers` (`use_double_wide_mlp`) : les couches qui réutilisent le KV ont une MLP de largeur `2 * intermediate-size`.
- `per-layer-input-size` et `kv-shared-layers` sont maintenant **supportés** (plus de refus).
- `model-config.mixture-of-experts()` doit renvoyer pour le 26B-A4B : `experts = 128`, `experts-per-token = 8`, `expert-intermediate-size = 704`, `normalize-top-k = true`, `shared-expert-intermediate-size = none`. Le Component refuse un expert partagé et `normalize-top-k = false`.

### Opérateurs

- `moe-experts` existe déjà (5 entrées, `experts`, `top_k`, `normalize_top_k`). Il faut ajouter l'attribut `activation = "gelu_tanh"` (par défaut : SiLU).
- `matmul` et `add` : attribut optionnel `output_scale` (float), résultat multiplié par ce facteur. Utilisé pour `hidden_size^-0.5` (projection du contexte) et `2^-0.5` (mélange token/contexte) des per-layer inputs.
- `embedding.scale` (déjà demandé au §2) sert aussi à `sqrt(hidden_size_per_layer_input)`.

### Loader : poids dérivés

| Nom logique | Construction depuis le checkpoint |
|---|---|
| `layers.N.router_norm` | vecteur de 1 de taille `hidden_size` (`Gemma4RMSNorm(with_scale=False)`) |
| `layers.N.router_proj` | `router.proj.weight` avec `router.scale * hidden_size^-0.5` multiplié sur la dimension d'entrée (exact, la projection est linéaire) |
| `layers.N.experts.gate_proj`, `up_proj` | moitiés de `experts.gate_up_proj` `[E, 2I, H]`, empilées en `[H, E*I]` |
| `layers.N.experts.down_proj` | `experts.down_proj` empilé en `[I, E*H]`, avec `router.per_expert_scale[e]` multiplié sur le bloc de l'expert `e` (exact, car appliqué après la pondération linéaire) |
| `layers.N.post_ffn_norm_1`, `post_ffn_norm_2`, `pre_ffn_norm_2` | `post_feedforward_layernorm_1`, `post_feedforward_layernorm_2`, `pre_feedforward_layernorm_2` |
| `layers.N.per_layer_embedding` | colonnes `[N*P, (N+1)*P)` de `embed_tokens_per_layer.weight` (`[vocab_per_layer, L*P]`) |
| `layers.N.per_layer_model_projection` | tranche `[N*P, (N+1)*P)` de `per_layer_model_projection.weight`, transposée en `[H, P]` |
| `per_layer_projection_norm` | `per_layer_projection_norm.weight` (`[P]`), partagé par toutes les couches |
| `layers.N.per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm` | mêmes noms HF |

Les couches qui réutilisent le KV n'ont **pas** de `k_proj`, `v_proj`, `k_norm`, `v_norm` : le Component ne les demande pas.

### Contraintes d'exécution

- **KV partagé** : une couche consommatrice lit les arêtes K/V (après RoPE/norme) du producteur, la dernière couche non partagée du même type d'attention. Le Runtime doit accepter qu'un nœud `attention` consomme l'arête portant la `kv-involvement` d'une autre couche et lui présenter l'historique complet. Seules les couches productrices allouent des ressources KV.
- **Segments (multi-Device)** : un segment doit contenir les producteurs de ses couches partagées (sinon trap). Avec per-layer inputs, un segment qui ne commence pas à la couche 0 déclare en plus l'entrée `token_ids` et recalcule les embeddings.
- **MoE** : `moe-experts` reçoit en entrée la sortie de `pre_ffn_norm_2` et les scores du routeur (`softmax` puis top-k puis renormalisation, côté Runtime) ; les poids empilés suivent la convention de `execute-and-cache-moe-experts`.

## 5 bis. Hors périmètre

- **Tours vision et audio** : demandent des opérateurs (patch embedding, RoPE 2-D, attention bidirectionnelle sur les tokens image, blocs conformer audio), l'injection d'embeddings aux positions des tokens image/audio, et des entrées non-token que `graph-builder` ne modélise pas. À traiter dans `explore-multimodal-and-embedding-components` avec leurs propres Components ; ce Component reste le décodeur texte.
- Attention bidirectionnelle sur les tokens image (`use_bidirectional_attention = "vision"`) : sans objet en texte seul.

## 6. Ce qui est livré côté Component

- Logique de graphe indépendante de l'hôte : `src/plan.rs`, 40 tests unitaires contre un `graph-builder` simulé, dont les formes réelles du 31B, du 26B-A4B et d'un E2B.
- Pont WIT : `src/component.rs`. Le binaire `wasm32-wasip2` se construit et n'importe que les trois Capabilities Magnetar, plus l'E/S standard wasip2 utilisée par les panics.
- Détail et commandes de vérification : `README.md` du dépôt.
