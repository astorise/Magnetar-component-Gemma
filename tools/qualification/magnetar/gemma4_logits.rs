//! Scratch: Gemma 4 Component vs `transformers` logits on a tiny bundle.
//!
//! GEMMA_COMPONENT  path of the built `.component.wasm`
//! GEMMA_BUNDLE     bundle dir made by make_tiny_gemma4_bundle.py
//! GEMMA_REFERENCE  json made by gemma4_reference_logits.py
//! GEMMA_MAX_ABS    optional tolerance (default 1e-3)

use magnetar_inference_qualification::numerics::{
    LogitRecorder, LogitReference, NumericalProfile, compare_logit_references,
};
use magnetar_loader_huggingface::{HuggingFaceIngestor, HuggingFaceTokenizer};
use magnetar_runtime::model::ModelTrustStore;
use magnetar_runtime::production_model_ingestion::{
    ProductionModelArtifactIngestor, ProductionModelSource,
};
use magnetar_runtime::tokenizer::Tokenizer;
use magnetar_runtime::{
    ComponentDigest, ComponentTrustStore, GenerationParameters, ModelArtifactSource,
    ProductionGenerationRequest, ProductionLoadedModel, PromptInput, ReferenceCpuProvider,
    StopConditions, production_model_fixture, register_inference_component_artifact,
};
use std::{fs, ops::ControlFlow, path::PathBuf, sync::Arc};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn manifest(digest: &str, families: &[&str]) -> Vec<u8> {
    let families: String = families.iter().map(|f| format!("    - {f}\n")).collect();
    format!(
        r#"schema: magnetar-component-artifact
schema_version: 1
artifact:
  kind: component
  digest:
    algorithm: sha256
    value: "{digest}"
component:
  name: "magnetar.gemma.real"
  version: "0.1.0"
  description: "Gemma 4 text decoder Model Component"
  role: "gemma-model-component"
runtime:
  magnetar:
    min_version: "0.1.2"
compatibility:
  architecture_families:
{families}  artifact_formats:
    - huggingface
    - gguf
wit:
  imports:
    - package: "magnetar:model-component-graph"
      interface: "graph-builder"
      version: "1.6.0"
    - package: "magnetar:model-component-graph"
      interface: "model-config"
      version: "1.6.0"
  exports: []
capabilities:
  requires: []
authority:
  requires: []
engine:
  profile: "native"
  features:
    - component-model
    - resource-limits
publisher:
  id: "local-dev"
  name: "Local Development"
source:
  kind: "local"
  uri: "./gemma.component.wasm"
signatures: []
"#
    )
    .into_bytes()
}

#[test]
fn gemma4_logits_match_transformers() {
    let component_bytes = fs::read(env("GEMMA_COMPONENT")).expect("component readable");
    let digest = ComponentDigest::sha256(&component_bytes);
    let trust = ComponentTrustStore::default().trust_digest(&digest.value);
    let registered = register_inference_component_artifact(
        component_bytes,
        manifest(&digest.value, &["gemma", "gemma4", "gemma4_text", "gemma4_unified"]),
        &trust,
    )
    .unwrap_or_else(|e| panic!("component registers: {e}"));

    let bundle = PathBuf::from(env("GEMMA_BUNDLE"));
    let gguf = std::env::var("GEMMA_FORMAT").is_ok_and(|f| f == "gguf");
    let source = ProductionModelSource::authorized_local_bundle(
        ModelArtifactSource::HuggingFace("tiny-gemma4-bundle".into()),
        bundle.clone(),
    );
    let corpus: LogitReference =
        serde_json::from_slice(&fs::read(env("GEMMA_REFERENCE")).unwrap()).unwrap();
    let n_new = corpus.steps.len();

    // The same production path for both formats: a Hugging Face bundle
    // (`config.json`, `tokenizer.json`, safetensors) or a GGUF file.
    let (ingested, tokenizer_metadata, tokenizer): (_, _, Arc<dyn Tokenizer + Send + Sync>) =
        if gguf {
            let ingested = magnetar_loader_gguf::GgufIngestor::new()
                .ingest(&source)
                .unwrap_or_else(|e| panic!("GGUF ingests: {e}"));
            let bytes = fs::read(bundle.join("model.gguf")).unwrap();
            let artifact = magnetar_format_gguf::parse(&bytes).unwrap();
            let t = magnetar_loader_gguf::GgufTokenizer::from_gguf_metadata(
                &artifact.metadata,
                "tiny-gemma4",
                Some(64),
            )
            .unwrap();
            let metadata = t.metadata().clone();
            (ingested, metadata, Arc::new(t))
        } else {
            let ingested = HuggingFaceIngestor::new()
                .ingest(&source)
                .unwrap_or_else(|e| panic!("bundle ingests: {e}"));
            let tokenizer_bytes = fs::read(bundle.join("tokenizer.json")).unwrap();
            let t = HuggingFaceTokenizer::from_bytes(&tokenizer_bytes, None, "tiny-gemma4", Some(64))
                .unwrap();
            let metadata = t.metadata().clone();
            (ingested, metadata, Arc::new(t))
        };
    eprintln!(
        "ingested family = {:?}, layers = {}",
        ingested.manifest.architecture.family,
        ingested.manifest.architecture_config.as_ref().map(|c| c.num_hidden_layers).unwrap_or(0)
    );
    let fixture =
        production_model_fixture(ingested.manifest.clone(), tokenizer_metadata, tokenizer).unwrap();
    let trust_store =
        ModelTrustStore::default().trust_digest(ingested.manifest.id.digest.value.clone());

    let mut model = ProductionLoadedModel::load_with_component(
        fixture,
        ingested.payload_source.as_ref(),
        trust_store,
        Arc::new(ReferenceCpuProvider::new()),
        Some(registered.digest),
    )
    .unwrap_or_else(|e| panic!("model loads with the Gemma component: {e}"));

    let mut recorder = LogitRecorder::default();
    let outcome = model
        .generate_streaming_with_logits(
            ProductionGenerationRequest {
                prompt: PromptInput::TokenIds(corpus.prompt_token_ids.clone()),
                parameters: GenerationParameters::greedy(),
                stop_conditions: StopConditions::default(),
                max_new_tokens: Some(n_new),
                max_generation_millis: None,
            },
            None,
            &mut |_| ControlFlow::Continue(()),
            &mut |logits| recorder.observe(logits),
        )
        .unwrap_or_else(|e| panic!("generation runs: {e}"));
    let candidate = recorder.finish(
        corpus.checkpoint.clone(),
        "magnetar-reference-cpu",
        "f32",
        corpus.prompt_token_ids.clone(),
        outcome.result.output.generated_token_ids.clone(),
    );

    let tolerance: f32 = std::env::var("GEMMA_MAX_ABS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    let profile = NumericalProfile {
        name: "gemma4-f32-vs-transformers".into(),
        dtype: "f32".into(),
        max_abs_logit_difference: tolerance,
        require_identical_tokens: true,
        compared_top_k: 8,
    };
    let comparison = compare_logit_references(&corpus, &candidate, &profile);
    eprintln!("reference tokens : {:?}", corpus.generated_token_ids);
    eprintln!("candidate tokens : {:?}", candidate.generated_token_ids);
    for s in &comparison.steps {
        eprintln!(
            "step {:2}: ref {} cand {} max|d| {:.3e} worst {:?}",
            s.step, s.reference_token_id, s.candidate_token_id, s.max_abs_difference, s.worst_token_id
        );
    }
    eprintln!("max |logit diff| overall: {:.3e}", comparison.max_abs_difference);
    assert!(comparison.passed(), "{:#?}", comparison.failures);
}
