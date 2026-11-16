"""Reference logits for a Gemma 4 text checkpoint, from `transformers`.

Loads a bundle laid out like `google/gemma-4-*-it` (`config.json` with a nested
`text_config`, `model.language_model.*` tensors), runs the canonical
`Gemma4ForCausalLM` in float32 and greedily generates `n_new` tokens,
recomputing the whole sequence at every step (no KV cache), so the reference
shares nothing with Magnetar's prefill/decode path.

Writes a `magnetar.inference-qualification.logit-reference/v1` document (see
`tools/inference-qualification/src/numerics.rs` in Magnetar), the format
Magnetar's qualification tests compare candidates against.

Usage:
  python3 gemma4_reference_logits.py <bundle_dir> <n_new> <out.json> <prompt ids...>
"""
import hashlib
import json
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

SNAPSHOT_TOP_K = 32
PREFIX = "model.language_model."


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(step, logits, sampled):
    order = sorted(range(len(logits)), key=lambda t: (-float(logits[t]), t))[:SNAPSHOT_TOP_K]
    return {
        "step": step,
        "sampled_token_id": sampled,
        "vocab_size": int(len(logits)),
        "top": [{"token_id": int(t), "logit": float(logits[t])} for t in order],
        "full_digest": "sha256:" + hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest(),
        "non_finite_count": int(np.count_nonzero(~np.isfinite(logits))),
    }


def load_model(bundle):
    cfg = json.load(open(f"{bundle}/config.json"))
    text = dict(cfg.get("text_config", cfg))
    text.pop("dtype", None)
    # `gemma4_unified_text` is the same decoder under another model type.
    text["model_type"] = "gemma4_text"
    model = Gemma4ForCausalLM(Gemma4TextConfig(**text)).float().eval()
    tensors = load_file(f"{bundle}/model.safetensors")
    state = {}
    for name, value in tensors.items():
        if not name.startswith(PREFIX):
            continue  # vision / audio towers are not part of the text decoder
        state["model." + name[len(PREFIX) :]] = value.float()
    # Tied output head: `lm_head.weight` is the embedding table.
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    return model


def main():
    bundle, n_new, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    prompt = [int(t) for t in sys.argv[4:]]
    model = load_model(bundle)
    generated, steps = [], []
    with torch.no_grad():
        for step in range(n_new):
            ids = torch.tensor([prompt + generated], dtype=torch.long)
            # An explicit index keeps `transformers` from slicing the sequence
            # before its KV-shared layers (an optimisation that assumes
            # per-layer inputs exist); every position is still computed.
            keep = torch.tensor([ids.shape[1] - 1])
            logits = model(input_ids=ids, logits_to_keep=keep).logits[0, -1].float().numpy()
            sampled = int(np.argmax(logits))
            steps.append(snapshot(step, logits, sampled))
            generated.append(sampled)
            print(f"step {step}: token {sampled} logit {logits[sampled]:.4f}", flush=True)
    weights = sha256_file(f"{bundle}/model.safetensors")
    corpus = {
        "schema_version": "magnetar.inference-qualification.logit-reference/v1",
        "checkpoint": {
            "name": "tiny-gemma4-bundle",
            "revision": None,
            "weights_digest": "sha256:" + weights,
            "digest_verified": True,
        },
        "provider": "transformers-f32",
        "dtype": "f32",
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "steps": steps,
    }
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)
        f.write("\n")
    print("generated", generated)


if __name__ == "__main__":
    main()
