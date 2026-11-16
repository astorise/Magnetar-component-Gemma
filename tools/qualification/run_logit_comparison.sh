#!/usr/bin/env bash
# Compares the Gemma 4 Component, run by Magnetar's Reference CPU Provider,
# with `transformers` on tiny random Gemma 4 bundles.
#
# Needs: a Rust toolchain with the wasm32-unknown-unknown target, `wasm-tools`,
# Python with torch (CPU), transformers >= 5.2x, safetensors and numpy
# (PYTHONPATH may point at them), and a Magnetar checkout at the revision in
# the README with `loaders/huggingface` and `formats/safetensors` populated
# (they are submodules). See tools/qualification/README.md.
#
#   (CARGO_FLAGS="--ignore-rust-version" if your rustc is older than the crate's rust-version)
#   (SKIP_RUNTIME_PATCH=1 once Magnetar no longer needs magnetar-runtime-gemma4-local.patch)
#   MAGNETAR_DIR=/path/to/magnetar tools/qualification/run_logit_comparison.sh [variant...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
: "${MAGNETAR_DIR:?set MAGNETAR_DIR to a Magnetar checkout}"
work="${WORK_DIR:-$(mktemp -d)}"; mkdir -p "$work"
variants=("$@"); [ ${#variants[@]} -gt 0 ] || variants=(dense shared)
prompt="5 17 33 8 41 2 60 12"
steps=12

echo "== building the Component"
(cd "$repo" && cargo build --release --target wasm32-unknown-unknown ${CARGO_FLAGS:-})
wasm-tools component new \
  "$repo/target/wasm32-unknown-unknown/release/magnetar_component_gemma.wasm" \
  -o "$work/gemma.component.wasm"

echo "== installing the test crate in $MAGNETAR_DIR/scratch-gemma-logits"
mkdir -p "$MAGNETAR_DIR/scratch-gemma-logits/tests"
cp "$here/magnetar/Cargo.toml" "$MAGNETAR_DIR/scratch-gemma-logits/Cargo.toml"
cp "$here/magnetar/gemma4_logits.rs" "$MAGNETAR_DIR/scratch-gemma-logits/tests/gemma4_logits.rs"
if [ -n "${SKIP_RUNTIME_PATCH:-}" ]; then
  echo "   (SKIP_RUNTIME_PATCH set: Magnetar is used as is)"
elif ! git -C "$MAGNETAR_DIR" apply --check "$here/magnetar/magnetar-runtime-gemma4-local.patch" 2>/dev/null; then
  echo "   (runtime patch already applied or not applicable; continuing)"
else
  git -C "$MAGNETAR_DIR" apply "$here/magnetar/magnetar-runtime-gemma4-local.patch"
fi

status=0
for v in "${variants[@]}"; do
  echo "== variant: $v"
  python3 "$here/make_tiny_gemma4_bundle.py" "$work/bundle_$v" --variant "$v" >/dev/null
  python3 "$here/gemma4_reference_logits.py" "$work/bundle_$v" "$steps" "$work/ref_$v.json" $prompt | tail -1
  (cd "$MAGNETAR_DIR/scratch-gemma-logits" && \
    GEMMA_COMPONENT="$work/gemma.component.wasm" GEMMA_BUNDLE="$work/bundle_$v" \
    GEMMA_REFERENCE="$work/ref_$v.json" \
    cargo test --test gemma4_logits -- --nocapture 2>&1 | grep -E "tokens :|max \|logit|test result|panicked" ) || status=1
done
exit $status
