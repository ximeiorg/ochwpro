#!/usr/bin/env bash
set -euo pipefail

if ! command -v wasm-pack &> /dev/null; then
  echo "wasm-pack not found, installing..."
  cargo install wasm-pack
fi

cd "$(dirname "$0")"

RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  wasm-pack build \
    --target web \
    -d ../ochw-web/pkg \
    --release \
    --no-opt
