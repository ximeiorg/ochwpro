# Build ochw-wasm for web
$ErrorActionPreference = "Stop"

if (-not (Get-Command "wasm-pack" -ErrorAction SilentlyContinue)) {
    Write-Host "wasm-pack not found, installing..."
    cargo install wasm-pack
}

$env:RUSTFLAGS = '--cfg getrandom_backend="wasm_js"'
wasm-pack build --target web -d ../ochw-web/pkg --release --no-opt
