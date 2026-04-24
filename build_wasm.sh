#!/usr/bin/env bash
set -euo pipefail
# getrandom wasm cfg: see .cargo/config.toml (same as packages/renderer; avoids duplicating RUSTFLAGS here).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cargo fetch --target wasm32-unknown-unknown
python3 "$SCRIPT_DIR/tools/patch_wgpu_webgpu_surface.py"

cargo build --no-default-features --target wasm32-unknown-unknown --lib --features npz --profile web-release
wasm-bindgen --out-dir public --web target/wasm32-unknown-unknown/web-release/web_splats.wasm --no-typescript

# Sync build output into apps/web/public/web-splat so the Next.js dev/prod server
# serves the freshly-built bundle. Copy only generated artifacts, not the static
# assets that already live in apps/web/public/web-splat (index.html, etc.).
WEB_PUBLIC="$SCRIPT_DIR/../../apps/web/public/web-splat"
cp -f "$SCRIPT_DIR/public/web_splats.js" "$WEB_PUBLIC/web_splats.js"
cp -f "$SCRIPT_DIR/public/web_splats_bg.wasm" "$WEB_PUBLIC/web_splats_bg.wasm"
echo "synced: web_splats.js + web_splats_bg.wasm -> apps/web/public/web-splat/"