#!/usr/bin/env bash
set -euo pipefail
# getrandom wasm cfg: see .cargo/config.toml (same as packages/renderer; avoids duplicating RUSTFLAGS here).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cargo fetch --target wasm32-unknown-unknown
python3 "$SCRIPT_DIR/tools/patch_wgpu_webgpu_surface.py"

cargo build --no-default-features --target wasm32-unknown-unknown --lib --features npz --profile web-release
wasm-bindgen --out-dir public --web target/wasm32-unknown-unknown/web-release/web_splats.wasm --no-typescript

# Sync build output into every Next.js public/web-splat folder so each dev/prod
# server serves the freshly-built bundle. Copy only generated artifacts, not the
# static assets that already live alongside (index.html, etc.).
#
# NOTE: BOTH apps must be synced. The admin portal's /ply-tuner and the customer
# Studio /scene-tuner both call set_perspective_camera_wasm via packages/viewer;
# a stale apps/admin-portal copy silently disables those controls (the
# `typeof mod.set_perspective_camera_wasm !== "function"` branch in
# packages/viewer/src/viewer-camera-bridge.ts::setCameraState).
for DEST in \
    "$SCRIPT_DIR/../../apps/web/public/web-splat" \
    "$SCRIPT_DIR/../../apps/admin-portal/public/web-splat"
do
    if [ ! -d "$DEST" ]; then
        echo "[build_wasm.sh] ERROR: destination folder missing: $DEST"
        exit 1
    fi
    cp -f "$SCRIPT_DIR/public/web_splats.js" "$DEST/web_splats.js"
    cp -f "$SCRIPT_DIR/public/web_splats_bg.wasm" "$DEST/web_splats_bg.wasm"
    echo "synced: web_splats.js + web_splats_bg.wasm -> $DEST"
done