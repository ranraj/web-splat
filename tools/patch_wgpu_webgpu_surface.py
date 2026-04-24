#!/usr/bin/env python3
"""
Patch the local Cargo registry copy of wgpu 29.0.1 before wasm builds.

`Instance::create_surface` on wasm used `dyn_into::<GpuCanvasContext>()` after
`getContext("webgpu")`. In several browser + wasm-bindgen combinations,
`instanceof GPUCanvasContext` fails even though the object is correct, which
panicked with "canvas context is not a GPUCanvasContext: ... JsValue(GPUCanvasContext)".

See: https://github.com/gfx-rs/wgpu/issues/3980 and related reports.

Run this before `cargo build` / `wasm-pack build` when targeting wasm32 so wgpu
is compiled from patched sources. Safe to run repeatedly (idempotent).
"""
from __future__ import annotations

import os
import pathlib
import sys

OLD = """        // Not returning this error because it is a type error that shouldn't happen unless
        // the browser, JS builtin objects, or wasm bindings are misbehaving somehow.
        let context: webgpu_sys::GpuCanvasContext = context
            .dyn_into()
            .expect("canvas context is not a GPUCanvasContext");"""

NEW = """        // Unchecked cast: avoid brittle `dyn_into` / `instanceof GPUCanvasContext` failures
        // when `getContext("webgpu")` still returned the real context (gfx-rs/wgpu#3980).
        let context: webgpu_sys::GpuCanvasContext =
            wasm_bindgen::JsCast::unchecked_into(context);"""


def cargo_home() -> pathlib.Path:
    if ch := os.environ.get("CARGO_HOME"):
        return pathlib.Path(ch)
    return pathlib.Path.home() / ".cargo"


def main() -> int:
    reg = cargo_home() / "registry" / "src"
    if not reg.is_dir():
        print(f"patch-wgpu: registry not found at {reg} (ok on first run before cargo fetch)")
        return 0

    hits = list(reg.glob("**/wgpu-29.0.1/src/backend/webgpu.rs"))
    if not hits:
        print("patch-wgpu: wgpu-29.0.1 not in registry yet (cargo will fetch it); skip")
        return 0

    path = hits[0]
    text = path.read_text(encoding="utf-8")

    if OLD not in text:
        if "wasm_bindgen::JsCast::unchecked_into(context)" in text and "create_surface_from_context" in text:
            print(f"patch-wgpu: already patched ({path})")
            return 0
        print(
            f"patch-wgpu: expected wgpu 29.0.1 surface snippet not found in {path}",
            file=sys.stderr,
        )
        return 1

    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patch-wgpu: patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
