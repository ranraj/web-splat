@echo off
setlocal
REM Always run from this script's directory (cargo/wasm-bindgen use relative paths).
cd /d "%~dp0"

REM getrandom wasm cfg is in .cargo/config.toml (same pattern as packages/renderer)
REM so we do not rely on RUSTFLAGS quoting in cmd.exe, which breaks the next line.

REM Ensure wgpu sources exist, then patch registry copy before compile (see tools/patch_wgpu_webgpu_surface.py).
cargo fetch --target wasm32-unknown-unknown
if errorlevel 1 exit /b 1
python "%~dp0tools\patch_wgpu_webgpu_surface.py"
if errorlevel 1 exit /b 1

cargo build --no-default-features --target wasm32-unknown-unknown --lib --features npz --profile web-release
if errorlevel 1 exit /b 1

set "WASM_IN=%~dp0target\wasm32-unknown-unknown\web-release\web_splats.wasm"
set "JS_OUT=%~dp0public\web_splats.js"
if not exist "%WASM_IN%" (
    echo [build_wasm.bat] ERROR: missing "%WASM_IN%"
    exit /b 1
)
where wasm-bindgen >nul 2>&1
if errorlevel 1 (
    echo [build_wasm.bat] ERROR: wasm-bindgen not on PATH. Install: cargo install wasm-bindgen-cli
    exit /b 1
)
wasm-bindgen --out-dir "%~dp0public" --web "%WASM_IN%" --no-typescript
if errorlevel 1 exit /b 1
if not exist "%JS_OUT%" (
    echo [build_wasm.bat] ERROR: wasm-bindgen did not write "%JS_OUT%"
    exit /b 1
)

REM Sync build output into apps/web/public/web-splat so the Next.js dev/prod server
REM serves the freshly-built bundle. Copy only the generated artifacts, not the
REM static assets that already live in apps/web/public/web-splat (index.html, etc.).
set "WEB_PUBLIC=%~dp0..\..\apps\web\public\web-splat"
if not exist "%WEB_PUBLIC%\" (
    echo [build_wasm.bat] ERROR: destination folder missing: "%WEB_PUBLIC%"
    exit /b 1
)
copy /Y "%~dp0public\web_splats.js" "%WEB_PUBLIC%\web_splats.js"
if errorlevel 1 exit /b 1
copy /Y "%~dp0public\web_splats_bg.wasm" "%WEB_PUBLIC%\web_splats_bg.wasm"
if errorlevel 1 exit /b 1
echo synced: web_splats.js + web_splats_bg.wasm -^> apps\web\public\web-splat\