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

REM Sync build output into every Next.js public/web-splat folder so each dev/prod
REM server serves the freshly-built bundle. Copy only the generated artifacts,
REM not the static assets that already live alongside (index.html, etc.).
REM
REM NOTE: BOTH apps must be synced. The admin portal's /ply-tuner and the
REM customer Studio /scene-tuner both call set_perspective_camera_wasm via
REM packages/viewer; a stale apps/admin-portal copy silently disables those
REM controls (typeof mod.set_perspective_camera_wasm !== "function" path in
REM packages/viewer/src/viewer-camera-bridge.ts::setCameraState).
for %%P in (
    "%~dp0..\..\apps\web\public\web-splat"
    "%~dp0..\..\apps\admin-portal\public\web-splat"
) do (
    if not exist "%%~P\" (
        echo [build_wasm.bat] ERROR: destination folder missing: %%~P
        exit /b 1
    )
    copy /Y "%~dp0public\web_splats.js" "%%~P\web_splats.js" >nul
    if errorlevel 1 exit /b 1
    copy /Y "%~dp0public\web_splats_bg.wasm" "%%~P\web_splats_bg.wasm" >nul
    if errorlevel 1 exit /b 1
    echo synced: web_splats.js + web_splats_bg.wasm -^> %%~P
)