# Coordinate frames: Marble vs gsplat trainer vs our WASM viewer

## What the WASM viewer assumes

The `engines/web-splat` renderer expects a **3DGS-style world** where:

- `+Y` is “up” (floor plane normals should point toward `+Y`)
- camera matrices use a conventional right-handed math setup:
  - we build a **world→view** matrix (`world2view(...)`)
  - we feed WGSL with `camera.view` / `camera.proj` and compute ray directions in that same world basis

To pick the scene “up” direction, we run `robust_scene_up(...)` which uses:

- PCA plane fitting from the loaded Gaussian centers (`pc.up()`)
- a small AABB fallback to avoid 90° camera tilts on wall-dominant scenes

## Marble assets (current “feels smooth” baseline)

Marble World Labs assets typically load correctly with the WASM viewer today. In practice, they:

- are already aligned so the floor/room plane normal is predominantly `+Y`
- have consistent “scene scale” so orbit/pan/zoom feel natural once we pivot around an AABB center and scale controls by scene radius

The key behavior we standardize on (for orbit/pan/rotate parity) is:

1. **Orbit pivot = AABB center** (`controller.center = pc.bbox().center()`)
2. **Control sensitivity scales by scene radius** (`scene_radius = pc.bbox().radius()`)
3. **near/far are clamped to the scene extent** (adaptive clip planes)

## gsplat `simple_trainer.py` outputs and `coordinate_transform.npz`

In `gsplat_3dgut/examples/simple_trainer.py`:

- The COLMAP parser is constructed with `normalize_world_space=True` by default.
- Training splats therefore live in the **parser-normalized world**, not raw COLMAP reconstruction world (unless configured otherwise).

The trainer always writes `result_dir/coordinate_transform.npz` via:

- `save_coordinate_metadata(parser.transform, ...)`

Keys we care about:

- `normalized_from_colmapworld`
- `colmapworld_from_normalized`

Meaning:

- These matrices map between the two coordinate systems that appear in gsplat training:
  - **COLMAP reconstruction world**
  - **normalized training world** (parser-normalized)

PLY export note:

- In gsplat, `save_ply(... export_colmap_coordinates=...)` controls what the PLY stores:
  - `export_colmap_coordinates=False` (default): Gaussian centers are still in **normalized** space.
  - `export_colmap_coordinates=True`: export applies `splats_colmap_world_from_normalized`, so centers are in **COLMAP** space.

So for any given gsplat PLY, the correct direction to apply depends on:

- whether the PLY centers were exported as normalized or as COLMAP coordinates.

## How the WASM viewer handles `coordinate_transform.npz`

When the browser provides `coordinate_transform.npz` bytes to the WASM module, we do:

1. Parse the NPZ matrices from:
   - `normalized_from_colmapworld`
   - `colmapworld_from_normalized`
2. Choose which transform “direction” to apply using a lightweight heuristic:
   - We compare a few candidates (including an “identity / no-conversion” option)
   - For each candidate, we transform a sampled subset of points
   - We score the result by how well the dominant plane aligns with **`+Y`**
3. Apply the chosen affine transform consistently to the entire splat:
   - Gaussian centers (`xyz`)
   - Gaussian covariances (`cov`)
   - Recompute derived metadata (`aabb`, `center`, `up`)
4. Then the camera is auto-framed from the transformed scene, so orbit/pan/zoom are consistent with the corrected coordinate basis.

Important: we do **not** change raster quality or shader logic; this is purely camera/scene-frame consistency.

## OrbitControls parity fixes we shipped

To make Marble and gsplat assets feel equally responsive:

- Orbit pivot is the AABB center (`pc.bbox().center()`), not the centroid.
- Pan/dolly feel is made scale-invariant by using:
  - `scene_radius = pc.bbox().radius()`
  - adaptive pan scaling and a scroll sensitivity scaling term
- `near` / `far` clip planes are clamped based on the same AABB radius, preventing near-plane precision issues when assets are in different unit scales.

## Minimal repro checklist (same controls, Marble vs gsplat)

### 1) Prepare a “sibling file” setup

Use a folder that contains, for the gsplat case:

- `point_cloud_*.ply`
- `coordinate_transform.npz` (same directory as the PLY)

For Marble:

- `marble.ply` (no coordinate transform required)

Host the directory over HTTP so the browser can fetch `coordinate_transform.npz` as a sibling of the gsplat PLY.

### 2) Load each asset in the WASM viewer

In the browser, load:

1. Marble PLY
2. gsplat `point_cloud_*.ply` (with its sibling `coordinate_transform.npz`)

### 3) Use the same gestures

Perform the same sequence for both:

- Orbit: left drag / orbit gesture
- Pan: right drag / shift gesture
- Dolly: mouse wheel / trackpad pinch

### 4) Expected parity

You should see:

- the scene rotates around a stable pivot (AABB center)
- pan speed feels proportional to the object scale
- zoom does not suddenly “jump” or lose the scene due to clip-plane problems

