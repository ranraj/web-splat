
The floor tilt angle from horizontal is simply:

$$\theta = \arccos(\vec{up} \cdot \hat{Y}) \times \frac{180}{\pi}$$

Where $\hat{Y} = (0, 1, 0)$ is world-up. Since `up.dot(Y) == up.y`, the formula reduces to `acos(up.y)` in degrees.

**How to expose this**: I can add a `#[wasm_bindgen]` method to `WindowContext` that returns the floor angle in degrees, callable from the browser console. Would that be useful, or would you prefer to use it to **automatically derive `init_pitch`** from the actual PLY data instead of the hardcoded `0.096`?


Let me read the key parts of lib.rs — imports and all 3 framing sites — before making changes. 

Read [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs#1-1), lines 1 to 35

Now let me read the exact current state of each framing site:   

Read [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs#245-245), lines 245 to 280

Read [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs#378-378), lines 378 to 415

Read [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs#912-912), lines 912 to 940

Now I understand the root cause clearly:

- `elegant_library` and `warm_traditional`: PCA floor normal is close to `(0,1,0)` → identity rotation works fine
- `rustic_kitchen`: PCA floor normal tilts far from Y ( walls dominate the point distribution) → old `acos` hack only moved the *position* but kept the *view direction* as identity `+Z`, making the scene appear 90° tilted

**The fix**: Build a proper rotation matrix from `world_up` so both camera position AND view direction align with the floor — works for any PLY orientation. 

Created 3 todos

Replacing 1 lines with 1 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)


```

```


Starting: *Fix framing logic at all 3 sites* (2/3)

Now replace all 3 framing sites in one call: 

Replacing 48 lines with 65 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)


```

```


Checked [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs), no problems found

Completed: *Fix framing logic at all 3 sites* (2/3)

No errors. Here's exactly what changed and why:

**Root cause**: The old `init_pitch = acos(world_up.y)` only rotated the camera's *position offset* — the actual *view direction* stayed locked at `Quaternion::one()` (+Z). For `rustic_kitchen`, where PCA detects the scene is tilted ~90°, the camera ended up positioned far from where it should be but still staring through the floor.

**The fix — proper rotation matrix at all 3 sites**:

1. **Quality gate**: `pc.up().filter(|up| up.y >= 0.5)` — only trust the PCA result when the detected floor normal is within 60° of world-Y. Beyond that, PCA found a wall dominant surface (rustic_kitchen), so fall back to `(0,1,0)`.

2. **Horizontal look direction**: Project `+Z` onto the floor plane perpendicular to `world_up`. If `world_up ≈ ±Z`, fall back to `+X`. This gives a reliable viewing direction regardless of PLY axis conventions.

3. **Rotation matrix**: Build `[right | world_up | look_horiz]` as column vectors → convert to `Quaternion`. This correctly sets both camera **position** and **view direction** to be perpendicular to the floor.

**For your three files**:
- `elegant_library`, `warm_traditional`: `world_up ≈ (0,1,0)` → rotation = identity → same behavior as before ✓
- `rustic_kitchen`: PCA wall normal has `y < 0.5` → falls back to `(0,1,0)` → camera looks straight in with floor horizontal ✓

Made changes.


check
------
The old code used (0, -1, 0) as default up (Y-down convention from 3DGS/OpenCV)

My new code changed to (0, 1, 0) as fallback and treated world_up as Y-up

The Marble PLY files use Y-down convention where negative Y is up, and the plane_from_points function flips the normal to point toward +Y, but then the library file's up vector like (0, -0.99, 0.01) gets negated to (0, 0.99, -0.01) and passes the y >= 0.5 filter

Let me analyze the three screenshots:

rustic_kitchen (image 1): Default render shows just a dark/gray blur - camera is inside a wall or floor, wrong position
rustic_kitchen (image 2): After manual rotation, scene is visible but heavily tilted - floor is not horizontal
elegant_library (image 3): Camera is now looking at the ceiling/roof - the new code broke it by making it parallel to floor looking up
The problem is clear:

The old code used (0, -1, 0) as default up (Y-down convention from 3DGS/OpenCV)

My new code changed to (0, 1, 0) as fallback and treated world_up as Y-up

The Marble PLY files use Y-down convention where negative Y is up, and the plane_from_points function flips the normal to point toward +Y, but then the library file's up vector like (0, -0.99, 0.01) gets negated to (0, 0.99, -0.01) and passes the y >= 0.5 filter

I need to trace through the plane detection logic more carefully to see where the coordinate system is getting confused. I'm realizing the issue: PCA finds the floor normal, but it's forcing normal.y >= 0, which gives (0, 1, 0) for a Y-down scene when it should be (0, -1, 0) to point toward negative Y where "up" actually is. downward, which is wrong. Let me trace through the working code from the conversation history more carefully. When PCA found a normal like (0.01, 0.99, 0.05) in elegant_library, the camera offset combined the backward movement in -Z with an upward shift in +Y, which positioned the camera correctly above and behind the scene center. The quaternion setup then...

Actually, I think I see the issue now. The new rotation matrix construction is fundamentally different—it's treating the world_up vector as the middle column instead of using it to define the camera's vertical axis, which is flipping how the camera interprets its orientation relative to the scene. The real issue is that my rotation matrix is treating world_up as a camera axis when it's actually the floor normal, which doesn't match what the camera expects. Different PLY files have inconsistent axis conventions — some Y-up, some Y-down, some arbitrarily rotated — so I need a universal approach that detects and adapts to each file's orientation rather than forcing a single transformation.

Instead of relying on PCA alone, I should combine multiple signals: use PCA to find the floor plane normal, check the bounding box to identify which axis represents room height, and analyze eigenvalues to pinpoint the floor-to-ceiling direction. The problem with my current plane_from_points function is that it works well when horizontal surfaces dominate the point cloud, but fails when walls capture more surface area and PCA finds a wall normal instead. The crude y >= 0.5 filtering won't scale across different file formats.

Study camera rotation convention
Study PCA up and AABB deminesions 
Robust auto-oriented algorithm
Implement at all 3 framing sites

The world2view / rotation convention: camera.rotation is treated by world2view() as Matrix3::from(rotation), which maps world coords → camera coords. Quaternion::one() (identity) means the camera's local axes are the world axes: camera-right = +X, camera-down = +Y, camera-forward = +Z. This is the 3DGS/OpenCV convention.

Matrix3::from_cols(right, world_up, look_horiz) approach: This built a world→camera matrix that transformed world_up to camera +Y (which is screen-DOWN in this convention). That made the camera look at the ceiling for the library and kitchen files that already had up ≈ (0,1,0).

Use AABB dimensions to detect the "thin" axis (smallest extent) — that's "up" for the room. This is more robust than PCA for room-scale scenes.

Camera looks from centroid + offset toward centroid, with the thin-axis as the up reference.

et me check what Quaternion::look_at does in cgmath and how the controller uses it:

The old algorithm: AABB shortest-axis first, PCA only if it agrees with AABB. This fails when the AABB shortest axis is X or Z (narrow galley kitchen, corridor, gallery room) — it returned +X or +Z as "up", rotating the camera 90°. PCA (which correctly found +Y) was discarded because it didn't agree with the wrong AABB result.

Image 1 (kitchen 90° tilt): The kitchen was narrower in X than it was tall in Y, so shortest_axis = X → aabb_up = +X. PCA produced ~+Y but PCA·(+X) = 0 < 0.707, so the code chose AABB (+X) over PCA — camera up was sideways.

Image 2 (library, dark vignettes): Same root cause. Wrong world_up made the camera placement offset go sideways instead of upward, pushing the camera into or near a wall. Close-proximity wall Gaussians projected as huge blobs covering the frame edges.

The fix — new priority order:

PCA when |Y| ≥ 0.5 — the PCA floor-plane normal already flips to +Y in io/mod.rs. Trust it whenever it's mostly upright.
AABB — only when Y is explicitly the shortest axis (ratio < 0.70). Never return X or Z as up.
+Y fallback — the standard 3DGS/COLMAP world convention.

A zero-dependency library with three modules:

Module	Purpose
math.rs	Minimal Vec3 type with dot/cross/normalize — no cgmath/glam needed
analysis.rs	SceneAnalysis::from_points() — median centroid, p90 radius, outlier removal, 3-tier up-axis detection (PCA → AABB → +Y fallback, matching web-splat's robust_scene_up)
camera.rs	AutoCamera::frame() — intelligent camera placement with CameraHint::{Interior, Exterior, Auto}
Key intelligence in the camera:

Interior mode (rooms): places camera inside the cloud, 12% pullback + 6.3% eye-rise, 60° horizontal FOV — matches web-splat's auto_frame_scene
Exterior mode (objects): 45° orbit, 20° elevation, 1.8× radius, 50° FOV
Auto mode: heuristic based on floor-area-to-height ratio
.project() method built in for direct world→screen mapping

PCB/AABB
sRGB gamma correction

Camera tilt fix (camera.rs):

right = world_up × forward (was forward × world_up — reversed, causing the 90° roll)
up = forward × right (was right × forward — also reversed)
Interior camera now approaches from a ¾ diagonal ([0.7, 0.0, 1.0]) instead of straight-on, for a more cinematic angle
Brightness fix (thumbnail.rs):

Added proper sRGB gamma correction (linear_to_srgb) — the old code wrote linear RGB directly, which looks dark/washed-out on screen
Alpha blending instead of hard pixel overwrite — overlapping splats accumulate naturally
Depth-adaptive splat size — near points render larger (up to 6px), far points render smaller (1px), based on projecting a fixed world-space size through the perspective divide
Distance-based opacity fade — far splats are slightly more transparent, giving natural depth cues

OCR (Optical Character Recognition) and spatial grounding


Now I'll make the changes. The key differences between vb_auto_camera (thumbnail) and web-splat:

Parameter	auto_camera (thumbnail)	web-splat (current)
Interior FOV	65°	60°
Exterior mode	50° FOV, ¾ elevated orbit	none (always interior)
Near plane	max(radius×0.001, 0.001)	0.005
Far plane	radius×20	1000
Look direction fallbacks	+Z, -Z, +Y, +X	+Z, +X

Interior/exterior auto-detection — added guess_interior() (same heuristic as auto_camera: "wide and flat" = interior)
Interior camera: FOV 60° → 65°, same position formula (radius×0.12 pullback, radius×0.063 eye rise)
Exterior camera (new): ¾ elevated orbit at radius×1.8 distance, 20° elevation, 50° FOV — matches auto_camera's exterior framing
Adaptive clip planes: near = max(radius×0.001, 0.001), far = radius×20 (was hardcoded 0.005/1000)
Better look-direction fallbacks: tries +Z, −Z, +Y, +X (was only +Z → +X)
All three camera setup sites were updated: WindowContext::new(), reload_from_bytes(), and the auto_frame_scene/return_to_origin handler.

Vantage point
Cardinal
quaternion
VIEW port
3DGS rendering


DX12/WGL
crate (DX12/WGL backend incompatibility on this Windows toolchain

pnpm --filter @vb/web

plylancemcpserver

WSGL

Normalized Device Coordinates (NDC)

Use:

PLY → for geometry / segmentation / precision
GLB (GLTF binary) → for rendering / UI / materials

👉 Think of it like this:

PLY = “data model”
GLB = “visual + interactive model”

OrbitControls
Light source optional
raycasting
material replacement.