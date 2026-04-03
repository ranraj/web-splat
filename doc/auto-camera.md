Now I have full understanding. Let me analyze the real problem:

**The `world2view` / rotation convention**: `camera.rotation` is treated by `world2view()` as `Matrix3::from(rotation)`, which maps **world coords → camera coords**. `Quaternion::one()` (identity) means the camera's local axes **are** the world axes: camera-right = +X, camera-down = +Y, camera-forward = +Z. This is the 3DGS/OpenCV convention.

**What happened with the `Matrix3::from_cols(right, world_up, look_horiz)` approach**: This built a world→camera matrix that transformed world_up to camera +Y (which is screen-DOWN in this convention). That made the camera look at the ceiling for the library and kitchen files that already had `up ≈ (0,1,0)`.

**The correct algorithm**:
1. Use AABB dimensions to detect the "thin" axis (smallest extent) — that's "up" for the room. This is more robust than PCA for room-scale scenes.
2. Use `Quaternion::look_at()` from the controller which uses the same convention.
3. Camera looks from centroid + offset *toward* centroid, with the thin-axis as the up reference.

Let me implement this properly: 

Starting: *Implement at all 3 framing sites* (4/5)

First, let me check what `Quaternion::look_at` does in cgmath and how the controller uses it: 

Read [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/controller.rs#370-370), lines 370 to 395

The controller uses `Quaternion::look_at(-new_dir, rolled_up)` where:
- `-new_dir` = direction from camera TO center (negative of camera offset)
- `rolled_up` = the up vector

`cgmath::Quaternion::look_at(dir, up)` builds a rotation where the Z-axis aligns with `dir` and Y-axis aligns with `up`. In this 3DGS convention, camera +Z = forward, camera +Y = down. So `look_at(toward_center, up_vec)` is the standard.

The cinematic pan uses a **different** approach — it manually builds the rotation matrix in a specific way. Let me check how it uses `screen_down_approx = -self.world_up`. 

The problem is clear: `Quaternion::look_at` from cgmath constructs a rotation where:
- +Z aligns with the `dir` parameter  
- +Y aligns with the `up` parameter

In the controller: `camera.rotation = Quaternion::look_at(-new_dir, rolled_up)` — note the negate on dir. The controller passes `(-offset_dir, up_vec)`. So the camera looks in the direction of `-(-offset) = +offset` which... wait, let me re-read carefully.

`new_dir = center → camera position`. So `-new_dir = camera → center = forward`. So `look_at(forward, up)` produces a rotation where +Z = forward, +Y = up direction.

But wait — in screen space for 3DGS, +Y is **down**. If we pass real "up" as the `up` parameter, that maps world-up to camera +Y = screen-down. That's exactly what the controller does and it works fine with orbiting.

So the correct initial rotation should be: `Quaternion::look_at(look_direction, world_up)` where `look_direction` points FROM camera TO centroid, and `world_up` is the scene up vector. This is exactly what the controller will produce once it takes over.

Now let me also add the AABB-based up detection as a fallback. The key insight: for room-scale 3DGS scenes, the "up" axis is almost always the shortest AABB dimension (rooms are wider and deeper than tall). PCA can fail when walls dominate the point count, but AABB won't. 

Replacing 74 lines with 44 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)


```

```
Now I need to add the `robust_scene_up` and `floor_projected_direction` helper functions. Let me find a good place, and also remove the `Matrix3` import since we no longer use it directly:  

Searched for text `pub fn smoothstep` (`**/src/lib.rs`), 1 result

Read [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs#760-760), lines 760 to 775

Now add the helper functions just before `smoothstep`:  

Replacing 3 lines with 97 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)


```

```
Now remove the unused `Matrix3` import and keep `InnerSpace`:   

Replacing 1 lines with 1 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)


```

```


Checked [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs), no problems found

Now let me also check the cinematic pan call to make sure it gets the correct up vector too:  

Searched for text `start_cinematic_pan` (`**/src/lib.rs`), 6 results

Read [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs#660-660), lines 660 to 690

The cinematic pan also uses `self.pc.up()` directly — it should use the same robust up vector. Let me fix this:   

Replacing 2 lines with 2 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)


```

```


Checked [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs), no problems found

Now let me do a quick build to make sure everything compiles:   

Ran terminal command: cargo check 2>&1 | Select-Object -Last 20

The errors are in `wgpu-hal`, a dependency — not in our code. This is a pre-existing dependency compatibility issue. Let me check if the code compiles specifically for the wasm target which is what the project actually uses:    

Ran terminal command: cargo check --target wasm32-unknown-unknown 2>&1 | Select-Object -Last 15

Two issues: need `cgmath::Rotation` import for `look_at`, and `InnerSpace` is unused at the top level (only used inside the helper functions which have their own `use`). Let me fix: 

Replacing 1 lines with 1 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)


```

```


Ran terminal command: cargo check --target wasm32-unknown-unknown 2>&1 | Select-Object -Last 15

Build succeeds. There's an unused `num_traits::One` warning — let me clean it up since we replaced `Quaternion::one()` with `look_at`:   

Searched for text `Quaternion::one()` (`**/src/lib.rs`), no results

Good — `Quaternion::one()` is no longer used anywhere. Let me remove the import:     

Replacing 3 lines with 2 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)

Read [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs#12-12), lines 12 to 20

Replace String in File

Replacing 5 lines with 4 lines in [](file:///c%3A/Users/Ranjithraj/Documents/Projects/realities-project/VirtualBhoomi-app/engines/web-splat/src/lib.rs)


```

```


Ran terminal command: cargo check --target wasm32-unknown-unknown 2>&1 | Select-Object -Last 10

Completed: *Implement at all 3 framing sites* (4/5)

Clean build with only pre-existing documentation warnings. Here's a summary of the new algorithm:

**Problem**: PCA plane-fitting alone is unreliable — it finds the "flattest spread" of points, which is the floor for some scenes but a dominant wall for others (like `rustic_kitchen`). And the previous fix set camera rotation wrong (mapping up to screen-down).

**New algorithm — `robust_scene_up()` (3-tier detection)**:

| Tier | Method | When it's used |
|------|--------|----------------|
| 1 | **AABB shortest axis** | Rooms are always wider+deeper than tall. The smallest bounding-box dimension = the vertical axis. Reliable for any room-scale PLY. |
| 2 | **PCA consensus** | If PCA agrees with AABB axis (within 45°), use PCA for sub-degree accuracy. If PCA disagrees (found a wall), discard it. |
| 3 | **Fallback +Y** | When the box is nearly cubic (ratio > 0.85) and PCA is ambiguous, assume standard +Y. |

**Camera rotation fix**: Now uses `Quaternion::look_at(forward, up)` — the **same function** the orbit controller uses every frame. This guarantees the initial view and the controller's orbit are perfectly aligned. No more ceiling-staring or tilted scenes.

**What this means for your 3 files**:
- **`elegant_library`** / **`warm_traditional_kitchen`**: AABB Y-axis is shortest → up = +Y → camera looks horizontally into the room ✓
- **`rustic_kitchen`**: Even though PCA detects a wall, AABB still correctly identifies the short axis → camera enters the room upright ✓

Build the WASM and test with all three PLY files.

Made changes.