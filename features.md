# Features — web-splat session

## On-screen Gamepad Overlay (`src/ui.rs`)

### Circle D-pad Controls
Replaced square button grid with mobile-game style circular d-pads (110×110 px each). Each pad renders a semi-transparent dark-red circle with V-shaped chevron arrows at North, South, West, and East. The active direction highlights in white.

### Three D-pad Layout (Row 1)
- **MOVE CAMERA** — maps to W/A/S/D. Boolean flags are updated every frame (`true` for active direction, `false` for all others) so switching directions works reliably without keys sticking.
- **ROTATE CAMERA** — applies rotation impulses (±2°) per frame. Camera icon drawn in the center (body rect + lens circle + viewfinder nub).
- **MOVE TARGET** — shifts the orbit pivot point.

### Auxiliary Controls (Row 2)
- **TILT CAMERA** — `< Z` / `X >` buttons for roll left/right.
- **ALTITUDE** — `Up` / `Dn` buttons for vertical movement (E/Q).
- **ZOOM** — `+` / `-` buttons driving the scroll accumulator.

### Labels Above Controls
Each control group has a bold uppercase label (`MOVE CAMERA`, `ROTATE CAMERA`, etc.) rendered at 10 px above its widget.

### U Key No Longer Hides Gamepad
All main UI windows (Render Stats, Settings, Scene, Keys) are wrapped in a `if state.ui_visible {}` guard. The gamepad panel sits outside that block and is controlled independently by the `V` key (`gamepad_visible` flag). Pressing U only toggles the main HUD.

### Always-render Shapes
`lib.rs` render call changed from `state.ui_visible.then_some(shapes)` to `Some(shapes)` so the gamepad overlay is composited onto the frame regardless of HUD visibility.

### Hint Line
A small `V · TOGGLE CONTROLS` hint is shown at the bottom of the gamepad panel.
