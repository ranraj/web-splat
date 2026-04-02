use cgmath::*;
use num_traits::Float;
use std::f32::consts::PI;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

use winit::keyboard::KeyCode;

use crate::camera::PerspectiveCamera;

#[derive(Debug, Clone)]
pub struct TouchState {
    pub touches: Vec<Touch>,
    pub last_touch_count: usize,
    pub last_pinch_distance: Option<f32>,
    pub last_touch_center: Option<(f32, f32)>,
}

#[derive(Debug, Clone)]
pub struct Touch {
    pub id: u64,
    pub position: (f32, f32),
    pub phase: TouchPhase,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TouchPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
}

impl TouchState {
    pub fn new() -> Self {
        Self {
            touches: Vec::new(),
            last_touch_count: 0,
            last_pinch_distance: None,
            last_touch_center: None,
        }
    }
}

#[derive(Debug)]
pub struct CameraController {
    pub center: Point3<f32>,
    pub up: Option<Vector3<f32>>,

    // Held-key booleans for FPS movement (set true on keydown, false on keyup)
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,
    roll_left: bool,
    roll_right: bool,

    // Impulse accumulators for smooth orbit (mouse/arrow keys) and pan (I/J/K/L)
    shift: Vector2<f32>,
    rotation: Vector3<f32>,
    scroll: f32,

    pub speed: f32,
    pub sensitivity: f32,

    /// Shift key held — multiplies walk speed by 3×
    pub shift_pressed: bool,
    /// When true, mouse motion always orbits without needing a button held
    pub natural_mouse: bool,
    /// When true, invert the right-drag pan direction
    pub invert_trackpad: bool,

    pub left_mouse_pressed: bool,
    pub right_mouse_pressed: bool,
    pub alt_pressed: bool,
    pub user_inptut: bool,

    // Touch support
    pub touch_state: TouchState,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            center: Point3::origin(),
            up: None,
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            move_up: false,
            move_down: false,
            roll_left: false,
            roll_right: false,
            shift: Vector2::zero(),
            rotation: Vector3::zero(),
            scroll: 0.0,
            speed,
            sensitivity,
            shift_pressed: false,
            natural_mouse: false,
            invert_trackpad: false,
            left_mouse_pressed: false,
            right_mouse_pressed: false,
            alt_pressed: false,
            user_inptut: false,
            touch_state: TouchState::new(),
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, pressed: bool) -> bool {
        let processed = match key {
            // FPS movement (boolean flags — set on press, clear on release)
            KeyCode::KeyW => { self.move_forward  = pressed; true }
            KeyCode::KeyS => { self.move_backward = pressed; true }
            KeyCode::KeyA => { self.move_left     = pressed; true }
            KeyCode::KeyD => { self.move_right    = pressed; true }
            KeyCode::KeyE | KeyCode::Space => { self.move_up   = pressed; true }
            KeyCode::KeyQ => { self.move_down = pressed; true }
            // O as alias for move backward
            KeyCode::KeyO => { self.move_backward = pressed; true }
            // Shift — speed multiplier
            KeyCode::ShiftLeft | KeyCode::ShiftRight => { self.shift_pressed = pressed; true }

            // Camera orbit — Arrow keys fire an impulse on press (decay naturally)
            KeyCode::ArrowLeft  => { if pressed { self.rotation.x -= 2.0; } true } // Pitch left
            KeyCode::ArrowRight => { if pressed { self.rotation.x += 2.0; } true } // Pitch right
            KeyCode::ArrowUp    => { if pressed { self.rotation.y += 2.0; } true } // Yaw up
            KeyCode::ArrowDown  => { if pressed { self.rotation.y -= 2.0; } true } // Yaw down

            // Camera roll (Z/X — boolean: held for continuous roll)
            KeyCode::KeyZ => { self.roll_left  = pressed; true }
            KeyCode::KeyX => { self.roll_right = pressed; true }

            // Move target / orbit pivot — impulse on press
            KeyCode::KeyI => { if pressed { self.shift.x += 1.0; } true } // Move target down
            KeyCode::KeyK => { if pressed { self.shift.x -= 1.0; } true } // Move target up
            KeyCode::KeyJ => { if pressed { self.shift.y -= 1.0; } true } // Move target left
            KeyCode::KeyL => { if pressed { self.shift.y += 1.0; } true } // Move target right

            _ => false,
        };
        // Flag user input when any movement key is actively held
        if processed {
            let moving = self.move_forward || self.move_backward
                || self.move_left || self.move_right
                || self.move_up   || self.move_down
                || self.roll_left || self.roll_right;
            if pressed || moving {
                self.user_inptut = true;
            }
        }
        processed
    }

    pub fn process_mouse(&mut self, mouse_dx: f32, mouse_dy: f32) {
        // Orbit: left-button drag, or always when natural_mouse is enabled
        if self.left_mouse_pressed || self.natural_mouse {
            self.rotation.x += mouse_dx;
            self.rotation.y += mouse_dy;
            self.user_inptut = true;
        }
        // Pan: right-button drag, with optional inversion for trackpads
        if self.right_mouse_pressed {
            let sign = if self.invert_trackpad { -1.0 } else { 1.0 };
            self.shift.y += -mouse_dx * sign;
            self.shift.x += mouse_dy * sign;
            self.user_inptut = true;
        }
    }

    pub fn process_scroll(&mut self, dy: f32) {
        self.scroll += -dy;
        self.user_inptut = true;
    }

    pub fn process_touch(&mut self, touch: Touch) {
        // Update touch state
        match touch.phase {
            TouchPhase::Started => {
                self.touch_state.touches.push(touch);
            }
            TouchPhase::Moved => {
                if let Some(existing_touch) = self
                    .touch_state
                    .touches
                    .iter_mut()
                    .find(|t| t.id == touch.id)
                {
                    existing_touch.position = touch.position;
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touch_state.touches.retain(|t| t.id != touch.id);
            }
        }

        self.handle_touch_gestures();
        self.user_inptut = true;
    }

    fn handle_touch_gestures(&mut self) {
        let touch_count = self.touch_state.touches.len();

        match touch_count {
            1 => {
                // Single touch - camera rotation
                let touch = &self.touch_state.touches[0];
                if let Some(last_center) = self.touch_state.last_touch_center {
                    let dx = touch.position.0 - last_center.0;
                    let dy = touch.position.1 - last_center.1;

                    // Scale the touch movement similar to mouse movement but with better mobile sensitivity
                    self.rotation.x += dx * 0.3; // Reduced sensitivity for more precise control
                    self.rotation.y += dy * 0.3;
                }
                self.touch_state.last_touch_center = Some(touch.position);
            }
            2 => {
                // Two touches - pinch to zoom and pan
                let touch1 = &self.touch_state.touches[0];
                let touch2 = &self.touch_state.touches[1];

                let center_x = (touch1.position.0 + touch2.position.0) / 2.0;
                let center_y = (touch1.position.1 + touch2.position.1) / 2.0;
                let current_center = (center_x, center_y);

                // Calculate distance for pinch gesture
                let dx = touch2.position.0 - touch1.position.0;
                let dy = touch2.position.1 - touch1.position.1;
                let current_distance = (dx * dx + dy * dy).sqrt();

                if let Some(last_distance) = self.touch_state.last_pinch_distance {
                    // Pinch to zoom with improved sensitivity
                    let distance_change = current_distance - last_distance;
                    let zoom_factor = distance_change * 0.005; // Adjusted for better mobile zoom control
                    self.scroll += zoom_factor;
                }

                if let Some(last_center) = self.touch_state.last_touch_center {
                    // Pan with two fingers - improved sensitivity for mobile
                    let center_dx = current_center.0 - last_center.0;
                    let center_dy = current_center.1 - last_center.1;

                    self.shift.y += -center_dx * 0.3; // Reduced sensitivity for more precise panning
                    self.shift.x += center_dy * 0.3;
                }

                self.touch_state.last_pinch_distance = Some(current_distance);
                self.touch_state.last_touch_center = Some(current_center);
            }
            _ => {
                // No touches or more than 2 touches - reset state
                self.touch_state.last_pinch_distance = None;
                self.touch_state.last_touch_center = None;
            }
        }

        self.touch_state.last_touch_count = touch_count;
    }

    pub fn clear_touch_state(&mut self) {
        self.touch_state.touches.clear();
        self.touch_state.last_touch_count = 0;
        self.touch_state.last_pinch_distance = None;
        self.touch_state.last_touch_center = None;
    }

    /// moves the controller center to the closest point on a line defined by the camera position and rotation
    /// ajusts the controller up vector by projecting the current up vector onto the plane defined by the camera right vector
    pub fn reset_to_camera(&mut self, camera: PerspectiveCamera) {
        let inv_view = camera.rotation.invert();
        let forward = inv_view * Vector3::unit_z();
        let right = inv_view * Vector3::unit_x();

        // move center point
        self.center = closest_point(camera.position, forward, self.center);
        // adjust up vector by projecting it onto the plane defined by the right vector of the camera
        if let Some(up) = &self.up {
            let new_up = up - up.project_on(right);
            self.up = Some(new_up.normalize());
        }
    }

    pub fn update_camera(&mut self, camera: &mut PerspectiveCamera, dt: Duration) {
        let dt: f32 = dt.as_secs_f32();
        if dt <= 0.0 {
            return;
        }

        let mut dir = camera.position - self.center;
        let distance = dir.magnitude();
        // Safety: prevent NaN from zero-length direction
        if !dir.x.is_finite() || !dir.y.is_finite() || !dir.z.is_finite() || distance < 1e-6 {
            dir = Vector3::new(0.0, 0.0, 1.0);
        }
        dir = dir.normalize_to((distance.ln() + self.scroll * dt * 10. * self.speed).exp());

        let view_t: Matrix3<f32> = camera.rotation.invert().into();
        let x_axis = view_t.x;
        let y_axis = self.up.unwrap_or(view_t.y);
        let z_axis = view_t.z;

        // 1. FPS WASD/QE/Space walk — moves both camera and orbit pivot together
        let fwd  = (self.move_forward  as i32 - self.move_backward as i32) as f32;
        let side = (self.move_right    as i32 - self.move_left     as i32) as f32;
        let vert = (self.move_up       as i32 - self.move_down     as i32) as f32;
        if fwd != 0.0 || side != 0.0 || vert != 0.0 {
            let speed_mult = if self.shift_pressed { 3.0 } else { 1.0 };
            let walk_speed = self.speed * 2.0 * dt * speed_mult;
            let raw = z_axis * fwd + x_axis * side - y_axis * vert;
            let move_vec = if raw.magnitude2() > 1e-6 {
                raw.normalize() * walk_speed
            } else {
                Vector3::zero()
            };
            camera.position += move_vec;
            self.center += move_vec;
            self.user_inptut = true;
        }

        // 2. Pan orbit pivot (I/J/K/L + right-mouse drag)
        let offset =
            (self.shift.y * x_axis - self.shift.x * y_axis) * dt * self.speed * 0.1 * distance;
        self.center += offset;
        camera.position += offset;

        // 3. Orbit rotation (arrow keys + mouse drag)
        let mut theta = Rad((self.rotation.x) * dt * self.sensitivity);
        let mut phi   = Rad((-self.rotation.y) * dt * self.sensitivity);

        // 4. Roll: Z/X keys (continuous while held) — always active, no alt required
        let roll_sign = (self.roll_left as i32 - self.roll_right as i32) as f32;
        let mut eta = Rad(roll_sign * dt * self.sensitivity * 2.0);

        if self.alt_pressed {
            // Alt overrides: use mouse/arrow Y axis for roll
            eta = Rad(-self.rotation.y * dt * self.sensitivity);
            theta = Rad::zero();
            phi = Rad::zero();
        }

        let rot_theta = Quaternion::from_axis_angle(y_axis, theta);
        let rot_phi   = Quaternion::from_axis_angle(x_axis, phi);
        let rot_eta   = Quaternion::from_axis_angle(z_axis, eta);
        let rot = rot_theta * rot_phi * rot_eta;

        let mut new_dir = rot.rotate_vector(dir);
        // Safety: prevent NaN after rotation
        if !new_dir.x.is_finite() || !new_dir.y.is_finite() || !new_dir.z.is_finite()
            || new_dir.magnitude() < 1e-6
        {
            new_dir = dir;
        }
        if angle_short(y_axis, new_dir) < Rad(0.1) {
            new_dir = dir;
        }
        camera.position = self.center + new_dir;

        // Apply roll (Z/X keys) by rotating the orbit up-vector around the view
        // forward axis.  This is the only way to get persistent roll — look_at()
        // always cancels roll if we just pass y_axis unchanged.
        let forward = (-new_dir).normalize();
        let rolled_up = if roll_sign != 0.0 {
            let roll_quat = Quaternion::from_axis_angle(forward, eta);
            let new_up = roll_quat.rotate_vector(y_axis);
            // Persist the rolled up so subsequent frames keep the tilt
            self.up = Some(new_up.normalize());
            new_up.normalize()
        } else {
            y_axis
        };
        camera.rotation = Quaternion::look_at(-new_dir, rolled_up);

        // Decay impulse accumulators
        let mut decay = (0.8_f32).powf(dt * 60.);
        if decay < 1e-4 {
            decay = 0.;
        }
        self.rotation *= decay;
        if self.rotation.magnitude() < 1e-4 {
            self.rotation = Vector3::zero();
        }
        self.shift *= decay;
        if self.shift.magnitude() < 1e-4 {
            self.shift = Vector2::zero();
        }
        self.scroll *= decay;
        if self.scroll.abs() < 1e-4 {
            self.scroll = 0.;
        }
        self.user_inptut = false;
    }
}

fn closest_point(orig: Point3<f32>, dir: Vector3<f32>, point: Point3<f32>) -> Point3<f32> {
    let dir = dir.normalize();
    let lhs = point - orig;

    let dot_p = lhs.dot(dir);
    // Return result
    return orig + dir * dot_p;
}

fn angle_short(a: Vector3<f32>, b: Vector3<f32>) -> Rad<f32> {
    let angle = a.angle(b);
    if angle > Rad(PI / 2.) {
        return Rad(PI) - angle;
    } else {
        return angle;
    }
}
