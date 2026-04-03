use splines::{Interpolate, Key};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

use cgmath::{EuclideanSpace, InnerSpace, MetricSpace, Point3, Quaternion, Rad, Rotation, Rotation3, Vector3, VectorSpace};

use crate::{PerspectiveProjection, camera::PerspectiveCamera};

pub trait Lerp {
    fn lerp(&self, other: &Self, amount: f32) -> Self;
}

pub trait Sampler {
    type Sample;

    fn sample(&self, v: f32) -> Self::Sample;
}

pub struct Transition<T> {
    from: T,
    to: T,
    interp_fn: fn(f32) -> f32,
}
impl<T: Lerp + Clone> Transition<T> {
    pub fn new(from: T, to: T, interp_fn: fn(f32) -> f32) -> Self {
        Self {
            from,
            to,
            interp_fn,
        }
    }
}

impl<T: Lerp + Clone> Sampler for Transition<T> {
    type Sample = T;
    fn sample(&self, v: f32) -> Self::Sample {
        self.from.lerp(&self.to, (self.interp_fn)(v))
    }
}

pub struct TrackingShot {
    spline: splines::Spline<f32, PerspectiveCamera>,
}

impl TrackingShot {
    pub fn from_cameras<C>(cameras: Vec<C>) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();

        let last_two = cameras.iter().skip(cameras.len() - 2).take(2);
        let first_two = cameras.iter().take(2);
        let spline = splines::Spline::from_iter(
            last_two
                .chain(cameras.iter())
                .chain(first_two)
                .enumerate()
                .map(|(i, c)| {
                    let v = (i as f32 - 1.) / (cameras.len()) as f32;
                    Key::new(v, c.clone(), splines::Interpolation::CatmullRom)
                }),
        );

        Self { spline }
    }

    pub fn num_control_points(&self) -> usize {
        self.spline.len()
    }
}

impl Sampler for TrackingShot {
    type Sample = PerspectiveCamera;
    fn sample(&self, v: f32) -> Self::Sample {
        match self.spline.sample(v) {
            Some(p) => p,
            None => panic!("spline sample failed at {}", v),
        }
    }
}

impl Interpolate<f32> for PerspectiveCamera {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        if t < threshold { a } else { b }
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        Self {
            position: Point3::from_vec(a.position.to_vec().lerp(b.position.to_vec(), t)),
            rotation: a.rotation.slerp(b.rotation, t),
            projection: a.projection.lerp(&b.projection, t),
        }
    }

    fn cosine(_t: f32, _a: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        // unroll quaternion rotations so that the animation always takes the shortest path
        // this is just a hack...
        let q_unrolled = unroll([x.1.rotation, a.1.rotation, b.1.rotation, y.1.rotation]);
        Self {
            position: Point3::from_vec(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.position.to_vec()),
                (a.0, a.1.position.to_vec()),
                (b.0, b.1.position.to_vec()),
                (y.0, y.1.position.to_vec()),
            )),
            rotation: Interpolate::cubic_hermite(
                t,
                (x.0, q_unrolled[0]),
                (a.0, q_unrolled[1]),
                (b.0, q_unrolled[2]),
                (y.0, q_unrolled[3]),
            )
            .normalize(),
            projection: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.projection),
                (a.0, a.1.projection),
                (b.0, b.1.projection),
                (y.0, y.1.projection),
            ),
        }
    }

    fn quadratic_bezier(_t: f32, _a: Self, _u: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier_mirrored(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }
}

impl Interpolate<f32> for PerspectiveProjection {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        if t < threshold { a } else { b }
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        return a.lerp(&b, t);
    }

    fn cosine(_t: f32, _a: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        Self {
            fovx: Rad(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fovx.0),
                (a.0, a.1.fovx.0),
                (b.0, b.1.fovx.0),
                (y.0, y.1.fovx.0),
            )),
            fovy: Rad(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fovy.0),
                (a.0, a.1.fovy.0),
                (b.0, b.1.fovy.0),
                (y.0, y.1.fovy.0),
            )),
            znear: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.znear),
                (a.0, a.1.znear),
                (b.0, b.1.znear),
                (y.0, y.1.znear),
            ),
            zfar: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.zfar),
                (a.0, a.1.zfar),
                (b.0, b.1.zfar),
                (y.0, y.1.zfar),
            ),
            fov2view_ratio: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fov2view_ratio),
                (a.0, a.1.fov2view_ratio),
                (b.0, b.1.fov2view_ratio),
                (y.0, y.1.fov2view_ratio),
            ),
        }
    }

    fn quadratic_bezier(_t: f32, _a: Self, _u: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier_mirrored(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }
}

pub struct Animation<T> {
    duration: Duration,
    time_left: Duration,
    looping: bool,
    sampler: Box<dyn Sampler<Sample = T>>,
}

impl<T> Animation<T> {
    pub fn new(duration: Duration, looping: bool, sampler: Box<dyn Sampler<Sample = T>>) -> Self {
        Self {
            duration,
            time_left: duration,
            looping,
            sampler,
        }
    }

    pub fn done(&self) -> bool {
        if self.looping {
            false
        } else {
            self.time_left.is_zero()
        }
    }

    pub fn update(&mut self, dt: Duration) -> T {
        match self.time_left.checked_sub(dt) {
            Some(new_left) => {
                // set time left
                self.time_left = new_left;
            }
            None => {
                if self.looping {
                    self.time_left = self.duration + self.time_left - dt;
                } else {
                    self.time_left = Duration::ZERO;
                }
            }
        }
        return self.sampler.sample(self.progress());
    }

    pub fn progress(&self) -> f32 {
        return 1. - self.time_left.as_secs_f32() / self.duration.as_secs_f32();
    }

    pub fn set_progress(&mut self, v: f32) {
        self.time_left = self.duration.mul_f32(1. - v);
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }

    pub fn set_duration(&mut self, duration: Duration) {
        let progress = self.progress();
        self.duration = duration;
        self.set_progress(progress);
    }
}

/// unroll quaternion rotations so that the animation always takes the shortest path
fn unroll(rot: [Quaternion<f32>; 4]) -> [Quaternion<f32>; 4] {
    let mut rot = rot;
    if rot[0].s < 0. {
        rot[0] = -rot[0];
    }
    for i in 1..4 {
        if rot[i].dot(rot[i - 1]) < 0. {
            rot[i] = -rot[i];
        }
    }
    return rot;
}

// ── Cinematic Pan ─────────────────────────────────────────────────────────────
//
// A looping, oscillating camera orbit that plays automatically on first load.
// The camera slowly sweeps left-to-right (yaw) and breathes slightly up-down
// (pitch) around the scene centroid, using a sine wave for perfectly smooth,
// natural-feeling movement.
//
// Plugs directly into the existing `Animation<PerspectiveCamera>` machinery:
//   let sampler = CinematicPan::new(...);
//   let anim = Animation::new(Duration::from_secs(10), true, Box::new(sampler));
//   state.animation = Some((anim, true));
//
// The animation is cancelled automatically the moment `controller.user_inptut`
// becomes true (mouse click, key press, touch, scroll) — see `update()` in lib.rs.

/// Smooth left-right (and subtle up-down) orbiting camera for cinematic intros.
pub struct CinematicPan {
    /// Frozen reference camera at the moment cinematic mode begins.
    initial_camera: PerspectiveCamera,
    /// Scene centroid — the point the camera always faces.
    centroid: Point3<f32>,
    /// World "up" direction (normalised).  Used as the yaw rotation axis.
    world_up: Vector3<f32>,
    /// Maximum horizontal sweep in radians (e.g. 0.35 ≈ ±20°).
    max_yaw_rad: f32,
    /// Maximum vertical lift as a fraction of orbit distance (e.g. 0.08 ≈ 8%).
    max_pitch_frac: f32,
    /// Distance from initial eye to centroid.
    orbit_distance: f32,
}

impl CinematicPan {
    /// Create a new cinematic pan sampler.
    ///
    /// * `initial_camera`  – camera state at the moment the pan begins
    /// * `centroid`        – the world-space point the camera always looks toward
    /// * `world_up`        – world up direction (from `PointCloud::up()`)
    /// * `max_yaw_deg`     – horizontal sweep half-width in degrees (recommend 15–25)
    /// * `max_pitch_frac`  – vertical breathing, fraction of orbit radius (recommend 0.05–0.10)
    pub fn new(
        initial_camera: PerspectiveCamera,
        centroid: Point3<f32>,
        world_up: Vector3<f32>,
        max_yaw_deg: f32,
        max_pitch_frac: f32,
    ) -> Self {
        let orbit_distance = initial_camera.position.distance(centroid);
        // Fall back to -Y if the scene has no stored up vector.
        let world_up = if world_up.magnitude2() < 1e-6 {
            Vector3::new(0.0, -1.0, 0.0)
        } else {
            world_up.normalize()
        };
        Self {
            initial_camera,
            centroid,
            world_up,
            max_yaw_rad: max_yaw_deg.to_radians(),
            max_pitch_frac,
            orbit_distance,
        }
    }
}

impl Sampler for CinematicPan {
    type Sample = PerspectiveCamera;

    /// `v` = animation progress 0.0..1.0 (loops).
    /// One full sin oscillation (left → centre → right → centre) per period.
    fn sample(&self, v: f32) -> PerspectiveCamera {
        use std::f32::consts::TAU;

        // ── 1. Compute new eye position ───────────────────────────────────────
        // Main horizontal sweep: one full sine cycle per animation period.
        let yaw_angle = self.max_yaw_rad * (v * TAU).sin();
        // Subtle vertical breathing at half frequency so it never lines up
        // with the horizontal extremes — keeps the motion feeling organic.
        let pitch_lift = self.max_pitch_frac * (v * TAU * 0.5).sin();

        // Orbit the initial offset vector around the world_up axis.
        let orbit_offset = self.initial_camera.position - self.centroid;
        let yaw_q = Quaternion::from_axis_angle(self.world_up, Rad(yaw_angle));
        let mut new_offset = yaw_q * orbit_offset;

        // Add vertical displacement proportional to orbit distance.
        new_offset += self.world_up * (pitch_lift * self.orbit_distance);

        let new_eye = self.centroid + new_offset;

        // ── 2. Build look-at rotation ───────────────────────────────────────
        // Use the same Quaternion::look_at convention the orbit controller uses
        // so the handoff from animation → interactive orbit is seamless:
        //   look_at(forward, up)  where forward = eye → centroid.
        let forward = (self.centroid - new_eye).normalize();
        let new_rotation = Quaternion::look_at(forward, self.world_up);

        PerspectiveCamera {
            position: new_eye,
            rotation: new_rotation,
            projection: self.initial_camera.projection,
        }
    }
}
