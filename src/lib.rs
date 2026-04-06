use std::{
    io::{Read, Seek},
    path::PathBuf,
    sync::Arc,
};

use egui_wgpu::EguiDisplayHandle;
use renderer::Display;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use cgmath::{EuclideanSpace, Point3, Quaternion, Rotation, UlpsEq, Vector2, Vector3};
use egui::FullOutput;
use vb_auto_camera::{AutoCamera, CameraHint, SceneAnalysis};

#[cfg(not(target_arch = "wasm32"))]
use utils::RingBuffer;
use utils::key_to_num;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, TouchPhase as WinitTouchPhase, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

mod animation;
mod ui;
pub use animation::{Animation, CinematicPan, Sampler, TrackingShot, Transition};
mod camera;
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection};
mod controller;
pub use controller::CameraController;
mod pointcloud;
pub use pointcloud::PointCloud;

pub mod io;

mod renderer;
pub use renderer::{GaussianRenderer, SplattingArgs};

mod scene;
use crate::utils::GPUStopwatch;

pub use self::scene::{Scene, SceneCamera, Split};

pub mod gpu_rs;
mod ui_renderer;
mod uniform;
mod utils;

pub struct RenderConfig {
    pub no_vsync: bool,
    pub hdr: bool,
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();
        log::info!("using apdater \"{}\"", adapter.get_info().name);

        #[cfg(target_arch = "wasm32")]
        let required_features = wgpu::Features::default();
        #[cfg(not(target_arch = "wasm32"))]
        let required_features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;

        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                #[cfg(not(target_arch = "wasm32"))]
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_storage_buffers_per_shader_stage: 12,
                    max_compute_workgroup_storage_size: 1 << 15,
                    ..adapter_limits
                },

                #[cfg(target_arch = "wasm32")]
                required_limits: wgpu::Limits {
                    max_compute_workgroup_storage_size: 1 << 15,
                    ..adapter_limits
                },
                label: None,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .unwrap();

        Self {
            device,
            queue,
            adapter,
        }
    }
}

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scale_factor: f32,

    pc: PointCloud,
    pointcloud_file_path: Option<PathBuf>,
    renderer: GaussianRenderer,
    animation: Option<(Animation<PerspectiveCamera>, bool)>,
    controller: CameraController,
    scene: Option<Scene>,
    scene_file_path: Option<PathBuf>,
    current_view: Option<usize>,
    ui_renderer: ui_renderer::EguiWGPU,
    fps: f32,
    ui_visible: bool,
    gamepad_visible: bool,

    #[cfg(not(target_arch = "wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
    display: Display,

    splatting_args: SplattingArgs,
    /// One-shot flag: when `true`, the next frame will re-render the scene even
    /// if `splatting_args` hasn't changed (e.g. wall-tint uniform update).
    force_scene_redraw: bool,

    saved_cameras: Vec<SceneCamera>,
    #[cfg(feature = "video")]
    cameras_save_path: String,
    stopwatch: Option<GPUStopwatch>,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R: Read + Seek>(
        window: Window,
        pc_file: R,
        render_config: &RenderConfig,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size();
        if size == PhysicalSize::new(0, 0) {
            size = PhysicalSize::new(800, 600);
        }

        let window = Arc::new(window);

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(window.clone_for_wgpu()),
        );

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        let device = &wgpu_context.device;
        let queue = &wgpu_context.queue;

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();

        let render_format = if render_config.hdr {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.remove_srgb_suffix()],
        };
        surface.configure(&device, &config);

        let pc_raw = io::GenericGaussianPointCloud::load(pc_file)?;
        // ── Auto-frame via vb_auto_camera (same algorithm as thumbnail.rs) ──────────────
        // Uses opacity-weighted scene analysis — more robust than the old centroid+radius
        // heuristic — so the initial view matches the thumbnail exactly.
        let (view_camera, centroid, world_up) =
            auto_frame_from_raw(&pc_raw, size.width, size.height);
        let pc = PointCloud::new(&device, pc_raw)?;
        log::info!("loaded point cloud with {:} points", pc.num_points());

        let renderer =
            GaussianRenderer::new(&device, &queue, render_format, pc.sh_deg(), pc.compressed())
                .await;

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = centroid;
        controller.up = Some(world_up);

        // Cache scene bounds so JS can read them via get_scene_bounds() at any time.
        #[cfg(target_arch = "wasm32")]
        SCENE_BOUNDS.with(|cell| {
            *cell.borrow_mut() = Some(pc.scene_bounds());
        });
        let ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);

        let display = Display::new(
            device,
            render_format,
            surface_format.remove_srgb_suffix(),
            size.width,
            size.height,
        );

        let stopwatch = if cfg!(not(target_arch = "wasm32")) {
            Some(GPUStopwatch::new(device, Some(3)))
        } else {
            None
        };

        Ok(Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            renderer,
            splatting_args: SplattingArgs {
                camera: view_camera,
                viewport: Vector2::new(size.width, size.height),
                gaussian_scaling: 1.,
                max_sh_deg: pc.sh_deg(),
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::ZERO,
                scene_center: None,
                scene_extend: None,
                background_color: wgpu::Color::BLACK,
            },
            pc,
            // camera: view_camera,
            controller,
            ui_renderer,
            fps: 0.,
            #[cfg(not(target_arch = "wasm32"))]
            history: RingBuffer::new(512),
            ui_visible: false,
            gamepad_visible: false,
            display,
            force_scene_redraw: false,
            saved_cameras: Vec::new(),
            #[cfg(feature = "video")]
            cameras_save_path: "cameras_saved.json".to_string(),
            animation: None,
            scene: None,
            current_view: None,
            pointcloud_file_path: None,
            scene_file_path: None,

            stopwatch,
        })
    }

    fn reload(&mut self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.pointcloud_file_path {
            log::info!("reloading volume from {:?}", file_path);
            let file = std::fs::File::open(file_path)?;
            let pc_raw = io::GenericGaussianPointCloud::load(file)?;
            self.pc = PointCloud::new(&self.wgpu_context.device, pc_raw)?;
        } else {
            return Err(anyhow::anyhow!("no pointcloud file path present"));
        }
        if let Some(scene_path) = &self.scene_file_path {
            log::info!("reloading scene from {:?}", scene_path);
            let file = std::fs::File::open(scene_path)?;

            self.set_scene(Scene::from_json(file)?);
        }
        Ok(())
    }

    /// Reload point cloud from raw bytes (used for in-place re-render on WASM without restart)
    #[cfg(target_arch = "wasm32")]
    fn reload_from_bytes(
        &mut self,
        pc_bytes: Vec<u8>,
        scene_bytes: Option<Vec<u8>>,
    ) -> anyhow::Result<()> {
        use std::io::Cursor;
        let pc_raw = io::GenericGaussianPointCloud::load(Cursor::new(pc_bytes))?;
        let (new_camera, centroid, world_up) =
            auto_frame_from_raw(&pc_raw, self.config.width, self.config.height);
        self.pc = PointCloud::new(&self.wgpu_context.device, pc_raw)?;
        self.splatting_args.camera = new_camera;
        // Mirror exactly what WindowContext::new does — just set center, no reset_to_camera.
        // Calling reset_to_camera with identity rotation corrupts the controller center
        // (projects it along +Z instead of toward the PC center) causing wrong camera orientation.
        self.controller.center = centroid;
        self.controller.up = Some(world_up);
        // Update the JS-readable bounds cache.
        SCENE_BOUNDS.with(|cell| {
            *cell.borrow_mut() = Some(self.pc.scene_bounds());
        });
        self.splatting_args.max_sh_deg = self.pc.sh_deg();
        self.animation = None;
        self.scene = None;
        self.current_view = None;
        self.splatting_args.scene_center = None;
        self.splatting_args.scene_extend = None;
        self.splatting_args.walltime = Duration::ZERO;

        if let Some(scene_bytes) = scene_bytes {
            match Scene::from_json(Cursor::new(scene_bytes)) {
                Ok(scene) => {
                    self.set_scene(scene);
                    self.set_scene_camera(0);
                }
                Err(e) => log::warn!("could not load scene on reload: {:?}", e),
            }
        }

        // Auto-start the cinematic intro pan so the scene feels alive on first load.
        // 18.75-second period, ±18° horizontal sweep.
        // Stops immediately when the user clicks, touches, scrolls, or presses a key.
        self.start_cinematic_pan(18.75, 18.0);

        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface
                .configure(&self.wgpu_context.device, &self.config);
            self.display
                .resize(&self.wgpu_context.device, new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
            self.splatting_args.viewport = Vector2::new(new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    /// returns whether redraw is required
    fn ui(&mut self) -> (bool, egui::FullOutput) {
        self.ui_renderer.begin_frame(&self.window);
        let request_redraw = ui::ui(self);

        let shapes = self.ui_renderer.end_frame(&self.window);

        return (request_redraw, shapes);
    }

    /// returns whether the sceen changed and we need a redraw
    fn update(&mut self, dt: Duration) {
        // ema fps update

        if self.splatting_args.walltime < Duration::from_secs(5) {
            self.splatting_args.walltime += dt;
        }
        if let Some((next_camera, playing)) = &mut self.animation {
            if self.controller.user_inptut {
                self.cancle_animation()
            } else {
                let dt = if *playing { dt } else { Duration::ZERO };
                self.splatting_args.camera = next_camera.update(dt);
                self.splatting_args
                    .camera
                    .projection
                    .resize(self.config.width, self.config.height);
                if next_camera.done() {
                    self.animation.take();
                    self.controller.reset_to_camera(self.splatting_args.camera);
                }
            }
        } else {
            self.controller
                .update_camera(&mut self.splatting_args.camera, dt);

            // check if camera moved out of selected view
            if let Some(idx) = self.current_view {
                if let Some(scene) = &self.scene {
                    if let Some(camera) = scene.camera(idx) {
                        let scene_camera: PerspectiveCamera = camera.into();
                        if !self.splatting_args.camera.position.ulps_eq(
                            &scene_camera.position,
                            1e-4,
                            f32::default_max_ulps(),
                        ) || !self.splatting_args.camera.rotation.ulps_eq(
                            &scene_camera.rotation,
                            1e-4,
                            f32::default_max_ulps(),
                        ) {
                            self.current_view.take();
                        }
                    }
                }
            }
        }

        let aabb = self.pc.bbox();
        self.splatting_args.camera.fit_near_far(aabb);
    }

    fn render(
        &mut self,
        redraw_scene: bool,
        shapes: Option<FullOutput>,
    ) -> Result<(), wgpu::CurrentSurfaceTexture> {
        self.stopwatch.as_mut().map(|s| s.reset());

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            err => return Err(err),
        };
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let view_srgb = output.texture.create_view(&Default::default());
        // do prepare stuff

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        if redraw_scene {
            self.renderer.prepare(
                &mut encoder,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &self.pc,
                self.splatting_args,
                (&mut self.stopwatch).into(),
            );
        }

        let ui_state = shapes.map(|shapes| {
            self.ui_renderer.prepare(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &mut encoder,
                shapes,
            )
        });

        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.start(&mut encoder, "rasterization").unwrap();
        }
        if redraw_scene {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.display.texture(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            self.renderer.render(&mut render_pass, &self.pc);
        }
        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.stop(&mut encoder, "rasterization").unwrap();
        }

        self.display.render(
            &mut encoder,
            &view_rgb,
            self.splatting_args.background_color,
            self.renderer.camera(),
            &self.renderer.render_settings(),
        );
        self.stopwatch.as_mut().map(|s| s.end(&mut encoder));

        if let Some(state) = &ui_state {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view_srgb,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.ui_renderer.render(&mut render_pass, state);
        }

        if let Some(ui_state) = ui_state {
            self.ui_renderer.cleanup(ui_state)
        }
        self.wgpu_context.queue.submit([encoder.finish()]);

        output.present();
        self.splatting_args.viewport = Vector2::new(self.config.width, self.config.height);
        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
        self.splatting_args.scene_extend = Some(scene.extend());
        let mut center = Point3::origin();
        for c in scene.cameras(None) {
            let z_axis: Vector3<f32> = c.rotation[2].into();
            center += Vector3::from(c.position) + z_axis * 2.;
        }
        center /= scene.num_cameras() as f32;

        self.controller.center = center;
        self.scene.replace(scene);
        if self.saved_cameras.is_empty() {
            self.saved_cameras = self
                .scene
                .as_ref()
                .unwrap()
                .cameras(Some(Split::Test))
                .clone();
        }
    }

    fn start_tracking_shot(&mut self) {
        if self.saved_cameras.len() > 1 {
            let shot = TrackingShot::from_cameras(self.saved_cameras.clone());
            let a: Animation<PerspectiveCamera> = Animation::new(
                Duration::from_secs_f32(self.saved_cameras.len() as f32 * 4.),
                true,
                Box::new(shot),
            );
            self.animation = Some((a, true));
        }
    }

    /// Starts the cinematic intro animation — a smooth sine-wave pan around
    /// the scene centroid.  Automatically stops the moment the user interacts
    /// (mouse, touch, keyboard, scroll) via the existing `user_inptut` flag.
    ///
    /// * `period_secs` — duration of one full left-right cycle (recommend 8–12).
    /// * `yaw_deg`     — horizontal sweep half-width in degrees (recommend 15–22).
    fn start_cinematic_pan(&mut self, period_secs: f32, yaw_deg: f32) {
        // Reset user-input flag so a previous interaction doesn't immediately
        // cancel the animation we are about to start.
        self.controller.user_inptut = false;

        let centroid = self.controller.center;
        let world_up = robust_scene_up(&self.pc);

        let sampler = CinematicPan::new(
            self.splatting_args.camera,
            centroid,
            world_up,
            yaw_deg,
            0.06, // subtle vertical breathing: 6% of orbit radius
        );

        let anim = Animation::new(
            Duration::from_secs_f32(period_secs),
            true, // looping — continues until user interacts
            Box::new(sampler),
        );
        self.animation = Some((anim, true));
    }

    fn cancle_animation(&mut self) {
        self.animation.take();
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn stop_animation(&mut self) {
        if let Some((_animation, playing)) = &mut self.animation {
            *playing = false;
        }
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn set_scene_camera(&mut self, i: usize) {
        if let Some(scene) = &self.scene {
            self.current_view.replace(i);
            log::info!("view moved to camera {i}");
            if let Some(camera) = scene.camera(i) {
                self.set_camera(camera, Duration::from_millis(200));
            } else {
                log::error!("camera {i} not found");
            }
        }
    }

    pub fn set_camera<C: Into<PerspectiveCamera>>(
        &mut self,
        camera: C,
        animation_duration: Duration,
    ) {
        let camera: PerspectiveCamera = camera.into();
        if animation_duration.is_zero() {
            self.update_camera(camera.into())
        } else {
            let target_camera = camera.into();
            let a = Animation::new(
                animation_duration,
                false,
                Box::new(Transition::new(
                    self.splatting_args.camera.clone(),
                    target_camera,
                    smoothstep,
                )),
            );
            self.animation = Some((a, true));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        self.splatting_args.camera = camera;
        self.splatting_args
            .camera
            .projection
            .resize(self.config.width, self.config.height);
    }

    fn save_view(&mut self) {
        let max_scene_id = if let Some(scene) = &self.scene {
            scene.cameras(None).iter().map(|c| c.id).max().unwrap_or(0)
        } else {
            0
        };
        let max_id = self.saved_cameras.iter().map(|c| c.id).max().unwrap_or(0);
        let id = max_id.max(max_scene_id) + 1;
        self.saved_cameras.push(SceneCamera::from_perspective(
            self.splatting_args.camera,
            id.to_string(),
            id,
            Vector2::new(self.config.width, self.config.height),
            Split::Test,
        ));
    }
}

/// Determine a robust "up" direction for the scene.
///
/// **Algorithm** (priority order):
///
/// 1. **PCA normal** — `plane_from_points` (in `io/mod.rs`) computes the minimum-
///    spread axis of the splat distribution and flips it toward +Y.  We trust it
///    when its Y component is large enough (|Y| ≥ 0.5, i.e. within 60° of +Y),
///    which covers every typical floor/ceiling-dominant capture.
///
/// 2. **AABB Y-axis check** — When PCA is unavailable or too horizontal, we fall
///    back to the AABB *only* if Y is clearly the shortest extent (ratio < 0.70).
///    We **never** return X or Z as the up axis: those are always floor-parallel
///    (walls / depth) and returning them as "up" causes a 90° camera tilt.
///
/// 3. **+Y fallback** — The standard 3DGS / COLMAP world convention is Y-up.
///    When neither PCA nor a clear AABB Y axis is available, +Y is the safest
///    default and correct for the vast majority of PLY captures.
fn robust_scene_up(pc: &PointCloud) -> Vector3<f32> {
    use cgmath::InnerSpace;

    let pca_up = pc.up();

    // ── Priority 1: PCA normal ──────────────────────────────────────────────
    // Trust PCA when it points mostly toward Y (within 60° → |Y| ≥ 0.5).
    // plane_from_points already flips the normal so Y ≥ 0; the belt-and-
    // suspenders sign check here handles any edge case.
    if let Some(pca) = pca_up {
        let n = pca.normalize();
        if n.y.abs() >= 0.5 {
            let up = if n.y >= 0.0 { n } else { -n };
            log::info!(
                "robust_scene_up: PCA ({:.4},{:.4},{:.4}) — using it",
                up.x, up.y, up.z
            );
            return up;
        }
        log::info!(
            "robust_scene_up: PCA ({:.4},{:.4},{:.4}) too horizontal (|Y|<0.5), skipping",
            n.x, n.y, n.z
        );
    }

    // ── Priority 2: AABB, only when Y is unambiguously shortest ────────────
    // Returning X or Z as "up" rotates the camera 90°, so we never do that.
    {
        let s = pc.bbox().size();
        let ay = s.y.abs();
        let ax = s.x.abs();
        let az = s.z.abs();
        if ay < ax && ay < az {
            let ratio = ay / ax.min(az).max(1e-6);
            if ratio < 0.70 {
                log::info!(
                    "robust_scene_up: AABB Y is shortest (ratio {:.3}), using +Y",
                    ratio
                );
                return Vector3::unit_y();
            }
        }
    }

    // ── Priority 3: +Y default ──────────────────────────────────────────────
    log::info!("robust_scene_up: fallback to +Y (3DGS convention)");
    Vector3::unit_y()
}

/// Project `base_dir` onto the floor plane (perpendicular to `up`) and normalise.
/// Falls back to +X if `base_dir` is nearly parallel to `up`.
fn floor_projected_direction(base_dir: Vector3<f32>, up: Vector3<f32>) -> Vector3<f32> {
    use cgmath::InnerSpace;
    let proj = base_dir - up * up.dot(base_dir);
    if proj.magnitude() > 0.1 {
        proj.normalize()
    } else {
        let alt = Vector3::unit_x();
        (alt - up * up.dot(alt)).normalize()
    }
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

/// Snap an arbitrary unit vector to the nearest cardinal axis (±X, ±Y, ±Z).
/// Used to keep the horizon level when PCA returns a slightly-tilted floor normal.
fn snap_to_cardinal_cgmath(v: Vector3<f32>) -> Vector3<f32> {
    let abs_x = v.x.abs();
    let abs_y = v.y.abs();
    let abs_z = v.z.abs();
    let sign_x = if v.x >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let sign_y = if v.y >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let sign_z = if v.z >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    if abs_x >= abs_y && abs_x >= abs_z {
        Vector3::new(sign_x, 0.0, 0.0)
    } else if abs_y >= abs_x && abs_y >= abs_z {
        Vector3::new(0.0, sign_y, 0.0)
    } else {
        Vector3::new(0.0, 0.0, sign_z)
    }
}

/// Build a `PerspectiveCamera` using `vb_auto_camera` — the same algorithm used
/// by `thumbnail.rs` — so the web viewer's initial framing matches the thumbnail.
fn auto_frame_from_raw(
    pc_raw: &io::GenericGaussianPointCloud,
    width: u32,
    height: u32,
) -> (PerspectiveCamera, Point3<f32>, Vector3<f32>) {
    use cgmath::Rad;
    let (positions, opacities) = pc_raw.positions_and_opacities();
    let weights: Option<&[f32]> = if opacities.is_empty() { None } else { Some(&opacities) };
    let analysis = SceneAnalysis::from_points(&positions, weights);
    let auto_cam = AutoCamera::frame(&analysis, width, height, CameraHint::Auto);

    let position = Point3::new(
        auto_cam.position.x(),
        auto_cam.position.y(),
        auto_cam.position.z(),
    );
    let look_dir = Vector3::new(
        auto_cam.forward.x(),
        auto_cam.forward.y(),
        auto_cam.forward.z(),
    );
    let raw_up = Vector3::new(
        auto_cam.world_up.x(),
        auto_cam.world_up.y(),
        auto_cam.world_up.z(),
    );
    let world_up = snap_to_cardinal_cgmath(raw_up);
    let rotation = Quaternion::look_at(look_dir, world_up);

    let aspect = width as f32 / height.max(1) as f32;
    let fov_y = Rad(auto_cam.fov_y_rad);
    let fov_x = Rad(2.0_f32 * ((auto_cam.fov_y_rad * 0.5).tan() * aspect).atan());

    let camera = PerspectiveCamera::new(
        position,
        rotation,
        PerspectiveProjection::new(
            Vector2::new(width, height),
            Vector2::new(fov_x, fov_y),
            auto_cam.near,
            auto_cam.far,
        ),
    );
    let centroid = Point3::new(
        auto_cam.target.x(),
        auto_cam.target.y(),
        auto_cam.target.z(),
    );
    (camera, centroid, world_up)
}

pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(
    file: R,
    scene_file: Option<R>,
    config: RenderConfig,
    pointcloud_file_path: Option<PathBuf>,
    scene_file_path: Option<PathBuf>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let scene = scene_file.and_then(|f| match Scene::from_json(f) {
        Ok(s) => Some(s),
        Err(err) => {
            log::error!("cannot load scene: {:?}", err);
            None
        }
    });

    // let window_size = if let Some(scene) = &scene {
    //     let camera = scene.camera(0).unwrap();
    //     let factor = 1200. / camera.width as f32;
    //     LogicalSize::new(
    //         (camera.width as f32 * factor) as u32,
    //         (camera.height as f32 * factor) as u32,
    //     )
    // } else {
    //     LogicalSize::new(800, 600)
    // };
    let window_size = LogicalSize::new(800, 600);
    let window_attributes = Window::default_attributes()
        .with_inner_size(window_size)
        .with_title(format!(
            "{} ({})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ));

    #[allow(deprecated)]
    let window = event_loop.create_window(window_attributes).unwrap();

    #[cfg(target_arch = "wasm32")]
    let my_generation: u32;
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        my_generation = RENDER_GENERATION.load(std::sync::atomic::Ordering::SeqCst);
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                doc.get_element_by_id("loading-display")
                    .unwrap()
                    .set_text_content(Some("Unpacking"));
                // Remove any existing canvas to prevent invisible stacking on re-render
                if let Some(old_canvas) = doc.get_element_by_id("window-canvas") {
                    if let Some(parent) = old_canvas.parent_node() {
                        let _ = parent.remove_child(&*old_canvas);
                    }
                }
                doc.body()
            })
            .and_then(|body| {
                let canvas = window.canvas().unwrap();
                canvas.set_id("window-canvas");
                canvas.set_width(body.client_width() as u32);
                canvas.set_height(body.client_height() as u32);
                let elm = web_sys::Element::from(canvas);
                elm.set_attribute("style", "width: 100%; height: 100%;")
                    .unwrap();
                body.append_child(&elm).ok()
            })
            .expect("couldn't append canvas to document body");
    }

    // limit the redraw rate to the monitor refresh rate
    let min_wait = window
        .current_monitor()
        .map(|m| {
            let hz = m.refresh_rate_millihertz().unwrap_or(60_000);
            Duration::from_millis(1000000 / hz as u64)
        })
        .unwrap_or(Duration::from_millis(17));

    let mut state = WindowContext::new(window, file, &config).await.unwrap();
    state.pointcloud_file_path = pointcloud_file_path;

    if let Some(scene) = scene {
        state.set_scene(scene);
        state.scene_file_path = scene_file_path;
    }

    #[cfg(target_arch = "wasm32")]
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            doc.get_element_by_id("spinner")
                .unwrap()
                .set_attribute("style", "display:none;")
                .unwrap();
            doc.body()
        });

    let mut last = Instant::now();

    // Auto-start the cinematic intro pan so the scene feels alive on first load.
    // Stops immediately the moment the user clicks, touches, scrolls, or presses a key.
    state.start_cinematic_pan(18.75, 18.0);

    #[allow(deprecated)]
    event_loop.run(move |event,target| {
        // Stop this event loop if a newer render has been started
        #[cfg(target_arch = "wasm32")]
        if RENDER_GENERATION.load(std::sync::atomic::Ordering::Relaxed) != my_generation {
            target.exit();
            return;
        }

        // On WASM: check if JS requested an in-place reload (new file uploaded while already rendering).
        // We must NOT create a new EventLoop for re-renders — instead we swap the point cloud here.
        #[cfg(target_arch = "wasm32")]
        if let Event::NewEvents(_) = &event {
            let pending = PENDING_RELOAD.with(|cell| cell.borrow_mut().take());
            if let Some((pc_bytes, scene_bytes)) = pending {
                match state.reload_from_bytes(pc_bytes, scene_bytes) {
                    Ok(()) => {
                        log::info!("point cloud hot-reloaded successfully");
                        // Hide the "Unpacking" spinner now that reload is complete
                        let _ = web_sys::window()
                            .and_then(|win| win.document())
                            .and_then(|doc| {
                                doc.get_element_by_id("spinner").map(|elem| {
                                    let _ = elem.set_attribute("style", "display:none;");
                                })
                            });
                        state.window.request_redraw();
                    }
                    Err(e) => log::error!("hot-reload failed: {:?}", e),
                }
            }

            // Check if JS or gamepad called auto_center_camera() / return_to_origin() to reset the view.
            let needs_center = PENDING_AUTO_CENTER.with(|cell| {
                std::mem::replace(&mut *cell.borrow_mut(), false)
            }) || PENDING_RETURN_TO_ORIGIN.with(|cell| {
                std::mem::replace(&mut *cell.borrow_mut(), false)
            });
            if needs_center {
                // Re-frame using the same interior-placement formula as WindowContext::new().
                let (centroid, radius) = state.pc.centroid_and_radius();
                let world_up = robust_scene_up(&state.pc);
                let back_dist = radius * 0.12;
                let eye_rise  = radius * 0.063;
                let look_dir = floor_projected_direction(Vector3::unit_z(), world_up);
                let camera_offset = -look_dir * back_dist + world_up * eye_rise;
                let camera_rotation = Quaternion::look_at(look_dir, world_up);
                state.splatting_args.camera.position = centroid + camera_offset;
                state.splatting_args.camera.rotation = camera_rotation;
                state.splatting_args.camera.fit_near_far(state.pc.bbox());
                state.controller.center = centroid;
                state.controller.up = Some(world_up);
                log::info!(
                    "auto_frame_scene: centroid={:?} radius={:.3}",
                    centroid,
                    radius
                );
                state.window.request_redraw();
            }

            // Check if JS called start_cinematic_pan_wasm() or stop_cinematic_pan_wasm().
            if let Some(start) = PENDING_CINEMATIC.with(|cell| cell.borrow_mut().take()) {
                if start {
                    state.start_cinematic_pan(18.75, 18.0);
                } else {
                    state.cancle_animation();
                }
                state.window.request_redraw();
            }

            // Check if JS called set_wall_tint_wasm() or clear_wall_tint_wasm().
            if let Some(tint) = PENDING_WALL_TINT.with(|cell| cell.borrow_mut().take()) {
                state.renderer.set_wall_tint(&state.wgpu_context.queue, tint);
                state.force_scene_redraw = true;
                state.window.request_redraw();
            }
        }

        match event {
            Event::NewEvents(e) =>  match e{
                winit::event::StartCause::ResumeTimeReached { .. }=>{
                    state.window.request_redraw();
                }
                _=>{}
            },
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() && !state.ui_renderer.on_event(&state.window,event) => match event {
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size, None);
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                ..
            } => {
                state.scale_factor = *scale_factor as f32;
            }
            WindowEvent::CloseRequested => {log::info!("close!");target.exit()},
            WindowEvent::ModifiersChanged(m)=>{
                state.controller.alt_pressed = m.state().alt_key();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key{
                if event.state == ElementState::Released{

                    if key == KeyCode::KeyT{
                        if state.animation.is_none(){
                            state.start_tracking_shot();
                        }else{
                            state.stop_animation()
                        }
                    }else if key == KeyCode::KeyU{
                        state.ui_visible = !state.ui_visible;
                    }else if key == KeyCode::KeyV{
                        state.gamepad_visible = !state.gamepad_visible;
                    }else if key == KeyCode::KeyC{
                        state.save_view();
                    } else  if key == KeyCode::KeyR && state.controller.alt_pressed{
                        if let Err(err) = state.reload(){
                            log::error!("failed to reload volume: {:?}", err);
                        }
                    }else if let Some(scene) = &state.scene{

                        let new_camera =
                        if let Some(num) = key_to_num(key){
                            Some(num as usize)
                        }
                        else if key == KeyCode::KeyR{
                            Some((rand::random::<u32>() as usize)%scene.num_cameras())
                        }else if key == KeyCode::KeyN{
                            scene.nearest_camera(state.splatting_args.camera.position,None)
                        }else if key == KeyCode::PageUp{
                            Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                        }else if key == KeyCode::KeyT{
                            Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                        }
                        else if key == KeyCode::PageDown{
                            Some(state.current_view.map_or(0, |v|v-1) % scene.num_cameras())
                        }else{None};

                        if let Some(new_camera) = new_camera{
                            state.set_scene_camera(new_camera);
                        }
                    }
                }
                state
                    .controller
                    .process_keyboard(key, event.state == ElementState::Pressed);
            }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                    state.controller.process_scroll(*dy )
                }
                winit::event::MouseScrollDelta::PixelDelta(p) => {
                    state.controller.process_scroll(p.y as f32 / 100.)
                }
            },
            WindowEvent::MouseInput { state:button_state, button, .. }=>{
                match button {
                    winit::event::MouseButton::Left =>                         state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                    winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                    _=>{}
                }
            }
            WindowEvent::Touch(touch) => {
                let touch_phase = match touch.phase {
                    WinitTouchPhase::Started => controller::TouchPhase::Started,
                    WinitTouchPhase::Moved => controller::TouchPhase::Moved,
                    WinitTouchPhase::Ended => controller::TouchPhase::Ended,
                    WinitTouchPhase::Cancelled => controller::TouchPhase::Cancelled,
                };
                let controller_touch = controller::Touch {
                    id: touch.id,
                    position: (touch.location.x as f32, touch.location.y as f32),
                    phase: touch_phase,
                };
                state.controller.process_touch(controller_touch);
            }
            WindowEvent::RedrawRequested => {
                if !config.no_vsync{
                    // make sure the next redraw is called with a small delay
                    target.set_control_flow(ControlFlow::wait_duration(min_wait));
                }
                let now = Instant::now();
                let dt = now-last;
                last = now;

                let old_settings = state.splatting_args.clone();
                state.update(dt);

                let (redraw_ui,shapes) = state.ui();

                let resolution_change = state.splatting_args.viewport != Vector2::new(state.config.width, state.config.height);

                let request_redraw = old_settings != state.splatting_args || resolution_change || state.force_scene_redraw;
                state.force_scene_redraw = false;

                if request_redraw || redraw_ui{
                    state.fps = (1. / dt.as_secs_f32()) * 0.05 + state.fps * 0.95;
                    match state.render(request_redraw, Some(shapes)) {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::CurrentSurfaceTexture::Suboptimal(_)) => state.resize(state.window.inner_size(), None),
                        Err(wgpu::CurrentSurfaceTexture::Lost) => state.resize(state.window.inner_size(), None),
                        // The system is out of memory, we should probably quit
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => println!("error: {:?}", e),
                    }
                }
                if config.no_vsync{
                    state.window.request_redraw();
                }
            }
            _ => {}
        },
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion{ delta, },
            .. // We're not using device_id currently
        } => {
            state.controller.process_mouse(delta.0 as f32, delta.1 as f32)
        }
        _ => {}
        }
    }).unwrap();
}

#[cfg(target_arch = "wasm32")]
static RENDER_GENERATION: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

// Thread-local storage for a pending point cloud reload requested from JS.
// On WASM, a second EventLoop cannot be created, so re-renders must be done
// by signalling the running event loop with new file bytes via reload_pc_wasm().
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_RELOAD: std::cell::RefCell<Option<(Vec<u8>, Option<Vec<u8>>)>> =
        std::cell::RefCell::new(None);
}

/// Signal from JS that the running event loop should re-run auto-centering on the
/// currently loaded point cloud (without reloading it).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_AUTO_CENTER: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
}

/// Signal from JS to start (true) or stop (false) the cinematic pan animation.
/// `None` means no pending request this frame.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_CINEMATIC: std::cell::RefCell<Option<bool>> =
        std::cell::RefCell::new(None);
}

/// Signal from gamepad or JS to return camera to origin (initial auto-framed position).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_RETURN_TO_ORIGIN: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
}

/// Caches [cx, cy, cz, radius] for the currently loaded point cloud so that
/// `get_scene_bounds()` can return them synchronously from JS.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static SCENE_BOUNDS: std::cell::RefCell<Option<[f32; 4]>> =
        std::cell::RefCell::new(None);
}

/// Called from JS when the user uploads a new file while a render is already active.
/// Stores the new data so the running event loop can swap the point cloud in-place.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn reload_pc_wasm(pc: Vec<u8>, scene: Option<Vec<u8>>) {
    PENDING_RELOAD.with(|cell| {
        *cell.borrow_mut() = Some((pc, scene));
    });
}

/// Signals the running event loop to reposition the camera so it frames the entire
/// loaded splat cloud (centroid + radius × 2.2 pull-back with a slight elevation).
/// Call this from JavaScript after `run_wasm` resolves for a guaranteed well-framed
/// first view, or at any time to reset without reloading the file.
///
/// Alias: `auto_frame_scene` is preferred for new code; `auto_center_camera` is kept
/// for backward compatibility.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn auto_center_camera() {
    PENDING_AUTO_CENTER.with(|cell| {
        *cell.borrow_mut() = true;
    });
}

/// Preferred alias for `auto_center_camera`.  Signals the event loop to reset the
/// camera to the optimal framing of the full scene (centroid, 2.2 × bounding radius,
/// slight elevation).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn auto_frame_scene() {
    PENDING_AUTO_CENTER.with(|cell| {
        *cell.borrow_mut() = true;
    });
}

/// Return the camera to the origin (same as auto_frame_scene).
/// Callable from JavaScript and internally from the gamepad "Return to Origin" button.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn return_to_origin() {
    PENDING_RETURN_TO_ORIGIN.with(|cell| {
        *cell.borrow_mut() = true;
    });
}

/// Starts the cinematic intro pan from JavaScript.
///
/// The camera smoothly oscillates left-to-right around the scene centroid with
/// a 10-second period and ±18° horizontal sweep.  The animation loops until
/// stopped by the user (any click, touch, key, or scroll) or until
/// `stop_cinematic_pan_wasm()` is called explicitly.
///
/// Safe to call at any time — replaces any currently running animation.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start_cinematic_pan_wasm() {
    PENDING_CINEMATIC.with(|cell| {
        *cell.borrow_mut() = Some(true);
    });
}

/// Stops the cinematic pan and hands control back to the user.
///
/// Call this if your UI wants to explicitly end the intro before the user
/// interacts — e.g. after an overlay is dismissed.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn stop_cinematic_pan_wasm() {
    PENDING_CINEMATIC.with(|cell| {
        *cell.borrow_mut() = Some(false);
    });
}

/// Pending wall tint change from JS.
/// `Some(Some([r, g, b, strength]))` → set tint; `Some(None)` → clear tint; `None` → no change.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_WALL_TINT: std::cell::RefCell<Option<Option<[f32; 4]>>> =
        std::cell::RefCell::new(None);
}

/// Apply a real-time RGBA wall color tint to the currently rendering scene.
///
/// * `r`, `g`, `b` — tint color in the 0.0–1.0 range.
/// * `strength`    — blend strength in the 0.0–1.0 range (0 = no effect, 1 = full replacement).
///
/// The tint is applied in the Gaussian splat fragment shader on the next rendered frame.
/// Call `clear_wall_tint_wasm()` to restore original colors.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_wall_tint_wasm(r: f32, g: f32, b: f32, strength: f32) {
    PENDING_WALL_TINT.with(|cell| {
        *cell.borrow_mut() = Some(Some([r, g, b, strength]));
    });
}

/// Remove the wall color tint and restore original splat colors.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn clear_wall_tint_wasm() {
    PENDING_WALL_TINT.with(|cell| {
        *cell.borrow_mut() = Some(None);
    });
}

/// Returns the scene bounds as a 4-element `Float32Array`: `[cx, cy, cz, radius]`.
///
/// `cx/cy/cz` — statistical centroid (mean position of all Gaussian splats).
/// `radius`   — radius of the bounding sphere centred on the centroid that
///              encloses every splat (computed from the AABB corners in O(1)).
///
/// Returns `null` if no point cloud has been loaded yet.
///
/// ### Usage in JavaScript / TypeScript
/// ```js
/// const bounds = get_scene_bounds();          // Float32Array [cx, cy, cz, r]
/// if (bounds) console.log("centroid:", bounds[0], bounds[1], bounds[2],
///                          "radius:", bounds[3]);
/// ```
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_scene_bounds() -> Option<Vec<f32>> {
    SCENE_BOUNDS.with(|cell| {
        cell.borrow().map(|b| b.to_vec())
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn run_wasm(
    pc: Vec<u8>,
    scene: Option<Vec<u8>>,
    pc_file: Option<String>,
    scene_file: Option<String>,
) {
    use std::{io::Cursor, str::FromStr};

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let _ = console_log::init(); // ignore error if logger already initialized (e.g. on re-render)
    // Increment generation to signal the currently running event loop (if any) to stop
    RENDER_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    wasm_bindgen_futures::spawn_local(open_window(
        pc_reader,
        scene_reader,
        RenderConfig {
            no_vsync: false,
            hdr: false,
        },
        pc_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
        scene_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
    ));
}
