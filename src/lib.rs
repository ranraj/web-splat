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

use cgmath::{
    EuclideanSpace, InnerSpace, Point3, Quaternion, Rotation, Rotation3, UlpsEq, Vector2, Vector3,
};
use egui::FullOutput;
use vb_auto_camera::{AutoCamera, SceneAnalysis};
pub use vb_auto_camera::CameraHint;

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
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection, VIEWPORT_Y_FLIP};
mod controller;
pub use controller::CameraController;
mod occupancy;
pub use occupancy::OccupancyGrid;
mod pointcloud;
pub use pointcloud::{Aabb, PointCloud};

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

#[cfg(not(target_arch = "wasm32"))]
mod headless_thumbnail;
#[cfg(not(target_arch = "wasm32"))]
pub use headless_thumbnail::{render_ply_panorama_strip_rgb, render_ply_thumbnail_rgb};

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
    /// Convenience constructor for headless (no-surface) native usage.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new_instance() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        Self::new(&instance, None).await
    }

    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> anyhow::Result<Self> {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .map_err(|e| anyhow::anyhow!(
                "No WebGPU adapter found: {:?}.\n\
                 Common causes:\n\
                 - The page is not a secure context. WebGPU only works on https://, \
                   http://localhost, http://127.0.0.1 or http://[::1]. A LAN IP \
                   like http://192.168.x.x:3000 will fail silently here. Open the \
                   app via http://localhost:<port> instead, or serve it over HTTPS.\n\
                 - The browser does not support WebGPU. Use Chrome 113+ or Edge 113+ \
                   on Windows / macOS / ChromeOS / Android.\n\
                 - The GPU is on the driver blocklist (try updating GPU drivers).",
                e
            ))?;
        log::info!("using adapter \"{}\"", adapter.get_info().name);

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
            .map_err(anyhow::Error::from)?;

        Ok(Self {
            device,
            queue,
            adapter,
        })
    }
}

/// How long a single "snap 90°" rotation takes to ease into the new yaw.
/// Picked to feel responsive but not snappy — short enough that four
/// taps back-to-back still feel like discrete steps, long enough that the
/// motion is legible (the original implementation teleported in one frame
/// and users couldn't tell which direction the orbit went).
const SNAP_ROTATE_DURATION_MS: u64 = 450;

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scale_factor: f32,

    pc: PointCloud,
    /// Same opacity-weighted inlier stats as `auto_frame_from_raw` / `vb_auto_camera`.
    /// Used for depth clip (`fit_near_far`) so a handful of far stray Gaussians do not
    /// dominate the projection matrix (otherwise the interior cluster can sit entirely
    /// inside the near-clip shell and nothing draws).
    scene_analysis: SceneAnalysis,
    pointcloud_file_path: Option<PathBuf>,
    renderer: GaussianRenderer,
    animation: Option<(Animation<PerspectiveCamera>, bool)>,
    /// True while `start_cinematic_pan` is the *currently active* animation
    /// (auto-sweep). Flips back to false when any other animation supersedes
    /// it OR when the user interacts with the canvas (which causes
    /// `cancle_animation` to drop the animation in the next frame). Mirrored
    /// to JS via the `vb-cinematic-pan` window event so the React play/pause
    /// icon can render its pressed-state accurately — see
    /// `get_cinematic_pan_running_wasm()` and
    /// `viewer-camera-bridge.ts::subscribeToCinematicSweep`.
    cinematic_active: bool,
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

        // Do not use `new_instance_with_webgpu_detection` on wasm: its probe can
        // return false negatives, and wgpu then drops `BROWSER_WEBGPU` from the
        // instance so `request_adapter` only asks for Vulkan/GL/Metal/DX12 —
        // impossible in the browser (matches NotFound with no WebGPU in
        // `requested_backends`). Gauzilla / vb-renderer use plain `Instance::new`.
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(window.clone_for_wgpu()),
        );

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await?;

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
        let (view_camera, centroid, world_up, scene_analysis) =
            auto_frame_from_raw(&pc_raw, size.width, size.height, CameraHint::Auto);
        let walls = wall_collision_for_raw(&pc_raw);
        let pc = PointCloud::new(&device, pc_raw)?;
        log::info!("loaded point cloud with {:} points", pc.num_points());

        let renderer =
            GaussianRenderer::new(&device, &queue, render_format, pc.sh_deg(), pc.compressed())
                .await;

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = centroid;
        controller.up = Some(world_up);
        controller.walls = walls;

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
            scene_analysis,
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
            cinematic_active: false,
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
            self.scene_analysis = scene_analysis_for_raw(&pc_raw);
            self.controller.walls = wall_collision_for_raw(&pc_raw);
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
        preserve_camera: bool,
    ) -> anyhow::Result<()> {
        use std::io::Cursor;
        let pc_raw = io::GenericGaussianPointCloud::load(Cursor::new(pc_bytes))?;
        let scene_analysis = scene_analysis_for_raw(&pc_raw);
        let (frame_camera, centroid, world_up, _) = auto_frame_from_raw(
            &pc_raw,
            self.config.width,
            self.config.height,
            CameraHint::Auto,
        );
        self.scene_analysis = scene_analysis;
        self.controller.walls = wall_collision_for_raw(&pc_raw);
        self.pc = PointCloud::new(&self.wgpu_context.device, pc_raw)?;
        self.controller.center = centroid;
        if preserve_camera {
            // Bake applies world2view — use the orbit-up vector embedded in
            // the tuned PLY (captured on source, rotated by the bake matrix)
            // so horizontal drag matches source navigation. Fall back to +Y
            // when JS did not pre-install a vector.
            let pca_up = robust_scene_up(&self.pc);
            let baked_up = PENDING_SET_ORBIT_UP.with(|cell| cell.borrow_mut().take());
            self.controller.up = Some(baked_up.unwrap_or_else(Vector3::unit_y));
            if let Some(pivot) = PENDING_SET_ORBIT_PIVOT.with(|cell| cell.borrow_mut().take())
            {
                self.controller.center = pivot;
            }
            let mut cam = self.splatting_args.camera.clone();
            cam.position = Point3::origin();
            cam.rotation = Quaternion::new(1., 0., 0., 0.);
            self.splatting_args.camera = cam;
            // Source tab cancels the intro cinematic on first orbit input, which
            // calls `reset_to_camera` and projects the pivot onto the view ray.
            // Tuned embed loads skip cinematic — without this projection the
            // pivot stays at the raw 3D centroid while the camera is at the
            // embed origin, so horizontal drag feels like swinging around a
            // point off to the side instead of orbiting the shot.
            let pivot_raw = self.controller.center;
            self.controller.reset_to_camera(self.splatting_args.camera);
            log::info!(
                "[web-splat] preserve_camera reload: pivot_raw=({:.4},{:.4},{:.4}) \
                 pivot_projected=({:.4},{:.4},{:.4}) orbit_radius={:.4} \
                 pca_up=({:.4},{:.4},{:.4}) controller.up=+Y",
                pivot_raw.x,
                pivot_raw.y,
                pivot_raw.z,
                self.controller.center.x,
                self.controller.center.y,
                self.controller.center.z,
                (self.splatting_args.camera.position - self.controller.center).magnitude(),
                pca_up.x,
                pca_up.y,
                pca_up.z,
            );
        } else {
            self.controller.up = Some(world_up);
            self.splatting_args.camera = frame_camera;
        }
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
            if let Err(e) = self.reload_scene_only(scene_bytes) {
                log::warn!("could not load scene on reload: {:?}", e);
            }
        }

        // Intro pan only when there is no cameras.json tour path (auto-walk is started from JS).
        // Skip when preserving camera — admin PLY-tuner tuned preview must hold the embed.
        if self.scene.is_none() && !preserve_camera {
            self.start_cinematic_pan(18.75, 18.0);
        }

        Ok(())
    }

    /// Attach or replace cameras.json for auto-walk without reframing the splat view.
    #[cfg(target_arch = "wasm32")]
    fn reload_scene_only(&mut self, scene_bytes: Vec<u8>) -> anyhow::Result<()> {
        use std::io::Cursor;
        let scene = Scene::from_json(Cursor::new(scene_bytes))?;
        self.splatting_args.scene_extend = Some(scene.extend());
        self.scene.replace(scene);
        self.saved_cameras = self
            .scene
            .as_ref()
            .unwrap()
            .cameras(Some(Split::Test))
            .clone();
        self.current_view = None;
        self.animation = None;
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
        #[cfg(target_arch = "wasm32")]
        let dt = dt.min(Duration::from_millis(250));

        #[cfg(target_arch = "wasm32")]
        if take_pending_viewer_chrome_toggle() {
            self.toggle_viewer_chrome();
        }

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
                    // Non-looping animation finished naturally — clear the
                    // cinematic flag too (defensive; the cinematic itself
                    // loops and never reaches this branch).
                    self.set_cinematic_active(false);
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

        let clip_aabb = clip_aabb_from_scene_analysis(&self.scene_analysis);
        self.splatting_args.camera.fit_near_far(&clip_aabb);
    }

    /// Re-frame the camera like initial load / JS `auto_frame_scene()` (gamepad + WASM).
    pub(crate) fn apply_auto_frame_scene(&mut self) {
        let (mut cam, centroid, world_up) = perspective_camera_from_scene_analysis(
            &self.scene_analysis,
            self.config.width,
            self.config.height,
            CameraHint::Auto,
        );
        cam.fit_near_far(&clip_aabb_from_scene_analysis(&self.scene_analysis));
        self.splatting_args.camera = cam;
        self.controller.center = centroid;
        self.controller.up = Some(world_up);
        log::info!(
            "auto_frame_scene: centroid={:?} radius={:.3} (vb_auto p90)",
            centroid,
            self.scene_analysis.radius
        );
        self.window.request_redraw();
    }

    /// Orbit the camera 90° clockwise around the scene up axis (gamepad + WASM).
    ///
    /// Instead of snapping the camera instantly, this builds the post-
    /// rotation target pose and pushes it through the same animation
    /// pipeline used by `set_camera()` — a `Transition<PerspectiveCamera>`
    /// driven by `smoothstep` (eased start + end). The animation lasts
    /// `SNAP_ROTATE_DURATION_MS` so each tap on the React/egui "snap 90°"
    /// button glides the view rather than teleporting it. Four consecutive
    /// taps still trace exactly 360° because position is slerp-ed with the
    /// rotation, leaving the orbit radius stable between snaps.
    pub(crate) fn apply_rotate_view_90_clockwise(&mut self) {
        let world_up = robust_scene_up(&self.pc);
        let center = self.controller.center;
        let rot90 =
            Quaternion::from_axis_angle(world_up, cgmath::Rad(-std::f32::consts::FRAC_PI_2));
        let cam_offset = self.splatting_args.camera.position - center;
        let new_offset = rot90.rotate_vector(cam_offset);
        let new_pos = center + new_offset;
        let new_look_dir = (center - new_pos).normalize();

        // Build the target pose. We clone the live camera so projection /
        // znear / zfar carry over; only position + rotation are rotated.
        let mut target = self.splatting_args.camera.clone();
        target.position = new_pos;
        target.rotation = Quaternion::look_at(new_look_dir, world_up);
        target.fit_near_far(&clip_aabb_from_scene_analysis(&self.scene_analysis));

        // Lock the world-up the controller orbits around BEFORE the
        // animation starts so the post-animation `reset_to_camera` keeps
        // the same up axis (otherwise PCA noise can flip the orbit pivot).
        self.controller.up = Some(world_up);

        // Clear any controller-side "user just typed" flag so the
        // animation we are about to install isn't cancelled on the very
        // next frame by the cancel-on-input branch in `update()`.
        self.controller.user_inptut = false;

        // `set_camera()` installs an `Animation<PerspectiveCamera>` using
        // `Transition` + `smoothstep`. The existing `update()` loop drives
        // it inside `state.update(dt)` — which runs after `old_settings`
        // is captured — so the render gate detects the change naturally
        // (no `force_scene_redraw` workaround needed for the animated
        // path).
        self.set_camera(target, Duration::from_millis(SNAP_ROTATE_DURATION_MS));
        self.window.request_redraw();
    }

    /// Toggle on-screen gamepad + notify JS measure chrome (V key).
    pub(crate) fn toggle_viewer_chrome(&mut self) {
        self.gamepad_visible = !self.gamepad_visible;
        #[cfg(target_arch = "wasm32")]
        set_viewer_chrome_for_js(self.gamepad_visible);
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

        #[cfg(target_arch = "wasm32")]
        {
            snapshot_camera_for_measure(
                &self.splatting_args.camera,
                self.config.width,
                self.config.height,
            );
            snapshot_orbit_axes_for_js(self);
            snapshot_tour_status(self);
            dispatch_splat_frame_event();
        }

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

    /// Update the cinematic-active mirror and notify JS if the value
    /// transitioned. Cheap to call from every animation-starting / clearing
    /// site even when nothing changes — the JS helper deduplicates.
    fn set_cinematic_active(&mut self, active: bool) {
        if self.cinematic_active == active {
            return;
        }
        self.cinematic_active = active;
        #[cfg(target_arch = "wasm32")]
        set_cinematic_active_for_js(active);
    }

    fn start_tracking_shot(&mut self) {
        self.start_tracking_shot_with_duration(None, 1.0);
    }

    /// Catmull-Rom spline through `saved_cameras`. `duration_secs` overrides the default
    /// (`len * 4` seconds per loop); `duration_scale` multiplies that default when `duration_secs` is None.
    fn start_tracking_shot_with_duration(
        &mut self,
        duration_secs: Option<f32>,
        duration_scale: f32,
    ) {
        if self.saved_cameras.len() <= 1 {
            return;
        }
        self.controller.user_inptut = false;
        let shot = TrackingShot::from_cameras(self.saved_cameras.clone());
        let default_secs = self.saved_cameras.len() as f32 * 4.0;
        let total_secs = duration_secs.unwrap_or(default_secs * duration_scale.max(0.1));
        let a: Animation<PerspectiveCamera> = Animation::new(
            Duration::from_secs_f32(total_secs),
            true,
            Box::new(shot),
        );
        self.animation = Some((a, true));
        // A different animation just superseded any cinematic sweep.
        self.set_cinematic_active(false);
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
        // We are now the active animation — sync JS so the play/pause
        // icon flips to "Pause" without waiting for a user click.
        self.set_cinematic_active(true);
    }

    fn cancle_animation(&mut self) {
        self.animation.take();
        self.controller.reset_to_camera(self.splatting_args.camera);
        // Whatever animation was running (including the cinematic auto-sweep)
        // is gone — make sure the React icon reflects "not playing".
        self.set_cinematic_active(false);
    }

    fn stop_animation(&mut self) {
        if let Some((_animation, playing)) = &mut self.animation {
            *playing = false;
        }
        self.controller.reset_to_camera(self.splatting_args.camera);
        self.set_cinematic_active(false);
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
            // Camera-snap animation supersedes any cinematic sweep.
            self.set_cinematic_active(false);
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
        if !n.x.is_finite() || !n.y.is_finite() || !n.z.is_finite() {
            log::info!("robust_scene_up: PCA direction non-finite, skipping");
        } else if n.y.abs() >= 0.5 {
            let up = if n.y >= 0.0 { n } else { -n };
            log::info!(
                "robust_scene_up: PCA ({:.4},{:.4},{:.4}) — using it",
                up.x, up.y, up.z
            );
            return up;
        } else {
            log::info!(
                "robust_scene_up: PCA ({:.4},{:.4},{:.4}) too horizontal (|Y|<0.5), skipping",
                n.x, n.y, n.z
            );
        }
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

/// Inlier axis-aligned box from `vb_auto_camera` (median + 2×p90 filter). Using this for
/// `PerspectiveCamera::fit_near_far` matches the numeric scale of the auto-framed cluster,
/// unlike [`PointCloud::bbox`] which includes arbitrary far outliers.
pub fn clip_aabb_from_scene_analysis(analysis: &SceneAnalysis) -> Aabb<f32> {
    Aabb::new(
        Point3::new(
            analysis.bb_min.x(),
            analysis.bb_min.y(),
            analysis.bb_min.z(),
        ),
        Point3::new(
            analysis.bb_max.x(),
            analysis.bb_max.y(),
            analysis.bb_max.z(),
        ),
    )
}

/// Wall grid from splat centers. None when the cloud is too sparse to form a surface.
fn wall_collision_for_raw(pc_raw: &io::GenericGaussianPointCloud) -> Option<OccupancyGrid> {
    let (positions, _) = pc_raw.positions_and_opacities();
    let walls = OccupancyGrid::build(&positions);
    match &walls {
        Some(grid) => log::info!(
            "[web-splat] wall collision: {} solid cells, cell {:.3}, {} centers",
            grid.solid_cells(),
            grid.cell(),
            positions.len()
        ),
        None => log::info!("[web-splat] wall collision: off (no surface in point cloud)"),
    }
    walls
}

fn scene_analysis_for_raw(pc_raw: &io::GenericGaussianPointCloud) -> SceneAnalysis {
    let (positions, opacities) = pc_raw.positions_and_opacities();
    let weights: Option<&[f32]> = if opacities.is_empty() {
        None
    } else {
        Some(&opacities)
    };
    SceneAnalysis::from_points(&positions, weights)
}

/// Build the same perspective camera as initial web load / thumbnail (`AutoCamera::frame`).
fn perspective_camera_from_scene_analysis(
    analysis: &SceneAnalysis,
    width: u32,
    height: u32,
    hint: CameraHint,
) -> (PerspectiveCamera, Point3<f32>, Vector3<f32>) {
    use cgmath::Rad;
    let auto_cam = AutoCamera::frame(analysis, width, height, hint);

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

/// World-space direction the camera looks along (same convention as [`CameraController`].
pub fn camera_world_forward(cam: &PerspectiveCamera) -> Vector3<f32> {
    use cgmath::Rotation;
    cam.rotation.invert().rotate_vector(Vector3::unit_z()).normalize()
}

/// Build the same perspective camera the WebGPU viewer uses for initial framing
/// (`vb_auto_camera` + `Quaternion::look_at`). Used by the `thumbnail` CLI tool.
pub fn auto_frame_point_cloud(
    pc_raw: &io::GenericGaussianPointCloud,
    width: u32,
    height: u32,
    hint: CameraHint,
) -> (PerspectiveCamera, Point3<f32>, Vector3<f32>, SceneAnalysis) {
    auto_frame_from_raw(pc_raw, width, height, hint)
}

/// Build a `PerspectiveCamera` using `vb_auto_camera` — the same algorithm used
/// by the `thumbnail` tool — so the web viewer's initial framing matches the thumbnail.
fn auto_frame_from_raw(
    pc_raw: &io::GenericGaussianPointCloud,
    width: u32,
    height: u32,
    hint: CameraHint,
) -> (PerspectiveCamera, Point3<f32>, Vector3<f32>, SceneAnalysis) {
    let analysis = scene_analysis_for_raw(pc_raw);
    let (camera, centroid, world_up) =
        perspective_camera_from_scene_analysis(&analysis, width, height, hint);
    (camera, centroid, world_up, analysis)
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
                // Next.js / React can call us before layout: `body.client_*` is often 0 here.
                // A 0×0 canvas makes `request_adapter` with a compatible surface fail (NotFound).
                let win = web_sys::window()?;
                let vw = win
                    .inner_width()
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|x| x.round().max(1.0) as u32)
                    .unwrap_or(800);
                let vh = win
                    .inner_height()
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|x| x.round().max(1.0) as u32)
                    .unwrap_or(600);
                let bw = body.client_width().max(0) as u32;
                let bh = body.client_height().max(0) as u32;
                let cw = bw.max(vw).max(1);
                let ch = bh.max(vh).max(1);
                canvas.set_width(cw);
                canvas.set_height(ch);
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

    let mut state = match WindowContext::new(window, file, &config).await {
        Ok(state) => state,
        Err(err) => {
            log::error!("failed to initialize renderer: {:?}", err);
            #[cfg(target_arch = "wasm32")]
            {
                let msg = format!("Failed to load point cloud: {}", err);
                let _ = web_sys::window()
                    .and_then(|win| win.document())
                    .map(|doc| {
                        if let Some(elm) = doc.get_element_by_id("loading-display") {
                            elm.set_text_content(Some(&msg));
                        }
                        if let Some(spinner) = doc.get_element_by_id("spinner") {
                            let _ = spinner.set_attribute("style", "display:none;");
                        }
                    });
            }
            return;
        }
    };
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

    // Intro pan only without a cameras.json scene; cinematic auto-walk uses start_tracking_shot_wasm.
    if state.scene.is_none() {
        state.start_cinematic_pan(18.75, 18.0);
    }

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
            let pending_scene = PENDING_SCENE_RELOAD.with(|cell| cell.borrow_mut().take());
            if let Some(scene_bytes) = pending_scene {
                match state.reload_scene_only(scene_bytes) {
                    Ok(()) => log::info!("scene hot-reloaded (view camera preserved)"),
                    Err(e) => log::error!("scene hot-reload failed: {:?}", e),
                }
            }

            let pending = PENDING_RELOAD.with(|cell| cell.borrow_mut().take());
            if let Some((pc_bytes, scene_bytes)) = pending {
                let preserve_camera = PENDING_RELOAD_PRESERVE_CAMERA.with(|cell| {
                    std::mem::replace(&mut *cell.borrow_mut(), false)
                });
                match state.reload_from_bytes(pc_bytes, scene_bytes, preserve_camera) {
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
            //
            // We force a render via `force_scene_redraw` because these snap
            // actions mutate `splatting_args.camera` directly, BEFORE the
            // RedrawRequested handler captures `old_settings`. By the time
            // the render gate compares old vs. new settings, both already
            // hold the snapped camera, so the inequality check would say
            // "nothing changed" and skip the render — leaving the user
            // staring at the stale pre-snap frame.
            let needs_center = PENDING_AUTO_CENTER.with(|cell| {
                std::mem::replace(&mut *cell.borrow_mut(), false)
            }) || PENDING_RETURN_TO_ORIGIN.with(|cell| {
                std::mem::replace(&mut *cell.borrow_mut(), false)
            });
            if needs_center {
                log::info!("[wasm] auto-center / return-to-origin drained");
                state.apply_auto_frame_scene();
                state.force_scene_redraw = true;
                state.controller.user_inptut = true;
                state.window.request_redraw();
            }

            // Check if JS called rotate_view_90_wasm() — orbit 90° around world_up.
            //
            // `apply_rotate_view_90_clockwise` now installs an
            // `Animation<PerspectiveCamera>` (smoothstep, ~450ms) instead
            // of mutating `splatting_args.camera` in-place. The animation
            // is sampled inside `state.update(dt)` — which runs after
            // `old_settings` is captured by the render gate — so the
            // gate's `old_settings != state.splatting_args` check fires
            // every frame the animation is running. We deliberately
            // DON'T set `controller.user_inptut = true` here: that flag
            // is the cancel-on-input signal for the animation branch in
            // `update()`, so toggling it on would cancel the snap before
            // the first sampled frame is ever drawn.
            if PENDING_ROTATE_90.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), false)) {
                log::info!("[wasm] snap-90 drained — starting eased rotate animation");
                state.apply_rotate_view_90_clockwise();
                state.window.request_redraw();
            }

            if PENDING_CLEAR_ORBIT_UP.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), false))
            {
                state.controller.up = None;
            }

            if PENDING_RESET_ORBIT_TO_CAMERA
                .with(|cell| std::mem::replace(&mut *cell.borrow_mut(), false))
            {
                state.controller.reset_to_camera(state.splatting_args.camera);
            }

            if let Some(up) =
                PENDING_SET_ORBIT_UP.with(|cell| cell.borrow_mut().take())
            {
                state.controller.up = Some(up);
            }

            if let Some(pivot) =
                PENDING_SET_ORBIT_PIVOT.with(|cell| cell.borrow_mut().take())
            {
                state.controller.center = pivot;
            }

            // JS-driven exact-camera install (Scene Tuner "Apply view",
            // pose persistence, "Capture & restore"). Animates via the
            // same Transition<PerspectiveCamera> pipeline as set_scene_camera
            // when animate_ms > 0; snaps otherwise. Cancels any running
            // animation (cinematic sweep, tour) — same semantics as
            // `set_scene_camera` because we go through `set_camera()`.
            if let Some([px, py, pz, qw, qx, qy, qz, fov_y, animate_ms]) =
                PENDING_SET_CAMERA.with(|cell| cell.borrow_mut().take())
            {
                let mut target = state.splatting_args.camera.clone();
                target.position = Point3::new(px, py, pz);
                target.rotation = Quaternion::new(qw, qx, qy, qz);
                if fov_y > 0.0 {
                    let prev_fov_y = target.projection.fovy.0;
                    if prev_fov_y > 1e-6 {
                        // Preserve aspect: fovx scales with the same ratio as fovy.
                        let scale = fov_y / prev_fov_y;
                        target.projection.fovy = cgmath::Rad(fov_y);
                        target.projection.fovx = cgmath::Rad(target.projection.fovx.0 * scale);
                    }
                }
                let dur = Duration::from_millis(animate_ms as u64);
                state.set_camera(target, dur);
                state.force_scene_redraw = true;
                state.controller.user_inptut = true;
                state.window.request_redraw();
            }

            // Mirror the React virtual-gamepad onto the controller using
            // edge-triggered dispatch: only overwrite a controller flag when
            // the React snapshot transitions vs. the last frame.
            //
            // This avoids two failure modes the original implementation hit:
            //   * OR'ing every frame — a release would never clear because
            //     `existing_true || new_false == true`. (Symptom: "Move
            //     camera doesn't stop after I let go.")
            //   * Always overwriting — clobbers the keyboard's flag every
            //     frame, so OS WASD input gets reset to false repeatedly.
            //
            // Edge dispatch lets each input source own its own transitions:
            // press → flag goes true once, release → flag goes false once,
            // and frames where React is idle leave the controller alone.
            let movement = PENDING_MOVEMENT_KEYS.with(|cell| *cell.borrow());
            let prev = LAST_APPLIED_JS_MOVEMENT.with(|cell| *cell.borrow());
            if movement != prev {
                if movement[0] != prev[0] { state.controller.move_forward  = movement[0]; }
                if movement[1] != prev[1] { state.controller.move_backward = movement[1]; }
                if movement[2] != prev[2] { state.controller.move_left     = movement[2]; }
                if movement[3] != prev[3] { state.controller.move_right    = movement[3]; }
                if movement[4] != prev[4] { state.controller.move_up       = movement[4]; }
                if movement[5] != prev[5] { state.controller.move_down     = movement[5]; }
                if movement[6] != prev[6] { state.controller.roll_left     = movement[6]; }
                if movement[7] != prev[7] { state.controller.roll_right    = movement[7]; }
                LAST_APPLIED_JS_MOVEMENT.with(|cell| *cell.borrow_mut() = movement);
                state.controller.user_inptut = true;
                state.window.request_redraw();
            }

            // Drain rotation / pan / zoom accumulators into the controller —
            // the controller decays them naturally each frame, matching how
            // the egui d-pad fed them when the panel was rendered in-WASM.
            let (yaw, pitch) = PENDING_ROTATION_IMPULSE.with(|cell| {
                std::mem::replace(&mut *cell.borrow_mut(), (0.0, 0.0))
            });
            if yaw != 0.0 || pitch != 0.0 {
                state.controller.rotation.x += yaw;
                state.controller.rotation.y += pitch;
                state.controller.user_inptut = true;
                state.window.request_redraw();
            }

            let (pan_x, pan_y) = PENDING_PAN_IMPULSE.with(|cell| {
                std::mem::replace(&mut *cell.borrow_mut(), (0.0, 0.0))
            });
            if pan_x != 0.0 || pan_y != 0.0 {
                state.controller.shift.x += pan_x;
                state.controller.shift.y += pan_y;
                state.controller.user_inptut = true;
                state.window.request_redraw();
            }

            let zoom_dy = PENDING_ZOOM_IMPULSE.with(|cell| {
                std::mem::replace(&mut *cell.borrow_mut(), 0.0)
            });
            if zoom_dy != 0.0 {
                state.controller.process_scroll(zoom_dy);
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

            if PENDING_TOUR_STOP.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), false)) {
                state.cancle_animation();
                state.window.request_redraw();
            }

            if let Some(scale) = PENDING_TRACKING_SHOT.with(|cell| cell.borrow_mut().take()) {
                state.start_tracking_shot_with_duration(None, scale);
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
                        #[cfg(target_arch = "wasm32")]
                        set_ui_visible_for_js(state.ui_visible);
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
                if !measure_picking_active() {
                    state
                        .controller
                        .process_keyboard(key, event.state == ElementState::Pressed);
                }
            }
            }
            WindowEvent::MouseWheel { delta, .. } if !measure_picking_active() => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                    state.controller.process_scroll(*dy )
                }
                winit::event::MouseScrollDelta::PixelDelta(p) => {
                    state.controller.process_scroll(p.y as f32 / 100.)
                }
            },
            WindowEvent::MouseInput { state:button_state, button, .. } if !measure_picking_active() => {
                match button {
                    winit::event::MouseButton::Left =>                         state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                    winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                    _=>{}
                }
            }
            WindowEvent::Touch(touch) if !measure_picking_active() => {
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
        } if !measure_picking_active() => {
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
    /// When true, the next `reload_pc_wasm` keeps the live camera (PLY-tuner
    /// tuned preview with `vb:initial_camera_*` embed).
    static PENDING_RELOAD_PRESERVE_CAMERA: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
    /// cameras.json only — keeps the current auto-framed view camera and orbit pivot.
    static PENDING_SCENE_RELOAD: std::cell::RefCell<Option<Vec<u8>>> =
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

/// Duration scale for the next `start_tracking_shot_wasm` (default 1.0).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_TRACKING_SHOT: std::cell::RefCell<Option<f32>> =
        std::cell::RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_TOUR_STOP: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
}

/// Latest tour status for JS: `[playing, cameraCount, hasScene, canPlay]`.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static TOUR_STATUS_SNAPSHOT: std::cell::RefCell<Option<Vec<f32>>> =
        std::cell::RefCell::new(None);
}

/// Signal from gamepad or JS to return camera to origin (initial auto-framed position).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_RETURN_TO_ORIGIN: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
}

/// Signal from JS to rotate the camera 90° clockwise around the scene up axis.
/// Each click increments a counter; 4 clicks = 360° = back to initial yaw.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_ROTATE_90: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
}

/// Held FPS-movement flags from a JS-side virtual gamepad. Mirrors the
/// `CameraController::move_*` / `roll_*` booleans 1-to-1 — the running event
/// loop overwrites the controller state from this snapshot every frame so a
/// React button can hold a key as long as the pointer is down.
///
/// Layout: `[forward, backward, left, right, up, down, roll_left, roll_right]`.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_MOVEMENT_KEYS: std::cell::RefCell<[bool; 8]> =
        std::cell::RefCell::new([false; 8]);
}

/// Snapshot of the JS movement flags as they were the last time the running
/// event loop applied them to the controller. Used for edge-triggered
/// dispatch: we only overwrite a controller boolean when the React snapshot
/// transitions, so OS-level keyboard input keeps owning the flag while React
/// is idle (and vice-versa). Without this, a momentary React press would
/// either get lost (if we always overwrote) or get stuck on (if we OR'd) —
/// the latter was the original "Move camera doesn't release" bug.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static LAST_APPLIED_JS_MOVEMENT: std::cell::RefCell<[bool; 8]> =
        std::cell::RefCell::new([false; 8]);
}

/// Accumulator for orbit-rotation impulses from JS (yaw, pitch). Drained
/// each frame and added to `controller.rotation` exactly like the gamepad
/// rotate-d-pad does internally. Decays naturally via the controller.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_ROTATION_IMPULSE: std::cell::RefCell<(f32, f32)> =
        std::cell::RefCell::new((0.0, 0.0));
}

/// Accumulator for orbit-pivot pan impulses from JS (shift_x, shift_y).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_PAN_IMPULSE: std::cell::RefCell<(f32, f32)> =
        std::cell::RefCell::new((0.0, 0.0));
}

/// Accumulator for zoom-scroll impulses from JS. Positive = zoom in, negative
/// = zoom out (matches the gamepad zoom buttons, which forward to
/// `process_scroll`).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_ZOOM_IMPULSE: std::cell::RefCell<f32> =
        std::cell::RefCell::new(0.0);
}

/// Caches [cx, cy, cz, radius] for the currently loaded point cloud so that
/// `get_scene_bounds()` can return them synchronously from JS.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static SCENE_BOUNDS: std::cell::RefCell<Option<[f32; 4]>> =
        std::cell::RefCell::new(None);
}

/// Snapshot of the live perspective camera, refreshed every frame.
///
/// Layout (11 floats):
/// `[px, py, pz, qw, qx, qy, qz, fov_y_rad, fov_x_rad, znear, zfar]`.
///
/// JS reads this via `get_perspective_camera_wasm()` to recover the human-
/// meaningful camera state (Scene Tuner "Capture current view" button,
/// pose persistence, etc.). The matrices in `CAMERA_SNAPSHOT` are kept for
/// raycasting; this one is purely the underlying perspective parameters
/// before any viewport-Y flip / projection bake.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PERSPECTIVE_CAMERA_SNAPSHOT: std::cell::RefCell<Option<[f32; 11]>> =
        std::cell::RefCell::new(None);
    /// Orbit-controller axes for JS diagnostics (`get_orbit_axis_debug_wasm`).
    static ORBIT_AXIS_SNAPSHOT: std::cell::RefCell<Option<Vec<f32>>> =
        std::cell::RefCell::new(None);
}

/// Pending JS request to install a specific perspective camera.
///
/// Layout: `[px, py, pz, qw, qx, qy, qz, fov_y_rad, animate_ms]`. A value
/// of `0.0` for `animate_ms` snaps; anything > 0 hands off to the smoothstep
/// `Animation<PerspectiveCamera>` pipeline used by `WindowContext::set_camera`.
/// `fov_y_rad <= 0.0` keeps the current fov.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_SET_CAMERA: std::cell::RefCell<Option<[f32; 9]>> =
        std::cell::RefCell::new(None);
}

/// Called from JS when the user uploads a new file while a render is already active.
/// Stores the new data so the running event loop can swap the point cloud in-place.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn reload_pc_wasm(pc: Vec<u8>, scene: Option<Vec<u8>>) {
    PENDING_RELOAD_PRESERVE_CAMERA.with(|cell| *cell.borrow_mut() = false);
    PENDING_RELOAD.with(|cell| {
        *cell.borrow_mut() = Some((pc, scene));
    });
}

/// Hot-swap the point cloud but keep the live camera. Used when the incoming PLY
/// carries a `vb:initial_camera_*` embed and JS will snap to that pose after load.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn reload_pc_wasm_preserve_camera(pc: Vec<u8>, scene: Option<Vec<u8>>) {
    PENDING_RELOAD_PRESERVE_CAMERA.with(|cell| *cell.borrow_mut() = true);
    PENDING_RELOAD.with(|cell| {
        *cell.borrow_mut() = Some((pc, scene));
    });
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_CLEAR_ORBIT_UP: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
    static PENDING_RESET_ORBIT_TO_CAMERA: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
    static PENDING_SET_ORBIT_UP: std::cell::RefCell<Option<Vector3<f32>>> =
        std::cell::RefCell::new(None);
    static PENDING_SET_ORBIT_PIVOT: std::cell::RefCell<Option<Point3<f32>>> =
        std::cell::RefCell::new(None);
}

/// Drop the orbit controller's locked world-up vector so yaw/pan/gamepad use
/// camera-relative axes. Call after a `vb:initial_camera_*` snap settles on
/// baked PLYs — their floor is view-aligned, not global +Y.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn clear_orbit_world_up_wasm() {
    PENDING_CLEAR_ORBIT_UP.with(|cell| *cell.borrow_mut() = true);
}

/// Project the orbit pivot onto the current camera view ray and align the
/// locked world-up vector to the camera plane — same as cancelling the intro
/// cinematic on the source tab. Call after a `vb:initial_camera_*` embed snap
/// settles so tuned PLY orbit matches source navigation.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn reset_orbit_to_camera_wasm() {
    PENDING_RESET_ORBIT_TO_CAMERA.with(|cell| *cell.borrow_mut() = true);
}

/// Lock horizontal orbit to a world-space up direction (need not be unit).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_orbit_world_up_wasm(x: f32, y: f32, z: f32) {
    let v = Vector3::new(x, y, z);
    let n = v.magnitude();
    if n < 1e-6 || !n.is_finite() {
        return;
    }
    PENDING_SET_ORBIT_UP.with(|cell| *cell.borrow_mut() = Some(v / n));
}

/// Install the orbit pivot (world-space point the camera swings around).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_orbit_pivot_wasm(x: f32, y: f32, z: f32) {
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return;
    }
    PENDING_SET_ORBIT_PIVOT.with(|cell| {
        *cell.borrow_mut() = Some(Point3::new(x, y, z));
    });
}

/// Hot-swap cameras.json for auto-walk without reloading the splat or moving the view.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn reload_scene_wasm(scene: Vec<u8>) {
    PENDING_SCENE_RELOAD.with(|cell| {
        *cell.borrow_mut() = Some(scene);
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

/// Rotate the camera 90° clockwise around the scene's vertical (world-up) axis.
///
/// The orbit pivot stays fixed at `controller.center` (the scene centroid).
/// Calling this 4 times brings the camera back to its original yaw.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn rotate_view_90_wasm() {
    PENDING_ROTATE_90.with(|cell| {
        *cell.borrow_mut() = true;
    });
}

/// Set the held FPS-movement key flags from a React/HTML virtual gamepad.
///
/// Each boolean maps 1-to-1 to a `CameraController` field — set them all to
/// `false` when the pointer leaves the button or comes up. The running event
/// loop overwrites the controller booleans from this snapshot every frame so
/// holding a button keeps the camera moving for as long as the press lasts.
///
/// * `forward` / `backward` → W / S (translate along floor-projected view forward)
/// * `left` / `right`       → A / D (strafe along floor-projected view right)
/// * `up` / `down`          → E / Q (altitude — translate along locked scene up)
/// * `roll_left` / `roll_right` → Z / X (tilt — roll around the view forward)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_movement_keys_wasm(
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    roll_left: bool,
    roll_right: bool,
) {
    PENDING_MOVEMENT_KEYS.with(|cell| {
        *cell.borrow_mut() = [
            forward, backward, left, right, up, down, roll_left, roll_right,
        ];
    });
}

/// Convenience wrapper to clear every movement key (useful on `pointerleave`,
/// `blur`, or when the React panel unmounts).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn clear_movement_keys_wasm() {
    PENDING_MOVEMENT_KEYS.with(|cell| *cell.borrow_mut() = [false; 8]);
}

/// Add to the orbit-rotation impulse accumulator (yaw, pitch). Mirrors the
/// arrow-key impulse the egui gamepad rotate-d-pad applied — call it on each
/// rAF while the React rotate button is held. The controller decays the
/// impulse naturally between frames so the motion stays smooth.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn apply_rotation_impulse_wasm(yaw: f32, pitch: f32) {
    if !yaw.is_finite() || !pitch.is_finite() {
        return;
    }
    PENDING_ROTATION_IMPULSE.with(|cell| {
        let mut v = cell.borrow_mut();
        v.0 += yaw;
        v.1 += pitch;
    });
}

/// Add to the orbit-pivot pan impulse accumulator (shift_x, shift_y).
/// Equivalent to the `I/J/K/L` keys.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn apply_pan_impulse_wasm(shift_x: f32, shift_y: f32) {
    if !shift_x.is_finite() || !shift_y.is_finite() {
        return;
    }
    PENDING_PAN_IMPULSE.with(|cell| {
        let mut v = cell.borrow_mut();
        v.0 += shift_x;
        v.1 += shift_y;
    });
}

/// Add to the zoom-scroll impulse accumulator. Positive zooms in, negative
/// zooms out (matches the gamepad zoom buttons, which call
/// `controller.process_scroll` with ±3.0 while held).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn apply_zoom_wasm(delta: f32) {
    if !delta.is_finite() {
        return;
    }
    PENDING_ZOOM_IMPULSE.with(|cell| *cell.borrow_mut() += delta);
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

/// Start a looping cinematic tour along imported scene cameras (TrackingShot spline).
/// Uses test-split cameras loaded from `cameras.json`. Cancels on user input.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start_tracking_shot_wasm() {
    start_tracking_shot_wasm_with_duration_scale(1.0);
}

/// Same as `start_tracking_shot_wasm` with duration multiplier (0.5 = faster, 2.0 = slower).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start_tracking_shot_wasm_with_duration_scale(duration_scale: f32) {
    PENDING_TRACKING_SHOT.with(|cell| {
        *cell.borrow_mut() = Some(duration_scale);
    });
}

/// Stop the camera tour and return control to the user.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn stop_tour_wasm() {
    PENDING_TOUR_STOP.with(|cell| *cell.borrow_mut() = true);
}

/// Tour status snapshot: `[playing, cameraCount, hasScene, canPlay]` or null before first frame.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_tour_status_wasm() -> Option<Vec<f32>> {
    TOUR_STATUS_SNAPSHOT.with(|c| c.borrow().clone())
}

/// Pending wall tint change from JS.
/// `Some(Some([r, g, b, strength]))` → set tint; `Some(None)` → clear tint; `None` → no change.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_WALL_TINT: std::cell::RefCell<Option<Option<[f32; 4]>>> =
        std::cell::RefCell::new(None);
}

/// Mirrors `WindowContext::ui_visible` (U key — egui stats/settings only).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static UI_VISIBLE_JS: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
}

/// Mirrors `gamepad_visible` for React measure panel (V key — toggled together).
#[cfg(target_arch = "wasm32")]
thread_local! {
    static VIEWER_CHROME_VISIBLE: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_VIEWER_CHROME_TOGGLE: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
}

#[cfg(target_arch = "wasm32")]
fn set_ui_visible_for_js(visible: bool) {
    UI_VISIBLE_JS.with(|c| *c.borrow_mut() = visible);
    if let Some(window) = web_sys::window() {
        if let Ok(event) = web_sys::Event::new("vb-ui-visible") {
            let _ = window.dispatch_event(&event);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn set_viewer_chrome_for_js(visible: bool) {
    VIEWER_CHROME_VISIBLE.with(|c| *c.borrow_mut() = visible);
    if let Some(window) = web_sys::window() {
        if let Ok(event) = web_sys::Event::new("vb-viewer-chrome") {
            let _ = window.dispatch_event(&event);
        }
    }
}

/// Mirrors `WindowContext::cinematic_active` so JS can poll on demand.
/// Updated through `set_cinematic_active_for_js`, which also dispatches
/// `vb-cinematic-pan` so React can subscribe instead of polling.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static CINEMATIC_PAN_ACTIVE_JS: std::cell::RefCell<bool> =
        std::cell::RefCell::new(false);
}

#[cfg(target_arch = "wasm32")]
fn set_cinematic_active_for_js(active: bool) {
    let changed = CINEMATIC_PAN_ACTIVE_JS.with(|c| {
        let prev = *c.borrow();
        if prev == active {
            return false;
        }
        *c.borrow_mut() = active;
        true
    });
    if !changed {
        return;
    }
    if let Some(window) = web_sys::window() {
        if let Ok(event) = web_sys::Event::new("vb-cinematic-pan") {
            let _ = window.dispatch_event(&event);
        }
    }
}

/// Returns whether the auto-sweep cinematic pan is currently the active
/// animation. Cheap getter used by React on viewer mount to initialize the
/// play/pause icon state correctly (WASM auto-starts the pan when no
/// cameras.json tour is loaded, so the icon must show "pause", not
/// "play", on first paint).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_cinematic_pan_running_wasm() -> bool {
    CINEMATIC_PAN_ACTIVE_JS.with(|c| *c.borrow())
}

#[cfg(target_arch = "wasm32")]
fn take_pending_viewer_chrome_toggle() -> bool {
    PENDING_VIEWER_CHROME_TOGGLE.with(|c| {
        let pending = *c.borrow();
        if pending {
            *c.borrow_mut() = false;
        }
        pending
    })
}

/// When true, orbit / FPS camera input is ignored so JS can place measure points.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static MEASURE_PICKING: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
}

/// Latest view + projection (column-major 16 + 16 floats) for JS raycasting.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static CAMERA_SNAPSHOT: std::cell::RefCell<Option<Vec<f32>>> =
        std::cell::RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
fn measure_picking_active() -> bool {
    MEASURE_PICKING.with(|c| *c.borrow())
}

#[cfg(target_arch = "wasm32")]
#[cfg(target_arch = "wasm32")]
fn snapshot_tour_status(state: &WindowContext) {
    let playing = state
        .animation
        .as_ref()
        .map(|(_, playing)| if *playing { 1.0 } else { 0.0 })
        .unwrap_or(0.0);
    let camera_count = state.saved_cameras.len() as f32;
    let has_scene = if state.scene.is_some() { 1.0 } else { 0.0 };
    let can_play = if state.saved_cameras.len() > 1 { 1.0 } else { 0.0 };
    TOUR_STATUS_SNAPSHOT.with(|c| {
        *c.borrow_mut() = Some(vec![playing, camera_count, has_scene, can_play]);
    });
}

fn snapshot_camera_for_measure(
    camera: &PerspectiveCamera,
    viewport_width: u32,
    viewport_height: u32,
) {
    let view = camera.view_matrix();
    let proj = VIEWPORT_Y_FLIP * camera.proj_matrix();
    let view_col: &[f32; 16] = AsRef::<[f32; 16]>::as_ref(&view);
    let proj_col: &[f32; 16] = AsRef::<[f32; 16]>::as_ref(&proj);
    let mut buf = Vec::with_capacity(34);
    buf.extend_from_slice(view_col);
    buf.extend_from_slice(proj_col);
    buf.push(viewport_width as f32);
    buf.push(viewport_height as f32);
    CAMERA_SNAPSHOT.with(|c| *c.borrow_mut() = Some(buf));

    // Mirror the underlying perspective parameters so JS can recover
    // position / quaternion / fov without inverting the view matrix.
    let q = camera.rotation;
    let snap: [f32; 11] = [
        camera.position.x,
        camera.position.y,
        camera.position.z,
        q.s,
        q.v.x,
        q.v.y,
        q.v.z,
        camera.projection.fovy.0,
        camera.projection.fovx.0,
        camera.projection.znear,
        camera.projection.zfar,
    ];
    PERSPECTIVE_CAMERA_SNAPSHOT.with(|c| *c.borrow_mut() = Some(snap));
}

/// Refresh the orbit-axis snapshot read by `get_orbit_axis_debug_wasm`.
///
/// Layout (26 floats):
///   0..3   orbit_pitch_axis  (camera right, mouse vertical orbit)
///   3..6   orbit_yaw_axis    (horizontal orbit pole)
///   6..9   look_forward      (camera view direction)
///  12..15  orbit_pivot
///  15..18  locked_world_up   (NaN when `controller.up` is None)
///  18      has_locked_up     (1.0 = locked, 0.0 = camera-relative)
///  19..22  robust_scene_up   (PCA/AABB floor normal heuristic)
///  22      orbit_radius      (camera ↔ pivot distance)
///  23      orbit_yaw_vs_world_y_deg
///  24      orbit_yaw_vs_scene_up_deg
///  25      orbit_yaw_vs_camera_down_deg (0 when using camera-relative up)
#[cfg(target_arch = "wasm32")]
fn snapshot_orbit_axes_for_js(state: &WindowContext) {
    use cgmath::{InnerSpace, Matrix3, MetricSpace};

    let camera = &state.splatting_args.camera;
    let controller = &state.controller;
    let view_t: Matrix3<f32> = camera.rotation.invert().into();
    let pitch_axis = view_t.x.normalize();
    let yaw_axis = controller
        .up
        .map(|u| u.normalize())
        .unwrap_or_else(|| view_t.y.normalize());
    let camera_back = view_t.z.normalize();
    let look_forward = (-camera_back).normalize();
    let robust_up = robust_scene_up(&state.pc).normalize();
    let world_y = Vector3::unit_y();
    let center = controller.center;
    let distance = camera.position.distance(center);
    let locked = controller
        .up
        .map(|u| u.normalize())
        .unwrap_or(Vector3::new(f32::NAN, f32::NAN, f32::NAN));
    let has_locked = if controller.up.is_some() { 1.0 } else { 0.0 };
    let camera_down = view_t.y.normalize();

    let angle_deg = |a: Vector3<f32>, b: Vector3<f32>| -> f32 {
        a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
    };

    let snap = vec![
        pitch_axis.x,
        pitch_axis.y,
        pitch_axis.z,
        yaw_axis.x,
        yaw_axis.y,
        yaw_axis.z,
        look_forward.x,
        look_forward.y,
        look_forward.z,
        camera_back.x,
        camera_back.y,
        camera_back.z,
        center.x,
        center.y,
        center.z,
        locked.x,
        locked.y,
        locked.z,
        has_locked,
        robust_up.x,
        robust_up.y,
        robust_up.z,
        distance,
        angle_deg(yaw_axis, world_y),
        angle_deg(yaw_axis, robust_up),
        angle_deg(yaw_axis, camera_down),
    ];
    ORBIT_AXIS_SNAPSHOT.with(|c| *c.borrow_mut() = Some(snap));
}

#[cfg(target_arch = "wasm32")]
fn dispatch_splat_frame_event() {
    if let Some(window) = web_sys::window() {
        if let Ok(event) = web_sys::Event::new("vb-splat-frame") {
            let _ = window.dispatch_event(&event);
        }
    }
}

/// Whether the main egui HUD is visible (toggled with U in the viewer).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_ui_visible_wasm() -> bool {
    UI_VISIBLE_JS.with(|c| *c.borrow())
}

/// Whether gamepad + JS measure chrome are visible (V key).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_viewer_chrome_visible_wasm() -> bool {
    VIEWER_CHROME_VISIBLE.with(|c| *c.borrow())
}

/// Toggle gamepad and JS measure chrome together (same as V in the viewer).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn toggle_viewer_chrome_wasm() {
    PENDING_VIEWER_CHROME_TOGGLE.with(|c| *c.borrow_mut() = true);
}

/// Enable/disable measure picking from JS (blocks orbit input only while placing points).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_measure_mode_wasm(enabled: bool) {
    MEASURE_PICKING.with(|c| *c.borrow_mut() = enabled);
}

/// Returns camera data for the JS measure overlay, or null before the first frame.
///
/// Layout: view[16] + projection[16] (column-major, projection includes VIEWPORT_Y_FLIP)
/// + viewport width + height (physical pixels).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_camera_matrices_wasm() -> Option<Vec<f32>> {
    CAMERA_SNAPSHOT.with(|c| c.borrow().clone())
}

/// Snapshot of the live perspective camera (Scene Tuner / pose persistence).
///
/// Returns an 11-element `Float32Array`:
/// `[px, py, pz, qw, qx, qy, qz, fov_y_rad, fov_x_rad, znear, zfar]`.
///
/// `(px, py, pz)`     — world-space camera position.
/// `(qw, qx, qy, qz)` — orientation quaternion (cgmath / Hamilton convention).
/// `fov_y_rad`        — vertical field of view in radians.
/// `fov_x_rad`        — horizontal field of view in radians (derived from aspect).
/// `znear` / `zfar`   — frustum near/far planes (auto-fitted to the AABB each frame).
///
/// Returns `null` until the first frame has rendered.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_perspective_camera_wasm() -> Option<Vec<f32>> {
    PERSPECTIVE_CAMERA_SNAPSHOT.with(|c| c.borrow().map(|s| s.to_vec()))
}

/// Orbit-controller axis snapshot for JS diagnostics (see `snapshot_orbit_axes_for_js`).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn get_orbit_axis_debug_wasm() -> Option<Vec<f32>> {
    ORBIT_AXIS_SNAPSHOT.with(|c| c.borrow().clone())
}

/// Install a specific perspective camera in the running event loop.
///
/// Used by Scene Tuner-style admin tools that need to land on an exact view.
/// Pass `animate_ms = 0` to snap, or any positive value to hand off to the
/// existing smoothstep `Transition<PerspectiveCamera>` animation pipeline
/// (same path that `set_scene_camera` and `rotate_view_90` go through).
///
/// Pass `fov_y_rad <= 0.0` to keep the current vertical field of view; the
/// horizontal fov is recomputed from the projection's locked aspect ratio.
///
/// Quaternion is `(w, x, y, z)` — cgmath / Hamilton convention.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_perspective_camera_wasm(
    px: f32,
    py: f32,
    pz: f32,
    qw: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    fov_y_rad: f32,
    animate_ms: f32,
) {
    if !px.is_finite()
        || !py.is_finite()
        || !pz.is_finite()
        || !qw.is_finite()
        || !qx.is_finite()
        || !qy.is_finite()
        || !qz.is_finite()
        || !fov_y_rad.is_finite()
        || !animate_ms.is_finite()
    {
        return;
    }
    PENDING_SET_CAMERA.with(|cell| {
        *cell.borrow_mut() = Some([px, py, pz, qw, qx, qy, qz, fov_y_rad, animate_ms.max(0.0)]);
    });
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
