use std::ops::RangeInclusive;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use crate::renderer::DEFAULT_KERNEL_SIZE;
use crate::{CameraController, SceneCamera, Split, WindowContext};
use cgmath::{Euler, Matrix3, Quaternion};
#[cfg(not(target_arch = "wasm32"))]
use egui::Vec2b;

#[cfg(target_arch = "wasm32")]
use egui::{Align2, Vec2};

use egui::{Color32, RichText, emath::Numeric};
use winit::keyboard::KeyCode;

#[cfg(not(target_arch = "wasm32"))]
use egui_plot::{Legend, PlotPoints};

pub(crate) fn ui(state: &mut WindowContext) -> bool {
    // Clone into an owned Context (cheap Arc bump) so `state` is no longer
    // borrowed when we later call gamepad_panel(state) mutably.
    let ctx = state.ui_renderer.winit.egui_ctx().clone();
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(stopwatch) = state.stopwatch.as_mut() {
        let durations = pollster::block_on(
            stopwatch.take_measurements(&state.wgpu_context.device, &state.wgpu_context.queue),
        );
        state.history.push((
            *durations.get("preprocess").unwrap_or(&Duration::ZERO),
            *durations.get("sorting").unwrap_or(&Duration::ZERO),
            *durations.get("rasterization").unwrap_or(&Duration::ZERO),
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    let num_drawn = pollster::block_on(
        state
            .renderer
            .num_visible_points(&state.wgpu_context.device, &state.wgpu_context.queue),
    );

    let mut new_camera: Option<SetCamera> = None;
    #[allow(unused_mut)]
    let mut toggle_tracking_shot = false;

    if state.ui_visible {

    #[cfg(not(target_arch = "wasm32"))]
    egui::Window::new("Render Stats")
        .default_width(200.)
        .default_height(100.)
        .show(&ctx, |ui| {
            use egui::TextStyle;
            egui::Grid::new("timing").num_columns(2).show(ui, |ui| {
                ui.colored_label(egui::Color32::WHITE, "FPS");
                ui.label(format!("{:}", state.fps as u32));
                ui.end_row();
                ui.colored_label(egui::Color32::WHITE, "Visible points");
                ui.label(format!(
                    "{:} ({:.2}%)",
                    format_thousands(num_drawn),
                    (num_drawn as f32 / state.pc.num_points() as f32) * 100.
                ));
            });
            let history = state.history.to_vec();
            let pre: Vec<f32> = history.iter().map(|v| v.0.as_secs_f32() * 1000.).collect();
            let sort: Vec<f32> = history.iter().map(|v| v.1.as_secs_f32() * 1000.).collect();
            let rast: Vec<f32> = history.iter().map(|v| v.2.as_secs_f32() * 1000.).collect();

            ui.label("Frame times (ms):");
            egui_plot::Plot::new("frame times")
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .y_axis_min_width(1.0)
                .y_axis_label("ms")
                .auto_bounds(Vec2b::TRUE)
                .show_axes([false, true])
                .legend(
                    Legend::default()
                        .text_style(TextStyle::Body)
                        .background_alpha(1.)
                        .position(egui_plot::Corner::LeftBottom),
                )
                .show(ui, |ui| {
                    let line = egui_plot::Line::new("preprocess", PlotPoints::from_ys_f32(&pre));
                    ui.line(line);
                    let line = egui_plot::Line::new("sorting", PlotPoints::from_ys_f32(&sort));
                    ui.line(line);
                    let line = egui_plot::Line::new("rasterize", PlotPoints::from_ys_f32(&rast));
                    ui.line(line);
                });
        });

    egui::Window::new("⚙ Render Settings").show(&ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Gaussian Scaling");
                ui.add(
                    egui::DragValue::new(&mut state.splatting_args.gaussian_scaling)
                        .range((1e-4)..=1.)
                        .clamp_existing_to_range(true)
                        .speed(1e-2),
                );
                ui.end_row();
                ui.label("Directional Color");
                let mut dir_color = state.splatting_args.max_sh_deg > 0;
                ui.add_enabled(
                    state.pc.sh_deg() > 0,
                    egui::Checkbox::new(&mut dir_color, ""),
                );
                state.splatting_args.max_sh_deg = if dir_color { state.pc.sh_deg() } else { 0 };

                ui.end_row();
                ui.add(egui::Label::new("Background Color"));
                let mut color = egui::Color32::from_rgba_premultiplied(
                    (state.splatting_args.background_color.r * 255.) as u8,
                    (state.splatting_args.background_color.g * 255.) as u8,
                    (state.splatting_args.background_color.b * 255.) as u8,
                    (state.splatting_args.background_color.a * 255.) as u8,
                );
                egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut color,
                    egui::color_picker::Alpha::BlendOrAdditive,
                );

                let color32 = color.to_normalized_gamma_f32();
                state.splatting_args.background_color.r = color32[0] as f64;
                state.splatting_args.background_color.g = color32[1] as f64;
                state.splatting_args.background_color.b = color32[2] as f64;
                state.splatting_args.background_color.a = color32[3] as f64;

                ui.end_row();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.label("Dilation Kernel Size");
                    optional_drag(
                        ui,
                        &mut state.splatting_args.kernel_size,
                        Some(0.0..=10.0),
                        Some(0.1),
                        Some(
                            state
                                .pc
                                .dilation_kernel_size()
                                .unwrap_or(DEFAULT_KERNEL_SIZE),
                        ),
                    );
                    ui.end_row();
                    ui.label("Mip Splatting");
                    optional_checkbox(
                        ui,
                        &mut state.splatting_args.mip_splatting,
                        state.pc.mip_splatting().unwrap_or(false),
                    );
                    ui.end_row();
                }
            });
    });

    egui::Window::new("ℹ Scene")
        .default_width(200.)
        .resizable(true)
        .default_height(100.)
        .show(&ctx, |ui| {
            egui::Grid::new("scene info")
                .num_columns(2)
                .striped(false)
                .show(ui, |ui| {
                    ui.strong("Gaussians:");
                    ui.label(format_thousands(state.pc.num_points()));
                    ui.end_row();
                    ui.strong("SH Degree:");
                    ui.label(state.pc.sh_deg().to_string());
                    ui.end_row();
                    ui.strong("Compressed:");
                    ui.label(state.pc.compressed().to_string());
                    ui.end_row();
                    ui.strong("Mip Splatting:");
                    ui.label(
                        state
                            .pc
                            .mip_splatting()
                            .map(|v| v.to_string())
                            .unwrap_or("-".to_string()),
                    );
                    ui.end_row();
                    ui.strong("Dilation Kernel Size:");
                    ui.label(
                        state
                            .pc
                            .dilation_kernel_size()
                            .map(|v| v.to_string())
                            .unwrap_or("-".to_string()),
                    );
                    ui.end_row();
                    if let Some(path) = &state.pointcloud_file_path {
                        ui.strong("File:");
                        let text = path.to_string_lossy().to_string();

                        ui.add(egui::Label::new(
                            path.file_name().unwrap().to_string_lossy().to_string(),
                        ))
                        .on_hover_text(text);
                        ui.end_row();
                    }
                    ui.end_row();
                });

            if let Some(scene) = &state.scene {
                let nearest = scene.nearest_camera(state.splatting_args.camera.position, None);
                ui.separator();
                ui.collapsing("Dataset Images", |ui| {
                    egui::Grid::new("image info")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("Images");
                            ui.label(scene.num_cameras().to_string());
                            ui.end_row();

                            ui.strong("Current View");

                            if let Some(c) = &mut state.current_view {
                                ui.horizontal(|ui| {
                                    let drag = ui.add(
                                        egui::DragValue::new(c)
                                            .range(0..=(scene.num_cameras().saturating_sub(1)))
                                            .clamp_existing_to_range(true),
                                    );
                                    if drag.changed() {
                                        new_camera = Some(SetCamera::ID(*c));
                                    }
                                    ui.label(scene.camera(*c as usize).unwrap().split.to_string());
                                });
                            } else {
                                ui.label("-");
                            }
                            if let Some(path) = &state.scene_file_path {
                                ui.end_row();
                                ui.strong("File:");
                                let text = path.to_string_lossy().to_string();

                                ui.add(egui::Label::new(
                                    path.file_name().unwrap().to_string_lossy().to_string(),
                                ))
                                .on_hover_text(text);
                            }
                        });

                    egui::ScrollArea::vertical()
                        .max_height(300.)
                        .show(ui, |ui| {
                            let cameras = scene.cameras(None);
                            let cameras2 = cameras.clone();
                            let curr_view = state.current_view;
                            egui::Grid::new("scene views grid")
                                .num_columns(4)
                                .striped(true)
                                .with_row_color(move |idx, _| {
                                    if let Some(view_id) = curr_view {
                                        if idx < cameras.len() && (&cameras)[idx].id == view_id {
                                            return Some(Color32::from_gray(64));
                                        }
                                    }
                                    return None;
                                })
                                .min_col_width(50.)
                                .show(ui, |ui| {
                                    let style = ui.style().clone();
                                    for c in cameras2 {
                                        ui.colored_label(
                                            style.visuals.strong_text_color(),
                                            c.id.to_string(),
                                        );
                                        ui.colored_label(
                                            match c.split {
                                                Split::Train => Color32::DARK_GREEN,
                                                Split::Test => Color32::LIGHT_GREEN,
                                            },
                                            c.split.to_string(),
                                        )
                                        .on_hover_text(
                                            RichText::new(format!(
                                                "{:#?}",
                                                Euler::from(Quaternion::from(Matrix3::from(
                                                    c.rotation
                                                )))
                                            )),
                                        );

                                        let resp =
                                            ui.add(egui::Label::new(c.img_name.clone()).truncate());
                                        if let Some(view_id) = curr_view {
                                            if c.id == view_id {
                                                resp.scroll_to_me(None);
                                            }
                                        }
                                        if ui.button("🎥").clicked() {
                                            new_camera = Some(SetCamera::ID(c.id));
                                        }
                                        ui.end_row();
                                    }
                                });
                        });
                    if let Some(nearest) = nearest {
                        ui.separator();
                        if ui.button(format!("Snap to closest ({nearest})")).clicked() {
                            new_camera = Some(SetCamera::ID(nearest));
                        }
                    }
                });
            }
        });

    #[cfg(target_arch = "wasm32")]
    egui::Window::new("🎮")
        .default_width(200.)
        .resizable(false)
        .default_height(100.)
        .default_open(false)
        .movable(false)
        .anchor(Align2::LEFT_BOTTOM, Vec2::new(10., -10.))
        .show(&ctx, |ui| {
            egui::Grid::new("controls")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {

                    ui.strong("Camera Controls");
                    ui.end_row();

                    // Desktop controls
                    ui.label("Move Camera");
                    ui.label("W/S (forward/back)  A/D (left/right)  E or Space (up)  Q (down)  Shift (faster)");
                    ui.end_row();

                    ui.label("Rotate Camera");
                    ui.label("Arrow Up/Down (Yaw)  Arrow Left/Right (Pitch)  Left click + drag");
                    ui.end_row();

                    ui.label("Tilt Camera");
                    ui.label("Z (tilt left)  X (tilt right)  Alt + drag mouse");
                    ui.end_row();

                    ui.label("Move Target/Center");
                    ui.label("I/K (up/down)  J/L (left/right)  Right click + drag / Two finger drag");
                    ui.end_row();

                    ui.label("Zoom");
                    ui.label("Mouse wheel / Pinch gesture");
                    ui.end_row();

                    ui.separator();
                    ui.end_row();

                    ui.strong("Mobile Touch Controls");
                    ui.end_row();
                    ui.label("Rotate");
                    ui.label("Single finger drag");
                    ui.end_row();
                    ui.label("Pan/Move");
                    ui.label("Two finger drag");
                    ui.end_row();
                    ui.label("Zoom");
                    ui.label("Pinch to zoom");
                    ui.end_row();

                    ui.separator();
                    ui.end_row();

                    ui.strong("Scene Navigation");
                    ui.end_row();
                    ui.label("Toggle UI");
                    ui.label("U");
                    ui.end_row();
                    ui.label("Views 0-9");
                    ui.label("0-9");
                    ui.end_row();
                    ui.label("Random view");
                    ui.label("R");
                    ui.end_row();
                    ui.label("Next View");
                    ui.label("Page Up");
                    ui.end_row();
                    ui.label("Previous View");
                    ui.label("Page Down");
                    ui.end_row();
                    ui.label("Snap to nearest view");
                    ui.label("N");
                    ui.end_row();
                    ui.label("Start/Pause Tracking shot");
                    ui.label("T");
                    ui.end_row();
                });
        });

    } // end ui_visible

    // ── Gamepad overlay panel ─────────────────────────────────────
    if state.gamepad_visible {
        gamepad_panel(&ctx, state);
    }

    let requested_repaint = ctx.has_requested_repaint();

    if let Some(c) = new_camera {
        match c {
            SetCamera::ID(id) => state.set_scene_camera(id),
            SetCamera::Camera(c) => state.set_camera(c, Duration::from_millis(200)),
        }
    }
    if toggle_tracking_shot {
        if let Some((_animation, playing)) = &mut state.animation {
            *playing = !*playing;
        } else {
            state.start_tracking_shot();
        }
    }
    return requested_repaint;
}

// ── Gamepad overlay panel ─────────────────────────────────────────────────────
//
// Renders an on-screen gaming console overlay anchored to the bottom-centre of
// the viewport.  Buttons directly mutate the CameraController fields, exactly as
// real keyboard/scroll events do — no synthetic browser events needed.
//
// Five groups matching the HTML overlay that was removed:
//   Move Camera   → W/A/S/D  (move_forward/back/left/right boolean flags)
//   Rotate Camera → Arrow keys (rotation impulse accumulator)
//   Tilt Camera   → Z/X (roll_left / roll_right booleans)
//   Altitude      → E/Q (move_up / move_down booleans)
//   Move Target   → I/J/K/L (shift impulse accumulator)
//   Zoom          → scroll accumulator
//
// Buttons are drawn with a dark semi-transparent style.  Held state is tracked
// via egui's pointer-down test: on each frame the flag is set while the button
// is being pressed and cleared when released.
fn gamepad_panel(ctx: &egui::Context, state: &mut WindowContext) {
    // Snapshot animation state BEFORE mutably borrowing controller, so the
    // egui closure below only captures `ctrl` (not `state` itself).
    let playing = state.animation.as_ref().map(|(_, p)| *p).unwrap_or(false);
    let mut cinematic_toggle: Option<bool> = None;

    let ctrl: &mut CameraController = &mut state.controller;
    let btn_stroke    = egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 26));
    let hint_col      = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 46);
    let panel_fill    = egui::Color32::from_rgba_unmultiplied(8, 10, 16, 158);
    let rounding12    = egui::CornerRadius::same(12);
    let active_fill   = egui::Color32::from_rgba_unmultiplied(79, 156, 249, 100);
    let active_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(99, 179, 237, 180));
    let lbl_col       = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 97);

    // Uppercase caption above each pod.
    let pod_lbl = |ui: &mut egui::Ui, text: &str| {
        ui.label(egui::RichText::new(text).size(10.0)
            .color(egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210)).strong());
    };

    // Helper: draws chevrons on a circle.
    let draw_dpad_circle = |painter: &egui::Painter, c: egui::Pos2, dir: Option<&str>| {
        let arm = 9.0_f32;
        let tip = 26.0_f32;
        for &d in &["up", "down", "left", "right"] {
            let active = dir == Some(d);
            let col = if active { egui::Color32::WHITE } else { egui::Color32::from_rgba_unmultiplied(255, 255, 255, 200) };
            let cs = egui::Stroke::new(2.5, col);
            let (px, py) = match d {
                "up"   => (c.x, c.y - tip),
                "down" => (c.x, c.y + tip),
                "left" => (c.x - tip, c.y),
                _      => (c.x + tip, c.y),
            };
            let tp = egui::pos2(px, py);
            let (a, b) = match d {
                "up"   => (egui::pos2(px - arm, py + arm * 0.65), egui::pos2(px + arm, py + arm * 0.65)),
                "down" => (egui::pos2(px - arm, py - arm * 0.65), egui::pos2(px + arm, py - arm * 0.65)),
                "left" => (egui::pos2(px + arm * 0.65, py - arm), egui::pos2(px + arm * 0.65, py + arm)),
                _      => (egui::pos2(px - arm * 0.65, py - arm), egui::pos2(px - arm * 0.65, py + arm)),
            };
            painter.line_segment([a, tp], cs);
            painter.line_segment([tp, b], cs);
        }
    };

    // Standard circular d-pad — 110×110, returns (up, dn, lt, rt).
    let dpad = |ui: &mut egui::Ui| -> (bool, bool, bool, bool) {
        let size = 110.0_f32;
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click_and_drag());
        let mut dir: Option<&str> = None;
        if ui.is_rect_visible(rect) {
            let c = rect.center();
            let radius = size / 2.0 - 2.0;
            if resp.is_pointer_button_down_on() {
                if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                    let dx = pos.x - c.x;
                    let dy = pos.y - c.y;
                    dir = Some(if dy.abs() > dx.abs() {
                        if dy < 0.0 { "up" } else { "down" }
                    } else {
                        if dx < 0.0 { "left" } else { "right" }
                    });
                }
            }
            let painter = ui.painter();
            painter.circle_filled(c, radius, egui::Color32::from_rgba_unmultiplied(110, 10, 10, 110));
            draw_dpad_circle(painter, c, dir);
        }
        (dir == Some("up"), dir == Some("down"), dir == Some("left"), dir == Some("right"))
    };

    // Wide button for Tilt/Altitude/Zoom.
    let wbtn = |ui: &mut egui::Ui, label: &str, active: bool| -> egui::Response {
        let (bg, stroke, fg) = if active {
            (active_fill, active_stroke, egui::Color32::WHITE)
        } else {
            (egui::Color32::from_rgba_unmultiplied(12, 14, 22, 160), btn_stroke, lbl_col)
        };
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(40.0, 28.0), egui::Sense::click_and_drag());
        if ui.is_rect_visible(rect) {
            ui.painter().rect(rect, egui::CornerRadius::same(8), bg, stroke, egui::StrokeKind::Inside);
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(11.0), fg);
        }
        resp
    };

    egui::Window::new("__gamepad__")
        .id(egui::Id::new("gpad_win"))
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .movable(false)
        .anchor(egui::Align2::CENTER_BOTTOM, egui::Vec2::new(0.0, -20.0))
        .frame(egui::Frame::new())
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);

            // ── Row 1: MOVE CAMERA · ROTATE CAMERA · MOVE TARGET ──────────
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(20.0, 0.0);

                // ── MOVE CAMERA  (W/A/S/D) — eye icon in center ───────────
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
                    pod_lbl(ui, "MOVE CAMERA");
                    let size = 110.0_f32;
                    let (rect, resp) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click_and_drag());
                    let mut dir: Option<&str> = None;
                    if ui.is_rect_visible(rect) {
                        let c = rect.center();
                        let radius = size / 2.0 - 2.0;
                        if resp.is_pointer_button_down_on() {
                            if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                                let dx = pos.x - c.x;
                                let dy = pos.y - c.y;
                                dir = Some(if dy.abs() > dx.abs() {
                                    if dy < 0.0 { "up" } else { "down" }
                                } else {
                                    if dx < 0.0 { "left" } else { "right" }
                                });
                            }
                        }
                        let painter = ui.painter();
                        painter.circle_filled(c, radius, egui::Color32::from_rgba_unmultiplied(110, 10, 10, 110));
                        draw_dpad_circle(painter, c, dir);
                    }
                    ctrl.process_keyboard(KeyCode::KeyW, dir == Some("up"));
                    ctrl.process_keyboard(KeyCode::KeyS, dir == Some("down"));
                    ctrl.process_keyboard(KeyCode::KeyA, dir == Some("left"));
                    ctrl.process_keyboard(KeyCode::KeyD, dir == Some("right"));
                });

                // ── ROTATE CAMERA  (rotation) — camera icon in center ─────
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
                    pod_lbl(ui, "ROTATE CAMERA");
                    let size = 110.0_f32;
                    let (rect, resp) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click_and_drag());
                    let mut dir: Option<&str> = None;
                    if ui.is_rect_visible(rect) {
                        let c = rect.center();
                        let radius = size / 2.0 - 2.0;
                        if resp.is_pointer_button_down_on() {
                            if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                                let dx = pos.x - c.x;
                                let dy = pos.y - c.y;
                                dir = Some(if dy.abs() > dx.abs() {
                                    if dy < 0.0 { "up" } else { "down" }
                                } else {
                                    if dx < 0.0 { "left" } else { "right" }
                                });
                            }
                        }
                        let painter = ui.painter();
                        painter.circle_filled(c, radius, egui::Color32::from_rgba_unmultiplied(110, 10, 10, 110));
                        draw_dpad_circle(painter, c, dir);
                        // Camera icon
                        let cam_w = 28.0_f32;
                        let cam_h = 16.0_f32;
                        let cam_rect = egui::Rect::from_center_size(c, egui::vec2(cam_w, cam_h));
                        let body_col = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 180);
                        ui.painter().rect(cam_rect, 3.0, body_col, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                        ui.painter().circle_filled(c, 5.0, egui::Color32::from_rgba_unmultiplied(110, 10, 10, 220));
                        let vf = egui::Rect::from_center_size(
                            egui::pos2(c.x - cam_w * 0.25, c.y - cam_h * 0.6),
                            egui::vec2(7.0, 5.0),
                        );
                        ui.painter().rect_filled(vf, 1.0, body_col);
                    }
                    if dir == Some("up")    { ctrl.rotation.y += 2.0; ctrl.user_inptut = true; }
                    if dir == Some("down")  { ctrl.rotation.y -= 2.0; ctrl.user_inptut = true; }
                    if dir == Some("left")  { ctrl.rotation.x -= 2.0; ctrl.user_inptut = true; }
                    if dir == Some("right") { ctrl.rotation.x += 2.0; ctrl.user_inptut = true; }
                });

                // ── MOVE TARGET  (shift) ───────────────────────────────────
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
                    pod_lbl(ui, "MOVE TARGET");
                    let (up, dn, lt, rt) = dpad(ui);
                    if up { ctrl.shift.x += 1.0; ctrl.user_inptut = true; }
                    if dn { ctrl.shift.x -= 1.0; ctrl.user_inptut = true; }
                    if lt { ctrl.shift.y -= 1.0; ctrl.user_inptut = true; }
                    if rt { ctrl.shift.y += 1.0; ctrl.user_inptut = true; }
                });
            });

            // ── Row 2: TILT · ALTITUDE · ZOOM ─────────────────────────────
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(10.0, 0.0);

                // TILT CAMERA  (Z / X)
                ui.vertical(|ui| {
                    pod_lbl(ui, "TILT CAMERA");
                    egui::Frame::new().fill(panel_fill).corner_radius(rounding12)
                        .inner_margin(egui::Margin { left: 10, right: 10, top: 7, bottom: 7 })
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                                let r = wbtn(ui, "< Z", ctrl.roll_left);
                                ctrl.process_keyboard(KeyCode::KeyZ, r.is_pointer_button_down_on());
                                let r = wbtn(ui, "X >", ctrl.roll_right);
                                ctrl.process_keyboard(KeyCode::KeyX, r.is_pointer_button_down_on());
                            });
                        });
                });

                // ALTITUDE  (E = Up, Q = Down)
                ui.vertical(|ui| {
                    pod_lbl(ui, "ALTITUDE");
                    egui::Frame::new().fill(panel_fill).corner_radius(rounding12)
                        .inner_margin(egui::Margin { left: 10, right: 10, top: 7, bottom: 7 })
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                                let r_up = wbtn(ui, "", ctrl.move_up);
                                ctrl.process_keyboard(KeyCode::KeyE, r_up.is_pointer_button_down_on());
                                let r_down = wbtn(ui, "", ctrl.move_down);
                                ctrl.process_keyboard(KeyCode::KeyQ, r_down.is_pointer_button_down_on());

                                // Draw arrows with the painter instead of text labels.
                                let painter = ui.painter();
                                let up_col = if ctrl.move_up { egui::Color32::WHITE } else { lbl_col };
                                let down_col = if ctrl.move_down { egui::Color32::WHITE } else { lbl_col };

                                let c_up = r_up.rect.center();
                                let tip_up = egui::pos2(c_up.x, c_up.y - 6.0);
                                let left_up = egui::pos2(c_up.x - 5.0, c_up.y + 4.0);
                                let right_up = egui::pos2(c_up.x + 5.0, c_up.y + 4.0);
                                painter.add(egui::Shape::convex_polygon(vec![tip_up, right_up, left_up], up_col, egui::Stroke::new(0.0, egui::Color32::TRANSPARENT)));

                                let c_dn = r_down.rect.center();
                                let tip_dn = egui::pos2(c_dn.x, c_dn.y + 6.0);
                                let left_dn = egui::pos2(c_dn.x - 5.0, c_dn.y - 4.0);
                                let right_dn = egui::pos2(c_dn.x + 5.0, c_dn.y - 4.0);
                                painter.add(egui::Shape::convex_polygon(vec![tip_dn, right_dn, left_dn], down_col, egui::Stroke::new(0.0, egui::Color32::TRANSPARENT)));
                            });
                        });
                });

                // ZOOM
                ui.vertical(|ui| {
                    pod_lbl(ui, "ZOOM");
                    egui::Frame::new().fill(panel_fill).corner_radius(rounding12)
                        .inner_margin(egui::Margin { left: 10, right: 10, top: 7, bottom: 7 })
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                                let r = wbtn(ui, "+", false);
                                if r.is_pointer_button_down_on() { ctrl.process_scroll(3.0); }
                                let r = wbtn(ui, "-", false);
                                if r.is_pointer_button_down_on() { ctrl.process_scroll(-3.0); }

                                // Play / Pause cinematic animation
                                let label = if playing { "Pause" } else { "Play" };
                                let r = wbtn(ui, label, playing);
                                if r.clicked() {
                                    cinematic_toggle = Some(!playing);
                                }
                            });
                        });
                });
            });

            // ── Hint ──────────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("V  ·  TOGGLE CONTROLS")
                    .size(7.0).color(hint_col));
            });
        });

    // Apply cinematic toggle AFTER the egui closure so `ctrl` borrow is released.
    if let Some(should_play) = cinematic_toggle {
        if should_play {
            state.start_cinematic_pan(18.75, 18.0);
        } else {
            state.stop_animation();
        }
    }
}

// ── old orphaned code removed ──

enum SetCamera {
    ID(usize),
    #[allow(dead_code)]
    Camera(SceneCamera),
}

/// 212312321 -> 212.312.321
fn format_thousands(n: u32) -> String {
    let mut n = n;
    let mut result = String::new();
    while n > 0 {
        let rem = n % 1000;
        n /= 1000;
        if n > 0 {
            result = format!(".{:03}", rem) + &result;
        } else {
            result = rem.to_string() + &result;
        }
    }
    result
}

#[allow(unused)]
fn optional_drag<T: Numeric>(
    ui: &mut egui::Ui,
    opt: &mut Option<T>,
    range: Option<RangeInclusive<T>>,
    speed: Option<impl Into<f64>>,
    default: Option<T>,
) {
    let mut placeholder = default.unwrap_or(T::from_f64(0.));
    let mut drag = if let Some(ref mut val) = opt {
        egui_winit::egui::DragValue::new(val)
    } else {
        egui_winit::egui::DragValue::new(&mut placeholder).custom_formatter(|_, _| {
            if let Some(v) = default {
                format!("{:.2}", v.to_f64())
            } else {
                "—".into()
            }
        })
    };
    if let Some(range) = range {
        drag = drag.range(range).clamp_existing_to_range(true);
    }
    if let Some(speed) = speed {
        drag = drag.speed(speed);
    }
    let changed = ui.add(drag).changed();
    if ui
        .add_enabled(opt.is_some(), egui::Button::new("↺"))
        .on_hover_text("Reset to default")
        .clicked()
    {
        *opt = None;
    }
    if changed && opt.is_none() {
        *opt = Some(placeholder);
    }
}

#[allow(unused)]
fn optional_checkbox(ui: &mut egui::Ui, opt: &mut Option<bool>, default: bool) {
    let mut val = default;
    let checkbox = if let Some(ref mut val) = opt {
        egui::Checkbox::new(val, "")
    } else {
        egui::Checkbox::new(&mut val, "")
    };
    let changed = ui.add(checkbox).changed();
    if ui
        .add_enabled(opt.is_some(), egui::Button::new("↺"))
        .on_hover_text("Reset to default")
        .clicked()
    {
        *opt = None;
    }
    if changed && opt.is_none() {
        *opt = Some(val);
    }
}
