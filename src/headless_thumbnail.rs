//! Headless thumbnail / panorama render using the same WebGPU path as the interactive viewer
//! (`GaussianRenderer` + `Display` + `SplattingArgs`), for native CLI tools.

use std::io::BufReader;
use std::path::Path;
use std::time::Duration;

use cgmath::{InnerSpace, Quaternion, Rad, Rotation, Rotation3, Vector2};
use image::RgbImage;

use crate::camera::{PerspectiveCamera, PerspectiveProjection};
use crate::io;
use crate::renderer::Display;
use crate::{
    auto_frame_point_cloud, camera_world_forward, clip_aabb_from_scene_analysis, CameraHint,
    GaussianRenderer, PointCloud, SplattingArgs, WGPUContext,
};

/// Renders one frame of a Gaussian PLY using the same pipeline as the WASM viewer
/// (default: SDR / `Rgba8Unorm` like `RenderConfig { hdr: false, .. }`).
///
/// Returns an `RgbImage` suitable for JPEG/PNG export (same tonemap path as in-window SDR).
pub async fn render_ply_thumbnail_rgb(
    ply_path: &Path,
    width: u32,
    height: u32,
    hint: CameraHint,
    hdr: bool,
) -> anyhow::Result<RgbImage> {
    let file = std::fs::File::open(ply_path)?;
    let mut reader = BufReader::new(file);
    let pc_raw = io::GenericGaussianPointCloud::load(&mut reader)?;

    let wgpu_context = WGPUContext::new_instance().await?;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    let (mut camera, _centroid, _world_up, scene_analysis) =
        auto_frame_point_cloud(&pc_raw, width, height, hint);
    camera.fit_near_far(&clip_aabb_from_scene_analysis(&scene_analysis));

    let pc = PointCloud::new(device, pc_raw)?;

    let render_format = hdr_format(hdr);

    let mut renderer = GaussianRenderer::new(
        device,
        queue,
        render_format,
        pc.sh_deg(),
        pc.compressed(),
    )
    .await;

    let display = Display::new(
        device,
        render_format,
        wgpu::TextureFormat::Rgba8Unorm,
        width,
        height,
    );

    let (output, output_view) = create_output_texture(device, width, height);

    let args = splat_args(camera, width, height, &pc);

    render_frame_to_rgb(
        &mut renderer,
        &display,
        device,
        queue,
        &output,
        &output_view,
        &pc,
        args,
    )
    .await
}

/// Horizontal **360° strip panorama**: `faces` perspective slices placed side-by-side.
///
/// Each slice uses the same vertical resolution (`slice_height`) and horizontal FOV `360° / faces`,
/// with the camera **fixed** at the auto-framed eye position and **yaw stepped** around
/// `world_up` so the directions cover a full turn (cylinder-style unwrap, not equirectangular).
///
/// Output size: `(slice_width * faces) × slice_height`.
///
/// `framing_width` / `framing_height` control [`auto_frame_point_cloud`] (eye position and
/// look direction). Defaults to each slice size; set them to the **same** values you use for
/// [`render_ply_thumbnail_rgb`] so the center column matches the thumbnail angle.
pub async fn render_ply_panorama_strip_rgb(
    ply_path: &Path,
    faces: u32,
    slice_width: u32,
    slice_height: u32,
    hint: CameraHint,
    hdr: bool,
    framing_width: Option<u32>,
    framing_height: Option<u32>,
) -> anyhow::Result<RgbImage> {
    anyhow::ensure!(
        faces >= 3 && faces <= 72,
        "panorama --faces must be between 3 and 72 (got {faces})"
    );

    let file = std::fs::File::open(ply_path)?;
    let mut reader = BufReader::new(file);
    let pc_raw = io::GenericGaussianPointCloud::load(&mut reader)?;

    let wgpu_context = WGPUContext::new_instance().await?;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    let fw = framing_width.unwrap_or(slice_width);
    let fh = framing_height.unwrap_or(slice_height);
    let (base_cam, _centroid, world_up, scene_analysis) =
        auto_frame_point_cloud(&pc_raw, fw, fh, hint);
    let clip_aabb = clip_aabb_from_scene_analysis(&scene_analysis);

    // Same forward as thumbnail / web viewer (`auto_cam.forward`), not eye→centroid (adds pitch).
    let forward0 = camera_world_forward(&base_cam);

    let hfov = Rad(std::f32::consts::TAU / faces as f32);
    let aspect = slice_width as f32 / slice_height.max(1) as f32;
    let vfov = Rad(2.0 * ((hfov.0 * 0.5).tan() / aspect).atan());

    let pc = PointCloud::new(device, pc_raw)?;
    let render_format = hdr_format(hdr);

    let mut renderer = GaussianRenderer::new(
        device,
        queue,
        render_format,
        pc.sh_deg(),
        pc.compressed(),
    )
    .await;

    let display = Display::new(
        device,
        render_format,
        wgpu::TextureFormat::Rgba8Unorm,
        slice_width,
        slice_height,
    );

    let (output, output_view) = create_output_texture(device, slice_width, slice_height);

    let mut slabs: Vec<RgbImage> = Vec::with_capacity(faces as usize);

    for i in 0..faces {
        let mut camera = if i == 0 {
            // First column: identical camera to thumbnail when framing + slice size match.
            let mut c = base_cam;
            c.projection.resize(slice_width, slice_height);
            c
        } else {
            let angle = std::f32::consts::TAU * (i as f32) / (faces as f32);
            let dir = Quaternion::from_axis_angle(world_up, Rad(angle)).rotate_vector(forward0);
            let dir = if dir.magnitude2() > 1e-12 {
                dir.normalize()
            } else {
                forward0
            };
            let rotation = Quaternion::look_at(dir, world_up);
            PerspectiveCamera::new(
                base_cam.position,
                rotation,
                PerspectiveProjection::new(
                    Vector2::new(slice_width, slice_height),
                    Vector2::new(hfov, vfov),
                    base_cam.projection.znear,
                    base_cam.projection.zfar,
                ),
            )
        };
        camera.fit_near_far(&clip_aabb);

        let args = splat_args(camera, slice_width, slice_height, &pc);

        let rgb = render_frame_to_rgb(
            &mut renderer,
            &display,
            device,
            queue,
            &output,
            &output_view,
            &pc,
            args,
        )
        .await?;
        slabs.push(rgb);
    }

    stitch_panorama_horizontal(&slabs)
}

fn hdr_format(hdr: bool) -> wgpu::TextureFormat {
    if hdr {
        wgpu::TextureFormat::Rgba16Float
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    }
}

fn splat_args(camera: PerspectiveCamera, w: u32, h: u32, pc: &PointCloud) -> SplattingArgs {
    SplattingArgs {
        camera,
        viewport: Vector2::new(w, h),
        gaussian_scaling: 1.,
        max_sh_deg: pc.sh_deg(),
        mip_splatting: None,
        kernel_size: None,
        clipping_box: None,
        walltime: Duration::from_secs(100),
        scene_center: None,
        scene_extend: None,
        background_color: wgpu::Color::BLACK,
    }
}

fn create_output_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = output.create_view(&wgpu::TextureViewDescriptor::default());
    (output, view)
}

async fn render_frame_to_rgb(
    renderer: &mut GaussianRenderer,
    display: &Display,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    output: &wgpu::Texture,
    output_view: &wgpu::TextureView,
    pc: &PointCloud,
    args: SplattingArgs,
) -> anyhow::Result<RgbImage> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless frame encoder"),
    });

    renderer.prepare(
        &mut encoder,
        device,
        queue,
        pc,
        args,
        &mut None,
    );

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("headless splats"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: display.texture(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer.render(&mut pass, pc);
    }

    display.render(
        &mut encoder,
        output_view,
        args.background_color,
        renderer.camera(),
        renderer.render_settings(),
    );

    queue.submit(std::iter::once(encoder.finish()));

    let rgba = download_texture_rgba8(output, device, queue).await;
    Ok(image::DynamicImage::from(rgba).into_rgb8())
}

fn stitch_panorama_horizontal(slabs: &[RgbImage]) -> anyhow::Result<RgbImage> {
    anyhow::ensure!(!slabs.is_empty(), "panorama: no slices");
    let slice_w = slabs[0].width();
    let h = slabs[0].height();
    for s in slabs.iter() {
        anyhow::ensure!(s.width() == slice_w && s.height() == h, "panorama: slice size mismatch");
    }
    let faces = slabs.len() as u32;
    let out_w = slice_w * faces;
    let mut buf = vec![0u8; (out_w * h * 3) as usize];
    for (i, slab) in slabs.iter().enumerate() {
        let ox = i as u32 * slice_w;
        let raw = slab.as_raw();
        for y in 0..h {
            let src_off = (y * slice_w * 3) as usize;
            let row = &raw[src_off..src_off + slice_w as usize * 3];
            let dst_off = ((y * out_w + ox) * 3) as usize;
            buf[dst_off..dst_off + slice_w as usize * 3].copy_from_slice(row);
        }
    }
    RgbImage::from_raw(out_w, h, buf).ok_or_else(|| anyhow::anyhow!("panorama: buffer size"))
}

async fn download_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    wait_idx: Option<wgpu::SubmissionIndex>,
) -> wgpu::BufferView {
    let slice = buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: wait_idx,
            timeout: None,
        })
        .unwrap();
    rx.receive().await.unwrap().unwrap();
    slice.get_mapped_range()
}

async fn download_texture_rgba8(
    texture: &wgpu::Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> image::RgbaImage {
    let fb_size = texture.size();
    let texel_size: u32 = texture.format().block_copy_size(None).unwrap();
    debug_assert_eq!(texel_size, 4);

    let align: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1;
    let bytes_per_row = (texel_size * fb_size.width + align) & !align;
    let output_buffer_size = (bytes_per_row * fb_size.height) as wgpu::BufferAddress;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: Some("headless readback"),
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless copy"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfoBase {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(fb_size.height),
            },
        },
        fb_size,
    );
    let sub_idx = queue.submit(std::iter::once(encoder.finish()));

    let view = download_buffer(device, &staging, Some(sub_idx)).await;
    let row_bytes = (texel_size * fb_size.width) as usize;
    let mut out = vec![0u8; row_bytes * fb_size.height as usize];
    for row in 0..fb_size.height as usize {
        let src_start = row * bytes_per_row as usize;
        let dst_start = row * row_bytes;
        out[dst_start..dst_start + row_bytes]
            .copy_from_slice(&view[src_start..src_start + row_bytes]);
    }
    drop(view);
    staging.unmap();

    image::RgbaImage::from_raw(fb_size.width, fb_size.height, out)
        .expect("rgba buffer size matches dimensions")
}
