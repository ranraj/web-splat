// SPZ file format reader
//
// SPZ is a gzip-compressed binary format for 3D Gaussian Splatting developed by Niantic Labs.
// Reference: https://github.com/nianticlabs/spz
//
// File structure (after gzip decompression):
//   Header (16 bytes):
//     magic:          u32 LE = 0x5053474e ('NGSP')
//     version:        u32 LE  (1, 2, or 3)
//     numPoints:      u32 LE
//     shDegree:       u8
//     fractionalBits: u8
//     flags:          u8  (bit 0 = antialiased, bit 1 = hasExtensions)
//     reserved:       u8
//   Data (in order):
//     positions:  numPoints * (9 bytes 24-bit fixed-point for v>=2, or 6 bytes f16 for v==1)
//     alphas:     numPoints bytes (uint8, 0-255)
//     colors:     numPoints * 3 bytes (uint8, DC SH component per channel)
//     scales:     numPoints * 3 bytes (uint8)
//     rotations:  numPoints * 4 bytes (smallest-three quaternion, v>=3) or
//                 numPoints * 3 bytes (first-three quaternion, v<3)
//     sh:         numPoints * shDim * 3 bytes (uint8, higher-order SH bands)
//
// SPZ internally stores data in RUB (Right-Up-Back) coordinates. Since this renderer
// expects the 3DGS/OpenCV convention (RDF: Right-Down-Forward), we apply the
// coordinate conversion: position (x, -y, -z), quaternion (w, x, -y, -z).
// SH higher-order coefficients are also sign-flipped for the Y and Z axis flip
// (equivalent to a π rotation around the X axis).

use std::io::Read;

use byteorder::{LittleEndian, ReadBytesExt};
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};
use half::f16;

use crate::{pointcloud::Gaussian, utils::build_cov};

use super::{GenericGaussianPointCloud, PointCloudReader};

// 'NGSP' = Niantic Gaussian SPz, stored as little-endian u32
const SPZ_MAGIC: u32 = 0x5053474e;

// colorScale from the SPZ reference: DC colour component packing factor
const COLOR_SCALE: f32 = 0.15;

// 1 / sqrt(2), used in quaternion smallest-three decoding
const SQRT_1_2: f32 = 0.70710678118654752;

// Sign flip table for real spherical harmonics under a π rotation around the X axis
// (i.e. the RUB → RDF coordinate conversion where Y and Z are negated).
// Indexed by SH coefficient index j (0-based, excluding DC):
//   j = 0..2  : degree 1 (m = -1, 0, +1)
//   j = 3..7  : degree 2 (m = -2, -1, 0, +1, +2)
//   j = 8..14 : degree 3 (m = -3, -2, -1, 0, +1, +2, +3)
const SH_FLIP_RUB_TO_RDF: [f32; 15] = [
    // degree 1: Y_1^{-1} ∝y (-1), Y_1^0 ∝z (-1), Y_1^1 ∝x (+1)
    -1.0, -1.0, 1.0,
    // degree 2: xy(-1), yz(+1), 2z²-x²-y²(+1), xz(-1), x²-y²(+1)
    -1.0, 1.0, 1.0, -1.0, 1.0,
    // degree 3: y(3x²-y²)(-1), xyz(+1), y(4z²-x²-y²)(-1),
    //           z(2z²-3x²-3y²)(-1), x(4z²-x²-y²)(+1), z(x²-y²)(-1), x(x²-3y²)(+1)
    -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
];

/// Number of higher-order SH coefficients per point per colour channel for a given degree.
/// This excludes the DC term (degree 0), which is stored separately in the `colors` array.
fn dim_for_degree(degree: u32) -> usize {
    match degree {
        0 => 0,
        1 => 3,
        2 => 8,
        3 => 15,
        4 => 24,
        _ => 0,
    }
}

/// Decode a uint8 24-bit-fixed-point position component with sign extension.
#[inline]
fn decode_fixed24(b: &[u8], scale_inv: f32) -> f32 {
    let v = (b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i32) << 16);
    // Sign-extend from 24-bit to 32-bit
    let v = if v & 0x80_0000 != 0 {
        v | -0x100_0000i32
    } else {
        v
    };
    v as f32 * scale_inv
}

/// Decode a packed smallest-three quaternion (version >= 3) into (x, y, z, w) components.
fn decode_quaternion_smallest_three(bytes: &[u8]) -> [f32; 4] {
    let comp: u32 = (bytes[0] as u32)
        | ((bytes[1] as u32) << 8)
        | ((bytes[2] as u32) << 16)
        | ((bytes[3] as u32) << 24);

    const C_MASK: u32 = (1u32 << 9) - 1; // 511

    let i_largest = (comp >> 30) as usize;
    let mut rotation = [0.0f32; 4];
    let mut sum_squares = 0.0f32;
    let mut c = comp;

    for i in (0..4usize).rev() {
        if i != i_largest {
            let mag = c & C_MASK;
            let negbit = (c >> 9) & 1;
            c >>= 10;
            let mut val = SQRT_1_2 * (mag as f32) / (C_MASK as f32);
            if negbit == 1 {
                val = -val;
            }
            rotation[i] = val;
            sum_squares += val * val;
        }
    }
    rotation[i_largest] = (1.0f32 - sum_squares).max(0.0).sqrt();
    rotation
}

/// Decode a packed first-three quaternion (version 1/2) into (x, y, z, w) components.
fn decode_quaternion_first_three(bytes: &[u8]) -> [f32; 4] {
    let x = bytes[0] as f32 / 127.5 - 1.0;
    let y = bytes[1] as f32 / 127.5 - 1.0;
    let z = bytes[2] as f32 / 127.5 - 1.0;
    let w = (1.0f32 - x * x - y * y - z * z).max(0.0).sqrt();
    [x, y, z, w]
}

/// Reader for the SPZ (Niantic Gaussian Splat) file format.
pub struct SpzReader {
    data: Vec<u8>, // fully decompressed gzip payload
}

impl SpzReader {
    /// Construct a reader by gzip-decompressing the full stream.
    pub fn new<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        use flate2::read::GzDecoder;
        let mut decoder = GzDecoder::new(reader);
        let mut data = Vec::new();
        decoder.read_to_end(&mut data)?;
        Ok(Self { data })
    }
}

impl PointCloudReader for SpzReader {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error> {
        let mut cursor = std::io::Cursor::new(&self.data);

        // ── Header ──────────────────────────────────────────────────────────
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != SPZ_MAGIC {
            return Err(anyhow::anyhow!(
                "SPZ: invalid magic bytes (expected 0x{:08x}, got 0x{:08x})",
                SPZ_MAGIC,
                magic
            ));
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        if version < 1 || version > 3 {
            return Err(anyhow::anyhow!("SPZ: unsupported version {}", version));
        }
        let num_points = cursor.read_u32::<LittleEndian>()? as usize;
        let sh_degree = cursor.read_u8()? as u32;
        let fractional_bits = cursor.read_u8()?;
        let flags = cursor.read_u8()?;
        let _reserved = cursor.read_u8()?;

        let antialiased = (flags & 0x1) != 0;
        let uses_float16 = version == 1;
        let uses_quaternion_smallest_three = version >= 3;
        let sh_dim = dim_for_degree(sh_degree);
        let scale_inv = 1.0f32 / ((1u32 << fractional_bits) as f32);

        log::info!(
            "SPZ: version={}, numPoints={}, shDegree={}, fractionalBits={}, antialiased={}",
            version,
            num_points,
            sh_degree,
            fractional_bits,
            antialiased
        );

        // ── Read raw data blocks ─────────────────────────────────────────────
        let pos_bytes = num_points * if uses_float16 { 6 } else { 9 };
        let mut positions_raw = vec![0u8; pos_bytes];
        cursor.read_exact(&mut positions_raw)?;

        let mut alphas_raw = vec![0u8; num_points];
        cursor.read_exact(&mut alphas_raw)?;

        let mut colors_raw = vec![0u8; num_points * 3];
        cursor.read_exact(&mut colors_raw)?;

        let mut scales_raw = vec![0u8; num_points * 3];
        cursor.read_exact(&mut scales_raw)?;

        let rot_bytes = num_points * if uses_quaternion_smallest_three { 4 } else { 3 };
        let mut rotations_raw = vec![0u8; rot_bytes];
        cursor.read_exact(&mut rotations_raw)?;

        let mut sh_raw = vec![0u8; num_points * sh_dim * 3];
        cursor.read_exact(&mut sh_raw)?;

        // ── Convert to Gaussian structs ──────────────────────────────────────
        let mut gaussians = Vec::with_capacity(num_points);
        let mut sh_coefs: Vec<[[f16; 3]; 16]> = Vec::with_capacity(num_points);

        for n in 0..num_points {
            // Position (RUB storage → convert to RDF by negating Y and Z)
            let (px, py, pz) = if uses_float16 {
                let p = &positions_raw[n * 6..];
                let x = f16::from_bits(u16::from_le_bytes([p[0], p[1]])).to_f32();
                let y = f16::from_bits(u16::from_le_bytes([p[2], p[3]])).to_f32();
                let z = f16::from_bits(u16::from_le_bytes([p[4], p[5]])).to_f32();
                (x, y, z)
            } else {
                let p = &positions_raw[n * 9..];
                let x = decode_fixed24(&p[0..3], scale_inv);
                let y = decode_fixed24(&p[3..6], scale_inv);
                let z = decode_fixed24(&p[6..9], scale_inv);
                (x, y, z)
            };
            // RUB → RDF: negate Y and Z
            let pos = Point3::new(px, -py, -pz);

            // Opacity: packed as sigmoid(logit) * 255 → actual opacity in [0,1]
            let opacity = alphas_raw[n] as f32 / 255.0;

            // Scale: packed as (log_scale + 10) * 16 → exp to get actual scale
            let s = &scales_raw[n * 3..];
            let scale = Vector3::new(
                (s[0] as f32 / 16.0 - 10.0).exp(),
                (s[1] as f32 / 16.0 - 10.0).exp(),
                (s[2] as f32 / 16.0 - 10.0).exp(),
            );

            // Rotation: decompress → xyzw in RUB → convert to cgmath Quaternion in RDF
            // SPZ quaternion components: rotation[0]=x, rotation[1]=y, rotation[2]=z, rotation[3]=w
            let rot_xyzw = if uses_quaternion_smallest_three {
                decode_quaternion_smallest_three(&rotations_raw[n * 4..])
            } else {
                decode_quaternion_first_three(&rotations_raw[n * 3..])
            };
            // RUB → RDF quaternion: (w, x, -y, -z)  [y and z components negate]
            let rot = Quaternion::new(rot_xyzw[3], rot_xyzw[0], -rot_xyzw[1], -rot_xyzw[2])
                .normalize();

            let cov = build_cov(rot, scale);

            gaussians.push(Gaussian::new(
                pos.cast().unwrap(),
                f16::from_f32(opacity),
                cov.map(|c| f16::from_f32(c)),
            ));

            // SH coefficients
            let mut sh_out = [[f16::ZERO; 3]; 16];

            // DC component (degree 0) from the colors array; no axis-flip (l=0 is invariant)
            let col = &colors_raw[n * 3..];
            sh_out[0][0] = f16::from_f32((col[0] as f32 / 255.0 - 0.5) / COLOR_SCALE);
            sh_out[0][1] = f16::from_f32((col[1] as f32 / 255.0 - 0.5) / COLOR_SCALE);
            sh_out[0][2] = f16::from_f32((col[2] as f32 / 255.0 - 0.5) / COLOR_SCALE);

            // Higher-order SH (stored as [shDim][3 channels] per point)
            if sh_dim > 0 {
                let base = n * sh_dim * 3;
                for j in 0..sh_dim {
                    let flip = SH_FLIP_RUB_TO_RDF.get(j).copied().unwrap_or(1.0);
                    let k = base + j * 3;
                    sh_out[j + 1][0] =
                        f16::from_f32(flip * (sh_raw[k] as f32 - 128.0) / 128.0);
                    sh_out[j + 1][1] =
                        f16::from_f32(flip * (sh_raw[k + 1] as f32 - 128.0) / 128.0);
                    sh_out[j + 1][2] =
                        f16::from_f32(flip * (sh_raw[k + 2] as f32 - 128.0) / 128.0);
                }
            }

            sh_coefs.push(sh_out);
        }

        Ok(GenericGaussianPointCloud::new(
            gaussians,
            sh_coefs,
            sh_degree,
            num_points,
            None,                // kernel_size
            Some(antialiased),   // mip_splatting (maps to SPZ antialiased flag)
            None,                // background_color
            None,                // covars
            None,                // quantization
        ))
    }

    /// Gzip magic bytes — the outer SPZ file is gzip-compressed.
    fn magic_bytes() -> &'static [u8] {
        &[0x1f, 0x8b]
    }

    fn file_ending() -> &'static str {
        "spz"
    }
}
