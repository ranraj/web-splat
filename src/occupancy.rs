//! Coarse solid grid built from Gaussian centers.
//!
//! A splat has no triangles, so the eye stops on the points themselves.
//! Cells that hold enough centers are walls, floor, and furniture. One
//! ring of neighbours is marked solid too, which closes thin walls and
//! keeps the eye a short distance short of the surface. Air stays empty,
//! including space outside the scene box — leaving the room is allowed;
//! entering a wall is not.
//!
//! Same rules as `packages/viewer/src/splat-occupancy.ts` (Three.js viewer).

use cgmath::{MetricSpace, Point3};

const CELLS_ALONG_LONGEST_AXIS: f32 = 64.0;
const MAX_CELLS_PER_AXIS: usize = 96;
const MIN_CENTERS_PER_CELL: u16 = 4;
const BBOX_PERCENTILE_LO: f32 = 0.005;
const BBOX_PERCENTILE_HI: f32 = 0.995;
/// Stride a 1.5M-splat room down to a few hundred thousand centers —
/// still dense enough for a wall cell to collect several of them.
const MAX_SAMPLES: usize = 400_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeMove {
    /// Walk and pan: each axis is stopped on its own, so the eye glides along a wall.
    Slide,
    /// Zoom and orbit: the eye halts at the last free spot along the move.
    Stop,
}

pub struct OccupancyGrid {
    /// Corner of cell (0, 0, 0). One empty cell of padding surrounds the
    /// scene box so the dilation ring never indexes out of bounds.
    origin: Point3<f32>,
    cell: f32,
    dims: [usize; 3],
    solid: Vec<bool>,
    solid_cells: usize,
}

impl std::fmt::Debug for OccupancyGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OccupancyGrid")
            .field("origin", &self.origin)
            .field("cell", &self.cell)
            .field("dims", &self.dims)
            .field("solid_cells", &self.solid_cells)
            .finish()
    }
}

fn percentile(positions: &[[f32; 3]], stride: usize, axis: usize, p: f32) -> f32 {
    let step = stride.max(positions.len() / 20_000).max(1);
    let mut values: Vec<f32> = positions
        .iter()
        .step_by(step)
        .map(|v| v[axis])
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    let k = values.len();
    let idx = (((k - 1) as f32) * p).floor() as usize;
    values[idx.min(k - 1)]
}

impl OccupancyGrid {
    /// Returns None when there are too few points or none of them cluster
    /// into a surface.
    pub fn build(positions: &[[f32; 3]]) -> Option<Self> {
        let n = positions.len();
        if n < 32 {
            return None;
        }
        let stride = (n / MAX_SAMPLES).max(1);

        let min = [
            percentile(positions, stride, 0, BBOX_PERCENTILE_LO),
            percentile(positions, stride, 1, BBOX_PERCENTILE_LO),
            percentile(positions, stride, 2, BBOX_PERCENTILE_LO),
        ];
        let max = [
            percentile(positions, stride, 0, BBOX_PERCENTILE_HI),
            percentile(positions, stride, 1, BBOX_PERCENTILE_HI),
            percentile(positions, stride, 2, BBOX_PERCENTILE_HI),
        ];
        let span = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let extent = span[0].max(span[1]).max(span[2]);
        if !extent.is_finite() || extent < 1e-6 {
            return None;
        }

        let cells_for = |cell: f32| span.map(|s| (s / cell).ceil().max(1.0) as usize);
        let mut cell = extent / CELLS_ALONG_LONGEST_AXIS;
        let mut core = cells_for(cell);
        while core.iter().copied().max().unwrap_or(0) > MAX_CELLS_PER_AXIS {
            cell *= 1.25;
            core = cells_for(cell);
        }

        // Core indices run 0..=core (a point on the max face lands on `core`),
        // shifted by one for the padding ring on each side.
        let dims = core.map(|c| c + 3);
        let origin = Point3::new(min[0] - cell, min[1] - cell, min[2] - cell);
        let index = |ix: usize, iy: usize, iz: usize| (ix * dims[1] + iy) * dims[2] + iz;

        let mut counts = vec![0u16; dims[0] * dims[1] * dims[2]];
        for p in positions.iter().step_by(stride) {
            let inside = (0..3).all(|a| p[a] >= min[a] && p[a] <= max[a]);
            if !inside {
                continue;
            }
            let ix = ((p[0] - min[0]) / cell).floor() as usize + 1;
            let iy = ((p[1] - min[1]) / cell).floor() as usize + 1;
            let iz = ((p[2] - min[2]) / cell).floor() as usize + 1;
            let c = &mut counts[index(ix, iy, iz)];
            *c = c.saturating_add(1);
        }

        let mut solid = vec![false; counts.len()];
        let mut any_core = false;
        for ix in 1..dims[0] - 1 {
            for iy in 1..dims[1] - 1 {
                for iz in 1..dims[2] - 1 {
                    if counts[index(ix, iy, iz)] < MIN_CENTERS_PER_CELL {
                        continue;
                    }
                    any_core = true;
                    for x in ix - 1..=ix + 1 {
                        for y in iy - 1..=iy + 1 {
                            for z in iz - 1..=iz + 1 {
                                solid[index(x, y, z)] = true;
                            }
                        }
                    }
                }
            }
        }
        if !any_core {
            return None;
        }

        let solid_cells = solid.iter().filter(|s| **s).count();
        Some(Self {
            origin,
            cell,
            dims,
            solid,
            solid_cells,
        })
    }

    pub fn cell(&self) -> f32 {
        self.cell
    }

    pub fn solid_cells(&self) -> usize {
        self.solid_cells
    }

    pub fn is_blocked(&self, p: Point3<f32>) -> bool {
        let fx = ((p.x - self.origin.x) / self.cell).floor();
        let fy = ((p.y - self.origin.y) / self.cell).floor();
        let fz = ((p.z - self.origin.z) / self.cell).floor();
        if !(fx >= 0.0 && fy >= 0.0 && fz >= 0.0) {
            return false;
        }
        let (ix, iy, iz) = (fx as usize, fy as usize, fz as usize);
        if ix >= self.dims[0] || iy >= self.dims[1] || iz >= self.dims[2] {
            return false;
        }
        self.solid[(ix * self.dims[1] + iy) * self.dims[2] + iz]
    }

    /// Walk the segment at half-cell steps so a move cannot tunnel through
    /// a thin wall. Returns the last free point. `from` itself is not tested.
    fn stop_at_wall(&self, from: Point3<f32>, to: Point3<f32>) -> Point3<f32> {
        let dist = from.distance(to);
        if !(dist >= 1e-8) {
            return to;
        }
        let steps = (dist / (self.cell * 0.5)).ceil().max(1.0) as usize;
        let mut last_free = from;
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let p = from + (to - from) * t;
            if self.is_blocked(p) {
                return last_free;
            }
            last_free = p;
        }
        to
    }

    /// Keep the eye out of solid cells. A start pose that is already inside
    /// a solid cell is allowed to move, so a bad opening view cannot trap
    /// the camera.
    pub fn resolve(&self, from: Point3<f32>, to: Point3<f32>, mode: EyeMove) -> Point3<f32> {
        if self.is_blocked(from) {
            return to;
        }
        match mode {
            EyeMove::Stop => self.stop_at_wall(from, to),
            EyeMove::Slide => {
                let delta = to - from;
                let mut order = [0usize, 1, 2];
                order.sort_by(|&a, &b| delta[b].abs().total_cmp(&delta[a].abs()));
                let mut pos = from;
                for axis in order {
                    let mut next = pos;
                    next[axis] = to[axis];
                    pos = self.stop_at_wall(pos, next);
                }
                pos
            }
        }
    }
}
