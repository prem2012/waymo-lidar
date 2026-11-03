# Detector training targets

Part of the [Phase 2 detector](README.md). Code: `src/waymo_lidar/det/targets.py`; tests:
`tests/test_det_targets.py`; settings: the `target` section of the [config](config.md).

The model's CenterPoint head predicts, on a 376 × 376 grid of 0.4 m cells:
- a **heatmap** per class: how likely an object centre is in each cell
- a **regression** at each centre: the box's exact position, size and heading

This component builds the ground truth for both from a sample's boxes. `reg_to_boxes` is the exact
inverse, and decoding will reuse it, so encoding and decoding can't drift apart.

## Usage

```python
from waymo_lidar.det.targets import TargetBuilder, reg_to_boxes

targets = TargetBuilder(cfg)                         # NumPy only, runs in the DataLoader workers
ds = FrameStream(root, "training", cfg, mode="train", transform=targets)
batch = next(iter(make_loader(ds, cfg)))
batch["heatmap"]                                     # (B, 3, 376, 376) float32
batch["target_ind"], batch["target_mask"], batch["target_cls"]   # (B, 500)
batch["target_reg"]                                  # (B, 500, 8) float32
```

The targets are added after the data stream's shuffle buffer, so buffered samples don't carry them.

## Cells

The [voxelizer's convention](voxelize.md#index-convention) applies at stride s = `target.out_stride`
= 4, so one heatmap cell is 4 × 4 voxels = 0.4 × 0.4 m. For a box centre (x, y):

```
u = (x − x_min) / 0.4     col = floor(u)    0 … 375   (heatmap column, along x)
v = (y − y_min) / 0.4     row = floor(v)    0 … 375   (heatmap row, along y)
```

The computation is in float64. A box whose centre falls outside the grid gets no target. In training
the data stream has already dropped those boxes; in eval mode the targets skip them. In a typical
frame that's 1–2 distant boxes just past ±75.2 m.

## What's encoded

| Key | Shape / dtype | Content |
|---|---|---|
| `heatmap` | (C, 376, 376) float32 | One channel per class, in `data.classes` order. A Gaussian peaks at exactly 1.0 at each object's centre cell; where Gaussians of the same class overlap, the max is kept. |
| `target_ind` | (500,) int64 | flat centre index `row · 376 + col` |
| `target_mask` | (500,) uint8 | 1 for real objects, 0 for padding |
| `target_cls` | (500,) int64 | class channel of each object |
| `target_reg` | (500, 8) float32 | `[dx, dy, z, log l, log w, log h, sin θ, cos θ]` |

- **Regression fields** (`REG_FIELDS`):
  - `dx = u − col` and `dy = v − row`, in [0, 1): where the true centre sits inside its 0.4 m cell.
    This recovers the precision lost to the stride.
  - `z`: box centre height, in metres in the vehicle frame.
  - `log l, log w, log h`: log of the size in metres. This keeps the scale manageable from
    pedestrians (~0.9 m) to trucks (~15 m).
  - `sin θ, cos θ`: heading as a unit vector, which avoids the ±π wrap.
- **Inverse,** `reg_to_boxes(rows, cols, reg, cfg)`, works on NumPy arrays or torch tensors:
  x = x_min + (col + dx) · 0.4 and y likewise; z = z; l, w, h = exp(·); θ = atan2(sin, cos).

## Heatmap Gaussians

Each object draws a Gaussian into its class channel:
- **Radius:** r = max(`target.min_radius`, ⌊`gaussian_radius`(l / 0.4, w / 0.4, `gaussian_overlap`)⌋).
  `gaussian_radius` is CornerNet's rule, used unchanged by CenterNet and CenterPoint: the largest
  radius at which a box displaced by r cells still overlaps the original by at least
  `gaussian_overlap` (0.1).
- **Shape:** a (2r + 1)² kernel with σ = (2r + 1) / 6, peak exactly 1, clipped at the grid edges.

Measured on the stage-1 segments:

| Class | Radius in cells (min / median / p95 / max) |
|---|---|
| vehicle | 2 / 3 / 5 / 6 |
| pedestrian | 2 / 2 / 2 / 2 (always the minimum) |
| cyclist | 2 / 2 / 2 / 2 |

So a car's peak spreads over ~7 × 7 cells (2.8 m), and a pedestrian's over 5 × 5 cells (2 m).

## Edge cases

- **Collisions:** two objects of the same class in the same 0.4 m cell share one heatmap peak, and
  each keeps its own regression slot (standard CenterPoint behaviour). On the stage-1 segments this
  happened **0 times in 990 objects**, so it isn't handled specially.
- **Capacity:** `target.max_objects` = 500 slots per frame. The busiest training frame in the full
  dataset has 237 vehicle + pedestrian + cyclist boxes (mean 45, p99.9 223), so nothing is ever
  truncated. If a frame ever had more, the first 500 would be kept with a warning.
- **Cost:** 0.47 ms per frame median (1.15 ms max). Memory is ~1.7 MB per sample, nearly all of it
  heatmap.

## Viewing targets

Add `--targets` to the [voxel viewer](voxelize.md#viewing-voxels):

```bash
uv run python -m waymo_lidar.det.viz_voxels ~/dev/data/waymo_v2 --split training --frame 50 --targets
```

A **BEV targets** panel (one tab per class) opens next to the 3D view:

| Entity | Content |
|---|---|
| `bev/<class>/heatmap` | the class heatmap as a 376 × 376 image (turbo colour, black = 0) |
| `bev/<class>/gt` | ground-truth box outlines, in cell coordinates |
| `bev/<class>/centres` | decoded centres (`col + dx`, `row + dy`), as magenta dots |
| `world/targets/<class>` | heatmap cells > 0.05 as thin tiles on the ground plane (z = −2 m) of the 3D view |
| `world/targets/decoded` | boxes rebuilt from the targets with `reg_to_boxes`, as magenta wireframes. They sit exactly on the GT boxes; a GT box without one has its centre outside the grid. |

- **BEV orientation:** the image's row 0 (top) is y = −75.2 m, the car's right. Drawn this way the car
  faces right with its **left side at the bottom**, mirrored relative to looking down in 3D.
- **Pixel alignment:** Rerun pixel *i* spans [*i*, *i* + 1), the same as `col = floor(u)`, so outlines
  and centres line up with the heatmap cells exactly.

## Tests

`uv run pytest tests/test_det_targets.py` (12 tests, ~20 s, pinned stage-1 segments):

- **Encoding:** a known centre gives the expected (row, col), offset, z, log size and sin/cos.
  The grid corner, just past a border, and outside the grid behave as documented. Heatmaps don't
  leak across classes.
- **Gaussians:** the peak is exactly 1, the kernel is symmetric, the radius grows with box size and
  is at least `min_radius`; overlaps take the max; clipping at the grid edge works.
- **Shapes and limits:** padding, empty frames, `max_objects` truncation with a warning, and
  NumPy vs torch `reg_to_boxes` agree.
- **Real frames:** the round trip reproduces every in-grid GT box (centre < 1e-4 m, size < 1e-5
  relative, heading < 1e-5 rad) in eval mode and in two augmented training epochs.
- **Through the stream:** `FrameStream(transform=TargetBuilder)` and collate give the documented
  batch shapes, and every target centre is a heatmap peak.
- **Viewer:** `--targets --save` writes a non-empty `.rrd`.

**Float32 note:** box coordinates are float32, and no cell border is exactly representable, because
−75.2 isn't. So a box "on" a border can fall either side by ~10⁻⁶ m. The tests use values clear of
borders, and float32(75.2) counts as inside the grid, matching the voxelizer.
