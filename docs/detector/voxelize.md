# Detector voxelizer

Part of the [Phase 2 detector](README.md). Code: `src/waymo_lidar/det/voxelize.py` (voxelizer) and
`src/waymo_lidar/det/viz_voxels.py` (viewer); tests: `tests/test_det_voxelize.py`; settings: the
`grid` section of the [config](config.md).

The voxelizer turns a batch of points from the [data stream](data.md) into the model's input image:
binary occupancy per 10 × 10 × 25 cm voxel, with the 24 height slices as channels of a 1504 × 1504
bird's-eye-view (BEV) image. It runs on the GPU in about 1 ms per batch and stores nothing on disk.

## Usage

```python
from waymo_lidar.det.voxelize import Voxelizer

vox = Voxelizer(cfg.grid).cuda()                       # no parameters; build once
bev = vox(batch["points"].cuda(), batch["batch_size"], dtype=torch.bfloat16)
bev.shape                                              # (B, 24, 1504, 1504), or 26 channels with extras
```

`batch["points"]` is the data stream's collated (ΣN, 5) tensor: `[batch index, x, y, z, intensity]`.

## Index convention

Every component (voxelizer, targets, decode) uses this one mapping, with points in the
[vehicle frame](../dataset.md#coordinate-frames):

```
ix = floor((x − x_min) / 0.10)    0 … 1503    → image column
iy = floor((y − y_min) / 0.10)    0 … 1503    → image row
iz = floor((z − z_min) / 0.25)    0 … 23      → channel (height slice)
```

- **Ranges are half-open:** x ∈ [−75.2, 75.2), y ∈ [−75.2, 75.2), z ∈ [−2, 4). Points outside are
  dropped. This is the same rule the data stream's crop uses.
- **Computed in float64.** In float32, points within ~10⁻⁵ of a voxel boundary can land in the
  neighbouring cell. On a real frame that's about 8 of 160k points, changing 3–4 voxels. The data
  stream's crop also compares in float64, so the two always agree.
- **Image orientation:**

  ```
               column ix →  (x: −75.2 behind the car … +75.2 ahead)
             ┌──────────────────────────────┐
  row iy     │ row 0       = y −75.2 (right) │
    ↓        │                              │
             │          car at (752, 752)    │
             │                              │
             │ row 1503    = y +75.2 (left)  │
             └──────────────────────────────┘
  ```
  Drawn with row 0 at the top, as image libraries do, the car faces right and its left side is at
  the **bottom**. Remember this when looking at BEV images.

Helpers:

| Function | Does |
|---|---|
| `points_to_voxel_indices(xyz, grid)` | (N, 3) metres → (N, 3) `[ix, iy, iz]` + in-range mask |
| `voxel_centers(idx, grid)` | indices → voxel centres in metres (inverse of the above) |
| `bev_cell_centers(grid, stride)` | y and x centres of BEV cells, `stride` voxels per cell (stride 4 → the 376 × 376 heatmap grid) |
| `channel_names(grid)` | `["z0:-2.00m", …, "z23:+3.75m"]` (+ `"density"`, `"intensity"`) |

## Channels

| Channel | Content | Range |
|---|---|---|
| 0 … 23 | **occupancy** of height slice *c* (z from −2 + 0.25c to −2 + 0.25(c+1) m): 1 if any point is in the voxel | {0, 1} |
| 24 *(if `grid.extra_channels`)* | **density**: log1p(points in the BEV cell, all heights) / `DENSITY_SCALE` | [0, ~1.8] |
| 25 *(if `grid.extra_channels`)* | **intensity**: max over the BEV cell of tanh(intensity) | [0, 1) |

Empty cells are 0 in every channel.

- **`DENSITY_SCALE` = log1p(32) ≈ 3.50,** set from real data. Points per occupied BEV cell have a
  median of 1, a 99th percentile of 30 and a max of 565, so cells with up to 32 points map to ≤ 1.
- **tanh** tames intensity's heavy tail (median ~0.1, retroreflectors > 1, max ~30) without per-LiDAR
  statistics. This detector only uses the top LiDAR, so the per-sensor calibration issue from
  [dataset.md](../dataset.md#gotchas) doesn't apply.

## How it works

1. Mask in-range points and compute `(b, ix, iy, iz)`.
2. **Occupancy:** write 1 at the flat index `((b·C + iz)·ny + iy)·nx + ix`. Several points in the
   same voxel just write the same 1.
3. **Extras:** `scatter_add_` counts points per (b, iy, ix), and `scatter_reduce_(amax)` takes the max
   tanh(intensity) per cell.

The output dtype is a parameter. Occupancy is exact in float32, bf16 and fp16. Training uses bf16.

## Performance

On an RTX 4090, a batch of 4 real frames (642k points):

| Channels | dtype | Output | Time |
|---|---|---|---|
| 24 | float32 | 869 MB | 1.3 ms |
| 24 | bf16 | 434 MB | 0.8 ms |
| 26 | float32 | 941 MB | 1.8 ms |
| 26 | bf16 | 470 MB | 1.3 ms |

A real frame occupies ~60–67k voxels, which is 0.1% of the grid. The input is very sparse, but a
dense tensor keeps the model a plain 2D CNN.

## Viewing voxels

`viz_voxels` loads a frame with the real data stream, voxelizes it with the real `Voxelizer`, and
draws the dense output's occupied voxels in Rerun. What you see is exactly the model's input.

### Workflow

1. **Pick a frame:** a split, a segment (copy a name from the Scenes table on `/analytics`, or leave
   it out to use the first segment) and a frame index.
2. **Run:**
   ```bash
   uv run python -m waymo_lidar.det.viz_voxels ~/dev/data/waymo_v2 \
       --split training --segment 10023947602400723454_1120_000_1140_000 --frame 50
   ```
   The Rerun desktop viewer opens, and the terminal prints the occupied-voxel count and voxelize
   time.
3. **Look around:** drag to orbit, scroll to zoom, right-drag to pan, and double-click to focus.
   Toggle entities with the eye icons in the left panel. Hover a voxel to see its `ix`, `iy`, `iz`,
   plus `density` and `intensity` with extras on.

### What's shown

| Entity | Content | Default |
|---|---|---|
| `world/voxels/z00` … `z23` | occupied voxels of each height slice, as solid boxes of true voxel size, coloured by slice (turbo: blue = low, red = high). One entity per slice, so you can view the model's input channels one at a time. | visible |
| `world/points` | raw points, coloured by intensity | hidden: turn it on to check the voxels line up |
| `world/boxes/<class>` | ground-truth boxes in class colours | visible |
| `world/grid` | outline of the grid volume (±75.2 m, z −2…4 m) | visible |

There's one entity per height slice for a second reason: Rerun draws at most **60,000 boxes per
entity** by default and silently drops the rest. A whole frame has ~61k voxels, but the largest slice
(the ground, z −0.25…0 m) has ~23k. A slice that ever exceeds 50k is split into `zNN/part0`,
`part1`, …

### Flags

| Flag | Effect |
|---|---|
| `--frames 0:200:10` | several frames on a `frame` timeline to scrub (instead of `--frame`) |
| `--augment --epoch 3` | the training-time augmented sample (flip, rotation, scale) for that epoch |
| `--set grid.extra_channels=true` | include the density and intensity channels (shown on hover) |
| `--config FILE`, `--set KEY=VALUE …` | any config file and overrides |
| `--as-points` | draw voxels as points instead of boxes |
| `--targets` | also show the CenterPoint training targets: BEV heatmaps per class and decoded boxes (see [targets.md](targets.md#viewing-targets)) |
| `--save out.rrd` | write a file instead of opening a window; view later with `uv run rerun out.rrd` |
| `--web [--web-port 9090] [--no-browser]` | serve to a browser instead of the desktop viewer; prints the URL |

A single frame is ~3 MB as `.rrd`; 5 frames with extras is ~16 MB.

## Tests

`uv run pytest tests/test_det_voxelize.py` (16 tests, ~11 s):

- **Exact cells:** lower bounds included, upper bounds dropped, a known point → known `(ix, iy, iz)`.
- **Layout:** one point lights exactly `bev[b, iz, iy, ix]`.
- **Batches and duplicates:** samples don't mix, and duplicate points give 1.
- **Extras:** values match a hand computation; empty cells are 0; there are 24 channels without
  extras.
- **Round trips:** centre → index; the BEV grid at stride 4.
- **dtypes:** float32, bf16 and fp16 have identical occupancy.
- **Real frame:** matches a NumPy reference voxel for voxel, on CPU and GPU, and every occupied voxel
  contains a real point.
- **Viewer:** `--save` writes a non-empty `.rrd`; the per-entity split respects Rerun's cap.
- **Data stream agreement:** the crop and the voxelizer agree on float32 points at the grid edges
  (in `tests/test_det_data.py`).
