# Detector data stream

Part of the [detector](README.md). Code: `src/waymo_lidar/det/data.py`; tests:
`tests/test_det_data.py`; settings: the `data` and `augment` sections of the [config](config.md).

The stream turns segments on disk into training and evaluation samples: top-LiDAR points plus
ground-truth boxes. Training samples are augmented and shuffled across segments. Turning a sample into
voxels and heatmap targets is done by later components, which plug in through the `transform` hook.

## Usage

```python
from waymo_lidar.det.config import load_config
from waymo_lidar.det.data import FrameStream, make_loader, load_sample

cfg = load_config("configs/bev_centerpoint.yaml")
ds = FrameStream("~/dev/data/waymo_v2", "training", cfg, mode="train")   # or mode="eval"
loader = make_loader(ds, cfg)               # batch_size, num_workers from the config
for epoch in range(cfg.train.epochs):
    ds.set_epoch(epoch)                     # before each epoch's iterator: new order, offsets, augmentation
    for batch in loader:
        ...

s = load_sample(root, "validation", segment, timestamp, cfg)   # one frame, no streaming
len(ds)                                     # frames per epoch (from Parquet metadata, no data read)
```

Benchmark: `uv run python -m waymo_lidar.det.data ~/dev/data/waymo_v2 [--split] [--workers N] [--mode eval]`.

## Frame

**Points and boxes are both in the vehicle frame at the frame timestamp**: x forward, y left, z up,
with the origin on the ground and most likely below the rear axle. See
[dataset.md → Coordinate frames](../dataset.md#coordinate-frames) for the definition and evidence.
The top LiDAR sits at (+1.43, 0, +2.18) m in this frame. Points arrive already corrected for the car
moving during the sweep, so no further transform is needed. Augmentation rotates, flips and scales
about the vehicle origin.

## Sample

One frame:

| Key | Shape / type | Content |
|---|---|---|
| `points` | (N, 4) float32 | x, y, z, intensity. Top LiDAR, `data.returns` merged, cropped to the grid. ~160k points. |
| `boxes` | (M, 7) float32 | x, y, z (box centre), length, width, height, heading in [−π, π) |
| `labels` | (M,) int64 | channel 0..C−1, the index into `data.classes` (default 0 vehicle, 1 pedestrian, 2 cyclist) |
| `num_points` | (M,) int32 | the dataset's `num_top_lidar_points_in_box` |
| `segment`, `timestamp`, `split` | str, int, str | identifies the frame |

Intensity is the raw sensor value. Any normalisation is left to the voxelizer.

**Crop:** points are kept if x ∈ [x_min, x_max), y ∈ [y_min, y_max) and z ∈ [z_min, z_max), the
same half-open ranges the voxelizer uses.

## Ground-truth rules

| Rule | train | eval |
|---|---|---|
| Keep only `data.classes` | ✓ | ✓ |
| Drop boxes with fewer than `data.min_points_train` top-LiDAR points (default 1) | ✓ | — |
| Drop boxes whose centre is outside the grid's x, y range (checked after augmentation) | ✓ | — |
| Augmentation | ✓ | — |

Eval keeps every box of the configured classes, exactly as labelled. Deciding what to ignore, such as
0-point boxes or boxes outside the grid, is the evaluator's job.

## Augmentation (train only)

Applied to points and boxes together, in this order, all about the vehicle origin:

1. **Flip** across the x axis with probability `augment.flip_y_prob`: y → −y, heading → −heading.
2. **Rotate** by θ ~ U(−r, r), r = `augment.rotation_range`: point and box xy are rotated, and
   heading += θ.
3. **Scale** by s ~ U(`augment.scale_range`): point xyz, box centres and box sizes are multiplied
   by s.

Heading is wrapped to [−π, π). Points are cropped *after* augmentation, so rotation never leaves
empty corners. Intensity is unchanged.

**Reproducible:** each frame's random draws are seeded from (`train.seed`, epoch, segment,
timestamp). The same epoch gives the same samples no matter which worker processes a frame.

## Sampling and shuffling

| | train | eval |
|---|---|---|
| Segments | shuffled each epoch (seed + epoch) | sorted |
| Frames per segment | every `frame_interval_train`-th frame, from a random offset per segment and epoch | every `frame_interval_eval`-th frame, from frame 0 |
| Order | shuffle buffer of `data.shuffle_buffer` frames per worker | segment, then time |

- **Only selected frames are decoded,** but each segment's `lidar` file is still read in full (see
  Performance).
- **Offsets:** with interval 5, successive epochs see different frames of each segment, so every
  frame is eventually used.
- **Sharding:** DataLoader worker *i* takes segments `i, i + W, i + 2W, …` of the epoch's order, so
  each segment is read once per epoch by exactly one worker.
- **Buffer:** a DataLoader batch comes from a single worker, so the buffer is what mixes segments
  within a batch. 128 frames at interval 5 is about 3 segments. Every frame is still yielded exactly
  once.
- **Incomplete segments:** a segment is used only if all of `lidar`, `lidar_pose`,
  `lidar_calibration`, `vehicle_pose` and `lidar_box` exist. Missing ones are skipped with a warning.
  A read error mid-segment, e.g. a file still downloading, skips the rest of that segment with a
  warning.

## Batching

`collate` (used by `make_loader`) turns a list of samples into:

| Key | Type |
|---|---|
| `points` | (ΣN, 5) float32 tensor: [batch index, x, y, z, intensity] |
| `boxes`, `labels`, `num_points` | lists of per-sample tensors (M varies per frame) |
| `meta` | list of (split, segment, timestamp) |
| `batch_size` | int |
| anything a `transform` adds | stacked into one tensor if every sample has the same shape, else a list |

`make_loader` pins memory when CUDA is available.

## Performance

Measured on the stage-1 segments (32-core machine, NVMe):

| | |
|---|---|
| Points per frame | ~160k (both returns) |
| Boxes per frame | ~40–45 (three classes) |
| Read range images | 96 ms per sampled frame (3.8 s per segment at interval 5) |
| Decode to points | 23 ms per frame |
| Throughput | ~10.6 frames/s per worker in isolation. Measured with 12 workers and targets: ~33 frames/s (see below) |
| Memory | ~2.3 GiB RSS per worker (reader + shuffle buffer) + ~1.1 GiB main → ~26 GiB for 12 workers |

**Reading is the bottleneck.** Each `lidar` row group stores all five LiDARs, and Parquet can't skip
rows within a row group, so the short-range LiDARs are read and thrown away. Their range images are
mostly empty but larger in total than the top LiDAR's (4 × 200 × 600 vs 64 × 2650 pixels). At
~125 frames/s this is still well ahead of what the model will consume. If it ever becomes the limit,
the fix is a one-time conversion to top-LiDAR-only files.

**Memory is set by decoding, not by the shuffle buffer.** The range images are Brotli-compressed ~10×
(a 42 MiB row-group column is 392 MiB decoded; a segment's `lidar_pose` is 28 MiB on disk and 732 MiB
decoded), and the Parquet decoder's transient buffers are a multiple of that again. Reading whole row
groups, plus all of `lidar_pose` at once, peaked at ~8 GiB per worker, and the Arrow memory pool keeps
its peak. 12 workers needed ~90 GB and the OOM killer took the machine's desktop down with it. The reader
now streams both files in 8-frame record batches and keeps only the wanted frames' poses: ~2.3 GiB per
worker, same speed, identical output. Size `data.num_workers` by that figure.

**Measured loader throughput is below the estimate.** 12 workers with targets delivered ~33 frames/s
in steady state on a 32-thread i9-13900K, under the ~49 frames/s the GPU trains at (`simple` backbone,
batch 4). Not yet investigated; the Arrow thread pool in every worker (oversubscription) is the first
suspect. Watch `train/data_wait_frac`.

## Tests

`uv run pytest tests/test_det_data.py` (12 tests, ~80 s, on the stage-1 segments pinned by name):

- **Pure functions:** angle wrapping, the shuffle buffer yields everything once, augmentation
  (flip, rotation, scale, heading, inputs not mutated, point-in-box counts preserved), and
  half-open cropping.
- **Real frames:**
  - Eval samples match the dataset's boxes exactly, and points lie inside the grid.
  - Decoded point counts track the dataset's `num_top_points` (median ratio > 0.9), so points and
    boxes share a frame.
  - Training rules hold; training samples reproduce for the same epoch and differ for a different
    epoch; augmentation keeps point-in-box counts within ±2 (float32 rounding at box faces).
- **Streaming:**
  - Eval is deterministic and covers each expected frame exactly once.
  - 2 workers give the same frames as 1, with no duplicates.
  - Epochs change order and offsets reproducibly.
  - Incomplete segments are skipped.
  - collate output is correct, and `len()` matches the frame counts.
