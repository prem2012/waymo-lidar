# Rotated IoU and NMS

Part of the [Phase 2 detector](README.md). Code: `src/waymo_lidar/det/iou.py`; tests:
`tests/test_det_iou.py`.

Exact **BEV IoU** and **3D IoU** between upright boxes, plus greedy **rotated NMS**, all vectorised
in torch on the GPU. Evaluation uses the IoUs to match detections to ground truth, and
[decode](decode.md) uses the NMS when `decode.nms: bev`. It was designed **not to become a
bottleneck**, and the benchmark below checks that.

Boxes are `[x, y, z, l, w, h, heading]` in the vehicle frame (z = centre), the same format as
everywhere else in the detector.

## Usage

```python
from waymo_lidar.det.iou import ious, bev_iou, iou_3d, overlapping_pairs, nms_bev, nms_bev_grouped

bev, iou3d = ious(dets, gts)                    # dense (N, M), one intersection pass for both
ia, ib, bev, iou3d = overlapping_pairs(dets, gts, det_frame, gt_frame)   # sparse, many frames at once
keep = nms_bev(boxes, scores, 0.7)              # kept indices, highest score first
keep = nms_bev_grouped(boxes, scores, groups, thresholds)  # many (frame, class) groups in one pass
```

Torch in gives torch out on the same device. NumPy in gives NumPy out. Empty inputs give empty
results of the right shape.

## Definitions

- **BEV IoU** = area(A ∩ B) / area(A ∪ B) of the two footprints (rotated rectangles in x-y).
- **3D IoU** = vol(A ∩ B) / vol(A ∪ B). For upright boxes, vol(A ∩ B) = BEV intersection area ×
  z overlap, where z overlap = max(0, min(top_A, top_B) − max(bottom_A, bottom_B)).
- **Heading:** a footprint is the same for θ and θ + π, so IoU doesn't depend on which way a box
  faces. If heading accuracy matters, score it separately.

## How the intersection is computed

The intersection of two rotated rectangles is a convex polygon with up to 8 vertices. Every one of
them is among **24 candidate points**:

```
   A's 4 corners that lie inside B          ┌───── B ─────┐
 + B's 4 corners that lie inside A          │   ╱╲        │
 + the 16 crossings of A's edges with       │  ╱  ╲ A     │      ● = candidates that end up
   B's edges (4 × 4)                        └─●────●──────┘        as polygon vertices
                                               ╲  ╱
                                                ╲╱
```

For a batch of P pairs, all candidates are computed at once as a (P, 24, 2) tensor with a validity
mask. The valid points are sorted by angle around their mean, and the area is the shoelace formula
over the sorted sequence. Invalid slots are replaced by the first valid point, so they add zero area.
There are no loops over pairs and no per-pair Python; the same code runs on CPU or GPU.

**Precision:**
- Computed in float32, with each pair re-centred on box A first, so boxes at the far edge of the grid
  (±75 m) lose no precision.
- **Parallel edges** are detected *relative to their lengths*: sin(angle) < 10⁻⁵. An absolute
  threshold failed. Float32 leaves ~10⁻⁷ of noise for parallel edges, and the "crossings" computed
  from it landed at random positions. The measured effect was IoU 0.14 for two boxes that only touch
  along an edge; the test suite caught it. Parallel overlaps are still handled, because their
  endpoints appear as corners lying inside the other box.

## Not doing unnecessary work

1. **Prefilter:** before any exact intersection, every (N × M) pair gets a cheap test: centre distance
   < the sum of the half-diagonals (the boxes' circumscribed circles). Typically only ~0.4% of pairs
   pass: 456 of 120,000 in the benchmark frame. The rest are 0 without further work.
2. **Groups:** `overlapping_pairs(..., a_group, b_group)` only pairs boxes of the same group (e.g. the
   same frame), so the evaluator can process all frames in one call without an all-frames × all-frames
   matrix.
3. **One pass for both IoUs:** BEV and 3D IoU share the intersection area.
4. **Batched NMS:** `nms_bev_grouped` handles every (frame, class) group of a decoded batch in one IoU
   pass. Groups are shifted 400 m apart along x so they can't overlap. The greedy pass then walks a
   sparse list of overlapping pairs instead of a dense N × N matrix.
   - The first version ran NMS separately per frame and class: 12 calls per batch, each with its own
     GPU syncs, and it measured 17 ms, missing its 10 ms target. Batching it brought that to 3.6 ms.

## Performance

`uv run python -m waymo_lidar.det.iou` on an RTX 4090:

| Case | Measured | Target |
|---|---|---|
| `ious`, 500 detections × 240 GT, one frame (456 candidate pairs) | **1.06 ms** | < 2 ms |
| exact intersection, 100k pairs | **0.85 ms** | < 5 ms |
| `decode` batch of 4 with `nms: bev`, 2,000 detections (worst case) | **3.61 ms** (1.14 ms without NMS) | < 10 ms |
| a whole validation pass: 8,000 frames × (300 detections, 45 GT), grouped by frame | **1.77 s** (571k candidate pairs) | ≪ 60 s |
| shapely (reference), the same 456 candidate pairs | 3.11 ms | — |
| shapely, all 120,000 pairs without the prefilter | 57 ms | — |

So the IoU adds about a millisecond per frame, and evaluating all of validation spends under
2 seconds on IoU.

## Correctness

`uv run pytest tests/test_det_iou.py` (27 tests). shapely is a **dev-only** dependency, used as the
exact reference:

- **Known values:**
  - identical → 1, disjoint → 0
  - 4 × 2 m boxes offset 2 m → 1/3
  - a unit square vs itself rotated 45° → 1/√2 (the overlap is a regular octagon)
  - z offset by half the height → 3D 1/3, BEV 1
  - nested boxes → the area ratio
  - touching edges or corners → 0
  - a 1 cm shift at (74.9, −74.9) m stays precise
- **Against shapely (float64):** 8,000 random pairs plus degenerate ones (shared edges, corners on
  edges, 6 cm boxes, aligned duplicates) all agree to within **1e-5**. The measured worst case on
  random pairs is 4 × 10⁻⁷.
- **Independent of shapely:** a Monte Carlo estimate (40k points per pair, with membership
  cross-checked against `data.boxes.points_in_boxes`) agrees to within 0.01 on 100+ overlapping
  pairs.
- **Prefilter:** results equal the unfiltered all-pairs computation.
- **Invariances:** rotating and translating both boxes, or adding π to a heading, leaves IoU
  unchanged; iou(a, b) = iou(b, a)ᵀ.
- **Devices and groups:** CPU equals GPU; grouped pairs never cross groups and equal the dense
  within-group values.
- **NMS:** matches a straightforward greedy NMS built on shapely, at thresholds 0.3 / 0.5 / 0.7.
  Grouped NMS equals per-group NMS, and groups never interact.
