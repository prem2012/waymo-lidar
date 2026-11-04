# Detector decode

Part of the [detector](README.md). Code: `src/waymo_lidar/det/decode.py`; tests:
`tests/test_det_decode.py`; settings: the `decode` section of the [config](config.md).

Decode turns the [model's](model.md) output maps into a list of 3D boxes per frame, each with a score
and a class. It's the inverse of the [targets](targets.md): the box math is the same `reg_to_boxes`,
so decoding "perfect" outputs gives back the ground truth exactly.

## Usage

```python
from waymo_lidar.det.decode import decode

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(bev)
dets = decode(out, cfg)            # list of Detections, one per frame, on the model's device
d = dets[0]
d.boxes                            # (N, 7) x, y, z, length, width, height, heading (vehicle frame)
d.scores                           # (N,)   in [0, 1], highest first
d.labels                           # (N,)   class channel: index into data.classes
d.numpy()                          # {"boxes", "scores", "labels"} as NumPy arrays
```

## Steps

1. **Scores:** p = sigmoid(heatmap logits), in float32.
2. **Peaks:** a cell survives only if it equals the maximum of its 3 × 3 neighbourhood
   (`max_pool2d`). This is CenterNet / CenterPoint's replacement for NMS. Ties keep every tied cell,
   so two objects of one class in adjacent cells can both be detected.
3. **Top-K:** the `decode.topk` (500) highest peaks per frame, jointly over all classes and cells,
   already sorted by score.
4. **Threshold:** drop peaks below `decode.score_threshold` (0.1).
5. **Boxes:** gather the 8 regression values at each peak and convert with `reg_to_boxes`:
   - x = x_min + (col + dx) · 0.4, and y likewise
   - z as predicted
   - size = exp(log size)
   - heading = atan2(sin, cos)

   Before the `exp`, the log sizes are clamped to [log 0.02, log 50] m. That only exists so an
   untrained model can't produce infinite or zero sizes. Real vehicle, pedestrian and cyclist boxes
   span **0.059 – 19.7 m** (measured over all 1,000 segments), well inside the clamp.

6. **Optional rotated NMS** (`decode.nms: bev`): after the threshold, greedy NMS on rotated BEV IoU
   within each frame and class, with thresholds from `decode.nms_iou` (vehicle 0.7, pedestrian 0.5,
   cyclist 0.5). Classes never suppress each other. The whole batch goes through one
   `nms_bev_grouped` call (see [iou.md](iou.md)). Detections stay sorted by score. The default `none`
   (peak picking only) is what CenterPoint uses for Waymo vehicles.

## Speed

RTX 4090, worst case (every frame returns the full 500 detections): **0.29 ms** for one frame,
**0.91 ms** for a batch of 4. With `nms: bev`, the batch of 4 (2,000 detections) takes **3.6 ms**.

## Checks

**Round trip.** `perfect_outputs(batch, cfg)` builds model-shaped outputs that encode the targets:
logit(heatmap) clipped to ±20, and the target regression at each object's centre. Decoding them gives
back every in-grid ground-truth box: centre within 1e-4 m, size within 1e-5 relative, heading within
1e-5 rad, all with score 1.0 and the right label, and no extra detections. This holds on real eval
frames and augmented training frames. The [voxel viewer's](voxelize.md#viewing-voxels) `--targets`
view uses this path for `world/targets/decoded`, so you can see it.

**After overfitting.** The model was trained for 300 steps on one fixed batch of 2 real frames
(the loss-test setup, final loss 0.05), then decoded:

| Frame | In-grid GT | Found within 0.3 m (same class) | Centre error | Unmatched detections > 0.3 |
|---|---|---|---|---|
| 1 | 60 | 60 / 60 | ≤ 0.014 m | 0 |
| 2 | 69 | 68 / 69 | | 0 |

The one miss is the batch's **only cyclist**. It's clearly visible (174 points), but after 300 steps the
model predicts p = 0.097 at its centre, just under the 0.1 threshold. With a single positive in its
heatmap channel, the cyclist gets far less learning signal than the 120 vehicles and pedestrians.
That's the dataset-wide cyclist imbalance (0.6% of boxes) showing up in miniature. The end-to-end test
therefore requires every vehicle and pedestrian with ≥ 5 points to be found within 0.3 m, and every
confident detection to sit on a real object of its class.

## Tests

`uv run pytest tests/test_det_decode.py` (13 tests, ~30 s):

- **Peaks:**
  - A single peak gives one detection at the right place, and a value below the threshold is
    dropped.
  - Adjacent equal peaks are both kept, while a lower neighbour is suppressed.
  - A target Gaussian has exactly one peak.
  - Peaks of different classes in one cell are both kept, sorted by score.
  - Top-K keeps the highest scores.
- **Clamp:** extreme log sizes give finite sizes at the clamp bounds.
- **NMS:**
  - `nms: bev` collapses overlapping same-class detections (4 × 2 m cars 0.8 m apart, IoU 0.667) at
    a threshold of 0.5, and keeps both at 0.7.
  - A different class at the same spot is never suppressed.
  - Frames in a batch are independent.
  - The perfect-output round trip is unchanged with NMS on.
- **Batching:** B = 2 equals two single-frame decodes.
- **Real data:** the perfect-output round trip (eval and augmented train), single-sample NumPy input,
  and overfit → decode finding vehicles and pedestrians (GPU).
