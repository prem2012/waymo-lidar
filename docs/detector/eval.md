# Detector evaluation

Part of the [Phase 2 detector](README.md). Code: `src/waymo_lidar/det/eval.py`; tests:
`tests/test_det_eval.py`; settings: the `eval` section of the [config](config.md).

Average precision (AP) per class (vehicle, pedestrian, cyclist), in three metric families:

| Family | A detection matches a GT box when… | Thresholds (`eval.*`) |
|---|---|---|
| **BEV AP** | the footprints' rotated IoU ≥ t | `iou_thresholds`: 0.5, 0.7, 0.8 |
| **3D AP** | the boxes' 3D IoU ≥ t | `iou_thresholds`: 0.5, 0.7, 0.8 |
| **Distance AP** | the BEV centres are within t metres (size and heading don't matter) | `center_distances`: 0.5 m, 1.0 m |

Each family is reported overall and by distance from the car (`distance_bins`: 0–30, 30–50, 50–75 m).
Distance AP separates *finding* an object from *getting its box right*. If distance AP is high but
IoU AP is low, the model finds objects but gets their size or heading wrong.

## Usage

```python
from waymo_lidar.det.eval import Evaluator

ev = Evaluator(cfg)
for batch in val_loader:                               # FrameStream(mode="eval")
    dets = decode(model(bev), cfg)
    ev.add(dets, samples)                              # Detections (or dicts) + the eval samples
res = ev.compute()
print(res.table())                                     # Markdown table, below
wandb.log(res.flat())                                  # flat metric names, below
res.pr_curves[("bev", 0.7, "vehicle")]                 # (precision, recall) arrays
```

`add` keeps only small NumPy arrays per frame. `compute` does all frames at once.

## Rules

**Ignored ground truth (don't care):**
- a box with fewer than `eval.min_points` (1) top-LiDAR points, i.e. boxes with no points, as in
  Waymo's rules
- a box whose centre is outside the grid, since the model can't predict it

A detection that matches only an ignored box is **dropped**: neither a true positive nor a false
positive. Ignored boxes don't count as misses.

**Matching:** per frame and class, separately for every family and threshold:
1. Detections are taken in descending score order. Each one takes the **unmatched, non-ignored** GT
   with the best affinity (the highest IoU, or the smallest centre distance) within the threshold.
   That's a TP.
2. If it has none, but it's within the threshold of an ignored GT, it's dropped.
3. Otherwise it's an FP.

Detections only match GT of their own class.

**AP:**
- Pool all frames' TPs and FPs for a class, sort by score, and compute precision and recall.
- AP is the area under the precision–recall curve after replacing precision with its running maximum
  from the right. This is **all-point interpolation**, as in VOC 2010+.
- A class with no valid GT gets NaN, which `mean` leaves out.

**Distance bins:** matching is done once, globally. A TP counts in the bin of **its GT**, and an FP in
the bin of the **detection**. A bin's ground truth is the valid boxes whose centre range falls in it.
- This differs from re-matching per bin (Waymo's breakdown) only when a detection and its GT sit on
  opposite sides of a bin edge, which is rare. In exchange, the bins cost almost nothing.

## Fast matching

Greedy matching is sequential in principle: a detection's choice depends on what higher-scoring
detections already took. Running that as a Python loop at validation scale would be slow:
~2.4M detections × 8 thresholds. `greedy_match` gets the **identical** result in vectorised rounds.
In each round, every unresolved detection proposes its best still-available GT. It's allowed to take
it only if it outranks every other unresolved detection that has that GT as a candidate. Each round
resolves at least the top unresolved detection, and in practice almost all of them, so it takes a
handful of NumPy passes. A test checks it against a plain sequential loop on 300 random cases,
including many affinity ties.

**Candidate pairs:** all frames go through **one grouped call** to `overlapping_pairs` (see
[iou.md](iou.md)), with group = (frame, class), so pairs never cross frames or classes. It uses
`min_reach` = the largest centre distance (1.0 m). The IoU prefilter alone (overlapping circumscribed
circles) would miss nearby small objects: two pedestrians 0.8 m apart can't overlap, but they're
within 1.0 m. Both families share the same candidates.

## Output

**`res.table()`**, here for the overfit check below:

| class | bev AP@0.5 | bev AP@0.7 | bev AP@0.8 | 3d AP@0.5 | 3d AP@0.7 | 3d AP@0.8 | dist AP@0.5m | dist AP@1m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vehicle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.969 | 1.000 | 1.000 |
| pedestrian | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cyclist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **mean** | **0.667** | **0.667** | **0.667** | **0.667** | **0.667** | **0.656** | **0.667** | **0.667** |

**`res.flat()`** names, sent to Weights & Biases unchanged:

| Name | Value |
|---|---|
| `eval/bev_ap@0.7/vehicle` | BEV AP at IoU 0.7 for vehicles (likewise `3d_ap@…`, `dist_ap@0.5m`, `dist_ap@1m`) |
| `eval/bev_ap@0.7/mean` | mean over classes with GT |
| `eval/bev_ap@0.7/vehicle/30-50m` | the same, in one distance bin |
| `eval/num_gt/vehicle`, `eval/num_ignored/…`, `eval/num_det/…` | counts |
| `eval/tp_center_m/vehicle` | mean centre error of TPs (m), at BEV IoU 0.5 |
| `eval/tp_size_rel/vehicle` | mean relative size error of TPs |
| `eval/tp_heading_rad/vehicle` | mean heading error of TPs (rad, wrapped to [0, π]) |
| `eval/tp_num_tp/vehicle` | number of TPs those errors are averaged over |

The TP errors explain *why* AP moves. For example, a good centre error with a large heading error
points at the rotation regression.

## Checks

- **Oracle:** perfect outputs (`perfect_outputs` → `decode`) on 4 real frames score **AP = 1.0** for
  every family, threshold and class present, with and without `nms: bev`.
- **Overfit:** after 300 steps on one fixed batch of 2 frames, the table above. 63 vehicles and 59
  pedestrians are found, with mean centre error 0.6–0.7 cm, size error 0.8–1.4% and heading error
  0.012–0.021 rad. The cyclist row is the batch's single cyclist, which the model hasn't learned yet
  at that point (see [decode.md](decode.md#checks)).

## Performance

`uv run python -m waymo_lidar.det.eval` evaluates a synthetic validation pass: 8,000 frames × (300
detections, 45 GT), 8 thresholds × 3 classes × (overall + 3 bins). **`compute()`: 9.8 s** on the
RTX 4090 machine, against a target of < 30 s. That's ~1.2 ms per frame for all metrics. A real model
produces fewer detections than this after the score threshold.

## Tests

`uv run pytest tests/test_det_eval.py` (15 tests, ~40 s):

- **AP by hand:** TP / FP / TP over 2 GT = 0.8333; perfect = 1; no detections = 0; only FPs = 0;
  no GT = NaN.
- **Matching:**
  - The vectorised greedy equals a sequential loop.
  - The higher score wins and a duplicate is an FP.
  - A detection on an ignored GT (0 points, or outside the grid) is dropped.
  - An IoU of 0.702 is a TP at 0.7, and 0.684 is an FP.
  - A class mismatch is never a TP.
- **Families:**
  - A z offset gives a BEV TP and a 3D FP.
  - 0.7 m off is a TP at 1.0 m and an FP at 0.5 m.
  - A wrong size or heading is still a distance TP.
  - Pedestrians 0.8 m apart match by distance even though they can't overlap.
- **Bins:** per-bin APs are right, and the bins' GT counts add up to the total.
- **Output:** `flat()` names, NaN handling in `mean`, `table()`, and distance AP turning off with
  `center_distances: []`.
- **Real data:** the oracle scores 1.0 (with and without NMS); the overfit model reaches BEV AP@0.5
  > 0.9 for vehicles and pedestrians (GPU).
