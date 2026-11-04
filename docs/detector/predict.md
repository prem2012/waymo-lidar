# Prediction viewer

Part of the [detector](README.md). Code: `src/waymo_lidar/det/predict.py`; tests:
`tests/test_det_predict.py`.

Runs a checkpoint over one segment and shows its predictions next to the ground truth in Rerun, colour-coded
by how the [evaluator](eval.md) scores them.

## Usage

```bash
uv run wl predict ~/dev/data/waymo_v2 --ckpt runs/baseline/checkpoints/best.pt --segment <name>
uv run wl predict ~/dev/data/waymo_v2 --ckpt ... --segment <name> --heatmaps --labels
uv run wl predict ~/dev/data/waymo_v2 --ckpt ... --segment <name> --save pred.rrd   # then: uv run rerun pred.rrd
uv run wl predict ~/dev/data/waymo_v2 --ckpt ... --segment <name> --web             # browser viewer on :9090
```

| Flag | Default | Meaning |
|---|---|---|
| `--segment` | first complete segment | Segment name in `--split`. |
| `--split` | validation | |
| `--every` | 1 | Use every Nth frame. A full segment (~198 frames) takes ~21 s on the RTX 4090 box and ~4 GB of RAM; a saved `.rrd` is ~450 MB with `--heatmaps`. |
| `--max-frames` | 0 (all) | Stop early. |
| `--min-score` | 0.3 | Predictions below this score are hidden **and left out of the per-frame matching**, so TP / FP / missed reflect what you'd get at that operating point. |
| `--iou` | 0.5 | BEV IoU for the colouring. |
| `--set` | | Overrides on the checkpoint's own config, e.g. `decode.nms=bev`, `eval.min_points=5`. Not `model.*`: the weights fix the architecture. |
| `--heatmaps` | off | Also show the GT and predicted BEV heatmaps side by side, one tab per class. |
| `--labels` | off | Draw each prediction's score next to it. Scores and IoUs are always visible on hover. |
| `--save` / `--web` | desktop viewer | Write an `.rrd`, or serve the web viewer (`--web-port`, `--no-browser`). |

## What you see

| Entity | Colour | Meaning |
|---|---|---|
| `world/pred/<class>/tp` | green | Matched a GT box of its class at BEV IoU ≥ `--iou`. |
| `world/pred/<class>/fp` | red | Matched nothing. |
| `world/pred/<class>/dropped` | grey | Matched nothing countable, but overlaps an *ignored* GT box. The evaluator counts it as neither TP nor FP. |
| `world/gt/<class>/matched` | class colour | Found. |
| `world/gt/<class>/missed` | yellow, thicker | Not found at this score and IoU. |
| `world/gt/<class>/ignored` | dim grey | Don't care: fewer than `eval.min_points` top-LiDAR points, or centre outside the grid. |
| `…/heading` | same | A line from each box's centre to the middle of its front face, so 180° heading flips are visible. |
| `world/points` | intensity | The frame's top-LiDAR points (vehicle frame). |
| `stats/<status>/<class>` | | Per-frame counts as time series: click a spike in `pred_fp` or `gt_missed` to jump to that frame. |
| `summary` | | Legend, the segment's AP table (as in [eval.md](eval.md)), and total TP / FP / missed shown. |
| `bev/<class>` | turbo | With `--heatmaps`: GT \| predicted heatmap (stride-4 cells). |

Each entity group toggles independently in Rerun's left panel. For example, hide `world/pred` to see only what
was missed, or hide everything but `fp` to look for patterns in false positives.

**Two scores, two thresholds.** The AP table in `summary` uses every decoded detection (down to
`decode.score_threshold`, 0.1), as in training validation. The colours use `--min-score` and `--iou`.
Matching is greedy in score order, so removing low-scoring detections never changes how higher-scoring ones
match. Only GT boxes that were matched by a now-hidden detection turn from matched to missed.

## Matching

`match_frame` applies the evaluator's rules to one frame and one threshold: detections in descending score
order (ties by index) each take the best-IoU unmatched, non-ignored GT box of their class. If there's none,
a detection on an ignored box is dropped, and otherwise it's a false positive. It uses the same
`greedy_match` as the evaluator. A test checks it against a plain sequential loop on 40 random frames with
score ties, wrong-class detections, 0-point and out-of-grid GT.
