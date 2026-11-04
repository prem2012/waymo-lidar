# Detector config

Part of the [detector](README.md). Code: `src/waymo_lidar/det/config.py`; default file: `configs/bev_centerpoint.yaml`; tests: `tests/test_det_config.py`.

Every number the detector uses lives in one typed, validated config. It is defined once, checked
before anything runs, and saved with each training run.

## Using it

```python
from waymo_lidar.det.config import load_config, save_config, build_optimizer

cfg = load_config("configs/bev_centerpoint.yaml", overrides=["train.batch_size=8"])
cfg.grid.nx, cfg.grid.ny, cfg.grid.nz        # 1504, 1504, 24
cfg.grid.in_channels                         # 24 (26 with grid.extra_channels)
cfg.target_out_size                          # (376, 376) heatmap rows, cols
cfg.data.class_ids                           # (1, 2, 4) dataset type IDs, in heatmap channel order
opt = build_optimizer(cfg.optim, model.parameters())
save_config(cfg, "runs/my_run/config.yaml")
```

- **File:** `configs/bev_centerpoint.yaml`. Every field is optional; anything left out takes its
  default from the dataclasses in `det/config.py`.
- **Overrides:** `section.key=value` strings, as from a future `--set` CLI flag. Each value is parsed
  as YAML: `true`, `8`, `1e-4`, `[0.5, 0.7]`, `null`.
- **Saved runs:** `save_config` writes plain YAML, and `load_config` on that file reproduces the exact
  config.
- **Not in the config:** the data root. It stays a command-line argument, as for `wl index` and `wl serve`.

## Validation

`load_config` raises `ConfigError` with the offending key in the message, for example
`grid.x_range: range 150.45 m is not a whole number of 0.1 m voxels`. It checks:

- **Keys and types:** no unknown keys or sections. Types must match: an integer isn't accepted as a
  bool, and `4.5` isn't accepted as an integer, but an integer is accepted for a float. Float fields
  also accept `1e-4`, which PyYAML would otherwise read as a string (it requires `1.0e-4`).
- **Grid:** each range has min < max and is a whole number of voxels. `nx` and `ny` must be divisible
  by the backbone's largest stride (16 for `simple`, 32 for a ResNet with `layer4`). `out_stride` must be a power of 2 that is at
  most that stride.
- **Classes and returns:** classes must be known (`vehicle`, `pedestrian`, `sign`, `cyclist`) and
  unique. `returns` must be a non-empty subset of {1, 2}.
- **List lengths:** `decode.nms_iou` needs one value per class, `loss.code_weights` needs 8, and
  `model.simple.stage_blocks` needs one value per stage.
- **Backbone:** `model.backbone` is one of the four; `model.resnet.layers` is a sorted, unique,
  non-empty subset of 1–4.
- **Value ranges:** probabilities and IoUs are within [0, 1], and counts are ≥ 1.
  `eval.distance_bins` starts at 0 and increases.
- **Loss:** `loss.smooth_l1_beta` can only be set with `reg_loss: smooth_l1`, and must be > 0.
- **Optimizer:** fields for one optimizer can't be set for another. For example, `momentum` with
  `adamw` is an error, not silently ignored. So to switch the YAML (which sets `betas`) to SGD, use
  `optim.name=sgd optim.betas=null`.

## Fields

**`grid`**: voxel grid in the vehicle frame (x forward, y left, z up), in metres. The BEV image has
`ny` rows and `nx` columns; z slices become channels.

| Field | Default | Effect |
|---|---|---|
| `x_range`, `y_range` | [−75.2, 75.2] | BEV extent. Points and boxes outside are dropped. |
| `z_range` | [−2.0, 4.0] | Height band. Points outside are dropped. |
| `voxel_size` | [0.1, 0.1, 0.25] | Cell size along x, y, z. |
| `extra_channels` | false | Adds 2 BEV channels: log1p(point count) and max intensity. |
| *derived* `nx`, `ny`, `nz` | 1504, 1504, 24 | Grid dimensions. |
| *derived* `in_channels` | 24 (26) | Model input channels. |

**`data`**: what each training sample contains.

| Field | Default | Effect |
|---|---|---|
| `classes` | [vehicle, pedestrian, cyclist] | Classes to detect. Their order is the heatmap channel order. |
| `returns` | [1, 2] | Top-LiDAR returns merged into the point cloud. |
| `frame_interval_train`, `frame_interval_eval` | 5, 5 | Use every Nth frame. Consecutive 10 Hz frames are near-duplicates; interval 5 gives ~31.6k training frames. |
| `shuffle_buffer` | 128 | Frames buffered per DataLoader worker, for shuffling across segments (~2.3 MB each, so 12 workers use ~3.5 GB). |
| `num_workers` | 12 | DataLoader processes that decode range images. Each takes ~2.3 GiB of RAM (see [data.md](data.md#performance)). |
| `min_points_train` | 1 | Training ignores boxes with fewer top-LiDAR points than this. |
| *derived* `class_ids` | (1, 2, 4) | Dataset `BoxType` IDs, in channel order. |

**`augment`**: training-time augmentation, applied to points and boxes together.

| Field | Default | Effect |
|---|---|---|
| `flip_y_prob` | 0.5 | Probability of mirroring across the x axis (y → −y, heading → −heading). |
| `rotation_range` | π/4 | Global yaw rotation, uniform in [−r, r] radians. |
| `scale_range` | [0.95, 1.05] | Global scale factor. |

**`target`**: CenterPoint training targets.

| Field | Default | Effect |
|---|---|---|
| `out_stride` | 4 | Heatmap cell = 4 voxels = 0.4 m. |
| `max_objects` | 500 | Objects per frame that get regression targets. |
| `gaussian_overlap` | 0.1 | CenterPoint's rule for the radius of each object's heatmap Gaussian. |
| `min_radius` | 2 | Smallest Gaussian radius, in heatmap cells. |
| *derived* `cfg.target_out_size` | (376, 376) | Heatmap rows, cols. |

**`model`**: the network (see [model.md](model.md)). The backbone is selectable; the neck and head are
shared by every backbone.

| Field | Default | Effect |
|---|---|---|
| `backbone` | simple | `simple` (this project's small residual CNN) or a torchvision ResNet: `resnet18`, `resnet34`, `resnet50`. |
| `simple.stem_channels` | 32 | Output channels of the stride-2 stem. Used when `backbone: simple`. |
| `simple.stage_channels` | [64, 128, 256] | Channels of the stages at strides 4, 8 and 16 (one stage per entry). |
| `simple.stage_blocks` | [2, 3, 3] | Residual blocks per stage. |
| `resnet.layers` | [1, 2, 3, 4] | ResNet layers that feed the neck. Layer *k* has stride 2^(k+1): 4, 8, 16, 32. Used when `backbone: resnet*`. |
| `resnet.pretrained` | false | Load ImageNet weights, with `conv1` adapted to the input channels. Experimental: RGB photos and BEV occupancy are very different. |
| `neck_channels` | 128 | Channels of each neck branch and of the merged map at the output stride. |
| `head_channels` | 64 | Width of the head's shared conv and branches. |
| `heatmap_bias` | −2.19 | Initial heatmap logit (sigmoid ≈ 0.1), which stabilises early training. |
| *derived* `stage_strides` | (4, 8, 16) | Strides of the feature maps the backbone hands to the neck: (4, 8, 16, 32) for a ResNet with the default layers. |
| *derived* `max_stride` | 16 | max(`stage_strides`): 16 for `simple`, 32 for a ResNet with `layer4`. `nx` and `ny` must be divisible by it (1504 / 32 = 47). |

Only the section for the selected backbone is used. The other one can stay in the file, so switching
is a one-line change: `--set model.backbone=resnet18`.

**`loss`**

| Field | Default | Effect |
|---|---|---|
| `focal_alpha`, `focal_beta` | 2, 4 | Gaussian focal loss exponents. |
| `reg_weight` | 0.25 | Weight of the regression loss relative to the heatmap loss. |
| `code_weights` | [1] × 8 | Per-value regression weights: offset x, y · z · log l, w, h · sin, cos. |
| `reg_loss` | l1 | Element loss for the regression: `l1` (CenterPoint) or `smooth_l1`. See [loss.md](loss.md). |
| `smooth_l1_beta` | 1/9 when unset | Smooth L1's switch point between quadratic and linear. Only valid with `reg_loss: smooth_l1`. |

**`decode`**: turning network outputs into boxes.

| Field | Default | Effect |
|---|---|---|
| `topk` | 500 | Peaks kept per frame. |
| `score_threshold` | 0.1 | Detections below this score are dropped. |
| `nms` | none | `none`: 3×3 max-pool peak picking only. `bev`: also rotated BEV NMS. |
| `nms_iou` | [0.7, 0.5, 0.5] | NMS IoU threshold per class, used when `nms: bev`. |

**`eval`**

| Field | Default | Effect |
|---|---|---|
| `iou_thresholds` | [0.5, 0.7, 0.8] | IoU-based AP is reported at each threshold, for both BEV and 3D IoU. |
| `distance_bins` | [0, 30, 50, 75] | Edges of the distance breakdown (m, xy distance from ego). |
| `center_distances` | [0.5, 1.0] | Centre-distance AP thresholds in metres; `[]` turns distance AP off. See [eval.md](eval.md). |
| `min_points` | 1 | GT boxes with fewer top-LiDAR points are ignored (neither missed nor matched). |

**`optim`**: the optimizer, built by `build_optimizer`.

| Field | Default | Applies to | Effect |
|---|---|---|---|
| `name` | adamw | all | `adamw`, `adam` or `sgd` |
| `lr` | 3e-3 | all | Peak learning rate. The schedule belongs to the training loop. |
| `weight_decay` | 0.01 | all | Decoupled for AdamW, L2 for Adam and SGD (torch semantics). |
| `betas` | (0.9, 0.99) when unset | adam, adamw | Moment coefficients. |
| `momentum` | 0.9 when unset | sgd | |
| `nesterov` | false when unset | sgd | Requires momentum > 0. |

**`train`**

| Field | Default | Effect |
|---|---|---|
| `batch_size` | 4 | Frames per step. To be tuned to GPU memory when the model exists. |
| `epochs` | 12 | |
| `grad_clip` | 35 | Max gradient norm. 0 turns clipping off. |
| `precision` | bf16 | `bf16`, `fp16` or `fp32` autocast. |
| `eval_every` | 2 | Validation evaluation interval, in epochs. |
| `seed` | 0 | |
| `runs_dir` | runs | Checkpoints, TensorBoard logs and the resolved config go to `<runs_dir>/<name>/`. |
| `lr_schedule` | onecycle | `onecycle`: warm up from lr/10 to `optim.lr`, then cosine down to lr/10⁴ (CenterPoint). `cosine`: cosine decay from `optim.lr`. `constant`. See [training.md](training.md). |
| `warmup_frac` | 0.4 | `onecycle`: fraction of all steps spent warming up. |
| `log_every` | 50 | Steps between TensorBoard scalar writes. |
| `val_max_frames` | 0 | Cap on validation frames per pass, for quick checks (0 = all). |
| `best_metric` | eval/bev_ap@0.7/mean | Metric (higher is better) that selects `checkpoints/best.pt`. |
