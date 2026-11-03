# Detector training

Part of the [Phase 2 detector](README.md). Code: `src/waymo_lidar/det/train.py`; tests:
`tests/test_det_train.py`; settings: the `train` and `optim` sections of the [config](config.md).

## Usage

```bash
uv run wl train ~/dev/data/waymo_v2 --name baseline                       # full run
uv run wl train ~/dev/data/waymo_v2 --name baseline --resume              # continue from last.pt
uv run wl train ~/dev/data/waymo_v2 --name r18 --set model.backbone=resnet18 train.batch_size=8
uv run wl train ~/dev/data/waymo_v2 --name smoke --overfit <segment> --max-minutes 5
uv run wl eval  ~/dev/data/waymo_v2 --ckpt runs/baseline/checkpoints/best.pt
uv run wl predict ~/dev/data/waymo_v2 --ckpt runs/baseline/checkpoints/best.pt --segment <name>   # Rerun: predict.md
uv run tensorboard --logdir runs                                          # http://localhost:6006
```

| Flag | Meaning |
|---|---|
| `--name` | Run name; outputs go to `<train.runs_dir>/<name>/`. Defaults to a timestamp. An existing run is refused unless `--resume`. |
| `--config`, `--set` | Config file (default `configs/bev_centerpoint.yaml`) and dotted overrides, as in [config.md](config.md). |
| `--resume` | Load model, optimizer, scheduler, epoch, step and best metric from `checkpoints/last.pt` and continue with the next epoch. |
| `--overfit SEGMENT` | Smoke test: train **and** validate on every frame of one training segment, with no augmentation. |
| `--max-minutes` | Stop cleanly after this much training time (setup excluded): validate, save `last.pt`, exit. Resume later with `--resume`. |

`wl eval` runs one checkpoint over a split (`--split`, default `validation`; `--max-frames` to cap it),
prints the metric table and writes it to `<run>/eval_<split>_<ckpt>.md`. The checkpoint carries its
own config, so no `--config` is needed.

## Run directory

```
runs/<name>/
  config.yaml              resolved config (after --set and --overfit adjustments)
  checkpoints/last.pt      written after every epoch (atomically: .tmp then rename)
  checkpoints/best.pt      the epoch with the highest train.best_metric so far
  eval_<epoch>.md          metric table of each validation pass
  tb/                      TensorBoard events
```

A checkpoint holds `model`, `optimizer`, `scheduler`, `config`, `epoch`, `step` and `best`.
`train.load_model(path)` rebuilds the model from the embedded config and loads the weights.

## Loop

Per step: voxelize on the GPU (no grad) → forward under autocast (`train.precision`: bf16 by default;
fp16 also enables a `GradScaler`) → [loss](loss.md) → backward → clip the gradient norm to
`train.grad_clip` → optimizer step → scheduler step.

- **Non-finite loss:** the step is skipped (no update) and counted in `train/skipped_steps`. Ten in a
  row raise `FloatingPointError`.
- **Validation** runs every `train.eval_every` epochs, after the last epoch, and on a `--max-minutes`
  stop. It decodes with [decode](decode.md), scores with the [evaluator](eval.md), and also
  averages the validation loss. `train.val_max_frames` caps it for quick checks.
- **Data:** the training split streams through `FrameStream` (re-shuffled per epoch with
  `set_epoch`), the validation split in eval mode. `--overfit` instead decodes the segment once into
  memory (~9 s in one pass over its files, versus ~10 min frame by frame) so the GPU is what's measured.

## Learning-rate schedule

`train.lr_schedule` multiplies `optim.lr` per step, over `epochs × steps_per_epoch` steps:

| Schedule | Shape |
|---|---|
| `onecycle` (default) | Cosine warm-up from lr/10 to lr over the first `warmup_frac` (0.4) of the steps, then cosine decay to lr × 10⁻⁴. The CenterPoint recipe. |
| `cosine` | Cosine decay from lr to lr × 10⁻⁴. |
| `constant` | lr throughout. |

For the streamed split the step count is an estimate (frames ÷ batch size, plus a small margin), and
the factor is clamped past the end, so a few extra steps just stay at the final lr.

## TensorBoard

| Tag | Every | What |
|---|---|---|
| `loss/total`, `loss/heatmap[/<class>]`, `loss/reg[/<field>]` | `log_every` steps | Training loss terms from the last step |
| `train/lr`, `train/grad_norm` | `log_every` steps | Current lr; gradient norm before clipping |
| `train/frames_per_s`, `train/data_wait_frac` | `log_every` steps | Throughput over the window; fraction of it spent waiting for the loader |
| `train/gpu_mem_gib` | `log_every` steps | Peak allocated memory in the window |
| `train/skipped_steps`, `train/epoch` | `log_every` steps | |
| `eval/...` | validation | Every metric of `EvalResult.flat()` (see [eval.md](eval.md)) |
| `val/loss/...`, `val/seconds` | validation | Validation loss terms; time the pass took |
| `val/heatmap/<class>` (image) | validation | Ground-truth heatmap \| predicted heatmap, first validation frame |
| `eval/table`, `config` (text) | validation / start | The metric table; the resolved config |

## Smoke result

`--overfit <segment>` (one training segment, every frame, `simple` backbone,
batch 4, lr 1e-3) on an RTX 4090: ~49 frames/s, 4.8 GiB peak, under 5 % data wait. After 40 epochs:

| class | bev AP@0.7 | 3d AP@0.7 | dist AP@1m |
|---|---:|---:|---:|
| vehicle | 0.997 | 0.990 | 1.000 |
| pedestrian | 0.992 | 0.979 | 1.000 |
| cyclist | 0.631 | 0.631 | 0.665 |

Vehicles and pedestrians are memorised, cyclists still rising (0 at epoch 9). That confirms targets,
loss, decode and evaluation agree end to end. It says nothing about generalisation.
