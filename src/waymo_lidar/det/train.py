"""Training loop with periodic validation, checkpoints and TensorBoard.

    uv run wl train ~/dev/data/waymo_v2 --name baseline
    uv run tensorboard --logdir runs          # http://localhost:6006

See docs/detector/training.md.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .config import DetectorConfig, build_optimizer, config_from_dict, config_to_dict, save_config
from .data import FrameStream, collate, make_loader
from .decode import decode
from .eval import EvalResult, Evaluator
from .loss import CenterPointLoss
from .model import BEVDetector
from .targets import TargetBuilder
from .voxelize import Voxelizer

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
MAX_CONSECUTIVE_SKIPS = 10


# --- schedule ----------------------------------------------------------------------------------------


def lr_factor(schedule: str, step: int, total: int, warmup_frac: float) -> float:
    """Multiplier on optim.lr at `step` of `total` (clamped, so extra steps are safe)."""
    p = min(step, total) / max(total, 1)
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        return 1e-4 + (1 - 1e-4) * 0.5 * (1 + math.cos(math.pi * p))
    # onecycle (CenterPoint): cosine ramp lr/10 -> lr over warmup_frac, then cosine down to lr * 1e-4
    if p < warmup_frac:
        q = p / warmup_frac
        return 0.1 + (1 - 0.1) * 0.5 * (1 - math.cos(math.pi * q))
    q = (p - warmup_frac) / (1 - warmup_frac)
    return 1e-4 + (1 - 1e-4) * 0.5 * (1 + math.cos(math.pi * q))


# --- data --------------------------------------------------------------------------------------------


class InMemoryFrames(Dataset):
    """Pre-loaded samples (with targets) for --overfit: decoded once, so the GPU is what gets measured."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        return self.samples[i]


def overfit_samples(root, segment: str, cfg: DetectorConfig) -> list[dict]:
    """Every frame of one training segment (eval mode: no augmentation) with targets, in ONE pass over its files.

    (Calling load_sample per frame re-reads the segment's lidar_pose file and a lidar row group for every
    frame: ~1.6 s each, ~10 min for a segment. One pass takes ~9 s.)
    """
    cfg1 = replace(cfg, data=replace(cfg.data, frame_interval_eval=1))
    stream = FrameStream(root, "training", cfg1, mode="eval", transform=TargetBuilder(cfg1), segments=[segment])
    if not stream.segments:
        raise ValueError(f"training segment {segment} not found or incomplete")
    return [stream.transform(s) for s in stream._frames([segment])]


def loaders(root, cfg: DetectorConfig, overfit: str | None):
    if overfit:
        samples = overfit_samples(root, overfit, cfg)
        ds = InMemoryFrames(samples)
        train = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate, drop_last=True,
                           generator=torch.Generator().manual_seed(cfg.train.seed))
        val = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate)
        return None, train, val, len(ds)
    tb = TargetBuilder(cfg)
    train_ds = FrameStream(root, "training", cfg, mode="train", transform=tb)
    val_ds = FrameStream(root, "validation", cfg, mode="eval", transform=tb)
    return train_ds, None, val_ds, len(train_ds)


# --- one step / validation -----------------------------------------------------------------------------


def forward(model, vox, batch, dev, amp):
    points = batch["points"].to(dev, non_blocking=True)
    with torch.no_grad():
        bev = vox(points, batch["batch_size"], dtype=amp or torch.float32).contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        return model(bev)


@torch.no_grad()
def validate(model, vox, loss_fn, loader, cfg: DetectorConfig, dev, amp, max_frames: int = 0):
    """Returns (EvalResult, mean validation loss logs, (gt, pred) heatmaps of the first frame, frames)."""
    model.eval()
    ev = Evaluator(cfg)
    sums, n_batches, frames, first = defaultdict(float), 0, 0, None
    for batch in loader:
        out = forward(model, vox, batch, dev, amp)
        for k, v in loss_fn(out, batch).scalars().items():
            sums[k] += v
        n_batches += 1
        samples = [{"boxes": b.numpy(), "labels": l.numpy(), "num_points": n.numpy()}
                   for b, l, n in zip(batch["boxes"], batch["labels"], batch["num_points"])]
        ev.add(decode(out, cfg), samples)
        if first is None:
            first = (batch["heatmap"][0].numpy(), torch.sigmoid(out["heatmap"][0].float()).cpu().numpy())
        frames += batch["batch_size"]
        if max_frames and frames >= max_frames:
            break
    model.train()
    return ev.compute(), {k: v / max(n_batches, 1) for k, v in sums.items()}, first, frames


# --- checkpoints -------------------------------------------------------------------------------------


def save_checkpoint(path: Path, model, opt, sched, cfg, epoch: int, step: int, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "config": config_to_dict(cfg), "epoch": epoch, "step": step, "best": best}, tmp)
    tmp.replace(path)


def load_model(ckpt_path: str | Path, device="cuda") -> tuple[BEVDetector, DetectorConfig, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    model = BEVDetector.from_config(cfg).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt


# --- logging helpers -------------------------------------------------------------------------------------


def log_eval(tb: SummaryWriter, res: EvalResult, val_loss: dict, first, step: int, run: Path, epoch, cfg) -> dict:
    flat = res.flat()
    for k, v in flat.items():
        if not math.isnan(v):
            tb.add_scalar(k, v, step)
    for k, v in val_loss.items():
        tb.add_scalar(f"val/{k}", v, step)
    table = res.table()
    tb.add_text("eval/table", table, step)
    if first is not None:
        gt, pred = first
        for c, name in enumerate(cfg.data.classes):
            tb.add_image(f"val/heatmap/{name}", np.concatenate([gt[c], pred[c]], axis=1)[None], step)  # GT | pred
    (run / f"eval_{epoch}.md").write_text(f"# Validation after epoch {epoch} (step {step})\n\n{table}\n")
    return flat


# --- main loop -------------------------------------------------------------------------------------------


def _print(*args) -> None:
    print(*args, flush=True)  # flush: progress must show up immediately, also when piped to a file


def train(root, cfg: DetectorConfig, name: str, resume: bool = False, overfit: str | None = None,
          max_minutes: float | None = None, log=_print) -> Path:
    t_setup = time.perf_counter()
    run = Path(cfg.train.runs_dir) / name
    ckpt_dir = run / "checkpoints"
    if run.exists() and not resume:
        raise FileExistsError(f"{run} exists; use --resume or another --name")
    if overfit:  # one segment, every frame; samples are loaded in eval mode, so there's no augmentation
        cfg = replace(cfg, data=replace(cfg.data, frame_interval_train=1, frame_interval_eval=1))
    run.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run / "config.yaml")

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    amp = AMP[cfg.train.precision]

    log(f"setting up {'overfit on ' + overfit if overfit else 'training'} ...")
    train_ds, train_dl, val_src, frames_per_epoch = loaders(root, cfg, overfit)
    log(f"data ready in {time.perf_counter() - t_setup:.1f} s: {frames_per_epoch} frames per epoch")
    steps_per_epoch = math.ceil(frames_per_epoch / cfg.train.batch_size) + (0 if overfit else 8)
    total_steps = cfg.train.epochs * steps_per_epoch

    vox = Voxelizer(cfg.grid).to(dev)
    model = BEVDetector.from_config(cfg).to(dev).to(memory_format=torch.channels_last)
    loss_fn = CenterPointLoss(cfg).to(dev)
    opt = build_optimizer(cfg.optim, model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(cfg.train.lr_schedule, s, total_steps, cfg.train.warmup_frac))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.precision == "fp16")

    start_epoch, step, best = 0, 0, -math.inf
    if resume and (ckpt_dir / "last.pt").is_file():
        ck = torch.load(ckpt_dir / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        log(f"resumed {run} at epoch {start_epoch}, step {step}")

    tb = SummaryWriter(str(run / "tb"))
    if step == 0:
        tb.add_text("config", "```yaml\n" + (run / "config.yaml").read_text() + "\n```", 0)
    log(f"run {run}: {frames_per_epoch} frames/epoch, {steps_per_epoch} steps/epoch, {total_steps} steps total, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    def val_loader():
        if overfit:
            return val_src
        return make_loader(val_src, cfg)

    def run_validation(epoch: int) -> None:
        nonlocal best
        t = time.perf_counter()
        res, vloss, first, n = validate(model, vox, loss_fn, val_loader(), cfg, dev, amp, cfg.train.val_max_frames)
        flat = log_eval(tb, res, vloss, first, step, run, epoch, cfg)
        dt = time.perf_counter() - t
        tb.add_scalar("val/seconds", dt, step)
        metric = flat.get(cfg.train.best_metric, float("nan"))
        log(f"validation after epoch {epoch} ({n} frames, {dt:.0f} s): {cfg.train.best_metric} = {metric:.4f}\n"
            f"{res.table()}")
        if not math.isnan(metric) and metric > best:
            best = metric
            save_checkpoint(ckpt_dir / "best.pt", model, opt, sched, cfg, epoch, step, best)

    model.train()
    skipped = consecutive = 0
    stop = False
    t_start = time.perf_counter()  # --max-minutes counts training time, not setup
    for epoch in range(start_epoch, cfg.train.epochs):
        if train_ds is not None:
            train_ds.set_epoch(epoch)
            loader = make_loader(train_ds, cfg)
        else:
            loader = train_dl
        win_t, win_frames, win_wait = time.perf_counter(), 0, 0.0
        t_wait = time.perf_counter()
        for batch in loader:
            win_wait += time.perf_counter() - t_wait
            out = forward(model, vox, batch, dev, amp)
            res = loss_fn(out, batch)
            if not torch.isfinite(res.total):
                skipped += 1
                consecutive += 1
                opt.zero_grad(set_to_none=True)
                if consecutive >= MAX_CONSECUTIVE_SKIPS:
                    raise FloatingPointError(f"{consecutive} consecutive non-finite losses at step {step}")
                t_wait = time.perf_counter()
                continue
            consecutive = 0
            opt.zero_grad(set_to_none=True)
            scaler.scale(res.total).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.train.grad_clip if cfg.train.grad_clip > 0 else float("inf"))
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            win_frames += batch["batch_size"]

            if step % cfg.train.log_every == 0:
                torch.cuda.synchronize()
                dt = time.perf_counter() - win_t
                s = res.scalars()
                for k, v in s.items():
                    tb.add_scalar(k, v, step)
                fps = win_frames / dt
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                tb.add_scalar("train/grad_norm", float(grad_norm), step)
                tb.add_scalar("train/frames_per_s", fps, step)
                tb.add_scalar("train/data_wait_frac", win_wait / dt, step)
                tb.add_scalar("train/gpu_mem_gib", torch.cuda.max_memory_allocated() / 2**30, step)
                tb.add_scalar("train/skipped_steps", skipped, step)
                tb.add_scalar("train/epoch", epoch, step)
                log(f"epoch {epoch} step {step}: loss {s['loss/total']:.4f} (hm {s['loss/heatmap']:.4f}, "
                    f"reg {s['loss/reg']:.4f}) lr {sched.get_last_lr()[0]:.2e} | {fps:.1f} frames/s, "
                    f"data wait {100 * win_wait / dt:.0f}%")
                torch.cuda.reset_peak_memory_stats()
                win_t, win_frames, win_wait = time.perf_counter(), 0, 0.0
            if max_minutes and time.perf_counter() - t_start > 60 * max_minutes:
                stop = True
                break
            t_wait = time.perf_counter()

        last_epoch = epoch == cfg.train.epochs - 1
        if stop or last_epoch or (epoch + 1) % cfg.train.eval_every == 0:
            run_validation(epoch)
        save_checkpoint(ckpt_dir / "last.pt", model, opt, sched, cfg, epoch, step, best)
        if stop:
            log(f"stopped after {max_minutes} min (epoch {epoch}, step {step}); resume with --resume")
            break
    tb.close()
    log(f"done: {run} ({(time.perf_counter() - t_setup) / 60:.1f} min including setup), "
        f"best {cfg.train.best_metric} = {best:.4f}")
    return run


def evaluate_checkpoint(root, ckpt: str | Path, split: str = "validation", max_frames: int = 0, log=_print) -> EvalResult:
    dev = torch.device("cuda")
    model, cfg, meta = load_model(ckpt, dev)
    tbuild = TargetBuilder(cfg)
    loader = make_loader(FrameStream(root, split, cfg, mode="eval", transform=tbuild), cfg)
    vox = Voxelizer(cfg.grid).to(dev)
    res, vloss, _, n = validate(model, vox, CenterPointLoss(cfg).to(dev), loader, cfg, dev, AMP[cfg.train.precision], max_frames)
    log(f"{ckpt} (epoch {meta['epoch']}, step {meta['step']}) on {split}, {n} frames: val loss {vloss.get('loss/total', float('nan')):.4f}")
    log(res.table())
    out = Path(ckpt).parent.parent / f"eval_{split}_{Path(ckpt).stem}.md"
    out.write_text(f"# {ckpt} on {split} ({n} frames)\n\n{res.table()}\n")
    return res
