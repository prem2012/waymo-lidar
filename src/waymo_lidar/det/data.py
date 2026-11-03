"""Training / evaluation samples: top-LiDAR points + ground-truth boxes, streamed from segments.

    ds = FrameStream(root, "training", cfg, mode="train")
    loader = make_loader(ds, cfg)
    for epoch in range(cfg.train.epochs):
        ds.set_epoch(epoch)
        for batch in loader: ...

Points and boxes are in the vehicle frame at the frame timestamp (docs/dataset.md#coordinate-frames).
See docs/detector/data.md for the sample format, sampling, shuffling and augmentation rules.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..data.boxes import Boxes
from ..data.dataset import Segment, WaymoDataset
from ..data.range_image import range_image_to_points
from ..data.types import Laser
from .config import AugmentConfig, DetectorConfig, GridConfig, load_config

REQUIRED_COMPONENTS = ("lidar", "lidar_pose", "lidar_calibration", "vehicle_pose", "lidar_box")
SAMPLE_KEYS = ("points", "boxes", "labels", "num_points", "segment", "timestamp", "split")

Sample = dict  # keys: SAMPLE_KEYS (+ anything a transform adds)


# --- per-frame building blocks ----------------------------------------------------------------------


def complete_segments(ds: WaymoDataset, split: str) -> tuple[list[str], list[str]]:
    """(segments with every required component, segments missing at least one)."""
    ok, missing = [], []
    for name in ds.segments(split):
        seg = ds.segment(split, name)
        (ok if all(seg.has(c) for c in REQUIRED_COMPONENTS) else missing).append(name)
    return ok, missing


def num_frames(seg: Segment) -> int:
    """Frame count from Parquet metadata, without reading any data."""
    return pq.ParquetFile(seg.path("vehicle_pose")).metadata.num_rows


def select_boxes(b: Boxes, class_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(boxes (M, 7) float32, labels (M,) int64 channel index, num_top_points (M,) int32)."""
    channel = {cid: i for i, cid in enumerate(class_ids)}
    keep = np.isin(b.type, class_ids)
    boxes = np.concatenate([b.center[keep], b.size[keep], b.heading[keep, None]], axis=1).astype(np.float32)
    labels = np.array([channel[int(t)] for t in b.type[keep]], dtype=np.int64)
    return boxes, labels, b.num_top_points[keep].astype(np.int32)


def top_points(seg: Segment, frame, returns: tuple[int, ...]) -> np.ndarray:
    """(N, 4) float32 x, y, z, intensity of the top LiDAR, vehicle frame at the frame timestamp."""
    pose = seg.pose_at(frame.timestamp)
    calib = seg.calibration[Laser.TOP]
    parts = []
    for r in returns:
        pc = range_image_to_points(frame.images[(Laser.TOP, r)], calib, frame.top_pose, pose)
        parts.append(np.concatenate([pc.xyz, pc.intensity[:, None]], axis=1))
    return np.concatenate(parts).astype(np.float32, copy=False)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def augment(points: np.ndarray, boxes: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Flip across x, rotate about z, scale — the same transform for points and boxes (copies).

    All about the vehicle origin. Returns the parameters used, for tests and debugging.
    """
    points, boxes = points.copy(), boxes.copy()
    flip = bool(rng.random() < cfg.flip_y_prob)
    theta = float(rng.uniform(-cfg.rotation_range, cfg.rotation_range)) if cfg.rotation_range > 0 else 0.0
    scale = float(rng.uniform(*cfg.scale_range))
    if flip:
        points[:, 1] *= -1
        boxes[:, 1] *= -1
        boxes[:, 6] = -boxes[:, 6]
    if theta:
        c, s = math.cos(theta), math.sin(theta)
        for arr in (points, boxes):
            x, y = arr[:, 0].copy(), arr[:, 1].copy()
            arr[:, 0] = c * x - s * y
            arr[:, 1] = s * x + c * y
        boxes[:, 6] += theta
    points[:, :3] *= scale
    boxes[:, :6] *= scale
    boxes[:, 6] = wrap_angle(boxes[:, 6])
    return points, boxes, {"flip": flip, "theta": theta, "scale": scale}


def crop_points(points: np.ndarray, grid: GridConfig) -> np.ndarray:
    """Keep points inside the grid (half-open ranges). Compared in float64, exactly like the voxelizer."""
    m = np.ones(len(points), dtype=bool)
    for axis, (lo, hi) in enumerate((grid.x_range, grid.y_range, grid.z_range)):
        v = points[:, axis].astype(np.float64)
        m &= (v >= lo) & (v < hi)
    return points[m]


def centres_in_grid(boxes: np.ndarray, grid: GridConfig) -> np.ndarray:
    x, y = boxes[:, 0].astype(np.float64), boxes[:, 1].astype(np.float64)
    return (x >= grid.x_range[0]) & (x < grid.x_range[1]) & (y >= grid.y_range[0]) & (y < grid.y_range[1])


def sample_rng(seed: int, epoch: int, segment: str, timestamp: int) -> np.random.Generator:
    """Per-frame RNG; independent of which worker handles the frame (crc32 is stable across processes)."""
    return np.random.default_rng([seed, epoch, zlib.crc32(segment.encode()), timestamp])


def make_sample(points: np.ndarray, boxes: np.ndarray, labels: np.ndarray, num_points: np.ndarray,
                segment: str, timestamp: int, split: str, cfg: DetectorConfig, mode: str,
                rng: np.random.Generator | None = None) -> Sample:
    """Apply the mode's ground-truth rules and augmentation, then crop points to the grid."""
    if mode == "train":
        keep = num_points >= cfg.data.min_points_train
        boxes, labels, num_points = boxes[keep], labels[keep], num_points[keep]
        points, boxes, _ = augment(points, boxes, rng, cfg.augment)
        keep = centres_in_grid(boxes, cfg.grid)
        boxes, labels, num_points = boxes[keep], labels[keep], num_points[keep]
    return {"points": crop_points(points, cfg.grid), "boxes": boxes, "labels": labels, "num_points": num_points,
            "segment": segment, "timestamp": int(timestamp), "split": split}


def load_sample(root: str | Path, split: str, segment: str, timestamp: int, cfg: DetectorConfig,
                mode: str = "eval", epoch: int = 0) -> Sample:
    """One frame, no streaming (tests, debugging, the viewer)."""
    seg = WaymoDataset(root).segment(split, segment)
    fr = next(seg.iter_range_images(returns=cfg.data.returns, lasers=[Laser.TOP], timestamps=[timestamp]))
    boxes, labels, npts = select_boxes(Boxes.from_table(seg.boxes_table()).at(timestamp), cfg.data.class_ids)
    rng = sample_rng(cfg.train.seed, epoch, segment, timestamp) if mode == "train" else None
    return make_sample(top_points(seg, fr, cfg.data.returns), boxes, labels, npts, segment, timestamp, split,
                       cfg, mode, rng)


# --- streaming --------------------------------------------------------------------------------------


def shuffle_buffer(items: Iterator, size: int, rng: np.random.Generator) -> Iterator:
    """Yield every item exactly once, in a locally shuffled order (window of `size`)."""
    buf: list = []
    for item in items:
        if len(buf) < size:
            buf.append(item)
            continue
        i = int(rng.integers(size))
        out, buf[i] = buf[i], item
        yield out
    rng.shuffle(buf)
    yield from buf


class FrameStream(IterableDataset):
    """Frames of one split, sharded by segment across DataLoader workers.

    mode="train": shuffled segment order per epoch, random frame offset per segment per epoch,
    augmentation, shuffle buffer. mode="eval": fixed order, offset 0, no augmentation, no shuffling.
    """

    def __init__(self, root: str | Path, split: str, cfg: DetectorConfig, mode: str = "train",
                 transform: Callable[[Sample], Sample] | None = None, segments: list[str] | None = None):
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.root, self.split, self.cfg, self.mode, self.transform = Path(root), split, cfg, mode, transform
        ds = WaymoDataset(root)
        ok, missing = complete_segments(ds, split)
        if missing:
            warnings.warn(f"{split}: skipping {len(missing)} segment(s) missing a component "
                          f"(e.g. {missing[0]}); {len(ok)} complete", stacklevel=2)
        self.segments = sorted(ok if segments is None else [s for s in segments if s in set(ok)])
        self.interval = cfg.data.frame_interval_train if mode == "train" else cfg.data.frame_interval_eval
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call before creating each epoch's DataLoader iterator (workers copy the dataset then)."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Frames per epoch (exact for eval; for train it varies by ±1 per segment with the offset)."""
        ds = WaymoDataset(self.root)
        return sum(math.ceil(num_frames(ds.segment(self.split, s)) / self.interval) for s in self.segments)

    def epoch_segments(self) -> list[str]:
        if self.mode == "eval":
            return list(self.segments)
        order = np.random.default_rng([self.cfg.train.seed, self.epoch]).permutation(len(self.segments))
        return [self.segments[i] for i in order]

    def frame_offset(self, segment: str) -> int:
        if self.mode == "eval":
            return 0
        return int(np.random.default_rng([self.cfg.train.seed, self.epoch, zlib.crc32(segment.encode()), 1])
                   .integers(self.interval))

    def _frames(self, segments: list[str]) -> Iterator[Sample]:
        ds, cfg = WaymoDataset(self.root), self.cfg
        for name in segments:
            try:
                seg = ds.segment(self.split, name)
                boxes = Boxes.from_table(seg.boxes_table())
                wanted = seg.timestamps[self.frame_offset(name)::self.interval]
                for fr in seg.iter_range_images(returns=cfg.data.returns, lasers=[Laser.TOP], timestamps=wanted):
                    b, labels, npts = select_boxes(boxes.at(fr.timestamp), cfg.data.class_ids)
                    rng = sample_rng(cfg.train.seed, self.epoch, name, fr.timestamp) if self.mode == "train" else None
                    yield make_sample(top_points(seg, fr, cfg.data.returns), b, labels, npts, name, fr.timestamp,
                                      self.split, cfg, self.mode, rng)
            except (OSError, ValueError, KeyError) as e:  # e.g. a file still being downloaded
                warnings.warn(f"{self.split}/{name}: skipped after error: {type(e).__name__}: {e}")

    def __iter__(self) -> Iterator[Sample]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        stream = self._frames(self.epoch_segments()[wid::nw])
        if self.mode == "train" and self.cfg.data.shuffle_buffer > 1:
            rng = np.random.default_rng([self.cfg.train.seed, self.epoch, wid, 2])
            stream = shuffle_buffer(stream, self.cfg.data.shuffle_buffer, rng)
        for sample in stream:
            yield self.transform(sample) if self.transform else sample


# --- batching ---------------------------------------------------------------------------------------


def collate(samples: list[Sample]) -> dict:
    """points -> (sum N, 5) [batch index, x, y, z, intensity]; per-sample lists for variable-size keys."""
    pts = [np.concatenate([np.full((len(s["points"]), 1), i, np.float32), s["points"]], axis=1)
           for i, s in enumerate(samples)]
    batch = {
        "points": torch.from_numpy(np.concatenate(pts)) if pts else torch.zeros((0, 5)),
        "boxes": [torch.from_numpy(s["boxes"]) for s in samples],
        "labels": [torch.from_numpy(s["labels"]) for s in samples],
        "num_points": [torch.from_numpy(s["num_points"]) for s in samples],
        "meta": [(s["split"], s["segment"], s["timestamp"]) for s in samples],
        "batch_size": len(samples),
    }
    for key in samples[0].keys() - set(SAMPLE_KEYS):  # added by a transform
        vals = [s[key] for s in samples]
        if all(isinstance(v, np.ndarray) and v.shape == vals[0].shape for v in vals):
            batch[key] = torch.from_numpy(np.stack(vals))
        else:
            batch[key] = vals
    return batch


def make_loader(ds: FrameStream, cfg: DetectorConfig, batch_size: int | None = None,
                num_workers: int | None = None) -> DataLoader:
    workers = cfg.data.num_workers if num_workers is None else num_workers
    return DataLoader(ds, batch_size=batch_size or cfg.train.batch_size, num_workers=workers,
                      collate_fn=collate, pin_memory=torch.cuda.is_available(),
                      prefetch_factor=2 if workers > 0 else None, persistent_workers=False)


# --- benchmark --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark the training data stream")
    p.add_argument("root", type=Path)
    p.add_argument("--split", default="training")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--mode", default="train", choices=["train", "eval"])
    p.add_argument("--frames", type=int, default=400, help="stop after this many frames")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    ds = FrameStream(a.root, a.split, cfg, mode=a.mode)
    workers = cfg.data.num_workers if a.workers is None else a.workers
    active = min(workers, len(ds.segments)) if workers else 1
    print(f"{a.split}: {len(ds.segments)} complete segments, ~{len(ds)} frames/epoch, "
          f"{workers} workers ({active} with work)")
    loader = make_loader(ds, cfg, batch_size=1, num_workers=workers)
    n = pts = boxes = 0
    t0 = first = None
    t0 = time.perf_counter()
    for batch in loader:
        if first is None:
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
        n += 1
        pts += len(batch["points"])
        boxes += sum(len(b) for b in batch["boxes"])
        if n >= a.frames:
            break
    dt = time.perf_counter() - t0
    print(f"first frame after {first:.1f}s; then {(n - 1) / dt:.1f} frames/s over {n - 1} frames; "
          f"{pts / n:,.0f} points/frame, {boxes / n:.1f} boxes/frame")


if __name__ == "__main__":
    main()
