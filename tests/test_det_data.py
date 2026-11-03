"""Data stream tests on the local stage-1 segments (pinned by name, so a running download doesn't matter)."""

import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from waymo_lidar.data.boxes import Boxes, points_in_boxes
from waymo_lidar.data.dataset import WaymoDataset
from waymo_lidar.det.config import load_config
from waymo_lidar.det.data import (REQUIRED_COMPONENTS, FrameStream, augment, collate, complete_segments, crop_points,
                                  load_sample, make_loader, num_frames, select_boxes, shuffle_buffer, wrap_angle)

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
TRAIN = ["10017090168044687777_6380_000_6400_000", "10023947602400723454_1120_000_1140_000"]
VAL = ["10203656353524179475_7625_000_7645_000"]
needs_data = pytest.mark.skipif(not all((ROOT / "training" / "lidar_box" / f"{s}.parquet").is_file() for s in TRAIN),
                                reason=f"stage-1 segments not at {ROOT}")


def cfg_with(*overrides):
    return load_config(overrides=["data.num_workers=0", *overrides])


def frames(ds, workers=0):
    return [b["meta"][0] + (len(b["points"]), len(b["boxes"][0])) for b in make_loader(ds, ds.cfg, 1, workers)]


# --- pure functions ---------------------------------------------------------------------------------


def test_wrap_angle():
    a = wrap_angle(np.array([np.pi, -np.pi, 3 * np.pi / 2, -3 * np.pi / 2, 0.1]))
    assert np.all(a >= -np.pi) and np.all(a < np.pi)
    assert np.allclose(a, [-np.pi, -np.pi, -np.pi / 2, np.pi / 2, 0.1])


def test_shuffle_buffer_yields_everything_once():
    rng = np.random.default_rng(0)
    out = list(shuffle_buffer(iter(range(1000)), 50, rng))
    assert sorted(out) == list(range(1000)) and out != list(range(1000))


def test_augment_transforms_points_and_boxes_identically():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 20, (5000, 4)).astype(np.float32)
    boxes = np.array([[5, 3, 0.5, 4, 2, 1.6, 0.3], [-8, -6, 1.0, 1, 1, 1.8, -2.9]], np.float32)
    cfg = cfg_with("augment.flip_y_prob=1.0").augment
    pts_in, boxes_in = pts.copy(), boxes.copy()
    p2, b2, params = augment(pts, boxes, np.random.default_rng(1), cfg)
    assert np.array_equal(pts, pts_in) and np.array_equal(boxes, boxes_in)  # inputs not mutated
    assert params["flip"]
    s, th = params["scale"], params["theta"]
    assert np.allclose(b2[:, 3:6], boxes[:, 3:6] * s, atol=1e-5)
    assert np.allclose(b2[:, 6], wrap_angle(-boxes[:, 6] + th), atol=1e-5)
    assert np.all(b2[:, 6] >= -np.pi) and np.all(b2[:, 6] < np.pi)
    assert np.allclose(p2[:, 3], pts[:, 3])  # intensity untouched
    before = points_in_boxes(pts[:, :3], boxes[:, :3].astype(float), boxes[:, 3:6].astype(float), boxes[:, 6].astype(float))
    after = points_in_boxes(p2[:, :3], b2[:, :3].astype(float), b2[:, 3:6].astype(float), b2[:, 6].astype(float))
    assert np.abs(before - after).max() <= 1


def test_crop_points_half_open():
    grid = cfg_with().grid
    pts = np.array([[-75.2, 0, 0, 1], [75.2, 0, 0, 1], [0, 0, -2.0, 1], [0, 0, 4.0, 1], [10, -10, 1, 1]], np.float64)
    assert len(crop_points(pts, grid)) == 3  # lower bounds kept, upper bounds dropped


def test_crop_agrees_with_voxelizer_on_float32_edges():
    # float32(75.2) = 75.19999695 < 75.2 and float32(-75.2) > -75.2: both are inside the grid, in both components.
    from waymo_lidar.det.voxelize import points_to_voxel_indices
    grid = cfg_with().grid
    pts = np.array([[75.2, 0, 0, 1], [-75.2, 0, 0, 1], [0, 0, 4.0, 1], [0, 0, -2.0, 1]], np.float32)
    _, mask = points_to_voxel_indices(torch.from_numpy(pts[:, :3]), grid)
    kept = np.isin(np.arange(len(pts)), [i for i, p in enumerate(pts) if len(crop_points(p[None], grid))])
    assert np.array_equal(kept, mask.numpy())


# --- on real data -----------------------------------------------------------------------------------


@needs_data
def test_eval_sample_shapes_and_frame():
    cfg = cfg_with()
    seg = WaymoDataset(ROOT).segment("training", TRAIN[0])
    ts = int(seg.timestamps[50])
    s = load_sample(ROOT, "training", TRAIN[0], ts, cfg, mode="eval")
    p, b = s["points"], s["boxes"]
    assert p.dtype == np.float32 and p.shape[1] == 4 and len(p) > 100_000
    g = cfg.grid
    assert (p[:, 0] >= g.x_range[0]).all() and (p[:, 0] < g.x_range[1]).all()
    assert (p[:, 2] >= g.z_range[0]).all() and (p[:, 2] < g.z_range[1]).all()
    assert b.shape[1] == 7 and b.dtype == np.float32 and s["labels"].dtype == np.int64
    assert set(s["labels"].tolist()) <= {0, 1, 2}
    # Eval keeps every box of the configured classes, unaugmented, equal to the dataset's boxes.
    ref, ref_labels, ref_n = select_boxes(Boxes.from_table(seg.boxes_table()).at(ts), cfg.data.class_ids)
    assert np.array_equal(b, ref) and np.array_equal(s["labels"], ref_labels) and np.array_equal(s["num_points"], ref_n)
    # Points and boxes share the frame: decoded top-LiDAR counts track the dataset's num_top_points.
    counts = points_in_boxes(p[:, :3], b[:, :3].astype(float), b[:, 3:6].astype(float), b[:, 6].astype(float))
    many = s["num_points"] >= 50
    assert np.median(counts[many] / s["num_points"][many]) > 0.9


@needs_data
def test_train_sample_rules_and_augmentation_geometry():
    cfg = cfg_with()
    seg = WaymoDataset(ROOT).segment("training", TRAIN[1])
    ts = int(seg.timestamps[100])
    ev = load_sample(ROOT, "training", TRAIN[1], ts, cfg, mode="eval")
    tr = load_sample(ROOT, "training", TRAIN[1], ts, cfg, mode="train", epoch=3)
    assert (tr["num_points"] >= cfg.data.min_points_train).all()
    g = cfg.grid
    assert ((tr["boxes"][:, 0] >= g.x_range[0]) & (tr["boxes"][:, 0] < g.x_range[1])).all()
    # Same seed/epoch/frame -> same sample.
    again = load_sample(ROOT, "training", TRAIN[1], ts, cfg, mode="train", epoch=3)
    assert np.array_equal(tr["points"], again["points"]) and np.array_equal(tr["boxes"], again["boxes"])
    other = load_sample(ROOT, "training", TRAIN[1], ts, cfg, mode="train", epoch=4)
    assert not np.array_equal(tr["boxes"], other["boxes"])
    # Augmentation keeps point-in-box counts for boxes well inside the grid.
    rng = np.random.default_rng(7)
    p2, b2, _ = augment(ev["points"], ev["boxes"], rng, cfg.augment)
    inner = np.hypot(ev["boxes"][:, 0], ev["boxes"][:, 1]) < 50
    def count(p, b):
        return points_in_boxes(p[:, :3], b[inner, :3].astype(float), b[inner, 3:6].astype(float), b[inner, 6].astype(float))
    before, after = count(ev["points"], ev["boxes"]), count(p2, b2)
    assert inner.sum() > 10
    assert np.abs(before - after).max() <= 2  # float32 rounding at box faces


@needs_data
def test_eval_stream_covers_each_frame_once_and_is_deterministic():
    cfg = cfg_with("data.frame_interval_eval=20")
    ds = FrameStream(ROOT, "training", cfg, mode="eval", segments=TRAIN)
    a, b = frames(ds), frames(ds)
    assert a == b  # deterministic order and content
    keys = [(seg, ts) for _, seg, ts, *_ in a]
    assert len(keys) == len(set(keys)) == len(ds)
    wds = WaymoDataset(ROOT)
    for name in TRAIN:
        expected = wds.segment("training", name).timestamps[::20].tolist()
        assert [ts for _, seg, ts, *_ in a if seg == name] == expected


@needs_data
def test_two_workers_shard_by_segment_without_duplicates():
    cfg = cfg_with("data.frame_interval_eval=25")
    ds = FrameStream(ROOT, "training", cfg, mode="eval", segments=TRAIN)
    single, multi = frames(ds, 0), frames(ds, 2)
    assert sorted(single) == sorted(multi)
    assert Counter((s, t) for _, s, t, *_ in multi).most_common(1)[0][1] == 1


@needs_data
def test_train_epochs_change_order_and_offsets_reproducibly():
    cfg = cfg_with("data.frame_interval_train=20", "data.shuffle_buffer=8")
    ds = FrameStream(ROOT, "training", cfg, mode="train", segments=TRAIN)
    ds.set_epoch(0); e0 = frames(ds)
    ds.set_epoch(0); e0b = frames(ds)
    ds.set_epoch(1); e1 = frames(ds)
    assert e0 == e0b
    assert e0 != e1
    offsets = set()
    for ep in range(6):
        ds.set_epoch(ep)
        offsets.add(tuple(ds.frame_offset(name) for name in TRAIN))
    assert len(offsets) > 1  # frame offsets vary across epochs
    for frames_ in (e0, e1):
        keys = [(s, t) for _, s, t, *_ in frames_]
        assert len(keys) == len(set(keys))
        assert abs(len(keys) - len(ds)) <= len(TRAIN)  # +-1 per segment depending on the offset


@needs_data
def test_incomplete_segments_are_skipped(tmp_path):
    for c in REQUIRED_COMPONENTS:
        (tmp_path / "training" / c).mkdir(parents=True)
        for name in TRAIN:
            if c == "lidar_pose" and name == TRAIN[1]:
                continue  # second segment is missing a component
            (tmp_path / "training" / c / f"{name}.parquet").symlink_to(ROOT / "training" / c / f"{name}.parquet")
    ok, missing = complete_segments(WaymoDataset(tmp_path), "training")
    assert ok == [TRAIN[0]] and missing == [TRAIN[1]]
    with pytest.warns(UserWarning, match="missing a component"):
        ds = FrameStream(tmp_path, "training", cfg_with(), mode="eval")
    assert ds.segments == [TRAIN[0]]


@needs_data
def test_collate_batch_index_and_lists():
    cfg = cfg_with("data.frame_interval_eval=50")
    ds = FrameStream(ROOT, "validation", cfg, mode="eval", segments=VAL)
    batch = next(iter(make_loader(ds, cfg, batch_size=3, num_workers=0)))
    assert batch["batch_size"] == 3 and batch["points"].shape[1] == 5
    idx = batch["points"][:, 0]
    assert set(idx.unique().tolist()) == {0.0, 1.0, 2.0}
    assert all(isinstance(b, torch.Tensor) and b.shape[1] == 7 for b in batch["boxes"])
    assert len(batch["labels"]) == len(batch["num_points"]) == len(batch["meta"]) == 3
    # transform-added fixed-shape arrays are stacked
    s = [{**load_sample(ROOT, "validation", VAL[0], int(t), cfg), "extra": np.zeros((2, 3), np.float32)}
         for t in WaymoDataset(ROOT).segment("validation", VAL[0]).timestamps[:2]]
    assert collate(s)["extra"].shape == (2, 2, 3)


@needs_data
def test_len_matches_frame_counts():
    cfg = cfg_with("data.frame_interval_eval=5")
    ds = FrameStream(ROOT, "training", cfg, mode="eval", segments=TRAIN)
    wds = WaymoDataset(ROOT)
    assert len(ds) == sum(math.ceil(num_frames(wds.segment("training", s)) / 5) for s in TRAIN)
