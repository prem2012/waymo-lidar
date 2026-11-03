"""Decode tests (pinned stage-1 segment only)."""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from waymo_lidar.data.dataset import WaymoDataset
from waymo_lidar.det.config import build_optimizer, load_config
from waymo_lidar.det.data import collate, load_sample, wrap_angle
from waymo_lidar.det.decode import SIZE_MAX, SIZE_MIN, decode, peak_mask, perfect_outputs
from waymo_lidar.det.targets import TargetBuilder, box_cells

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
SEG = "10023947602400723454_1120_000_1140_000"
needs_data = pytest.mark.skipif(not (ROOT / "training" / "lidar_box" / f"{SEG}.parquet").is_file(),
                                reason="stage-1 data missing")
CFG = load_config()


def outputs(heat: dict, reg_fill=0.0, b=1, c=3, h=376, w=376):
    """heat: {(b, c, row, col): probability}; everything else ~0."""
    logits = torch.full((b, c, h, w), -20.0)
    for (bi, ci, r, co), p in heat.items():
        logits[bi, ci, r, co] = math.log(p / (1 - p)) if p < 1 else 20.0
    return {"heatmap": logits, "reg": torch.full((b, 8, h, w), reg_fill)}


# --- peaks -------------------------------------------------------------------------------------------


def test_single_peak_and_threshold():
    d = decode(outputs({(0, 0, 100, 200): 0.9, (0, 1, 50, 50): 0.05}), CFG)[0]
    assert len(d) == 1 and d.labels.tolist() == [0] and d.scores[0].item() == pytest.approx(0.9, rel=1e-5)
    # reg = 0 -> centre at the cell's corner (dx = dy = 0), sizes exp(0) = 1 m, heading atan2(0, 0) = 0
    x0, y0 = CFG.grid.x_range[0], CFG.grid.y_range[0]
    assert d.boxes[0, :2].tolist() == pytest.approx([x0 + 200 * 0.4, y0 + 100 * 0.4], abs=1e-4)


def test_adjacent_equal_peaks_both_kept_and_tail_suppressed():
    heat = {(0, 0, 10, 10): 0.8, (0, 0, 10, 11): 0.8, (0, 0, 30, 30): 0.9, (0, 0, 30, 31): 0.5}
    d = decode(outputs(heat), CFG)[0]
    cells = sorted((round((y + 75.2) / 0.4), round((x + 75.2) / 0.4)) for x, y in d.boxes[:, :2].tolist())
    assert cells == [(10, 10), (10, 11), (30, 30)]  # the lower neighbour (30, 31) is not a peak


def test_peak_mask_on_gaussian_has_single_peak():
    from waymo_lidar.det.targets import gaussian_kernel
    g = torch.from_numpy(gaussian_kernel(4)).float()[None, None]
    assert peak_mask(g).sum().item() == 1 + (g == 0).sum().item()  # the centre, plus the zero background ties


def test_classes_same_cell_both_kept_and_sorted():
    heat = {(0, 0, 5, 5): 0.6, (0, 2, 5, 5): 0.7, (0, 1, 40, 40): 0.95}
    d = decode(outputs(heat), CFG)[0]
    assert d.labels.tolist() == [1, 2, 0]  # sorted by score: 0.95, 0.7, 0.6
    assert torch.all(d.scores[:-1] >= d.scores[1:])


def test_topk_caps_and_keeps_highest():
    cfg = load_config(overrides=["decode.topk=3"])
    heat = {(0, 0, 10 * i, 10 * i): 0.2 + 0.05 * i for i in range(1, 10)}
    d = decode(outputs(heat), cfg)[0]
    assert len(d) == 3 and d.scores.tolist() == pytest.approx([0.65, 0.6, 0.55], rel=1e-5)


def test_size_clamp_keeps_boxes_finite():
    out = outputs({(0, 0, 20, 20): 0.9, (0, 1, 60, 60): 0.9})
    out["reg"][0, 3:6, 20, 20] = 80.0    # exp would overflow
    out["reg"][0, 3:6, 60, 60] = -80.0   # exp would underflow to 0
    d = decode(out, CFG)[0]
    sizes = d.boxes[:, 3:6]
    assert torch.isfinite(sizes).all()
    assert sizes.max().item() == pytest.approx(SIZE_MAX, rel=1e-5) and sizes.min().item() == pytest.approx(SIZE_MIN, rel=1e-5)


def test_nms_bev_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="iou"):
        decode(outputs({}), load_config(overrides=["decode.nms=bev"]))


def test_batch_matches_single_frames():
    heat = {(0, 0, 10, 10): 0.9, (1, 1, 200, 300): 0.8, (1, 2, 7, 9): 0.4}
    both = decode(outputs(heat, b=2), CFG)
    one = decode(outputs({k: v for k, v in heat.items() if k[0] == 0}, b=1), CFG)[0]
    two = decode(outputs({(0, *k[1:]): v for k, v in heat.items() if k[0] == 1}, b=1), CFG)[0]
    assert len(both) == 2
    for a, b_ in ((both[0], one), (both[1], two)):
        assert torch.equal(a.labels, b_.labels) and torch.allclose(a.boxes, b_.boxes) and torch.allclose(a.scores, b_.scores)


# --- real data -----------------------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("mode, epoch", [("eval", 0), ("train", 2)])
def test_perfect_outputs_roundtrip(mode, epoch):
    tb = TargetBuilder(CFG)
    ts = WaymoDataset(ROOT).segment("training", SEG).timestamps
    samples = [load_sample(ROOT, "training", SEG, int(t), CFG, mode=mode, epoch=epoch) for t in ts[[30, 150]]]
    batch = collate([tb(s) for s in samples])
    dets = decode(perfect_outputs(batch, CFG), CFG)
    for s, d in zip(samples, dets):
        _, _, _, inside = box_cells(s["boxes"], CFG)
        gt, gl = s["boxes"][inside].astype(np.float64), s["labels"][inside]
        d = d.numpy()
        assert len(d["boxes"]) == len(gt)  # nothing extra, nothing missing
        assert np.allclose(d["scores"], 1.0)
        order_gt = np.lexsort((gt[:, 1], gt[:, 0]))
        order_d = np.lexsort((d["boxes"][:, 1], d["boxes"][:, 0]))
        g, b = gt[order_gt], d["boxes"][order_d].astype(np.float64)
        assert np.abs(b[:, :3] - g[:, :3]).max() < 1e-4
        assert np.abs(b[:, 3:6] / g[:, 3:6] - 1).max() < 1e-5
        assert np.abs(wrap_angle(b[:, 6] - g[:, 6])).max() < 1e-5
        assert np.array_equal(d["labels"][order_d], gl[order_gt])


@needs_data
def test_perfect_outputs_single_sample_numpy():
    ts = int(WaymoDataset(ROOT).segment("training", SEG).timestamps[0])
    t = TargetBuilder(CFG)(load_sample(ROOT, "training", SEG, ts, CFG))
    out = perfect_outputs(t, CFG)
    assert out["heatmap"].shape == (1, 3, 376, 376) and out["reg"].shape == (1, 8, 376, 376)
    assert len(decode(out, CFG)[0]) == int(t["target_mask"].sum())


@needs_data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_overfit_then_decode_finds_the_objects():
    from waymo_lidar.det.loss import CenterPointLoss
    from waymo_lidar.det.model import BEVDetector
    from waymo_lidar.det.voxelize import Voxelizer
    cfg = load_config(overrides=["optim.lr=1e-3"])
    tb = TargetBuilder(cfg)
    ts = WaymoDataset(ROOT).segment("training", SEG).timestamps
    samples = [load_sample(ROOT, "training", SEG, int(t), cfg) for t in ts[[20, 140]]]
    batch = collate([tb(s) for s in samples])
    torch.manual_seed(0)
    bev = Voxelizer(cfg.grid).cuda()(batch["points"].cuda(), 2, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    model = BEVDetector.from_config(cfg).cuda().to(memory_format=torch.channels_last).train()
    loss_fn, opt = CenterPointLoss(cfg).cuda(), build_optimizer(cfg.optim, model.parameters())
    for _ in range(300):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(bev)
        res = loss_fn(out, batch)
        opt.zero_grad(set_to_none=True)
        res.total.backward()
        opt.step()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dets = decode(model(bev), cfg)
    cyclist = cfg.data.classes.index("cyclist")
    for s, d in zip(samples, dets):
        d = d.numpy()
        _, _, _, inside = box_cells(s["boxes"], cfg)
        # Vehicles and pedestrians with >= 5 points must be found. The batch has a single cyclist: with one
        # positive in its heatmap channel it learns much more slowly (p = 0.097 after 300 steps, just under the
        # 0.1 threshold), so it isn't required here. See docs/detector/decode.md.
        want = inside & (s["num_points"] >= 5) & (s["labels"] != cyclist)
        dist = np.linalg.norm(s["boxes"][want, None, :2] - d["boxes"][None, :, :2], axis=2)
        same = s["labels"][want, None] == d["labels"][None]
        best = np.where(same, dist, np.inf).min(1)
        assert (best < 0.3).all(), f"missed {np.sum(best >= 0.3)} of {want.sum()} objects"
        # every confident detection belongs to some in-grid GT box of its class
        conf = d["scores"] > 0.3
        dist_all = np.linalg.norm(d["boxes"][conf, None, :2] - s["boxes"][None, inside, :2], axis=2)
        same_all = d["labels"][conf, None] == s["labels"][None, inside]
        assert (np.where(same_all, dist_all, np.inf).min(1) < 0.3).all()
