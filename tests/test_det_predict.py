"""Prediction viewer: per-frame matching (vs a sequential reference) and an end-to-end .rrd write."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from waymo_lidar.det import predict as P
from waymo_lidar.det.config import load_config
from waymo_lidar.det.iou import ious, jitter, random_boxes

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
SEG = "10023947602400723454_1120_000_1140_000"
CFG = load_config()


def reference_match(det, sample, cfg, thr):
    """Plain sequential version of the evaluator's rules for one frame and BEV IoU threshold."""
    db, ds, dl = det["boxes"], det["scores"], det["labels"]
    gb, gl = sample["boxes"], sample["labels"]
    ign = P.ignored_gt(sample, cfg)
    iou = ious(db, gb)[0] if len(db) and len(gb) else np.zeros((len(db), len(gb)))
    det_status, gt_status = np.full(len(ds), P.FP), np.where(ign, P.IGNORED, P.MISSED)
    taken = np.zeros(len(gb), bool)
    for d in np.lexsort((np.arange(len(ds)), -ds)):
        cand = [g for g in range(len(gb)) if gl[g] == dl[d] and iou[d, g] >= thr and not ign[g] and not taken[g]]
        if cand:
            g = max(cand, key=lambda g: (iou[d, g], -g))  # best IoU, ties to the lowest index
            taken[g], det_status[d], gt_status[g] = True, P.TP, P.MATCHED
        elif any(gl[g] == dl[d] and iou[d, g] >= thr and ign[g] for g in range(len(gb))):
            det_status[d] = P.DROPPED
    return det_status, gt_status


def random_frame(rng, n_gt, n_det):
    gt = random_boxes(n_gt, rng, spread=20.0).astype(np.float32)
    labels = rng.integers(0, 3, n_gt)
    pts = rng.integers(0, 3, n_gt)  # some 0-point GT -> ignored
    src = rng.integers(0, max(n_gt, 1), n_det)
    det = jitter(gt[src], rng, pos=0.4) if n_gt else random_boxes(n_det, rng, spread=20.0)
    det = np.concatenate([det, random_boxes(n_det // 3, rng, spread=20.0)]).astype(np.float32)  # some pure FPs
    dl = np.concatenate([labels[src] if n_gt else rng.integers(0, 3, n_det), rng.integers(0, 3, n_det // 3)])
    dl[rng.random(len(dl)) < 0.1] = rng.integers(0, 3)  # a few wrong-class detections
    scores = np.round(rng.random(len(dl)), 1).astype(np.float32)  # rounding -> many score ties
    return ({"boxes": det, "scores": scores, "labels": dl.astype(np.int64)},
            {"boxes": gt, "labels": labels.astype(np.int64), "num_points": pts.astype(np.int64)})


@pytest.mark.parametrize("seed", range(40))
def test_match_frame_equals_sequential_reference(seed):
    rng = np.random.default_rng(seed)
    det, sample = random_frame(rng, int(rng.integers(0, 25)), int(rng.integers(0, 40)))
    thr = float(rng.choice([0.3, 0.5, 0.7]))
    fm = P.match_frame(det, sample, CFG, thr)
    ref_det, ref_gt = reference_match(det, sample, CFG, thr)
    np.testing.assert_array_equal(fm.det_status, ref_det)
    np.testing.assert_array_equal(fm.gt_status, ref_gt)
    assert (fm.det_iou[fm.det_status == P.TP] >= thr).all() and (fm.det_iou[fm.det_status != P.TP] == 0).all()


def test_ground_truth_as_detections():
    rng = np.random.default_rng(0)
    gt = random_boxes(12, rng, spread=20.0).astype(np.float32)
    gt[-1, 0] = 500.0  # outside the grid -> ignored
    sample = {"boxes": gt, "labels": np.arange(12) % 3, "num_points": np.r_[np.ones(10), 0, 5].astype(np.int64)}
    det = {"boxes": gt.copy(), "scores": np.linspace(1, 0.5, 12, dtype=np.float32), "labels": np.arange(12) % 3}
    fm = P.match_frame(det, sample, CFG, 0.7)
    assert (fm.det_status[:10] == P.TP).all() and (fm.det_status[10:] == P.DROPPED).all()
    assert (fm.gt_status[:10] == P.MATCHED).all() and (fm.gt_status[10:] == P.IGNORED).all()


def test_empty_frames():
    empty_det = {"boxes": np.zeros((0, 7), np.float32), "scores": np.zeros(0, np.float32), "labels": np.zeros(0, np.int64)}
    det, sample = random_frame(np.random.default_rng(1), 5, 5)
    assert (P.match_frame(empty_det, sample, CFG).gt_status != P.MATCHED).all()
    no_gt = {"boxes": np.zeros((0, 7), np.float32), "labels": np.zeros(0, np.int64), "num_points": np.zeros(0, np.int64)}
    assert (P.match_frame(det, no_gt, CFG).det_status == P.FP).all()


def test_heading_ticks_point_forward():
    b = np.array([[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, np.pi / 2]])
    np.testing.assert_allclose(P.heading_ticks(b)[0], [[1, 2, 0.5], [1, 4, 0.5]], atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not (ROOT / "training" / "lidar_box" / f"{SEG}.parquet").is_file(), reason="stage-1 data missing")
def test_run_writes_recording(tmp_path):
    from waymo_lidar.det.model import BEVDetector
    from waymo_lidar.det.train import save_checkpoint

    cfg = load_config(overrides=["grid.x_range=[-25.6, 25.6]", "grid.y_range=[-25.6, 25.6]"])
    model = BEVDetector.from_config(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    save_checkpoint(tmp_path / "ck.pt", model, opt, sched, cfg, epoch=0, step=0, best=0.0)
    out = P.run(ROOT, tmp_path / "ck.pt", SEG, split="training", every=50, min_score=0.0, heatmaps=True,
                save=tmp_path / "p.rrd", log=lambda *a: None)
    assert out["frames"] == 4  # 199 frames, every 50th
    assert (tmp_path / "p.rrd").stat().st_size > 1_000_000
    n_gt = sum(v for k, v in out["totals"].items() if k.startswith("gt_"))
    assert n_gt > 0 and out["result"].counts["vehicle"]["num_gt"] > 0
