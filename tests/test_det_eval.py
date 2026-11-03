"""Evaluation tests (pinned stage-1 segment only)."""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from waymo_lidar.data.dataset import WaymoDataset
from waymo_lidar.det.config import build_optimizer, load_config
from waymo_lidar.det.data import collate, load_sample
from waymo_lidar.det.decode import decode, perfect_outputs
from waymo_lidar.det.eval import Evaluator, average_precision, greedy_match
from waymo_lidar.det.targets import TargetBuilder

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
SEG = "10023947602400723454_1120_000_1140_000"
needs_data = pytest.mark.skipif(not (ROOT / "training" / "lidar_box" / f"{SEG}.parquet").is_file(),
                                reason="stage-1 data missing")
CFG = load_config()
CAR = [4.0, 2.0, 1.5]


def box(x, y, z=1.0, size=CAR, heading=0.0):
    return [x, y, z, *size, heading]


def frame(dets, gts, det_labels=None, gt_labels=None, points=None):
    """dets: [(box, score)], gts: [box]."""
    db = np.array([d for d, _ in dets], np.float32).reshape(-1, 7)
    ds = np.array([s for _, s in dets], np.float32)
    gb = np.array(gts, np.float32).reshape(-1, 7)
    return ({"boxes": db, "scores": ds, "labels": np.array(det_labels or [0] * len(dets))},
            {"boxes": gb, "labels": np.array(gt_labels or [0] * len(gts)), "num_points": np.array(points or [100] * len(gts))})


def evaluate(frames, cfg=CFG):
    ev = Evaluator(cfg)
    for d, s in frames:
        ev.add([d], [s])
    return ev.compute(device="cpu")


# --- AP --------------------------------------------------------------------------------------------


def test_average_precision_by_hand():
    # TP (0.9), FP (0.8), TP (0.7) over 2 GT: precision 1, .5, .667; recall .5, .5, 1 -> AP = .5*1 + .5*.667
    ap, _, _ = average_precision(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7]), 2)
    assert ap == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert average_precision(np.array([1, 1]), np.array([0.9, 0.8]), 2)[0] == 1.0
    assert average_precision(np.zeros(0), np.zeros(0), 3)[0] == 0.0
    assert average_precision(np.array([0, 0]), np.array([0.9, 0.8]), 3)[0] == 0.0
    assert math.isnan(average_precision(np.array([0]), np.array([0.5]), 0)[0])


# --- matching ----------------------------------------------------------------------------------------


def sequential_greedy(rank, pd, pg, pa, n_det, n_gt):
    matched, taken = np.full(n_det, -1), np.zeros(n_gt, bool)
    for d in np.argsort(rank):
        cand = [(a, -g) for dd, g, a in zip(pd, pg, pa) if dd == d and not taken[g]]
        if cand:
            a, g = max(cand)
            matched[d], taken[-g] = -g, True
    return matched


def test_vectorised_greedy_equals_sequential():
    rng = np.random.default_rng(0)
    for trial in range(300):
        n_det, n_gt = rng.integers(1, 25), rng.integers(1, 15)
        k = rng.integers(0, n_det * n_gt + 1)
        pairs = rng.choice(n_det * n_gt, size=k, replace=False)
        pd, pg = pairs // n_gt, pairs % n_gt
        pa = rng.choice([0.3, 0.5, 0.7, 0.9], size=k) if trial % 2 else rng.random(k)  # odd trials: many affinity ties
        rank = rng.permutation(n_det)
        assert np.array_equal(greedy_match(rank, pd, pg, pa, n_det, n_gt), sequential_greedy(rank, pd, pg, pa, n_det, n_gt))


def test_higher_score_wins_duplicate_is_fp():
    r = evaluate([frame([(box(0, 0), 0.6), (box(0.1, 0), 0.9)], [box(0, 0)])])
    # 0.9 matches; 0.6 duplicate is an FP ranked second -> precision 1 then 0.5, recall 1 -> AP 1
    assert r.ap[("bev", 0.7, "vehicle", None)] == 1.0
    r2 = evaluate([frame([(box(0, 0), 0.9), (box(0.1, 0), 0.95)], [box(0, 0)])])
    assert r2.ap[("bev", 0.7, "vehicle", None)] == 1.0  # whichever scores higher takes the GT; still one TP first


def test_ignored_gt_drops_detection():
    # The only GT has 0 points: ignored. A detection on it is neither TP nor FP; one FP elsewhere remains.
    r = evaluate([frame([(box(0, 0), 0.9), (box(30, 30), 0.8)], [box(0, 0), box(-20, 0)], points=[0, 100])])
    # valid GT = the one at (-20, 0), not detected -> AP 0; the FP at (30, 30) is counted, the (0, 0) det is not
    assert r.ap[("bev", 0.5, "vehicle", None)] == 0.0
    assert r.counts["vehicle"] == {"num_gt": 1, "num_ignored": 1, "num_det": 2}
    r2 = evaluate([frame([(box(0, 0), 0.9), (box(-20, 0), 0.5)], [box(0, 0), box(-20, 0)], points=[0, 100])])
    assert r2.ap[("bev", 0.5, "vehicle", None)] == 1.0  # the dropped det can't lower precision


def test_out_of_grid_gt_is_ignored():
    r = evaluate([frame([(box(76.0, 0), 0.9)], [box(76.0, 0), box(10, 0)])])
    assert r.counts["vehicle"]["num_ignored"] == 1 and r.ap[("bev", 0.5, "vehicle", None)] == 0.0


def test_threshold_edges_and_class_mismatch():
    # 4 x 2 boxes shifted along x: IoU = (4 - s) / (4 + s). s = 0.7 -> 0.702 (TP at 0.7), s = 0.75 -> 0.684 (FP)
    tp = evaluate([frame([(box(0.7, 0), 0.9)], [box(0, 0)])])
    fp = evaluate([frame([(box(0.75, 0), 0.9)], [box(0, 0)])])
    assert tp.ap[("bev", 0.7, "vehicle", None)] == 1.0 and fp.ap[("bev", 0.7, "vehicle", None)] == 0.0
    wrong = evaluate([frame([(box(0, 0), 0.9)], [box(0, 0)], det_labels=[1])])
    assert wrong.ap[("bev", 0.5, "vehicle", None)] == 0.0                # the vehicle GT is missed
    assert math.isnan(wrong.ap[("bev", 0.5, "pedestrian", None)])         # no pedestrian GT: the det can't count as TP


def test_3d_vs_bev():
    r = evaluate([frame([(box(0, 0, z=1.8), 0.9)], [box(0, 0, z=1.0)])])  # same footprint, z off by 0.8 m of 1.5
    assert r.ap[("bev", 0.7, "vehicle", None)] == 1.0 and r.ap[("3d", 0.7, "vehicle", None)] == 0.0


def test_center_distance_matching():
    r = evaluate([frame([(box(0.7, 0), 0.9)], [box(0, 0)])])
    assert r.ap[("dist", 1.0, "vehicle", None)] == 1.0 and r.ap[("dist", 0.5, "vehicle", None)] == 0.0
    # wrong size and heading, right centre: TP by distance, FP by IoU 0.7
    odd = evaluate([frame([(box(0.1, 0, size=[2.0, 1.0, 1.0], heading=1.2), 0.9)], [box(0, 0)])])
    assert odd.ap[("dist", 0.5, "vehicle", None)] == 1.0 and odd.ap[("bev", 0.7, "vehicle", None)] == 0.0


def test_center_distance_nearest_wins_and_reaches_beyond_iou_prefilter():
    # Two pedestrians 0.9 m apart; each detection sits 0.8 m from "its" GT and 1.7 m / 0.1 m from the other.
    ped = [0.5, 0.5, 1.7]
    gts = [box(0, 0, size=ped), box(0.9, 0, size=ped)]
    dets = [(box(-0.8, 0, size=ped), 0.9), (box(1.7, 0, size=ped), 0.8)]  # 0.8 m from GT 0 and GT 1 respectively
    r = evaluate([frame(dets, gts, det_labels=[1, 1], gt_labels=[1, 1])])
    # circumradii ~0.35 each: the IoU prefilter alone would never pair objects 0.8 m apart
    assert r.ap[("dist", 1.0, "pedestrian", None)] == 1.0
    assert r.ap[("bev", 0.5, "pedestrian", None)] == 0.0


def test_distance_bins():
    gts = [box(10, 0), box(40, 0), box(60, 0)]
    dets = [(box(10, 0), 0.9), (box(40.1, 0), 0.8), (box(65, 5), 0.7)]  # third is an FP at ~65 m
    r = evaluate([frame(dets, gts)])
    a = lambda b: r.ap[("bev", 0.5, "vehicle", b)]  # noqa: E731
    assert a((0.0, 30.0)) == 1.0 and a((30.0, 50.0)) == 1.0 and a((50.0, 75.0)) == 0.0
    # bin GT counts add up to the total
    g = np.hypot(np.array(gts)[:, 0], np.array(gts)[:, 1])
    assert sum(((g >= lo) & (g < hi)).sum() for lo, hi in r.bins) == r.counts["vehicle"]["num_gt"]


def test_flat_and_table():
    r = evaluate([frame([(box(0, 0), 0.9)], [box(0, 0)])])
    f = r.flat()
    for name in ("eval/bev_ap@0.7/vehicle", "eval/3d_ap@0.5/mean", "eval/dist_ap@1m/pedestrian",
                 "eval/bev_ap@0.7/vehicle/30-50m", "eval/num_gt/vehicle", "eval/tp_heading_rad/vehicle"):
        assert name in f
    assert f["eval/bev_ap@0.7/vehicle"] == 1.0 and math.isnan(f["eval/bev_ap@0.7/cyclist"])
    assert f["eval/bev_ap@0.7/mean"] == 1.0  # NaN classes (no GT) are excluded from the mean
    t = r.table()
    assert t.count("\n") == 5 and "bev AP@0.7" in t and "dist AP@1m" in t


def test_distance_ap_off_when_empty():
    r = evaluate([frame([(box(0, 0), 0.9)], [box(0, 0)])], load_config(overrides=["eval.center_distances=[]"]))
    assert "dist" not in r.thresholds and not any("dist" in k for k in r.flat())


# --- real data -----------------------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("nms", ["none", "bev"])
def test_oracle_scores_one(nms):
    cfg = load_config(overrides=[f"decode.nms={nms}"])
    tb = TargetBuilder(cfg)
    ts = WaymoDataset(ROOT).segment("training", SEG).timestamps
    samples = [load_sample(ROOT, "training", SEG, int(t), cfg) for t in ts[[0, 60, 120, 180]]]
    dets = decode(perfect_outputs(collate([tb(s) for s in samples]), cfg), cfg)
    ev = Evaluator(cfg)
    ev.add(dets, samples)
    r = ev.compute()
    for fam, thrs in r.thresholds.items():
        for t in thrs:
            for c in cfg.data.classes:
                v = r.ap[(fam, t, c, None)]
                assert math.isnan(v) or v == pytest.approx(1.0), (fam, t, c, v)
    assert r.counts["vehicle"]["num_gt"] > 0 and r.counts["pedestrian"]["num_gt"] > 0


@needs_data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_overfit_model_ap():
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
    ev = Evaluator(cfg)
    ev.add(dets, samples)
    r = ev.compute()
    assert r.ap[("bev", 0.5, "vehicle", None)] > 0.9 and r.ap[("bev", 0.5, "pedestrian", None)] > 0.9
