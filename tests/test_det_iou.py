"""Rotated IoU / NMS tests. shapely (dev dependency) is the exact reference."""

import math

import numpy as np
import pytest
import shapely
import torch

from waymo_lidar.data.boxes import points_in_boxes
from waymo_lidar.det.iou import (bev_iou, box_corners_2d, intersection_area, iou_3d, ious, jitter, nms_bev,
                                 nms_bev_grouped, overlapping_pairs, random_boxes)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def B(*rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 7)


def shapely_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact float64 BEV IoU for matching rows a[i], b[i]."""
    pa = shapely.polygons(box_corners_2d(torch.from_numpy(a).double()).numpy())
    pb = shapely.polygons(box_corners_2d(torch.from_numpy(b).double()).numpy())
    inter = shapely.area(shapely.intersection(pa, pb))
    return inter / (shapely.area(pa) + shapely.area(pb) - inter)


# --- known values ------------------------------------------------------------------------------------


@pytest.mark.parametrize("a, b, bev, iou3", [
    (B([0, 0, 1, 4, 2, 1.5, 0]), B([0, 0, 1, 4, 2, 1.5, 0]), 1.0, 1.0),                 # identical
    (B([0, 0, 1, 4, 2, 1.5, 0]), B([10, 0, 1, 4, 2, 1.5, 0]), 0.0, 0.0),                # disjoint
    (B([0, 0, 1, 4, 2, 1.5, 0]), B([2, 0, 1, 4, 2, 1.5, 0]), 1 / 3, 1 / 3),             # half overlap
    (B([0, 0, 0, 1, 1, 1, 0]), B([0, 0, 0, 1, 1, 1, math.pi / 4]), 1 / math.sqrt(2), 1 / math.sqrt(2)),  # octagon
    (B([0, 0, 1, 4, 2, 2, 0]), B([0, 0, 2, 4, 2, 2, 0]), 1.0, 1 / 3),                   # z offset h/2
    (B([0, 0, 1, 4, 2, 2, 0.3]), B([0, 0, 1, 2, 1, 1, 0.3]), 0.25, 0.125),              # nested
    (B([0, 0, 1, 2, 2, 1, 0]), B([2, 0, 1, 2, 2, 1, 0]), 0.0, 0.0),                     # touching edges
    (B([0, 0, 1, 2, 2, 1, 0]), B([2, 2, 1, 2, 2, 1, 0]), 0.0, 0.0),                     # touching corners
])
@pytest.mark.parametrize("device", DEVICES)
def test_known_values(a, b, bev, iou3, device):
    got_bev, got_3d = ious(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device))
    assert got_bev.item() == pytest.approx(bev, abs=1e-6) and got_3d.item() == pytest.approx(iou3, abs=1e-6)


def test_far_from_origin_keeps_precision():
    a = B([74.9, -74.9, 1, 4.5, 1.9, 1.6, 1.1])
    b = a.copy()
    b[0, 0] += 0.01  # 1 cm shift
    expected = shapely_iou(a.astype(np.float64), b.astype(np.float64))[0]
    assert bev_iou(a, b)[0, 0] == pytest.approx(expected, abs=1e-5)


# --- against references ------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_matches_shapely_on_random_and_degenerate_pairs(device):
    rng = np.random.default_rng(0)
    a = random_boxes(8000, rng, spread=75)
    b = jitter(a, rng, pos=1.0, size=0.3, ang=0.6)
    # near-degenerate: shared edges, corners on edges, tiny boxes, axis-aligned duplicates
    sq = B(*[[0, 0, 0, 2, 2, 1, 0]] * 4)
    sq2 = B([2, 0, 0, 2, 2, 1, 0], [1, 1, 0, 2, 2, 1, 0], [0, 0, 0, 2, 2, 1, math.pi / 2], [1, 0, 0, 2, 2, 1, 0])
    tiny = B([5, 5, 0, 0.06, 0.08, 0.1, 0.3])
    tiny2 = B([5.01, 5.0, 0, 0.06, 0.08, 0.1, -0.2])
    edge_a = random_boxes(1000, rng, spread=75)
    edge_b = edge_a.copy()
    edge_b[:, 0] += edge_a[:, 3] * np.cos(edge_a[:, 6])  # shifted by exactly one length along heading: edge contact
    edge_b[:, 1] += edge_a[:, 3] * np.sin(edge_a[:, 6])
    A = np.concatenate([a, sq, tiny, edge_a])
    Bx = np.concatenate([b, sq2, tiny2, edge_b])
    got = intersection_area(torch.from_numpy(A).to(device), torch.from_numpy(Bx).to(device)).cpu().numpy()
    union = A[:, 3] * A[:, 4] + Bx[:, 3] * Bx[:, 4] - got
    ref = shapely_iou(A.astype(np.float64), Bx.astype(np.float64))
    assert np.abs(got / union - ref).max() < 1e-5


def test_matches_monte_carlo_independent_of_shapely():
    """Sample points over each pair's bounding area; IoU ~ |in A and in B| / |in A or in B|."""
    rng = np.random.default_rng(1)
    a = random_boxes(200, rng, spread=20)
    b = jitter(a, rng, pos=0.6, size=0.2, ang=0.4)
    got = np.diagonal(bev_iou(a, b))

    def member(pts, box):  # footprint membership, written independently of iou.py
        d = pts[:, :2] - box[:2]
        c, s = np.cos(box[6]), np.sin(box[6])
        return (np.abs(d[:, 0] * c + d[:, 1] * s) <= box[3] / 2) & (np.abs(-d[:, 0] * s + d[:, 1] * c) <= box[4] / 2)

    checked = 0
    for i in np.flatnonzero(got > 0.05):
        corners = box_corners_2d(torch.from_numpy(np.stack([a[i], b[i]]))).numpy().reshape(-1, 2)
        pts = np.c_[rng.uniform(corners.min(0), corners.max(0), (40_000, 2)), np.zeros(40_000)]
        ma, mb = member(pts, a[i]), member(pts, b[i])
        # cross-check the membership against data.boxes.points_in_boxes (tall box so z never excludes)
        box = a[i:i + 1].astype(np.float64)
        assert points_in_boxes(pts, np.c_[box[:, :2], [[0.0]]], np.c_[box[:, 3:5], [[10.0]]], box[:, 6])[0] == ma.sum()
        assert got[i] == pytest.approx((ma & mb).sum() / (ma | mb).sum(), abs=0.01)
        checked += 1
    assert checked > 100


def test_prefilter_equals_all_pairs():
    rng = np.random.default_rng(2)
    a, b = random_boxes(300, rng, spread=15), random_boxes(200, rng, spread=15)
    dense = bev_iou(a, b)
    ai, bi = np.meshgrid(np.arange(300), np.arange(200), indexing="ij")
    inter = intersection_area(torch.from_numpy(a[ai.ravel()]), torch.from_numpy(b[bi.ravel()])).numpy()
    union = a[ai.ravel(), 3] * a[ai.ravel(), 4] + b[bi.ravel(), 3] * b[bi.ravel(), 4] - inter
    assert np.allclose(dense, (inter / union).reshape(300, 200), atol=1e-6)
    assert (dense > 0).sum() > 50  # the test actually has overlaps


# --- invariances, shapes, types -----------------------------------------------------------------------


def test_invariances_and_symmetry():
    rng = np.random.default_rng(3)
    a = random_boxes(100, rng, spread=10)
    b = jitter(a, rng, pos=0.8, ang=0.5)
    base_bev, base_3d = ious(a, b)
    th = 0.9
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    ra, rb = a.copy(), b.copy()
    for x in (ra, rb):
        x[:, :2] = x[:, :2] @ R.T + [3.0, -7.0]
        x[:, 6] += th
    assert np.allclose(ious(ra, rb)[0], base_bev, atol=1e-5)
    flipped = b.copy()
    flipped[:, 6] += np.pi
    assert np.allclose(ious(a, flipped)[0], base_bev, atol=1e-5) and np.allclose(ious(a, flipped)[1], base_3d, atol=1e-5)
    assert np.allclose(ious(b, a)[0], base_bev.T, atol=1e-6)


def test_shapes_types_and_devices():
    empty = np.zeros((0, 7), np.float32)
    one = B([0, 0, 0, 1, 1, 1, 0])
    assert bev_iou(empty, one).shape == (0, 1) and iou_3d(one, empty).shape == (1, 0)
    assert isinstance(bev_iou(one, one), np.ndarray)
    t = bev_iou(torch.from_numpy(one), torch.from_numpy(one))
    assert isinstance(t, torch.Tensor)
    if torch.cuda.is_available():
        rng = np.random.default_rng(4)
        a, b = random_boxes(400, rng, 10), random_boxes(300, rng, 10)
        cpu = bev_iou(torch.from_numpy(a), torch.from_numpy(b))
        gpu = bev_iou(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()).cpu()
        assert torch.allclose(cpu, gpu, atol=1e-5)


def test_grouped_pairs_never_cross_groups():
    rng = np.random.default_rng(5)
    a = random_boxes(60, rng, spread=3)
    b = jitter(a, rng)
    ga = np.repeat(np.arange(3), 20)
    gb = np.repeat(np.arange(3), 20)[::-1].copy()  # reversed groups: overlapping boxes are in different groups now
    ia, ib, _, _ = overlapping_pairs(a, b, ga, gb)
    assert np.all(ga[ia] == gb[ib])
    ia2, ib2, bev2, _ = overlapping_pairs(a, b, ga, ga)
    dense = bev_iou(a, b)
    same = ga[:, None] == ga[None]
    assert np.allclose(np.sort(bev2[bev2 > 0]), np.sort(dense[same & (dense > 0)]), atol=1e-6)


# --- NMS -----------------------------------------------------------------------------------------------


def greedy_reference(boxes, scores, thr):
    order = np.argsort(-scores, kind="stable")
    keep = []
    for i in order:
        if all(shapely_iou(boxes[i:i + 1].astype(np.float64), boxes[j:j + 1].astype(np.float64))[0] <= thr for j in keep):
            keep.append(i)
    return np.array(keep)


def test_nms_basic():
    boxes = B([0, 0, 1, 4, 2, 1.5, 0], [0.2, 0, 1, 4, 2, 1.5, 0.05], [10, 0, 1, 4, 2, 1.5, 0], [2.5, 0, 1, 4, 2, 1.5, 0])
    scores = np.array([0.6, 0.9, 0.8, 0.5], np.float32)
    keep = nms_bev(boxes, scores, 0.5)
    assert keep.tolist() == [1, 2, 3]  # 0 overlaps 1 (higher) above 0.5; 3 overlaps 1 by < 0.5
    assert nms_bev(boxes, scores, 0.95).tolist() == [1, 2, 0, 3]  # nothing above 0.95: all kept, by score


def test_nms_matches_greedy_reference():
    rng = np.random.default_rng(6)
    base = random_boxes(40, rng, spread=12)
    boxes = np.concatenate([base, jitter(base, rng, pos=0.4), jitter(base, rng, pos=0.4)])
    scores = rng.random(len(boxes)).astype(np.float32)
    for thr in (0.3, 0.5, 0.7):
        assert nms_bev(boxes, scores, thr).tolist() == greedy_reference(boxes, scores, thr).tolist()


def test_grouped_nms_equals_per_group():
    rng = np.random.default_rng(7)
    base = random_boxes(30, rng, spread=8)
    boxes = np.concatenate([base, jitter(base, rng, pos=0.3)] * 2)
    scores = rng.random(len(boxes)).astype(np.float32)
    groups = np.repeat([0, 1], len(boxes) // 2)  # identical geometry in both groups: must not interact
    thr = np.where(groups == 0, 0.5, 0.3).astype(np.float32)
    keep = set(nms_bev_grouped(boxes, scores, groups, thr).tolist())
    for g, t in ((0, 0.5), (1, 0.3)):
        idx = np.flatnonzero(groups == g)
        assert keep & set(idx.tolist()) == set(idx[nms_bev(boxes[idx], scores[idx], t)].tolist())
