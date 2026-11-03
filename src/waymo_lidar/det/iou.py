"""Rotated BEV IoU, 3D IoU and rotated NMS for upright boxes [x, y, z, l, w, h, heading], on the GPU.

    bev, iou3d = ious(dets, gts)               # dense (N, M), one intersection pass for both
    ia, ib, bev, iou3d = overlapping_pairs(dets, gts, det_frame, gt_frame)   # sparse, many frames at once
    keep = nms_bev(boxes, scores, 0.7)

The intersection of two rotated rectangles is exact: its vertices are among the 24 candidates
(4 corners of A inside B, 4 corners of B inside A, 16 edge crossings), sorted by angle and measured
with the shoelace formula, for all pairs at once. A circumscribed-circle prefilter skips pairs that
can't overlap. See docs/detector/iou.md.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

PAIR_CHUNK = 262_144  # pairs per exact-intersection chunk (bounds memory: ~0.3 KB per pair)


def _as_tensor(x, device=None) -> tuple[torch.Tensor, bool]:
    if isinstance(x, torch.Tensor):
        return (x if device is None else x.to(device)), False
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    return (t if device is None else t.to(device)), True


def box_corners_2d(boxes: torch.Tensor) -> torch.Tensor:
    """(N, 7) -> (N, 4, 2) footprint corners, counter-clockwise from front-left (as data.boxes.box_corners)."""
    signs = boxes.new_tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    local = signs[None] * (boxes[:, None, 3:5] / 2)
    c, s = torch.cos(boxes[:, 6])[:, None], torch.sin(boxes[:, 6])[:, None]
    x = local[..., 0] * c - local[..., 1] * s
    y = local[..., 0] * s + local[..., 1] * c
    return torch.stack([x, y], -1) + boxes[:, None, :2]


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside(points: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """(P, K, 2) points vs (P, 7) boxes (same pair) -> (P, K) bool, boundary inclusive."""
    d = points - boxes[:, None, :2]
    c, s = torch.cos(boxes[:, 6])[:, None], torch.sin(boxes[:, 6])[:, None]
    u = d[..., 0] * c + d[..., 1] * s
    v = -d[..., 0] * s + d[..., 1] * c
    return (u.abs() <= boxes[:, None, 3] / 2 + eps) & (v.abs() <= boxes[:, None, 4] / 2 + eps)


def intersection_area(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Footprint intersection area of pairs a[i], b[i]: (P, 7), (P, 7) -> (P,)."""
    if len(a) == 0:
        return a.new_zeros(0)
    # Re-centre each pair on box A so float32 keeps sub-mm precision anywhere in the grid.
    a, b = a.clone(), b.clone()
    origin = a[:, :2].clone()
    a[:, :2] -= origin
    b[:, :2] -= origin
    ca, cb = box_corners_2d(a), box_corners_2d(b)                       # (P, 4, 2)

    a_in_b, b_in_a = _inside(ca, b), _inside(cb, a)                      # (P, 4)
    # edge crossings: A edge i (ca[i] -> ca[i+1]) with B edge j
    a0, a1 = ca[:, :, None], ca.roll(-1, dims=1)[:, :, None]            # (P, 4, 1, 2)
    b0, b1 = cb[:, None], cb.roll(-1, dims=1)[:, None]                  # (P, 1, 4, 2)
    da, db = a1 - a0, b1 - b0
    denom = _cross(da, db)                                               # (P, 4, 4) = |da||db| sin(angle)
    # Parallel test relative to the edge lengths: in float32, parallel edges leave |denom| ~ 1e-7 of noise,
    # and noise/noise "crossings" would land anywhere on the edges (measured: IoU 0.14 for two boxes that only
    # share an edge). Parallel overlaps are still covered: their endpoints show up as corners inside the other box.
    parallel = denom.abs() <= 1e-5 * torch.linalg.norm(da, dim=-1) * torch.linalg.norm(db, dim=-1)
    safe = torch.where(parallel, torch.ones_like(denom), denom)
    t = _cross(b0 - a0, db) / safe
    u = _cross(b0 - a0, da) / safe
    cross_ok = ~parallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    crossings = (a0 + t[..., None] * da).reshape(-1, 16, 2)

    pts = torch.cat([ca, cb, crossings], dim=1)                          # (P, 24, 2)
    valid = torch.cat([a_in_b, b_in_a, cross_ok.reshape(-1, 16)], dim=1)  # (P, 24)
    n = valid.sum(1)
    vf = valid.float()[..., None]
    centre = (pts * vf).sum(1) / n.clamp_min(1)[:, None]
    rel = pts - centre[:, None]
    ang = torch.atan2(rel[..., 1], rel[..., 0]).masked_fill(~valid, 10.0)  # invalid sort last
    order = ang.argsort(dim=1)
    sp = torch.gather(rel, 1, order[..., None].expand(-1, -1, 2))
    sv = torch.gather(valid, 1, order)
    sp = torch.where(sv[..., None], sp, sp[:, :1].expand_as(sp))       # invalid -> first point: zero area
    area = 0.5 * _cross(sp, sp.roll(-1, dims=1)).sum(1).abs()
    return torch.where(n >= 3, area, torch.zeros_like(area))


def _candidates(a: torch.Tensor, b: torch.Tensor, a_group: torch.Tensor | None, b_group: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pairs (ia, ib) whose circumscribed circles overlap (and, with groups, share a group)."""
    ra, rb = torch.linalg.norm(a[:, 3:5], dim=1) / 2, torch.linalg.norm(b[:, 3:5], dim=1) / 2
    if a_group is None:
        close = torch.cdist(a[:, :2].double(), b[:, :2].double()) < (ra[:, None] + rb[None]).double() + 1e-6
        return torch.nonzero(close, as_tuple=True)
    # Grouped (e.g. many frames): enumerate pairs only within each group, then distance-filter in chunks.
    ga, oa = torch.sort(a_group, stable=True)
    gb, ob = torch.sort(b_group, stable=True)
    groups = torch.unique(torch.cat([ga, gb]))
    na = torch.searchsorted(ga, groups, right=True) - torch.searchsorted(ga, groups)
    nb = torch.searchsorted(gb, groups, right=True) - torch.searchsorted(gb, groups)
    sa, sb = torch.searchsorted(ga, groups), torch.searchsorted(gb, groups)
    out_a, out_b = [], []
    for g in range(len(groups)):  # per group: small dense block (dets x gts of one frame)
        if na[g] == 0 or nb[g] == 0:
            continue
        ia = oa[sa[g]:sa[g] + na[g]]
        ib = ob[sb[g]:sb[g] + nb[g]]
        close = torch.cdist(a[ia, :2].double(), b[ib, :2].double()) < (ra[ia, None] + rb[None, ib]).double() + 1e-6
        i, j = torch.nonzero(close, as_tuple=True)
        out_a.append(ia[i])
        out_b.append(ib[j])
    if not out_a:
        e = torch.zeros(0, dtype=torch.long, device=a.device)
        return e, e
    return torch.cat(out_a), torch.cat(out_b)


def overlapping_pairs(a, b, a_group=None, b_group=None):
    """Sparse IoUs of every pair that can overlap: (ia, ib, bev_iou, iou_3d). Groups keep frames apart."""
    a, np_in = _as_tensor(a)
    b, _ = _as_tensor(b, a.device)
    a, b = a.float(), b.float()
    if a_group is not None:
        a_group, _ = _as_tensor(a_group, a.device)
        b_group, _ = _as_tensor(b_group, a.device)
        a_group, b_group = a_group.long(), b_group.long()
    ia, ib = _candidates(a, b, a_group, b_group)
    inter = torch.cat([intersection_area(a[ia[k:k + PAIR_CHUNK]], b[ib[k:k + PAIR_CHUNK]])
                       for k in range(0, len(ia), PAIR_CHUNK)]) if len(ia) else a.new_zeros(0)
    area_a, area_b = a[ia, 3] * a[ia, 4], b[ib, 3] * b[ib, 4]
    bev = (inter / (area_a + area_b - inter).clamp_min(1e-12)).clamp(0, 1)
    za0, za1 = a[ia, 2] - a[ia, 5] / 2, a[ia, 2] + a[ia, 5] / 2
    zb0, zb1 = b[ib, 2] - b[ib, 5] / 2, b[ib, 2] + b[ib, 5] / 2
    inter3 = inter * (torch.minimum(za1, zb1) - torch.maximum(za0, zb0)).clamp_min(0)
    vol_a, vol_b = area_a * a[ia, 5], area_b * b[ib, 5]
    iou3 = (inter3 / (vol_a + vol_b - inter3).clamp_min(1e-12)).clamp(0, 1)
    if np_in:
        return ia.cpu().numpy(), ib.cpu().numpy(), bev.cpu().numpy(), iou3.cpu().numpy()
    return ia, ib, bev, iou3


def ious(a, b):
    """Dense (N, M) BEV and 3D IoU (one intersection pass). NumPy in -> NumPy out."""
    at, np_in = _as_tensor(a)
    bt, _ = _as_tensor(b, at.device)
    ia, ib, bev_v, iou3_v = overlapping_pairs(at, bt)
    bev = at.new_zeros((len(at), len(bt)))
    iou3 = at.new_zeros((len(at), len(bt)))
    bev[ia, ib], iou3[ia, ib] = bev_v, iou3_v
    return (bev.cpu().numpy(), iou3.cpu().numpy()) if np_in else (bev, iou3)


def bev_iou(a, b):
    return ious(a, b)[0]


def iou_3d(a, b):
    return ious(a, b)[1]


def nms_bev(boxes, scores, iou_threshold: float):
    """Greedy rotated NMS on BEV IoU. Returns kept indices in descending score order (same type as input)."""
    bt, np_in = _as_tensor(boxes)
    st, _ = _as_tensor(scores, bt.device)
    if len(bt) == 0:
        keep = torch.zeros(0, dtype=torch.long, device=bt.device)
        return keep.cpu().numpy() if np_in else keep
    order = torch.argsort(st, descending=True)
    sb = bt[order].float()
    ia, ib, bev, _ = overlapping_pairs(sb, sb)
    over = torch.zeros((len(sb), len(sb)), dtype=torch.bool, device=bt.device)
    hit = (bev > iou_threshold) & (ia != ib)
    over[ia[hit], ib[hit]] = True
    over_np = over.cpu().numpy()  # one small transfer; the greedy pass is sequential by nature
    removed = np.zeros(len(sb), dtype=bool)
    kept = []
    for i in range(len(sb)):
        if removed[i]:
            continue
        kept.append(i)
        removed |= over_np[i]
    keep = order[torch.as_tensor(kept, device=bt.device)]
    return keep.cpu().numpy() if np_in else keep


GROUP_SPACING = 400.0  # m: wider than the grid (150.4 m) plus any box, so shifted groups can't overlap


def nms_bev_grouped(boxes, scores, groups, thresholds):
    """Greedy rotated NMS for many independent groups (e.g. frame x class) in one pass.

    boxes (N, 7), scores (N,), groups (N,) int, thresholds (N,) per box (its group's IoU threshold).
    Groups are shifted 400 m apart along x so a single candidate search can't pair boxes across groups.
    The greedy pass walks a sparse adjacency (only overlapping pairs), not a dense N x N matrix.
    Returns kept indices sorted by descending score.
    """
    bt, np_in = _as_tensor(boxes)
    dev = bt.device
    st, _ = _as_tensor(scores, dev)
    gt = torch.as_tensor(groups, device=dev).long()
    tt, _ = _as_tensor(thresholds, dev)
    if len(bt) == 0:
        keep = torch.zeros(0, dtype=torch.long, device=dev)
        return keep.cpu().numpy() if np_in else keep
    order = torch.argsort(st, descending=True)
    sb = bt[order].float().clone()
    sg = gt[order]
    sb[:, 0] += (sg - sg.min()).float() * GROUP_SPACING
    ia, ib, bev, _ = overlapping_pairs(sb, sb)
    hit = (ia < ib) & (bev > tt[order][ia])          # i outranks j (sorted order) and overlaps above i's threshold
    ia, ib = ia[hit].cpu().numpy(), ib[hit].cpu().numpy()
    n = len(sb)
    srt = np.argsort(ia, kind="stable")
    ia, ib = ia[srt], ib[srt]
    indptr = np.searchsorted(ia, np.arange(n + 1))
    removed = np.zeros(n, dtype=bool)
    kept = []
    for i in range(n):
        if removed[i]:
            continue
        kept.append(i)
        removed[ib[indptr[i]:indptr[i + 1]]] = True
    keep = order[torch.as_tensor(kept, device=dev, dtype=torch.long)]
    return keep.cpu().numpy() if np_in else keep


# --- benchmark --------------------------------------------------------------------------------------


def random_boxes(n: int, rng: np.random.Generator, spread: float = 75.0) -> np.ndarray:
    """Vehicle-ish and pedestrian-ish boxes scattered over the grid."""
    car = rng.random(n) < 0.6
    l = np.where(car, rng.uniform(3.5, 6.0, n), rng.uniform(0.5, 1.0, n))
    w = np.where(car, rng.uniform(1.6, 2.3, n), rng.uniform(0.5, 1.0, n))
    h = np.where(car, rng.uniform(1.4, 2.2, n), rng.uniform(1.5, 1.9, n))
    return np.stack([rng.uniform(-spread, spread, n), rng.uniform(-spread, spread, n), rng.uniform(0.5, 1.5, n),
                     l, w, h, rng.uniform(-np.pi, np.pi, n)], 1).astype(np.float32)


def jitter(boxes: np.ndarray, rng: np.random.Generator, pos: float = 0.3, size: float = 0.1, ang: float = 0.1) -> np.ndarray:
    out = boxes.copy()
    out[:, :2] += rng.normal(0, pos, (len(out), 2))
    out[:, 3:6] *= 1 + rng.normal(0, size, (len(out), 3)).clip(-0.5, 0.5)
    out[:, 6] += rng.normal(0, ang, len(out))
    return out


def _timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark rotated IoU / NMS on the GPU (and shapely for reference)")
    p.add_argument("--frames", type=int, default=8000, help="frames in the validation-pass case")
    a = p.parse_args(argv)
    rng = np.random.default_rng(0)
    dev = "cuda"

    # 1) one frame: 240 GT, 500 detections = 240 near-GT (jittered) + 260 random
    gt = random_boxes(240, rng)
    det = np.concatenate([jitter(gt, rng), random_boxes(260, rng)])
    g, d = torch.from_numpy(gt).to(dev), torch.from_numpy(det).to(dev)
    ms = _timed(lambda: ious(d, g))
    n_pairs = len(overlapping_pairs(d, g)[0])
    print(f"ious 500 det x 240 GT ({n_pairs} candidate pairs): {ms:.2f} ms   [target < 2 ms]")

    # 2) raw exact intersection throughput
    base = torch.from_numpy(random_boxes(100_000, rng, spread=5)).to(dev)
    other = torch.from_numpy(jitter(base.cpu().numpy(), rng, pos=1.0)).to(dev)
    ms = _timed(lambda: intersection_area(base, other))
    print(f"intersection_area, 100k pairs: {ms:.2f} ms   [target < 5 ms]")

    # 3) decode + NMS worst case: batch 4, 500 detections per frame
    from .config import load_config
    from .decode import decode
    cfg = load_config(overrides=["decode.nms=bev"])
    out = {"heatmap": torch.randn(4, 3, 376, 376, device=dev) - 2.5, "reg": torch.randn(4, 8, 376, 376, device=dev) * 0.3}
    n_det = sum(len(x) for x in decode(out, cfg))
    ms_nms = _timed(lambda: decode(out, cfg), reps=10)
    ms_plain = _timed(lambda: decode(out, load_config()), reps=10)
    print(f"decode batch 4 ({n_det} detections): {ms_plain:.2f} ms without NMS, {ms_nms:.2f} ms with nms: bev   [target < 10 ms]")

    # 4) a whole validation pass, batched: per frame ~45 GT, 300 detections (one class pass, all frames at once)
    nf = a.frames
    gts = [random_boxes(45, rng) for _ in range(nf)]
    dets = [np.concatenate([jitter(x, rng), random_boxes(255, rng)]) for x in gts]
    G = torch.from_numpy(np.concatenate(gts)).to(dev)
    D = torch.from_numpy(np.concatenate(dets)).to(dev)
    gg = torch.repeat_interleave(torch.arange(nf, device=dev), 45)
    dg = torch.repeat_interleave(torch.arange(nf, device=dev), 300)
    torch.cuda.synchronize()
    t = time.perf_counter()
    res = overlapping_pairs(D, G, dg, gg)
    torch.cuda.synchronize()
    print(f"validation pass, {nf} frames x (300 det, 45 GT), batched: {time.perf_counter() - t:.2f} s, "
          f"{len(res[0]):,} candidate pairs   [target well under 60 s]")

    # 5) shapely reference on case 1, for comparison
    try:
        import shapely
        from shapely.geometry import Polygon
    except ImportError:
        print("shapely not installed (dev dependency): skipping the comparison")
        return
    cd, cg = box_corners_2d(d).cpu().numpy(), box_corners_2d(g).cpu().numpy()
    pd, pg = shapely.polygons(cd), shapely.polygons(cg)
    ia, ib = (x.cpu().numpy() for x in overlapping_pairs(d, g)[:2])
    t = time.perf_counter()
    for _ in range(5):
        inter = shapely.area(shapely.intersection(pd[ia], pg[ib]))
    ms_shp = (time.perf_counter() - t) / 5 * 1000
    t = time.perf_counter()
    allp = shapely.area(shapely.intersection(pd[:, None], pg[None]))  # no prefilter, all 120k pairs
    ms_all = (time.perf_counter() - t) * 1000
    print(f"shapely, same {len(ia)} candidate pairs: {ms_shp:.2f} ms; all 120,000 pairs without prefilter: {ms_all:.0f} ms")
    del inter, allp, Polygon


if __name__ == "__main__":
    main()
