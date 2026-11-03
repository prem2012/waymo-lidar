"""Detection evaluation: AP per class for BEV IoU, 3D IoU and centre-distance matching, by distance bin.

    ev = Evaluator(cfg)
    for dets, samples in ...:            # decoded detections + the matching eval-mode samples
        ev.add(dets, samples)
    res = ev.compute()
    print(res.table()); wandb.log(res.flat())

Rules (docs/detector/eval.md): GT with < eval.min_points top-LiDAR points or a centre outside the grid is
ignored (detections on it are dropped, neither TP nor FP); greedy matching by score; all-point AP.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import DetectorConfig, load_config
from .iou import overlapping_pairs

FAMILIES = ("bev", "3d", "dist")  # BEV IoU, 3D IoU, centre distance


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def average_precision(tp: np.ndarray, scores: np.ndarray, num_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    """All-point interpolated AP (VOC 2010+): area under the precision envelope. Returns (ap, precision, recall)."""
    if num_gt == 0:
        return float("nan"), np.zeros(0), np.zeros(0)
    if len(tp) == 0:
        return 0.0, np.zeros(0), np.zeros(0)
    order = np.argsort(-scores, kind="stable")
    t = tp[order].astype(np.float64)
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    recall = ctp / num_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    env = np.maximum.accumulate(precision[::-1])[::-1]  # monotone non-increasing envelope
    r = np.concatenate([[0.0], recall])
    return float(np.sum((r[1:] - r[:-1]) * env)), precision, recall


def greedy_match(rank: np.ndarray, pair_det: np.ndarray, pair_gt: np.ndarray, pair_aff: np.ndarray,
                 n_det: int, n_gt: int) -> np.ndarray:
    """Greedy one-to-one matching: in score order, each detection takes its best available GT.

    rank: (n_det,) unique ints, 0 = highest score. Pairs are (det, gt, affinity), already filtered by the
    threshold; higher affinity is better. Returns matched GT per detection (-1 = none).

    Vectorised in rounds, identical to the sequential loop: a detection may claim its best available GT only
    if it outranks every other unresolved detection that has that GT as a candidate. Each round resolves at
    least the highest-ranked unresolved detection, and usually almost all of them.
    """
    matched = np.full(n_det, -1, dtype=np.int64)
    if len(pair_det) == 0:
        return matched
    # Sort pairs once by (det, affinity desc, gt) so "best available" is the first active pair per det.
    order = np.lexsort((pair_gt, -pair_aff, pair_det))
    pd, pg = pair_det[order], pair_gt[order]
    resolved = np.zeros(n_det, dtype=bool)
    taken = np.zeros(n_gt, dtype=bool)
    big = np.iinfo(np.int64).max
    while True:
        active = ~resolved[pd] & ~taken[pg]
        if not active.any():
            break
        ad, ag = pd[active], pg[active]
        first = np.ones(len(ad), dtype=bool)
        first[1:] = ad[1:] != ad[:-1]
        best_det, best_gt = ad[first], ag[first]          # each unresolved det with options: its best GT
        top = np.full(n_gt, big, dtype=np.int64)
        np.minimum.at(top, ag, rank[ad])                  # best rank among unresolved dets wanting each GT
        win = rank[best_det] == top[best_gt]
        matched[best_det[win]] = best_gt[win]
        resolved[best_det[win]] = True
        taken[best_gt[win]] = True
    return matched


@dataclass
class EvalResult:
    classes: tuple[str, ...]
    thresholds: dict[str, tuple[float, ...]]          # family -> thresholds
    bins: tuple[tuple[float, float], ...]
    ap: dict = field(default_factory=dict)            # (family, thr, class, bin|None) -> AP
    counts: dict = field(default_factory=dict)        # class -> {"num_gt", "num_det", "num_ignored"}
    errors: dict = field(default_factory=dict)        # class -> {"center_m", "size_rel", "heading_rad", "num_tp"}
    pr_curves: dict = field(default_factory=dict)     # (family, thr, class) -> (precision, recall)
    max_recall: dict = field(default_factory=dict)    # (family, thr, class) -> recall at the lowest score

    @staticmethod
    def label(family: str, thr: float) -> str:
        return f"{family}_ap@{thr:g}m" if family == "dist" else f"{family}_ap@{thr:g}"

    def mean(self, family: str, thr: float, b=None) -> float:
        vals = [self.ap[(family, thr, c, b)] for c in self.classes]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def flat(self, prefix: str = "eval") -> dict[str, float]:
        out = {}
        for fam, thrs in self.thresholds.items():
            for t in thrs:
                name = f"{prefix}/{self.label(fam, t)}"
                for c in self.classes:
                    out[f"{name}/{c}"] = self.ap[(fam, t, c, None)]
                    for b in self.bins:
                        out[f"{name}/{c}/{b[0]:g}-{b[1]:g}m"] = self.ap[(fam, t, c, b)]
                out[f"{name}/mean"] = self.mean(fam, t)
        for c in self.classes:
            for k, v in self.counts[c].items():
                out[f"{prefix}/{k}/{c}"] = float(v)
            for k, v in self.errors[c].items():
                out[f"{prefix}/tp_{k}/{c}"] = float(v)
        return out

    def table(self) -> str:
        cols = [(f, t) for f, ts in self.thresholds.items() for t in ts]
        head = "| class | " + " | ".join(self.label(f, t).replace("_ap", " AP") for f, t in cols) + " |"
        rows = [head, "|---|" + "---:|" * len(cols)]
        fmt = lambda v: "–" if math.isnan(v) else f"{v:.3f}"  # noqa: E731
        for c in self.classes:
            rows.append(f"| {c} | " + " | ".join(fmt(self.ap[(f, t, c, None)]) for f, t in cols) + " |")
        rows.append("| **mean** | " + " | ".join(f"**{fmt(self.mean(f, t))}**" for f, t in cols) + " |")
        return "\n".join(rows)


class Evaluator:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.classes = tuple(cfg.data.classes)
        self.frames: list[dict] = []

    def reset(self) -> None:
        self.frames.clear()

    def add(self, dets, samples) -> None:
        """dets: list of Detections (or dicts with boxes/scores/labels); samples: matching eval-mode samples."""
        for d, s in zip(dets, samples, strict=True):
            d = d.numpy() if hasattr(d, "numpy") and not isinstance(d, dict) else d
            self.frames.append({
                "det_boxes": np.asarray(d["boxes"], np.float32).reshape(-1, 7),
                "det_scores": np.asarray(d["scores"], np.float32).reshape(-1),
                "det_labels": np.asarray(d["labels"], np.int64).reshape(-1),
                "gt_boxes": np.asarray(s["boxes"], np.float32).reshape(-1, 7),
                "gt_labels": np.asarray(s["labels"], np.int64).reshape(-1),
                "gt_points": np.asarray(s["num_points"], np.int64).reshape(-1),
            })

    def compute(self, device: str | None = None) -> EvalResult:
        cfg, ec, g = self.cfg, self.cfg.eval, self.cfg.grid
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        nc = len(self.classes)
        cat = lambda k: np.concatenate([f[k] for f in self.frames]) if self.frames else None  # noqa: E731
        db, ds, dl = cat("det_boxes"), cat("det_scores"), cat("det_labels")
        gb, gl, gp = cat("gt_boxes"), cat("gt_labels"), cat("gt_points")
        if db is None:
            db, ds, dl = np.zeros((0, 7), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
            gb, gl, gp = np.zeros((0, 7), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
        dframe = np.repeat(np.arange(len(self.frames)), [len(f["det_scores"]) for f in self.frames])
        gframe = np.repeat(np.arange(len(self.frames)), [len(f["gt_labels"]) for f in self.frames])

        inside = ((gb[:, 0] >= g.x_range[0]) & (gb[:, 0] < g.x_range[1]) &
                  (gb[:, 1] >= g.y_range[0]) & (gb[:, 1] < g.y_range[1]))
        ignored = (gp < ec.min_points) | ~inside
        g_range, d_range = np.hypot(gb[:, 0], gb[:, 1]), np.hypot(db[:, 0], db[:, 1])
        bins = tuple(zip(ec.distance_bins[:-1], ec.distance_bins[1:]))

        # One grouped candidate search for every family: group = (frame, class); reach covers centre distance.
        reach = max(ec.center_distances, default=0.0)
        dev_t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
        ia, ib, bev, iou3 = overlapping_pairs(dev_t(db), dev_t(gb), dev_t(dframe * nc + dl), dev_t(gframe * nc + gl),
                                              min_reach=reach)
        ia, ib, bev, iou3 = (x.cpu().numpy() for x in (ia, ib, bev, iou3))
        dist = np.hypot(db[ia, 0] - gb[ib, 0], db[ia, 1] - gb[ib, 1])
        rank = np.empty(len(ds), np.int64)
        rank[np.lexsort((np.arange(len(ds)), -ds))] = np.arange(len(ds))  # 0 = highest score, ties by index

        thresholds = {"bev": tuple(ec.iou_thresholds), "3d": tuple(ec.iou_thresholds)}
        if ec.center_distances:
            thresholds["dist"] = tuple(ec.center_distances)
        res = EvalResult(classes=self.classes, thresholds=thresholds, bins=bins)

        for fam, thrs in thresholds.items():
            for t in thrs:
                if fam == "dist":
                    ok, aff = dist <= t, -dist
                else:
                    val = bev if fam == "bev" else iou3
                    ok, aff = val >= t, val
                valid_pair = ok & ~ignored[ib]
                m = greedy_match(rank, ia[valid_pair], ib[valid_pair], aff[valid_pair], len(ds), len(gb))
                hits_ignored = np.zeros(len(ds), bool)
                hits_ignored[ia[ok & ignored[ib]]] = True
                tp = m >= 0
                counted = tp | ~hits_ignored          # dropped: unmatched but on an ignored GT
                for ci, c in enumerate(self.classes):
                    dc = (dl == ci) & counted
                    gt_valid = (gl == ci) & ~ignored
                    ap, prec, rec = average_precision(tp[dc], ds[dc], int(gt_valid.sum()))
                    res.ap[(fam, t, c, None)] = ap
                    res.pr_curves[(fam, t, c)] = (prec, rec)
                    res.max_recall[(fam, t, c)] = float(rec[-1]) if len(rec) else 0.0
                    for b in bins:
                        # TP -> bin of its GT; FP -> bin of the detection; out-of-bin GT is ignored
                        in_bin_det = np.where(tp, (g_range[np.maximum(m, 0)] >= b[0]) & (g_range[np.maximum(m, 0)] < b[1]),
                                              (d_range >= b[0]) & (d_range < b[1]))
                        sel = dc & in_bin_det
                        n_gt_bin = int((gt_valid & (g_range >= b[0]) & (g_range < b[1])).sum())
                        res.ap[(fam, t, c, b)] = average_precision(tp[sel], ds[sel], n_gt_bin)[0]
                if fam == "bev" and t == min(thrs):  # TP error statistics at the loosest BEV threshold
                    for ci, c in enumerate(self.classes):
                        sel = tp & (dl == ci)
                        d_, g_ = db[sel].astype(np.float64), gb[m[sel]].astype(np.float64)
                        res.errors[c] = {
                            "center_m": float(np.hypot(*(d_[:, :2] - g_[:, :2]).T).mean()) if len(d_) else float("nan"),
                            "size_rel": float(np.abs(d_[:, 3:6] / g_[:, 3:6] - 1).mean()) if len(d_) else float("nan"),
                            "heading_rad": float(np.abs(wrap_pi(d_[:, 6] - g_[:, 6])).mean()) if len(d_) else float("nan"),
                            "num_tp": int(sel.sum()),
                        }
        for ci, c in enumerate(self.classes):
            res.counts[c] = {"num_gt": int(((gl == ci) & ~ignored).sum()), "num_ignored": int(((gl == ci) & ignored).sum()),
                             "num_det": int((dl == ci).sum())}
        return res


# --- benchmark --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    from .iou import jitter, random_boxes
    p = argparse.ArgumentParser(description="Benchmark Evaluator.compute on a synthetic validation pass")
    p.add_argument("--frames", type=int, default=8000)
    p.add_argument("--gt", type=int, default=45)
    p.add_argument("--dets", type=int, default=300)
    a = p.parse_args(argv)
    cfg = load_config()
    rng = np.random.default_rng(0)
    ev = Evaluator(cfg)
    for _ in range(a.frames):
        gt = random_boxes(a.gt, rng)
        gl = rng.integers(0, 3, a.gt)
        near = jitter(gt, rng, pos=0.3, size=0.1, ang=0.1)
        far = random_boxes(a.dets - a.gt, rng)
        boxes = np.concatenate([near, far])
        labels = np.concatenate([gl, rng.integers(0, 3, a.dets - a.gt)])
        scores = np.concatenate([rng.uniform(0.4, 1.0, a.gt), rng.uniform(0.1, 0.6, a.dets - a.gt)]).astype(np.float32)
        ev.add([{"boxes": boxes, "scores": scores, "labels": labels}],
               [{"boxes": gt, "labels": gl, "num_points": rng.integers(0, 200, a.gt)}])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t = time.perf_counter()
    res = ev.compute()
    dt = time.perf_counter() - t
    n_metrics = sum(len(v) for v in res.thresholds.values())
    print(f"compute(): {dt:.1f} s for {a.frames} frames x ({a.dets} det, {a.gt} GT), {n_metrics} thresholds x "
          f"{len(res.classes)} classes x (all + {len(res.bins)} bins)   [target < 30 s]")
    print(res.table())


if __name__ == "__main__":
    main()
