"""Run a checkpoint over one segment and view predictions next to ground truth in Rerun.

    uv run wl predict ~/dev/data/waymo_v2 --ckpt runs/baseline/checkpoints/best.pt --segment <name>
    uv run wl predict ~/dev/data/waymo_v2 --ckpt ... --segment <name> --save pred.rrd   # then: uv run rerun pred.rrd

Every frame is matched to its ground truth with the evaluator's rules (greedy by score, per class, BEV IoU
>= --iou; GT with too few points or outside the grid is ignored), so the viewer shows why AP is what it is:
predictions are true positives (green), false positives (red) or dropped on an ignored box (grey); ground
truth is matched (class colour), missed (yellow) or ignored (dim). See docs/detector/predict.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from ..data.dataset import WaymoDataset
from ..data.types import BOX_TYPE_COLORS
from ..viz.colors import intensity_colors, turbo
from .config import DetectorConfig, config_from_dict
from .data import FrameStream, collate, complete_segments
from .decode import decode
from .eval import Evaluator, greedy_match
from .iou import ious
from .targets import TargetBuilder
from .train import AMP, forward, load_model
from .viz_voxels import grid_outline
from .voxelize import Voxelizer

# Per-detection and per-GT match status.
TP, FP, DROPPED = 0, 1, 2
MATCHED, MISSED, IGNORED = 0, 1, 2
DET_STATUS = ("tp", "fp", "dropped")
GT_STATUS = ("matched", "missed", "ignored")
DET_COLORS = {TP: (60, 220, 90), FP: (235, 45, 45), DROPPED: (150, 150, 150)}
MISSED_COLOR, IGNORED_COLOR = (255, 210, 0), (90, 90, 90)


# --- matching ----------------------------------------------------------------------------------------


@dataclass
class FrameMatch:
    det_status: np.ndarray  # (N,) TP / FP / DROPPED
    det_iou: np.ndarray  # (N,) IoU with the matched GT (0 if unmatched)
    gt_status: np.ndarray  # (M,) MATCHED / MISSED / IGNORED
    gt_ignored: np.ndarray  # (M,) bool


def ignored_gt(sample: dict, cfg: DetectorConfig) -> np.ndarray:
    """The evaluator's don't-care rule: too few top-LiDAR points, or centre outside the grid."""
    b, g = sample["boxes"], cfg.grid
    inside = (b[:, 0] >= g.x_range[0]) & (b[:, 0] < g.x_range[1]) & (b[:, 1] >= g.y_range[0]) & (b[:, 1] < g.y_range[1])
    return (sample["num_points"] < cfg.eval.min_points) | ~inside


def match_frame(det: dict, sample: dict, cfg: DetectorConfig, iou_thr: float = 0.5, family: str = "bev") -> FrameMatch:
    """Match one frame's detections to its GT exactly as `Evaluator` does for one (family, threshold).

    Matching is greedy in score order, so dropping detections below a score cut doesn't change how the
    remaining ones match: filtering before or after gives the same statuses for the kept detections.
    """
    db, ds, dl = det["boxes"], det["scores"], det["labels"]
    gb, gl = sample["boxes"], sample["labels"]
    n, m = len(ds), len(gl)
    ign = ignored_gt(sample, cfg)
    det_status = np.full(n, FP, np.int64)
    det_iou = np.zeros(n, np.float32)
    gt_status = np.where(ign, IGNORED, MISSED)
    if n == 0 or m == 0:
        return FrameMatch(det_status, det_iou, gt_status, ign)
    bev, iou3 = ious(np.asarray(db, np.float32), np.asarray(gb, np.float32))
    aff = bev if family == "bev" else iou3
    ok = (aff >= iou_thr) & (dl[:, None] == gl[None, :])
    ia, ib = np.nonzero(ok & ~ign[None, :])
    rank = np.empty(n, np.int64)
    rank[np.lexsort((np.arange(n), -ds))] = np.arange(n)
    matched = greedy_match(rank, ia, ib, aff[ia, ib], n, m)
    tp = matched >= 0
    det_status[~tp & (ok & ign[None, :]).any(1)] = DROPPED
    det_status[tp] = TP
    det_iou[tp] = aff[np.flatnonzero(tp), matched[tp]]
    gt_status[matched[tp]] = MATCHED
    return FrameMatch(det_status, det_iou, gt_status, ign)


# --- inference ----------------------------------------------------------------------------------------


def segment_samples(root, split: str, segment: str, cfg: DetectorConfig, every: int = 1) -> Iterator[dict]:
    """Eval-mode samples (with targets) of one segment, in time order, in one pass over its files."""
    cfg1 = replace(cfg, data=replace(cfg.data, frame_interval_eval=every))
    stream = FrameStream(root, split, cfg1, mode="eval", transform=TargetBuilder(cfg1), segments=[segment])
    if not stream.segments:
        raise SystemExit(f"{split} segment {segment} not found or incomplete")
    for s in stream._frames([segment]):
        yield stream.transform(s)


@torch.no_grad()
def predict(model, vox, sample: dict, cfg: DetectorConfig, dev) -> tuple[dict, np.ndarray]:
    """One frame -> (detections as NumPy, (C, H, W) predicted heatmap probabilities)."""
    out = forward(model, vox, collate([sample]), dev, AMP[cfg.train.precision])
    return decode(out, cfg)[0].numpy(), torch.sigmoid(out["heatmap"][0].float()).cpu().numpy()


# --- Rerun logging --------------------------------------------------------------------------------------


def _quats(h: np.ndarray) -> np.ndarray:
    return np.stack([np.zeros_like(h), np.zeros_like(h), np.sin(h / 2), np.cos(h / 2)], 1)


def heading_ticks(b: np.ndarray) -> np.ndarray:
    """(N, 2, 3) segments from each box centre to the middle of its front face: shows heading (and 180° flips)."""
    c = b[:, :3].astype(np.float64)
    tip = c + np.stack([np.cos(b[:, 6]) * b[:, 3] / 2, np.sin(b[:, 6]) * b[:, 3] / 2, np.zeros(len(b))], 1)
    return np.stack([c, tip], 1)


def log_boxes(rec: rr.RecordingStream, path: str, b: np.ndarray, color, radius: float,
              labels: list[str] | None = None, show_labels: bool = False, **values) -> None:
    if len(b) == 0:
        rec.log(path, rr.Clear(recursive=True))
        return
    b = b.astype(np.float64)
    rec.log(path, rr.Boxes3D(centers=b[:, :3], half_sizes=b[:, 3:6] / 2, quaternions=_quats(b[:, 6]), colors=color,
                             radii=rr.Radius.ui_points(radius), labels=labels, show_labels=show_labels))
    if values:
        rec.log(path, rr.AnyValues(**values))
    rec.log(f"{path}/heading", rr.LineStrips3D(heading_ticks(b), colors=color, radii=rr.Radius.ui_points(radius)))


def log_frame(rec: rr.RecordingStream, sample: dict, det: dict, fm: FrameMatch, cfg: DetectorConfig,
              show_labels: bool = False) -> dict[str, int]:
    """Points, GT and predictions (by class and status) for one frame; returns status counts."""
    p = sample["points"]
    rec.log("world/points", rr.Points3D(p[:, :3], colors=intensity_colors(p[:, 3]), radii=rr.Radius.ui_points(1.5)))
    gb, gl, gp = sample["boxes"], sample["labels"], sample["num_points"]
    db, ds, dl = det["boxes"], det["scores"], det["labels"]
    counts: dict[str, int] = {}
    for ch, name in enumerate(cfg.data.classes):
        colors = {MATCHED: BOX_TYPE_COLORS[cfg.data.class_ids[ch]], MISSED: MISSED_COLOR, IGNORED: IGNORED_COLOR}
        for st, label in enumerate(GT_STATUS):
            k = (gl == ch) & (fm.gt_status == st)
            log_boxes(rec, f"world/gt/{name}/{label}", gb[k], colors[st], 2.0 if st == MISSED else 1.5,
                      num_points=gp[k])
            counts[f"gt_{label}/{name}"] = int(k.sum())
        for st, label in enumerate(DET_STATUS):
            k = (dl == ch) & (fm.det_status == st)
            log_boxes(rec, f"world/pred/{name}/{label}", db[k], DET_COLORS[st], 1.0,
                      labels=[f"{s:.2f}" for s in ds[k]], show_labels=show_labels, score=ds[k], iou=fm.det_iou[k])
            counts[f"pred_{label}/{name}"] = int(k.sum())
    for key, v in counts.items():
        rec.log(f"stats/{key}", rr.Scalars(v))
    return counts


def log_heatmaps(rec: rr.RecordingStream, pred: np.ndarray, gt: np.ndarray, classes) -> None:
    """Per class: GT heatmap | predicted heatmap, side by side (row = y cell, column = x cell)."""
    for c, name in enumerate(classes):
        img = np.concatenate([gt[c], pred[c]], axis=1)
        rgb = turbo(img.ravel()).reshape(*img.shape, 3)
        rec.log(f"bev/{name}", rr.Image(rgb))


def blueprint(classes: tuple[str, ...], heatmaps: bool) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(origin="world", name="GT vs predictions")
    stats = rrb.TimeSeriesView(origin="stats", name="Per-frame counts")
    side = [stats, rrb.TextDocumentView(origin="summary", name="Segment AP")]
    if heatmaps:
        side.insert(0, rrb.Tabs(*[rrb.Spatial2DView(origin=f"bev/{c}", name=f"heatmap {c} (GT | pred)")
                                  for c in classes]))
    main = rrb.Horizontal(view3d, rrb.Vertical(*side), column_shares=[3, 2])
    return rrb.Blueprint(main, rrb.TimePanel(state="expanded"), collapse_panels=False)


LEGEND = """**Predictions**: green = true positive, red = false positive, grey = dropped (overlaps an ignored GT box).
**Ground truth**: class colour = matched, yellow = missed, dim grey = ignored (too few points / outside grid).
The short line in each box points to its front (heading). Toggle entities under `world/gt` and `world/pred`."""


# --- main ----------------------------------------------------------------------------------------------


def run(root, ckpt, segment: str | None = None, split: str = "validation", every: int = 1, max_frames: int = 0,
        min_score: float = 0.3, iou: float = 0.5, overrides: list[str] | None = None, heatmaps: bool = False,
        show_labels: bool = False, save: Path | None = None, web: bool = False, web_port: int = 9090,
        open_browser: bool = True, log=print) -> dict:
    root = Path(root).expanduser()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, meta = load_model(ckpt, dev)
    if overrides:  # e.g. decode.nms=bev; the model's architecture comes from the checkpoint
        cfg = config_from_dict(meta["config"], overrides)
    model.eval()
    if segment is None:
        complete, _ = complete_segments(WaymoDataset(root), split)
        if not complete:
            raise SystemExit(f"no complete segments in {split}")
        segment = complete[0]
    ts_all = WaymoDataset(root).segment(split, segment).timestamps

    rec = rr.RecordingStream("waymo_lidar_predict", recording_id=f"{segment}-{time.time_ns()}")
    bp = blueprint(cfg.data.classes, heatmaps)
    if save:
        rec.save(save, default_blueprint=bp)
    elif web:
        url = rec.serve_grpc(default_blueprint=bp)
        rr.serve_web_viewer(web_port=web_port, open_browser=open_browser, connect_to=url)
        log(f"web viewer: http://localhost:{web_port}/?url={quote(url, safe='')}")
    else:
        rec.spawn(default_blueprint=bp)
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("world/grid", rr.LineStrips3D(grid_outline(cfg.grid), colors=(120, 120, 120),
                                          radii=rr.Radius.ui_points(1)), static=True)
    rec.log("summary", rr.TextDocument(f"{LEGEND}\n\n*running...*", media_type=rr.MediaType.MARKDOWN), static=True)

    vox = Voxelizer(cfg.grid).to(dev)
    ev = Evaluator(cfg)
    totals: dict[str, int] = {}
    n, t0 = 0, time.perf_counter()
    log(f"{split}/{segment}: {Path(ckpt)} (epoch {meta['epoch']}, step {meta['step']}), every {every} frame(s), "
        f"min score {min_score}, BEV IoU {iou}")
    for sample in segment_samples(root, split, segment, cfg, every):
        det, heat = predict(model, vox, sample, cfg, dev)
        ev.add([det], [sample])  # AP over every decoded detection (decode.score_threshold), as in training
        keep = det["scores"] >= min_score
        shown = {k: v[keep] for k, v in det.items()}
        fm = match_frame(shown, sample, cfg, iou)
        rec.set_time("frame", sequence=int(np.searchsorted(ts_all, sample["timestamp"])))
        rec.set_time("frame_time", timestamp=np.datetime64(int(sample["timestamp"]), "us"))
        counts = log_frame(rec, sample, shown, fm, cfg, show_labels)
        if heatmaps:
            log_heatmaps(rec, heat, sample["heatmap"], cfg.data.classes)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        n += 1
        if max_frames and n >= max_frames:
            break
    res = ev.compute()
    table = res.table()
    shown_summary = "\n".join(
        f"| {c} | {totals.get(f'pred_tp/{c}', 0)} | {totals.get(f'pred_fp/{c}', 0)} | {totals.get(f'gt_missed/{c}', 0)} |"
        for c in cfg.data.classes)
    rec.log("summary", rr.TextDocument(
        f"{LEGEND}\n\n### {split}/{segment}: {n} frames\n\n{table}\n\n"
        f"### Shown (score >= {min_score}, BEV IoU {iou})\n\n| class | TP | FP | missed |\n|---|---:|---:|---:|\n"
        f"{shown_summary}\n", media_type=rr.MediaType.MARKDOWN), static=True)
    rec.flush()
    log(f"{n} frames in {time.perf_counter() - t0:.1f} s\n{table}")
    if save:
        rec.disconnect()
        log(f"saved {save}; view with: uv run rerun {save}")
    elif web:
        log("Serving; press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return {"frames": n, "totals": totals, "result": res}


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("root", type=Path)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--segment", help="default: first complete segment of the split")
    p.add_argument("--split", default="validation")
    p.add_argument("--every", type=int, default=1, help="use every Nth frame (default 1: all ~199)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = all)")
    p.add_argument("--min-score", type=float, default=0.3, help="hide predictions below this score (default 0.3)")
    p.add_argument("--iou", type=float, default=0.5, help="BEV IoU for TP / FP / missed colouring (default 0.5)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides on the checkpoint's config, e.g. decode.nms=bev (not model.*)")
    p.add_argument("--heatmaps", action="store_true", help="also show GT | predicted BEV heatmaps")
    p.add_argument("--labels", action="store_true", help="draw scores next to predicted boxes")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--save", type=Path, help="write an .rrd instead of opening a window")
    out.add_argument("--web", action="store_true", help="serve in the browser instead of the desktop viewer")
    p.add_argument("--web-port", type=int, default=9090)
    p.add_argument("--no-browser", action="store_true", help="with --web: print the URL instead of opening a browser")


def main_args(a: argparse.Namespace) -> None:
    if a.every < 1:
        raise SystemExit("--every must be >= 1")
    run(a.root, a.ckpt, a.segment, a.split, a.every, a.max_frames, a.min_score, a.iou, a.set, a.heatmaps, a.labels,
        a.save, a.web, a.web_port, not a.no_browser)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="wl predict", description="View a checkpoint's predictions next to ground truth in Rerun")
    add_arguments(p)
    main_args(p.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
