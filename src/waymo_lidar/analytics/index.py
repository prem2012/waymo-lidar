"""Build the analytics cache: small per-segment tables, computed once and reused by the web tool.

Layout: <cache>/index/<kind>/<split>__<segment>.parquet with kinds
  boxes     one row per labelled box (every frame)
  points    one row per (sampled frame, laser, return): point counts, NLZ count, extents
  hist      one row per (laser, return, metric): pre-binned histogram over the sampled frames
  segments  one row per segment (written last; its presence marks the segment as indexed)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from ..data.boxes import Boxes
from ..data.dataset import WaymoDataset
from ..data.range_image import frame_to_points
from ..data.types import BOX_TYPE_NAMES, LASER_NAMES

KINDS = ("boxes", "points", "hist", "segments")
INDEX_VERSION = 1

# Fixed bins so histograms from different segments can simply be summed.
HIST_BINS: dict[str, np.ndarray] = {
    "intensity_log10": np.linspace(-3.0, 1.5, 46),
    "elongation": np.linspace(0.0, 2.0, 41),
    "z": np.linspace(-3.0, 7.0, 51),
    "range": np.linspace(0.0, 80.0, 81),
}


def default_cache_dir(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    return root.parent / "cache" / root.name


def _hist(values: np.ndarray, edges: np.ndarray) -> list[int]:
    """Counts with out-of-range values clipped into the first/last bin."""
    v = np.clip(values, edges[0], np.nextafter(edges[-1], -np.inf))
    return np.histogram(v, bins=edges)[0].tolist()


def _file(cache: Path, kind: str, split: str, segment: str) -> Path:
    return cache / "index" / kind / f"{split}__{segment}.parquet"


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def is_indexed(cache: Path, split: str, segment: str) -> bool:
    return all(_file(cache, k, split, segment).is_file() for k in KINDS)


def index_segment(root: Path, cache: Path, split: str, name: str, stride: int) -> dict:
    t0 = time.perf_counter()
    seg = WaymoDataset(root).segment(split, name)
    ts_all = seg.timestamps
    frame_idx = {int(t): i for i, t in enumerate(ts_all)}

    # Boxes: every frame (cheap, no range images).
    b = Boxes.from_table(seg.boxes_table())
    boxes = pl.DataFrame({
        "split": split, "segment": name,
        "timestamp": b.timestamp,
        "frame": np.array([frame_idx.get(int(t), -1) for t in b.timestamp], dtype=np.int32),
        "object_id": b.object_id.astype(str),
        "type": pl.Series(b.type.astype(np.int8)).replace_strict(BOX_TYPE_NAMES, return_dtype=pl.String),
        "x": b.center[:, 0], "y": b.center[:, 1], "z": b.center[:, 2],
        "length": b.size[:, 0], "width": b.size[:, 1], "height": b.size[:, 2],
        "heading": b.heading,
        "range": np.hypot(b.center[:, 0], b.center[:, 1]),
        "bearing": np.arctan2(b.center[:, 1], b.center[:, 0]),
        "speed": np.linalg.norm(b.speed[:, :2], axis=1),
        "num_points": b.num_points, "num_top_points": b.num_top_points,
        "difficulty": b.difficulty.astype(np.int8),
    })

    # Point statistics: every `stride`-th frame, both returns.
    sampled = ts_all[::stride]
    point_rows: list[dict] = []
    hist_acc: dict[tuple[int, int, str], np.ndarray] = {}
    for fr in seg.iter_range_images(returns=(1, 2), timestamps=sampled):
        pcs = frame_to_points(fr, seg.calibration, seg.pose_at(fr.timestamp))
        for (laser, ret), pc in pcs.items():
            n = len(pc)
            q = np.percentile(pc.xyz, [1, 99], axis=0) if n else np.full((2, 3), np.nan)
            point_rows.append({
                "split": split, "segment": name, "timestamp": fr.timestamp, "frame": frame_idx[fr.timestamp],
                "laser": LASER_NAMES[laser], "return": ret, "num_points": n, "num_nlz": int(pc.nlz.sum()),
                "x_p01": q[0, 0], "x_p99": q[1, 0], "y_p01": q[0, 1], "y_p99": q[1, 1],
                "z_p01": q[0, 2], "z_p99": q[1, 2],
                "range_max": float(pc.range.max()) if n else np.nan,
            })
            values = {
                "intensity_log10": np.log10(np.maximum(pc.intensity, 1e-6)),
                "elongation": pc.elongation,
                "z": pc.xyz[:, 2],
                "range": pc.range,
            }
            for metric, v in values.items():
                key = (laser, ret, metric)
                h = np.asarray(_hist(v, HIST_BINS[metric]), dtype=np.int64)
                hist_acc[key] = hist_acc.get(key, 0) + h

    points = pl.DataFrame(point_rows)
    hist = pl.DataFrame([
        {"split": split, "segment": name, "laser": LASER_NAMES[l], "return": r, "metric": m, "counts": c.tolist()}
        for (l, r, m), c in sorted(hist_acc.items())
    ], schema={"split": pl.String, "segment": pl.String, "laser": pl.String, "return": pl.Int64,
               "metric": pl.String, "counts": pl.List(pl.Int64)})

    scene = seg.scene()
    segments = pl.DataFrame([{
        "split": split, "segment": name, "num_frames": len(ts_all),
        "duration_s": float((ts_all[-1] - ts_all[0]) / 1e6) if len(ts_all) else 0.0,
        "num_boxes": len(b), "num_objects": int(len(np.unique(b.object_id))),
        "sampled_frames": len(sampled), "stride": stride, **scene,
    }])

    _write(boxes, _file(cache, "boxes", split, name))
    _write(points, _file(cache, "points", split, name))
    _write(hist, _file(cache, "hist", split, name))
    _write(segments, _file(cache, "segments", split, name))  # last: marks completion
    return {"split": split, "segment": name, "seconds": time.perf_counter() - t0}


def build_index(root: str | Path, cache: str | Path | None = None, stride: int = 10,
                workers: int | None = None, force: bool = False, log=print) -> Path:
    root = Path(root).expanduser().resolve()
    cache = Path(cache).expanduser().resolve() if cache else default_cache_dir(root)
    ds = WaymoDataset(root)
    todo = [(s, n) for s, n in ds.all_segments() if force or not is_indexed(cache, s, n)]
    total = len(ds.all_segments())
    log(f"data: {root}\ncache: {cache}\n{total} segments, {total - len(todo)} already indexed, {len(todo)} to do "
        f"(point stats every {stride} frames)")

    (cache / "index").mkdir(parents=True, exist_ok=True)
    (cache / "index" / "meta.json").write_text(json.dumps({
        "version": INDEX_VERSION, "root": str(root), "stride": stride,
        "hist_bins": {k: v.tolist() for k, v in HIST_BINS.items()},
    }, indent=1))

    if todo:
        workers = workers or max(1, min(len(todo), (os.cpu_count() or 2) // 2))
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(index_segment, root, cache, s, n, stride) for s, n in todo]
            for i, f in enumerate(as_completed(futures), 1):
                r = f.result()
                log(f"[{i}/{len(todo)}] {r['split']}/{r['segment']}  {r['seconds']:.1f}s")
        log(f"done in {time.perf_counter() - t0:.1f}s with {workers} workers")
    return cache
