"""Aggregations over the analytics cache, shaped for the web tool's charts (all JSON-serialisable)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from ..data.types import BOX_TYPE_COLORS, BOX_TYPE_NAMES, LASER_NAMES

CLASSES = ["vehicle", "pedestrian", "cyclist", "sign"]  # display order; "unknown" never appears in labels
CLASS_COLORS = {BOX_TYPE_NAMES[k]: "#%02x%02x%02x" % v for k, v in BOX_TYPE_COLORS.items()}
LASERS = [LASER_NAMES[i] for i in sorted(LASER_NAMES)]


class Stats:
    def __init__(self, cache: Path):
        self.cache = Path(cache)
        idx = self.cache / "index"
        if not (idx / "segments").is_dir():
            raise FileNotFoundError(f"no index at {idx}; run `wl index <data_root>` first")
        self.meta = json.loads((idx / "meta.json").read_text())
        read = lambda kind: pl.read_parquet(idx / kind / "*.parquet")  # noqa: E731
        self.segments = read("segments").sort("split", "segment")
        self.boxes = read("boxes").with_columns(
            # Waymo eval: unset difficulty -> L2 if <= 5 points, else L1.
            pl.when((pl.col("difficulty") == 2) | (pl.col("num_points") <= 5)).then(pl.lit("L2"))
            .otherwise(pl.lit("L1")).alias("level")
        )
        self.points = read("points")
        self.hist = read("hist")

    # --- helpers -------------------------------------------------------------------------------

    @staticmethod
    def _f(df: pl.DataFrame, split: str) -> pl.DataFrame:
        return df if split == "all" else df.filter(pl.col("split") == split)

    def splits(self) -> list[str]:
        return self.segments["split"].unique(maintain_order=True).to_list()

    def classes(self, boxes: pl.DataFrame) -> list[str]:
        present = set(boxes["type"].unique().to_list())
        return [c for c in CLASSES if c in present]

    # --- sections ------------------------------------------------------------------------------

    def overview(self, split: str = "all") -> dict:
        seg, box = self._f(self.segments, split), self._f(self.boxes, split)
        per_split = (
            self.segments.group_by("split").agg(
                pl.len().alias("segments"), pl.col("num_frames").sum().alias("frames"),
                pl.col("num_boxes").sum().alias("boxes"), pl.col("num_objects").sum().alias("objects"),
                (pl.col("duration_s").sum() / 60).alias("minutes"),
            ).sort("split").to_dicts()
        )
        return {
            "per_split": per_split,
            "totals": {
                "segments": seg.height, "frames": int(seg["num_frames"].sum()), "boxes": box.height,
                "objects": int(seg["num_objects"].sum()), "minutes": float(seg["duration_s"].sum() / 60),
            },
            "class_counts": self._class_counts(box),
            "stride": self.meta["stride"], "root": self.meta["root"], "cache": str(self.cache),
        }

    def _class_counts(self, box: pl.DataFrame) -> list[dict]:
        df = box.group_by("type").agg(
            pl.len().alias("boxes"), pl.col("object_id").n_unique().alias("objects"),
            (pl.col("level") == "L1").sum().alias("L1"), (pl.col("level") == "L2").sum().alias("L2"),
            (pl.col("num_points") == 0).sum().alias("empty"),
        )
        rows = {r["type"]: r for r in df.to_dicts()}
        return [rows[c] for c in CLASSES if c in rows]

    def labels(self, split: str = "all") -> dict:
        box = self._f(self.boxes, split)
        by_split = box.group_by("split", "type").agg(pl.len().alias("boxes")).to_dicts()
        out = {"classes": self.classes(box), "colors": CLASS_COLORS, "by_split": by_split,
               "class_counts": self._class_counts(box), "size": {}, "heading": {}}
        heading_edges = np.linspace(-np.pi, np.pi, 37)
        for c in out["classes"]:
            b = box.filter(pl.col("type") == c)
            out["size"][c] = {dim: _hist_auto(b[dim].to_numpy()) for dim in ("length", "width", "height")}
            out["size"][c]["summary"] = {
                dim: {q: float(b[dim].quantile(v)) for q, v in (("p10", .1), ("p50", .5), ("p90", .9))}
                for dim in ("length", "width", "height")
            }
            out["heading"][c] = {"edges": heading_edges.tolist(), "counts": _hist(b["heading"].to_numpy(), heading_edges)}
        return out

    def spatial(self, split: str = "all") -> dict:
        box = self._f(self.boxes, split)
        classes = self.classes(box)
        range_edges = np.arange(0, 82, 2.0)
        bev_edges = np.arange(-80, 81, 2.0)
        pts_edges = np.arange(0, 85, 5.0)
        out = {"classes": classes, "colors": CLASS_COLORS, "range": {}, "bev": {}, "points_vs_range": {}}
        for c in classes:
            b = box.filter(pl.col("type") == c)
            r = b["range"].to_numpy()
            out["range"][c] = {"edges": range_edges.tolist(), "counts": _hist(r, range_edges)}
            h, _, _ = np.histogram2d(b["x"].to_numpy(), b["y"].to_numpy(), bins=[bev_edges, bev_edges])
            out["bev"][c] = {"edges": bev_edges.tolist(), "counts": h.T.astype(int).tolist()}  # rows = y
            n = b["num_points"].to_numpy()
            idx = np.digitize(r, pts_edges) - 1
            rows = []
            for i in range(len(pts_edges) - 1):
                m = idx == i
                if m.sum() < 5:
                    continue
                rows.append({"range": float(pts_edges[i] + 2.5), "n": int(m.sum()),
                             "p10": float(np.percentile(n[m], 10)), "p50": float(np.percentile(n[m], 50)),
                             "p90": float(np.percentile(n[m], 90)), "empty_frac": float(np.mean(n[m] == 0)),
                             "le5_frac": float(np.mean(n[m] <= 5))})
            out["points_vs_range"][c] = rows
        return out

    def pointcloud(self, split: str = "all") -> dict:
        pts, hist = self._f(self.points, split), self._f(self.hist, split)
        per_laser = (
            pts.group_by("laser", "return").agg(
                pl.col("num_points").mean().alias("mean_points"),
                (pl.col("num_nlz").sum() / pl.col("num_points").sum()).alias("nlz_frac"),
                *[pl.col(f"{a}_{p}").median().alias(f"{a}_{p}") for a in "xyz" for p in ("p01", "p99")],
                pl.col("range_max").median().alias("range_max"),
                pl.len().alias("frames"),
            ).to_dicts()
        )
        order = {l: i for i, l in enumerate(LASERS)}
        per_laser.sort(key=lambda r: (order[r["laser"]], r["return"]))
        hists: dict[str, dict] = {}
        for (metric, laser, ret), g in hist.group_by("metric", "laser", "return"):
            counts = np.sum(np.stack(g["counts"].to_list()), axis=0)
            hists.setdefault(metric, {}).setdefault(laser, {})[str(ret)] = counts.tolist()
        return {"lasers": LASERS, "per_laser": per_laser, "hist": hists, "bins": self.meta["hist_bins"],
                "stride": self.meta["stride"]}

    def scenes(self, split: str = "all") -> dict:
        seg, box = self._f(self.segments, split), self._f(self.boxes, split)
        cats = {}
        for field in ("time_of_day", "weather", "location"):
            cats[field] = seg.with_columns(pl.col(field).fill_null("unknown")).group_by(field, "split") \
                .agg(pl.len().alias("segments")).sort(field).to_dicts()
        per_frame = box.group_by("split", "segment", "frame", "type").agg(pl.len().alias("n"))
        classes = self.classes(box)
        objects_per_frame = {}
        for c in classes:
            n = per_frame.filter(pl.col("type") == c)["n"].to_numpy()
            # Frames without any box of this class count as zero.
            total_frames = int(seg["num_frames"].sum())
            zeros = max(0, total_frames - len(n))
            n = np.concatenate([n, np.zeros(zeros, dtype=n.dtype)])
            edges = np.arange(0, max(int(n.max()) + 2, 2))
            objects_per_frame[c] = {"edges": edges.tolist(), "counts": _hist(n, edges),
                                    "mean": float(n.mean()), "max": int(n.max())}
        table = seg.select("split", "segment", "num_frames", "num_boxes", "num_objects", "time_of_day",
                           "weather", "location").to_dicts()
        return {"categories": cats, "objects_per_frame": objects_per_frame, "classes": classes,
                "colors": CLASS_COLORS, "segments": table}

    def segment_list(self) -> list[dict]:
        return self.segments.select("split", "segment", "num_frames", "num_boxes", "time_of_day", "weather",
                                    "location").to_dicts()


def _hist(values: np.ndarray, edges: np.ndarray) -> list[int]:
    v = np.clip(values, edges[0], np.nextafter(edges[-1], -np.inf))
    return np.histogram(v, bins=edges)[0].astype(int).tolist()


def _hist_auto(values: np.ndarray, bins: int = 40) -> dict:
    if len(values) == 0:
        return {"edges": [0, 1], "counts": [0]}
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi - lo < 0.05 * max(abs(hi), 1e-3):  # near-constant (e.g. a single object): show a +-10% window
        mid = (lo + hi) / 2
        lo, hi = mid - 0.1 * max(abs(mid), 0.1), mid + 0.1 * max(abs(mid), 0.1)
    edges = np.linspace(lo, hi, bins + 1)
    return {"edges": edges.tolist(), "counts": _hist(values, edges)}


@lru_cache(maxsize=4)
def _load(cache: str, stamp: float) -> Stats:
    return Stats(Path(cache))


def load_stats(cache: Path) -> Stats:
    """Cached Stats; reloads automatically when the index changes (e.g. after `wl index` adds segments)."""
    seg_dir = Path(cache) / "index" / "segments"
    stamp = max((p.stat().st_mtime for p in seg_dir.glob("*.parquet")), default=0.0)
    return _load(str(cache), stamp)
