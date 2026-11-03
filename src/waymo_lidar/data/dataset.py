"""Discover and read segments of Waymo Open Dataset v2 (Parquet, one file per segment per component).

Layout: <root>/<split>/<component>/<segment>.parquet. Components are joined on the ``key.*`` columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .types import SPLITS, Laser

KEY_SEGMENT = "key.segment_context_name"
KEY_TS = "key.frame_timestamp_micros"
KEY_LASER = "key.laser_name"
STREAM_FRAMES = 8  # frames per record batch when streaming lidar / lidar_pose (see iter_range_images)


def col(component: str, field: str) -> str:
    """Column name of a component field, e.g. col("LiDAR", "range_image_return1.values")."""
    return f"[{component}Component].{field}"


def list_column(arr: pa.ChunkedArray | pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Flat values and row offsets of a list column, without per-row Python objects."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if pa.types.is_fixed_size_list(arr.type):
        n = arr.type.list_size
        values = arr.flatten().to_numpy(zero_copy_only=False)
        return values, np.arange(0, len(values) + 1, n)
    # List offsets index directly into `values`, also for sliced arrays.
    offsets = arr.offsets.to_numpy()
    values = arr.values.to_numpy(zero_copy_only=False)
    return values, offsets


@dataclass(frozen=True)
class LaserCalibration:
    laser: int
    extrinsic: np.ndarray  # (4, 4) vehicle_from_sensor
    inclination_min: float
    inclination_max: float
    inclinations: np.ndarray | None  # (H,) ascending, only provided for the top LiDAR


@dataclass
class FrameRangeImages:
    timestamp: int
    # (laser, return) -> (H, W, 4) float32: range, intensity, elongation, is_in_nlz
    images: dict[tuple[int, int], np.ndarray]
    # (H, W, 6) float32 per-pixel world_from_vehicle pose (roll, pitch, yaw, x, y, z) for the top LiDAR
    top_pose: np.ndarray | None


def _row_groups_with(pf: pq.ParquetFile, column: str, wanted: set[int] | None) -> list[int]:
    """Row groups whose [min, max] statistics for `column` contain at least one wanted value (all if None).

    Rows are sorted by timestamp and a frame can straddle two consecutive groups; both are kept, and the
    reader's frame assembly joins them. Groups without statistics are always read.
    """
    groups = range(pf.num_row_groups)
    if wanted is None:
        return list(groups)
    md = pf.metadata
    idx = [md.schema.column(i).path for i in range(md.num_columns)].index(column)
    w = np.fromiter(wanted, dtype=np.int64)
    keep = []
    for g in groups:
        st = md.row_group(g).column(idx).statistics
        if st is None or not st.has_min_max or np.any((w >= st.min) & (w <= st.max)):
            keep.append(g)
    return keep


class WaymoDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"data root not found: {self.root}")

    def splits(self) -> list[str]:
        return [s for s in SPLITS if (self.root / s).is_dir()]

    def segments(self, split: str) -> list[str]:
        for component in ("lidar_box", "lidar"):
            d = self.root / split / component
            if d.is_dir():
                return sorted(p.stem for p in d.glob("*.parquet"))
        return []

    def all_segments(self) -> list[tuple[str, str]]:
        return [(split, seg) for split in self.splits() for seg in self.segments(split)]

    def segment(self, split: str, name: str) -> Segment:
        return Segment(self.root, split, name)


class Segment:
    def __init__(self, root: Path, split: str, name: str):
        self.root, self.split, self.name = Path(root), split, name

    def __repr__(self) -> str:
        return f"Segment({self.split}/{self.name})"

    def path(self, component: str) -> Path:
        return self.root / self.split / component / f"{self.name}.parquet"

    def has(self, component: str) -> bool:
        return self.path(component).is_file()

    def read(self, component: str, columns: list[str] | None = None, filters=None) -> pa.Table:
        return pq.read_table(self.path(component), columns=columns, filters=filters)

    # --- small per-segment tables -------------------------------------------------------------

    @cached_property
    def calibration(self) -> dict[int, LaserCalibration]:
        c = "LiDARCalibration"
        rows = self.read("lidar_calibration").to_pylist()
        out = {}
        for r in rows:
            incl = r[col(c, "beam_inclination.values")]
            out[r[KEY_LASER]] = LaserCalibration(
                laser=r[KEY_LASER],
                extrinsic=np.asarray(r[col(c, "extrinsic.transform")], dtype=np.float64).reshape(4, 4),
                inclination_min=r[col(c, "beam_inclination.min")],
                inclination_max=r[col(c, "beam_inclination.max")],
                inclinations=None if incl is None else np.asarray(incl, dtype=np.float64),
            )
        return out

    @cached_property
    def _vehicle_poses(self) -> tuple[np.ndarray, np.ndarray]:
        t = self.read("vehicle_pose").sort_by(KEY_TS)
        ts = t.column(KEY_TS).to_numpy()
        flat, _ = list_column(t.column(col("VehiclePose", "world_from_vehicle.transform")))
        return ts, flat.astype(np.float64).reshape(-1, 4, 4)

    @property
    def timestamps(self) -> np.ndarray:
        """Sorted frame timestamps (microseconds)."""
        return self._vehicle_poses[0]

    @property
    def vehicle_poses(self) -> np.ndarray:
        """(F, 4, 4) world_from_vehicle, aligned with ``timestamps``."""
        return self._vehicle_poses[1]

    def pose_at(self, ts: int) -> np.ndarray:
        i = int(np.searchsorted(self.timestamps, ts))
        if i >= len(self.timestamps) or self.timestamps[i] != ts:
            raise KeyError(f"no vehicle pose for timestamp {ts}")
        return self.vehicle_poses[i]

    def boxes_table(self) -> pa.Table:
        return self.read("lidar_box")

    def scene(self) -> dict[str, str | None]:
        """Segment-level metadata from the stats component (constant within a segment)."""
        if not self.has("stats"):
            return {"time_of_day": None, "location": None, "weather": None}
        c = "Stats"
        t = self.read("stats", columns=[col(c, "time_of_day"), col(c, "location"), col(c, "weather")])
        row = t.slice(0, 1).to_pylist()[0]
        return {k: row[col(c, k)] for k in ("time_of_day", "location", "weather")}

    # --- range images ---------------------------------------------------------------------------

    def iter_range_images(
        self,
        returns: Iterable[int] = (1,),
        lasers: Iterable[int] | None = None,
        timestamps: Iterable[int] | None = None,
    ) -> Iterator[FrameRangeImages]:
        """Yield range images frame by frame, in time order.

        Streams the LiDAR file in small record batches (``STREAM_FRAMES`` frames each), so the first
        frame is available quickly and memory stays bounded. ``timestamps`` restricts which frames are
        kept, and skips row groups that hold none of them.

        Batches, not whole row groups: a row group is ~50 frames whose range images are ~10x larger
        decoded than on disk (Brotli), and the decoder's transient buffers are a multiple of that again.
        Reading row groups whole peaked at ~8 GiB per stream; batches keep it near 2 GiB at the same speed.
        """
        returns = tuple(returns)
        lasers = set(Laser) if lasers is None else {int(l) for l in lasers}
        wanted = None if timestamps is None else set(int(t) for t in timestamps)
        c = "LiDAR"
        columns = [KEY_TS, KEY_LASER]
        for r in returns:
            columns += [col(c, f"range_image_return{r}.values"), col(c, f"range_image_return{r}.shape")]

        top_pose = self._top_pose_reader(wanted) if Laser.TOP in lasers else None

        pf = pq.ParquetFile(self.path("lidar"))
        pending: FrameRangeImages | None = None
        for t in pf.iter_batches(batch_size=STREAM_FRAMES * len(Laser), columns=columns,
                                 row_groups=_row_groups_with(pf, KEY_TS, wanted)):
            ts = t.column(KEY_TS).to_numpy()
            ln = t.column(KEY_LASER).to_numpy()
            data = {}
            for r in returns:
                vals, offs = list_column(t.column(col(c, f"range_image_return{r}.values")))
                shapes = list_column(t.column(col(c, f"range_image_return{r}.shape")))[0].reshape(-1, 3)
                data[r] = (vals, offs, shapes)
            for i in range(t.num_rows):
                frame_ts, laser = int(ts[i]), int(ln[i])
                if pending is not None and pending.timestamp != frame_ts:
                    yield pending
                    pending = None
                if (wanted is not None and frame_ts not in wanted) or laser not in lasers:
                    continue
                if pending is None:
                    pending = FrameRangeImages(frame_ts, {}, top_pose(frame_ts) if top_pose else None)
                for r, (vals, offs, shapes) in data.items():
                    pending.images[(laser, r)] = vals[offs[i]:offs[i + 1]].reshape(shapes[i])
        if pending is not None:
            yield pending

    def _top_pose_reader(self, wanted: set[int] | None = None):
        """Returns ts -> (H, W, 6) top-LiDAR pixel pose (return 1 pose applies to both returns).

        Keeps only the `wanted` frames (all if None). The file is streamed in batches like the LiDAR
        file: decoded whole it is ~730 MiB of floats per segment and peaked at ~3 GiB.
        """
        c = "LiDARPose"
        cols = [KEY_TS, col(c, "range_image_return1.values"), col(c, "range_image_return1.shape")]
        pf = pq.ParquetFile(self.path("lidar_pose"))
        poses: dict[int, np.ndarray] = {}
        for b in pf.iter_batches(batch_size=STREAM_FRAMES, columns=cols,
                                 row_groups=_row_groups_with(pf, KEY_TS, wanted)):
            ts = b.column(KEY_TS).to_numpy()
            vals, offs = list_column(b.column(cols[1]))
            shapes = list_column(b.column(cols[2]))[0].reshape(-1, 3)
            for i, frame_ts in enumerate(ts):
                if wanted is None or int(frame_ts) in wanted:  # copy: don't pin the whole batch
                    poses[int(frame_ts)] = vals[offs[i]:offs[i + 1]].reshape(shapes[i]).copy()
        return poses.get
