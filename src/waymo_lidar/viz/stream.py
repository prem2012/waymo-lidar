"""Stream a segment from Parquet straight into the Rerun web viewer (no intermediate .rrd files).

One Rerun web viewer is hosted for the process. Each opened segment gets its own recording with its own
in-process gRPC server; opening another segment cancels the worker and disconnects the old recording,
which shuts its server down, so only one segment is held in memory.
"""

from __future__ import annotations

import socket
import threading
import time
import traceback
from dataclasses import dataclass, field
from urllib.parse import quote

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ..data.boxes import Boxes
from ..data.dataset import WaymoDataset
from ..data.range_image import frame_to_points
from ..data.types import BOX_TYPE_COLORS, BOX_TYPE_NAMES, LASER_NAMES

APP_ID = "waymo_lidar"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _turbo(x: np.ndarray) -> np.ndarray:
    """Turbo colormap (polynomial approximation), x in [0, 1] -> (N, 3) uint8."""
    x = np.clip(x, 0.0, 1.0)[:, None]
    coeffs = np.array([
        [0.13572138, 0.09140261, 0.10667330],
        [4.61539260, 2.19418839, 12.64194608],
        [-42.66032258, 4.84296658, -60.58204836],
        [132.13108234, -14.18503333, 110.36276771],
        [-152.94239396, 4.27729857, -89.90310912],
        [59.28637943, 2.82956604, 27.34824973],
    ])
    rgb = sum(coeffs[i] * x ** i for i in range(6))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def intensity_colors(intensity: np.ndarray) -> np.ndarray:
    # Intensity is heavy-tailed (retroreflectors >> asphalt); colour on a log scale.
    return _turbo((np.log10(np.maximum(intensity, 1e-3)) + 2.5) / 3.0)


def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial3DView(origin="world/ego", name="Ego frame"),
            rrb.Spatial3DView(origin="world", name="World frame"),
        ),
        rrb.TimePanel(state="expanded"),
        collapse_panels=False,
    )


@dataclass
class StreamState:
    split: str | None = None
    segment: str | None = None
    returns: tuple[int, ...] = (1,)
    grpc_url: str | None = None
    viewer_url: str | None = None
    frames_total: int = 0
    frames_sent: int = 0
    state: str = "idle"  # idle | streaming | done | cancelled | error
    error: str | None = None
    started: float = 0.0
    elapsed_s: float = 0.0
    first_frame_s: float | None = None
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class Streamer:
    def __init__(self, dataset: WaymoDataset, web_port: int = 9090, memory_limit: str = "2GiB", host: str = "localhost"):
        self.ds, self.web_port, self.memory_limit, self.host = dataset, web_port, memory_limit, host
        self._lock = threading.Lock()
        self._rec: rr.RecordingStream | None = None
        self._thread: threading.Thread | None = None
        self.status = StreamState()
        rr.serve_web_viewer(web_port=web_port, open_browser=False)

    def open(self, split: str, segment: str, returns: tuple[int, ...] = (1,)) -> dict:
        if segment not in self.ds.segments(split):
            raise KeyError(f"unknown segment {split}/{segment}")
        with self._lock:
            self._stop_current()
            rec = rr.RecordingStream(APP_ID, recording_id=f"{segment}-{time.time_ns()}")
            port = _free_port()
            origins = [f"http://{h}:{self.web_port}" for h in ("localhost", "127.0.0.1")]
            url = rec.serve_grpc(grpc_port=port, server_memory_limit=self.memory_limit,
                                 cors_allow_origin=origins, default_blueprint=blueprint())
            rec.send_recording_name(segment)
            seg = self.ds.segment(split, segment)
            st = StreamState(split=split, segment=segment, returns=returns, grpc_url=url,
                             viewer_url=f"http://{self.host}:{self.web_port}/?url={quote(url, safe='')}",
                             frames_total=len(seg.timestamps), state="streaming", started=time.perf_counter())
            self._rec, self.status = rec, st
            self._thread = threading.Thread(target=self._run, args=(rec, seg, st), daemon=True)
            self._thread.start()
            return st.public()

    def _stop_current(self) -> None:
        if self._thread is not None:
            self.status._stop.set()
            self._thread.join(timeout=10)
            if self.status.state == "streaming":
                self.status.state = "cancelled"
        if self._rec is not None:
            self._rec.disconnect()
        self._rec, self._thread = None, None

    def close(self) -> None:
        with self._lock:
            self._stop_current()

    def _run(self, rec: rr.RecordingStream, seg, st: StreamState) -> None:
        try:
            self._stream(rec, seg, st)
            if not st._stop.is_set():
                st.state = "done"
        except Exception as e:  # surfaced to the page via /api/view/status
            st.state, st.error = "error", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            st.elapsed_s = time.perf_counter() - st.started

    def _stream(self, rec: rr.RecordingStream, seg, st: StreamState) -> None:
        rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rec.log("world", rr.AnnotationContext([
            rr.ClassDescription(info=rr.AnnotationInfo(id=k, label=BOX_TYPE_NAMES[k], color=BOX_TYPE_COLORS[k]))
            for k in BOX_TYPE_NAMES
        ]), static=True)

        boxes = Boxes.from_table(seg.boxes_table())
        frame_of = {int(t): i for i, t in enumerate(seg.timestamps)}
        # World poses are large (UTM-like) numbers; re-centre on the first frame for float32 precision.
        origin = seg.vehicle_poses[0][:3, 3].copy()

        for fr in seg.iter_range_images(returns=st.returns):
            if st._stop.is_set():
                return
            ts = fr.timestamp
            rec.set_time("frame_time", timestamp=np.datetime64(ts, "us"))
            rec.set_time("frame", sequence=frame_of[ts])

            pose = seg.pose_at(ts)
            rec.log("world/ego", rr.Transform3D(translation=pose[:3, 3] - origin, mat3x3=pose[:3, :3]))

            for (laser, ret), pc in sorted(frame_to_points(fr, seg.calibration, pose).items()):
                rec.log(f"world/ego/lidar/{LASER_NAMES[laser]}/return{ret}",
                        rr.Points3D(pc.xyz, colors=intensity_colors(pc.intensity), radii=rr.Radius.ui_points(1.5)))

            b = boxes.at(ts)
            for k in np.unique(b.type):
                m = b.type == k
                h = b.heading[m]
                quats = np.stack([np.zeros_like(h), np.zeros_like(h), np.sin(h / 2), np.cos(h / 2)], axis=1)
                rec.log(f"world/ego/boxes/{BOX_TYPE_NAMES[int(k)]}",
                        rr.Boxes3D(centers=b.center[m], half_sizes=b.size[m] / 2, quaternions=quats,
                                   class_ids=np.full(int(m.sum()), int(k)), show_labels=False))
            # Clear classes absent from this frame so stale boxes don't linger.
            for k in set(BOX_TYPE_NAMES) - set(int(v) for v in np.unique(b.type)):
                rec.log(f"world/ego/boxes/{BOX_TYPE_NAMES[k]}", rr.Clear(recursive=False))

            st.frames_sent += 1
            if st.first_frame_s is None:
                st.first_frame_s = time.perf_counter() - st.started
            st.elapsed_s = time.perf_counter() - st.started
