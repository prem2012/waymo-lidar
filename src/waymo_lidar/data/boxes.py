"""3D box labels: table -> arrays, box geometry, and point-in-box tests. Boxes are in the vehicle frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .dataset import KEY_TS, col

_C = "LiDARBox"


@dataclass
class Boxes:
    timestamp: np.ndarray  # (N,) int64
    object_id: np.ndarray  # (N,) object (str)
    center: np.ndarray  # (N, 3) float64
    size: np.ndarray  # (N, 3) float64: length (x), width (y), height (z)
    heading: np.ndarray  # (N,) float64, rotation about +z
    type: np.ndarray  # (N,) int8, see types.BoxType
    num_points: np.ndarray  # (N,) int64, all LiDARs (from the dataset)
    num_top_points: np.ndarray  # (N,) int64, top LiDAR only
    difficulty: np.ndarray  # (N,) int8, detection difficulty (1 = L1, 2 = L2)
    speed: np.ndarray  # (N, 3) float64, m/s

    def __len__(self) -> int:
        return len(self.timestamp)

    def select(self, mask: np.ndarray) -> Boxes:
        return Boxes(**{f: getattr(self, f)[mask] for f in self.__dataclass_fields__})

    def at(self, ts: int) -> Boxes:
        return self.select(self.timestamp == ts)

    @staticmethod
    def from_table(t: pa.Table) -> Boxes:
        def num(field: str) -> np.ndarray:
            return t.column(col(_C, field)).to_numpy()

        def vec(prefix: str) -> np.ndarray:
            return np.stack([num(f"{prefix}.{a}") for a in "xyz"], axis=1)

        return Boxes(
            timestamp=t.column(KEY_TS).to_numpy(),
            object_id=np.asarray(t.column("key.laser_object_id").to_pylist(), dtype=object),
            center=vec("box.center"),
            size=vec("box.size"),
            heading=num("box.heading"),
            type=num("type"),
            num_points=num("num_lidar_points_in_box"),
            num_top_points=num("num_top_lidar_points_in_box"),
            difficulty=num("difficulty_level.detection"),
            speed=vec("speed"),
        )


def box_corners(center: np.ndarray, size: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """(N, 8, 3) corners; first four are the bottom face, counter-clockwise from front-left."""
    signs = np.array([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
                      [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]], dtype=np.float64)
    local = signs[None] * (size[:, None, :] / 2)
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    x = local[..., 0] * c - local[..., 1] * s
    y = local[..., 0] * s + local[..., 1] * c
    return np.stack([x, y, local[..., 2]], axis=-1) + center[:, None, :]


def points_in_boxes(xyz: np.ndarray, center: np.ndarray, size: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """(N,) number of points inside each box (boundary inclusive)."""
    counts = np.zeros(len(center), dtype=np.int64)
    if len(xyz) == 0:
        return counts
    xyz = xyz.astype(np.float64, copy=False)
    for i in range(len(center)):
        half = size[i] / 2
        radius = np.linalg.norm(half)
        d = xyz - center[i]
        near = np.abs(d[:, 0]) <= radius
        near &= np.abs(d[:, 1]) <= radius
        near &= np.abs(d[:, 2]) <= half[2]
        if not near.any():
            continue
        dn = d[near]
        c, s = np.cos(heading[i]), np.sin(heading[i])
        lx = dn[:, 0] * c + dn[:, 1] * s
        ly = -dn[:, 0] * s + dn[:, 1] * c
        counts[i] = np.count_nonzero((np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1]))
    return counts
