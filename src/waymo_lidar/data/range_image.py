"""Range image -> point cloud, following the geometry of the official waymo_open_dataset utils.

- Row 0 of a range image is the highest beam; column 0 is azimuth +pi (behind the sensor), sweeping clockwise.
- Points are returned in the vehicle frame at the frame timestamp. For the top LiDAR, the per-pixel
  pose corrects for the vehicle moving during the sweep (rolling shutter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import FrameRangeImages, LaserCalibration
from .types import CH_ELONGATION, CH_INTENSITY, CH_NLZ, CH_RANGE, Laser


@dataclass
class PointCloud:
    xyz: np.ndarray  # (N, 3) float32, vehicle frame
    intensity: np.ndarray  # (N,) float32
    elongation: np.ndarray  # (N,) float32
    nlz: np.ndarray  # (N,) bool, point lies in a no-label zone
    range: np.ndarray  # (N,) float32, metres from the sensor

    def __len__(self) -> int:
        return len(self.xyz)

    def select(self, mask: np.ndarray) -> PointCloud:
        return PointCloud(self.xyz[mask], self.intensity[mask], self.elongation[mask], self.nlz[mask], self.range[mask])

    @staticmethod
    def concat(clouds: list[PointCloud]) -> PointCloud:
        if not clouds:
            z = np.zeros(0, np.float32)
            return PointCloud(np.zeros((0, 3), np.float32), z, z, np.zeros(0, bool), z)
        return PointCloud(*(np.concatenate([getattr(c, f) for c in clouds])
                            for f in ("xyz", "intensity", "elongation", "nlz", "range")))


def beam_inclinations(calib: LaserCalibration, height: int) -> np.ndarray:
    """(H,) inclination per range image row, row 0 = highest beam."""
    if calib.inclinations is not None:
        incl = calib.inclinations
    else:  # side LiDARs: beams spaced uniformly between min and max
        incl = calib.inclination_min + (calib.inclination_max - calib.inclination_min) * (np.arange(height) + 0.5) / height
    return incl[::-1]


def beam_azimuths(extrinsic: np.ndarray, width: int) -> np.ndarray:
    """(W,) azimuth per range image column, in the sensor frame."""
    az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (width - np.arange(width) - 0.5) / width
    return (ratios * 2.0 - 1.0) * np.pi - az_correction


def rpy_to_rotation(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    r = np.empty(roll.shape + (3, 3), dtype=np.float64)
    r[..., 0, 0] = cy * cp
    r[..., 0, 1] = cy * sp * sr - sy * cr
    r[..., 0, 2] = cy * sp * cr + sy * sr
    r[..., 1, 0] = sy * cp
    r[..., 1, 1] = sy * sp * sr + cy * cr
    r[..., 1, 2] = sy * sp * cr - cy * sr
    r[..., 2, 0] = -sp
    r[..., 2, 1] = cp * sr
    r[..., 2, 2] = cp * cr
    return r


def range_image_to_points(
    image: np.ndarray,
    calib: LaserCalibration,
    pixel_pose: np.ndarray | None = None,
    frame_pose: np.ndarray | None = None,
) -> PointCloud:
    """Decode one (H, W, 4) range image into vehicle-frame points (pixels with range <= 0 are dropped).

    ``pixel_pose`` (H, W, 6) and ``frame_pose`` (4, 4 world_from_vehicle) must be given together; they
    apply the rolling-shutter correction used for the top LiDAR.
    """
    h, w, _ = image.shape
    rng = image[..., CH_RANGE]
    rows, cols = np.nonzero(rng > 0)
    r = rng[rows, cols].astype(np.float64)

    incl = beam_inclinations(calib, h)[rows]
    az = beam_azimuths(calib.extrinsic, w)[cols]
    cos_incl = np.cos(incl)
    p = np.stack([np.cos(az) * cos_incl * r, np.sin(az) * cos_incl * r, np.sin(incl) * r], axis=1)

    e = calib.extrinsic
    p = p @ e[:3, :3].T + e[:3, 3]  # sensor -> vehicle (at pixel time)

    if pixel_pose is not None:
        if frame_pose is None:
            raise ValueError("pixel_pose requires frame_pose")
        pp = pixel_pose[rows, cols].astype(np.float64)
        rot = rpy_to_rotation(pp[:, 0], pp[:, 1], pp[:, 2])
        p = np.einsum("nij,nj->ni", rot, p) + pp[:, 3:6]  # -> world
        inv = np.linalg.inv(frame_pose)
        p = p @ inv[:3, :3].T + inv[:3, 3]  # world -> vehicle (at frame time)

    return PointCloud(
        xyz=p.astype(np.float32),
        intensity=image[rows, cols, CH_INTENSITY].astype(np.float32),
        elongation=image[rows, cols, CH_ELONGATION].astype(np.float32),
        nlz=image[rows, cols, CH_NLZ] > 0,
        range=r.astype(np.float32),
    )


def frame_to_points(
    frame: FrameRangeImages,
    calibration: dict[int, LaserCalibration],
    frame_pose: np.ndarray,
) -> dict[tuple[int, int], PointCloud]:
    """Decode every range image of a frame. Keys are (laser, return)."""
    out = {}
    for (laser, ret), image in frame.images.items():
        pose = frame.top_pose if laser == Laser.TOP else None
        out[(laser, ret)] = range_image_to_points(image, calibration[laser], pose, frame_pose if pose is not None else None)
    return out
