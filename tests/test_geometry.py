"""Geometry checks against the dataset's own per-box point counts.

The dataset's `num_*_points_in_box` counts come from the raw point cloud, which has a few percent more
points than the range images (points that share a pixel are dropped when rasterising). So decoded counts
should track the reference almost perfectly but sit slightly below it, and rarely above.
Data root: $WAYMO_ROOT or ~/dev/data/waymo_v2.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from waymo_lidar.data.boxes import Boxes, box_corners, points_in_boxes
from waymo_lidar.data.dataset import WaymoDataset
from waymo_lidar.data.range_image import PointCloud, frame_to_points, range_image_to_points
from waymo_lidar.data.types import Laser

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
# Pinned stage-1 segments: tests must not scale with how much data is on disk (the full download is 1,000
# segments and would take hours here).
SEGMENTS = [("training", "10017090168044687777_6380_000_6400_000"),
            ("training", "10023947602400723454_1120_000_1140_000"),
            ("validation", "10203656353524179475_7625_000_7645_000")]
needs_data = pytest.mark.skipif(
    not all((ROOT / split / "lidar_box" / f"{name}.parquet").is_file() for split, name in SEGMENTS),
    reason=f"stage-1 segments not at {ROOT}")


@pytest.fixture(scope="module")
def counts():
    """Reference vs decoded counts for every 25th frame of the pinned segments."""
    ds = WaymoDataset(ROOT)
    rows = []
    for split, name in SEGMENTS:
        seg = ds.segment(split, name)
        boxes = Boxes.from_table(seg.boxes_table())
        for fr in seg.iter_range_images(returns=(1, 2), timestamps=seg.timestamps[::25]):
            b = boxes.at(fr.timestamp)
            pcs = frame_to_points(fr, seg.calibration, seg.pose_at(fr.timestamp))
            top = PointCloud.concat([pcs[(Laser.TOP, 1)], pcs[(Laser.TOP, 2)]])
            top_no_rs = PointCloud.concat([range_image_to_points(fr.images[(Laser.TOP, r)], seg.calibration[Laser.TOP])
                                           for r in (1, 2)])
            all_pts = PointCloud.concat(list(pcs.values()))
            rows.append(np.stack([
                b.num_top_points,
                points_in_boxes(top.xyz, b.center, b.size, b.heading),
                points_in_boxes(top_no_rs.xyz, b.center, b.size, b.heading),
                b.num_points,
                points_in_boxes(all_pts.xyz, b.center, b.size, b.heading),
                np.linalg.norm(b.speed[:, :2], axis=1),
            ], axis=1))
    return np.concatenate(rows)


@needs_data
@pytest.mark.parametrize("ref_col, ours_col", [(0, 1), (3, 4)], ids=["top", "all_lidars"])
def test_points_in_box_track_reference(counts, ref_col, ours_col):
    ref, ours = counts[:, ref_col], counts[:, ours_col]
    many = ref >= 50
    ratio = ours[many] / ref[many]
    assert np.corrcoef(ours, ref)[0, 1] > 0.999
    assert 0.9 < np.median(ratio) <= 1.0
    assert np.mean(ours > ref + 2) < 0.03


@needs_data
def test_rolling_shutter_helps_moving_objects(counts):
    ref, with_rs, without_rs, speed = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 5]
    moving = (speed > 5) & (ref >= 50)
    assert moving.sum() >= 10
    assert np.median(with_rs[moving] / ref[moving]) > np.median(without_rs[moving] / ref[moving])


def test_points_in_boxes_rotated():
    center, size, heading = np.array([[10.0, 0, 1]]), np.array([[4.0, 2, 2]]), np.array([np.pi / 2])
    # After a 90 degree turn the box is 2 m along x and 4 m along y.
    pts = np.array([[10, 1.9, 1], [10, -1.9, 1], [10.9, 0, 1], [11.5, 0, 1], [10, 0, 2.5]])
    assert points_in_boxes(pts, center, size, heading)[0] == 3


def test_box_corners_extent():
    c = box_corners(np.array([[0.0, 0, 0]]), np.array([[4.0, 2, 1]]), np.array([0.0]))[0]
    assert np.allclose(c.min(0), [-2, -1, -0.5]) and np.allclose(c.max(0), [2, 1, 0.5])
