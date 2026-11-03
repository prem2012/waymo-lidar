"""Target encoding tests (pinned stage-1 segments only)."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from waymo_lidar.data.dataset import WaymoDataset
from waymo_lidar.det.config import load_config
from waymo_lidar.det.data import FrameStream, load_sample, make_loader, wrap_angle
from waymo_lidar.det.targets import (REG_FIELDS, TargetBuilder, box_cells, draw_gaussian, gaussian_kernel,
                                     gaussian_radius, reg_to_boxes)

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
TRAIN = ["10017090168044687777_6380_000_6400_000", "10023947602400723454_1120_000_1140_000"]
needs_data = pytest.mark.skipif(not all((ROOT / "training" / "lidar_box" / f"{s}.parquet").is_file() for s in TRAIN),
                                reason=f"stage-1 segments not at {ROOT}")
CFG = load_config(overrides=["data.num_workers=0"])
TB = TargetBuilder(CFG)


def sample_with(boxes, labels):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 7)
    return {"points": np.zeros((0, 4), np.float32), "boxes": boxes, "labels": np.asarray(labels, np.int64),
            "num_points": np.ones(len(boxes), np.int32), "segment": "synthetic", "timestamp": 0, "split": "x"}


def check_roundtrip(sample, out):
    m = out["target_mask"].astype(bool)
    rows, cols = np.divmod(out["target_ind"][m], 376)
    rebuilt = reg_to_boxes(rows, cols, out["target_reg"][m].astype(np.float64), CFG)
    _, _, _, inside = box_cells(sample["boxes"], CFG)
    gt = sample["boxes"][inside].astype(np.float64)
    assert len(rebuilt) == len(gt)
    assert np.abs(rebuilt[:, :3] - gt[:, :3]).max() < 1e-4                       # centre, metres
    assert np.abs(rebuilt[:, 3:6] / gt[:, 3:6] - 1).max() < 1e-5                 # size, relative
    assert np.abs(wrap_angle(rebuilt[:, 6] - gt[:, 6])).max() < 1e-5             # heading, wrapped
    assert np.array_equal(out["target_cls"][m], sample["labels"][inside])


# --- encoding rules --------------------------------------------------------------------------------


def test_cell_and_offset_of_known_centre():
    # x = 10.3 -> u = (10.3 + 75.2) / 0.4 = 213.75 -> col 213, dx 0.75; y = -2.1 -> v = 182.75 -> row 182, dy 0.75
    out = TB(sample_with([[10.3, -2.1, 0.8, 4.5, 2.0, 1.6, 0.3]], [0]))
    assert out["target_mask"].sum() == 1
    assert out["target_ind"][0] == 182 * 376 + 213
    assert out["target_reg"][0, :2] == pytest.approx([0.75, 0.75], abs=1e-5)
    assert out["target_reg"][0, 2:].tolist() == pytest.approx([0.8, math.log(4.5), math.log(2.0), math.log(1.6),
                                                               math.sin(0.3), math.cos(0.3)], abs=1e-6)
    assert out["heatmap"][0, 182, 213] == 1.0 and out["heatmap"][0].max() == 1.0
    assert out["heatmap"][1:].max() == 0.0  # no leakage into other classes


def test_grid_corner_and_out_of_grid():
    # The grid corner maps to cell (0, 0), offset ~0 (float32(-75.2) is 3e-6 m inside the grid).
    # Just past a border -> the next cell. Outside -> skipped. (Not 75.2 itself: float32(75.2) = 75.19999695
    # is inside the grid, consistently with the voxelizer.)
    out = TB(sample_with([[-75.2, -75.2, 0, 1, 1, 1, 0], [-74.79, -74.79, 0, 1, 1, 1, 0],
                          [75.25, 0, 0, 1, 1, 1, 0], [0, -75.3, 0, 1, 1, 1, 0]], [1, 1, 1, 1]))
    m = out["target_mask"].astype(bool)
    assert m.sum() == 2
    rows, cols = np.divmod(out["target_ind"][m], 376)
    assert (rows[0], cols[0]) == (0, 0) and (rows[1], cols[1]) == (1, 1)
    assert out["target_reg"][m][0, :2] == pytest.approx([0, 0], abs=1e-4)
    assert out["target_reg"][m][1, :2] == pytest.approx([0.025, 0.025], abs=1e-4)


def test_gaussian_kernel_and_radius():
    for r in (2, 3, 6):
        g = gaussian_kernel(r)
        assert g.shape == (2 * r + 1, 2 * r + 1) and g[r, r] == 1.0 and g.max() == 1.0
        assert np.allclose(g, g.T) and np.allclose(g, g[::-1, ::-1])
    assert gaussian_radius(11, 5, 0.1) > gaussian_radius(2, 2, 0.1)
    assert TB.radius(0.8, 0.8) == CFG.target.min_radius  # a pedestrian gets the minimum
    assert TB.radius(4.8, 2.1) > CFG.target.min_radius   # a car is larger


def test_overlapping_gaussians_take_max_and_clip_at_border():
    hm = np.zeros((20, 20), np.float32)
    draw_gaussian(hm, 10, 10, 3)
    draw_gaussian(hm, 10, 12, 3)
    assert hm.max() == 1.0 and hm[10, 11] < 1.0  # max, not sum
    edge = np.zeros((20, 20), np.float32)
    draw_gaussian(edge, 0, 19, 4)  # corner: clipped, no error
    assert edge[0, 19] == 1.0


def test_padding_and_shapes():
    out = TB(sample_with(np.zeros((0, 7)), []))
    assert out["heatmap"].shape == (3, 376, 376) and out["heatmap"].dtype == np.float32
    assert out["target_ind"].shape == (500,) and out["target_reg"].shape == (500, len(REG_FIELDS))
    assert out["target_mask"].sum() == 0 and out["heatmap"].max() == 0


def test_too_many_objects_warns_and_truncates():
    small = load_config(overrides=["target.max_objects=2"])
    boxes = [[i * 3.0, 0, 0, 1, 1, 1, 0] for i in range(3)]
    with pytest.warns(UserWarning, match="max_objects"):
        out = TargetBuilder(small)(sample_with(boxes, [0, 0, 0]))
    assert out["target_mask"].sum() == 2


def test_reg_to_boxes_numpy_and_torch_agree():
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 376, 50), rng.integers(0, 376, 50)
    reg = rng.uniform(-1, 1, (50, 8))
    a = reg_to_boxes(rows, cols, reg, CFG)
    b = reg_to_boxes(torch.from_numpy(rows), torch.from_numpy(cols), torch.from_numpy(reg), CFG).numpy()
    assert np.allclose(a, b)


# --- real frames -----------------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("mode, epoch", [("eval", 0), ("train", 0), ("train", 5)])
def test_roundtrip_on_real_frames(mode, epoch):
    ts = WaymoDataset(ROOT).segment("training", TRAIN[1]).timestamps
    for t in ts[[0, 120]]:
        s = load_sample(ROOT, "training", TRAIN[1], int(t), CFG, mode=mode, epoch=epoch)
        check_roundtrip(s, TB(s))


@needs_data
def test_targets_through_the_stream():
    cfg = load_config(overrides=["data.num_workers=0", "data.frame_interval_eval=50"])
    ds = FrameStream(ROOT, "training", cfg, mode="eval", segments=TRAIN[:1], transform=TargetBuilder(cfg))
    batch = next(iter(make_loader(ds, cfg, batch_size=2, num_workers=0)))
    assert batch["heatmap"].shape == (2, 3, 376, 376) and batch["heatmap"].dtype == torch.float32
    assert batch["target_ind"].shape == (2, 500) and batch["target_reg"].shape == (2, 500, 8)
    assert batch["target_mask"].shape == (2, 500) and batch["target_cls"].shape == (2, 500)
    # every peak of 1.0 in the heatmap is a target centre
    for b in range(2):
        m = batch["target_mask"][b].bool()
        peaks = {tuple(p) for p in torch.nonzero(batch["heatmap"][b] == 1.0).tolist()}
        centres = {(int(c), int(i) // 376, int(i) % 376) for c, i in zip(batch["target_cls"][b][m], batch["target_ind"][b][m])}
        assert centres <= peaks


@needs_data
def test_viewer_saves_targets(tmp_path):
    out = tmp_path / "targets.rrd"
    r = subprocess.run([sys.executable, "-m", "waymo_lidar.det.viz_voxels", str(ROOT), "--split", "training",
                        "--segment", TRAIN[1], "--frame", "10", "--targets", "--save", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 100_000
