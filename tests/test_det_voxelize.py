import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from waymo_lidar.data.dataset import WaymoDataset
from waymo_lidar.det.config import load_config
from waymo_lidar.det.data import collate, load_sample
from waymo_lidar.det.voxelize import (DENSITY_SCALE, Voxelizer, bev_cell_centers, channel_names,
                                      points_to_voxel_indices, voxel_centers)

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
SEG = "10023947602400723454_1120_000_1140_000"
needs_data = pytest.mark.skipif(not (ROOT / "training" / "lidar_box" / f"{SEG}.parquet").is_file(),
                                reason=f"stage-1 data not at {ROOT}")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

CFG = load_config()
G = CFG.grid
CFG_X = load_config(overrides=["grid.extra_channels=true"])


def pts(rows, b=0):
    """rows of (x, y, z[, intensity]) -> (N, 5) tensor with batch index b."""
    a = np.array(rows, dtype=np.float32)
    if a.shape[1] == 3:
        a = np.concatenate([a, np.zeros((len(a), 1), np.float32)], axis=1)
    return torch.from_numpy(np.concatenate([np.full((len(a), 1), b, np.float32), a], axis=1))


def numpy_reference(points: np.ndarray, grid) -> np.ndarray:
    """Straightforward loop-free NumPy voxelization of one sample's (N, 4) points -> (nz, ny, nx) bool."""
    lo = np.array([grid.x_range[0], grid.y_range[0], grid.z_range[0]])
    hi = np.array([grid.x_range[1], grid.y_range[1], grid.z_range[1]])
    xyz = points[:, :3].astype(np.float64)
    m = ((xyz >= lo) & (xyz < hi)).all(1)
    idx = np.floor((xyz[m] - lo) / np.array(grid.voxel_size)).astype(int)
    idx = np.minimum(idx, [grid.nx - 1, grid.ny - 1, grid.nz - 1])
    out = np.zeros((grid.nz, grid.ny, grid.nx), bool)
    out[idx[:, 2], idx[:, 1], idx[:, 0]] = True
    return out


# --- index convention ----------------------------------------------------------------------------


def test_exact_cells_and_bounds():
    x0, y0, z0 = G.x_range[0], G.y_range[0], G.z_range[0]
    x1, y1, z1 = G.x_range[1], G.y_range[1], G.z_range[1]
    xyz = torch.tensor([
        [x0, y0, z0],                          # first voxel
        [x1 - 1e-3, y1 - 1e-3, z1 - 1e-3],     # last voxel
        [x1, 0, 0], [0, y1, 0], [0, 0, z1],    # on an upper bound -> dropped
        [x0 - 1e-3, 0, 0],                     # below a lower bound -> dropped
        [0.05, -0.05, 0.1],                    # near the origin
    ], dtype=torch.float64)  # the convention is defined exactly; float32 can't represent 75.2
    idx, mask = points_to_voxel_indices(xyz, G)
    assert mask.tolist() == [True, True, False, False, False, False, True]
    assert idx[0].tolist() == [0, 0, 0]
    assert idx[1].tolist() == [G.nx - 1, G.ny - 1, G.nz - 1]
    # x = 0.05 is cell 752 (75.2 / 0.1 = 752 cells below 0); y = -0.05 is cell 751; z = 0.1 is slice 8.
    assert idx[6].tolist() == [752, 751, 8]


@pytest.mark.parametrize("device", DEVICES)
def test_single_point_lights_one_cell(device):
    vox = Voxelizer(G)
    bev = vox(pts([[10.04, -3.27, 1.3]]).to(device), batch_size=1)
    ix, iy, iz = int((10.04 + 75.2) / 0.1), int((-3.27 + 75.2) / 0.1), int((1.3 + 2.0) / 0.25)
    assert bev.shape == (1, 24, 1504, 1504) and bev.dtype == torch.float32
    assert bev.sum().item() == 1 and bev[0, iz, iy, ix].item() == 1


@pytest.mark.parametrize("device", DEVICES)
def test_batches_do_not_mix_and_duplicates_are_one(device):
    p = torch.cat([pts([[1, 1, 0]] * 5, b=0), pts([[-20, 30, 2]], b=1)]).to(device)
    bev = Voxelizer(G)(p, batch_size=2)
    assert bev[0].sum().item() == 1 and bev[1].sum().item() == 1  # 5 points in one voxel -> 1
    idx, _ = points_to_voxel_indices(torch.tensor([[-20.0, 30, 2]]), G)
    ix, iy, iz = idx[0].tolist()
    assert bev[1, iz, iy, ix].item() == 1 and bev[0, iz, iy, ix].item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_extra_channels(device):
    # 3 points in one BEV cell at different heights + 1 point alone; intensities 0.2, 2.0, 0.5 and 0.1.
    p = pts([[5.01, 5.01, 0.1, 0.2], [5.02, 5.03, 1.1, 2.0], [5.05, 5.05, 3.0, 0.5], [-7, 2, 0.5, 0.1]]).to(device)
    bev = Voxelizer(CFG_X.grid)(p, batch_size=1)
    assert bev.shape[1] == 26 and channel_names(CFG_X.grid)[-2:] == ["density", "intensity"]
    idx, _ = points_to_voxel_indices(p[:, 1:4].cpu(), CFG_X.grid)
    (ix, iy, _), (jx, jy, _) = idx[0].tolist(), idx[3].tolist()
    assert bev[0, 24, iy, ix].item() == pytest.approx(np.log1p(3) / DENSITY_SCALE, rel=1e-6)
    assert bev[0, 25, iy, ix].item() == pytest.approx(np.tanh(2.0), rel=1e-6)  # max over the cell
    assert bev[0, 24, jy, jx].item() == pytest.approx(np.log1p(1) / DENSITY_SCALE, rel=1e-6)
    assert bev[0, 25, jy, jx].item() == pytest.approx(np.tanh(0.1), rel=1e-6)
    assert bev[0, :24, iy, ix].sum().item() == 3  # three different slices occupied
    assert torch.count_nonzero(bev[0, 24]).item() == 2 and torch.count_nonzero(bev[0, 25]).item() == 2  # empty cells 0


def test_no_extras_means_nz_channels():
    assert Voxelizer(G)(pts([[0, 0, 0]]), 1).shape[1] == G.nz == 24


def test_centers_roundtrip():
    idx = torch.tensor([[0, 0, 0], [G.nx - 1, G.ny - 1, G.nz - 1], [752, 751, 8]])
    c = voxel_centers(idx, G)
    assert torch.allclose(c[0], torch.tensor([-75.15, -75.15, -1.875]), atol=1e-5)
    assert torch.allclose(c[1], torch.tensor([75.15, 75.15, 3.875]), atol=1e-5)
    back, mask = points_to_voxel_indices(c, G)
    assert mask.all() and torch.equal(back, idx)
    cy, cx = bev_cell_centers(G, stride=4)
    assert len(cy) == len(cx) == 376 and cx[0].item() == pytest.approx(-75.0) and cx[-1].item() == pytest.approx(75.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtypes_same_occupancy(dtype):
    rng = np.random.default_rng(0)
    p = pts(rng.uniform([-80, -80, -3, 0], [80, 80, 5, 3], (20000, 4)))
    ref = Voxelizer(CFG_X.grid)(p, 1)
    got = Voxelizer(CFG_X.grid)(p, 1, dtype=dtype)
    assert got.dtype == dtype and torch.equal(got[:, :24].float(), ref[:, :24])


# --- real data -----------------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("device", DEVICES)
def test_real_frame_matches_numpy_reference(device):
    ts = int(WaymoDataset(ROOT).segment("training", SEG).timestamps[60])
    s = load_sample(ROOT, "training", SEG, ts, CFG)
    bev = Voxelizer(G)(collate([s])["points"].to(device), 1).cpu()[0].bool()
    ref = numpy_reference(s["points"], G)
    assert bev.shape == ref.shape and torch.equal(bev, torch.from_numpy(ref))
    assert 30_000 < bev.sum().item() < 300_000
    # every occupied voxel's centre is within half a voxel of some point
    occ = torch.nonzero(bev)[:, [2, 1, 0]]  # (iz, iy, ix) -> (ix, iy, iz)
    centers = voxel_centers(occ, G)
    idx, mask = points_to_voxel_indices(torch.from_numpy(s["points"][:, :3]), G)
    key = lambda i: (i[:, 2] * G.ny + i[:, 1]) * G.nx + i[:, 0]  # noqa: E731
    assert torch.isin(key(occ), key(idx[mask])).all()
    assert (centers[:, 2] >= G.z_range[0]).all() and (centers[:, 2] < G.z_range[1]).all()


@needs_data
def test_viewer_saves_rrd(tmp_path):
    out = tmp_path / "frame.rrd"
    r = subprocess.run([sys.executable, "-m", "waymo_lidar.det.viz_voxels", str(ROOT), "--split", "training",
                        "--segment", SEG, "--frame", "0", "--save", str(out)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 100_000
    assert "occupied voxels" in r.stdout


def test_split_for_rerun_respects_instance_cap():
    from waymo_lidar.det.viz_voxels import MAX_INSTANCES_PER_ENTITY, split_for_rerun
    for n in (0, 1, 49_999, 50_000, 50_001, 123_456):
        parts = split_for_rerun(n)
        covered = [i for p in parts for i in range(n)[p]]
        assert covered == list(range(n)) and all(p.stop - p.start <= MAX_INSTANCES_PER_ENTITY for p in parts)
    assert MAX_INSTANCES_PER_ENTITY < 60_000  # Rerun's default per-entity cap
