"""See the voxelizer's output (and, with --targets, the training targets) in Rerun: one command, one window.

    uv run python -m waymo_lidar.det.viz_voxels ~/dev/data/waymo_v2 --split training --segment <name> --frame 50

Each frame is loaded with the real data stream and voxelized with the real Voxelizer; the occupied voxels
drawn are taken from the dense model input itself. See docs/detector/voxelize.md#viewing-voxels.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from ..data.boxes import box_corners
from ..data.dataset import WaymoDataset
from ..data.types import BOX_TYPE_COLORS
from ..viz.colors import intensity_colors, turbo
from .config import GridConfig, load_config
from .data import collate, complete_segments, load_sample
from .targets import TargetBuilder, reg_to_boxes
from .voxelize import Voxelizer, bev_cell_centers, voxel_centers


def parse_frames(frame: int | None, frames: str | None, n: int) -> list[int]:
    if frames:
        parts = [int(p) if p else None for p in frames.split(":")]
        out = list(range(n))[slice(*parts)]
    else:
        out = [frame or 0]
    bad = [i for i in out if not 0 <= i < n]
    if bad or not out:
        raise SystemExit(f"frame index out of range 0..{n - 1}: {bad or frames}")
    return out


def grid_outline(grid: GridConfig) -> list[np.ndarray]:
    (x0, x1), (y0, y1), (z0, z1) = grid.x_range, grid.y_range, grid.z_range
    bottom = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z0)]
    top = [(x, y, z1) for x, y, _ in bottom]
    verticals = [np.array([(x, y, z0), (x, y, z1)]) for x, y, _ in bottom[:4]]
    return [np.array(bottom), np.array(top), *verticals]


def blueprint(multi_frame: bool, target_classes: tuple[str, ...] = ()) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(origin="world", name="Voxels",
                               overrides={"world/points": rrb.EntityBehavior(visible=False)})
    main = view3d
    if target_classes:  # BEV heatmaps, one tab per class, next to the 3D view
        bev = rrb.Tabs(*[rrb.Spatial2DView(origin=f"bev/{c}", name=f"BEV {c}") for c in target_classes],
                       name="BEV targets")
        main = rrb.Horizontal(view3d, bev, column_shares=[3, 2])
    return rrb.Blueprint(main, rrb.TimePanel(state="expanded" if multi_frame else "collapsed"), collapse_panels=False)


HEAT_TILE_MIN = 0.05  # heatmap cells drawn as 3D ground tiles


def log_targets(rec: rr.RecordingStream, sample: dict, targets: dict, cfg) -> None:
    """Heatmaps as BEV images (+ GT outlines and decoded centres in cell coordinates) and as 3D ground tiles."""
    g, s = cfg.grid, cfg.target.out_stride
    cell = (g.voxel_size[0] * s, g.voxel_size[1] * s)
    hm, m = targets["heatmap"], targets["target_mask"].astype(bool)
    rows, cols = np.divmod(targets["target_ind"][m], hm.shape[2])
    reg, tcls = targets["target_reg"][m].astype(np.float64), targets["target_cls"][m]
    cy, cx = (t.numpy() for t in bev_cell_centers(g, stride=s))
    for ch, name in enumerate(cfg.data.classes):
        h = hm[ch]
        rgb = turbo(h.ravel()).reshape(*h.shape, 3)
        rgb[h == 0] = 0
        rec.log(f"bev/{name}/heatmap", rr.Image(rgb))
        b = sample["boxes"][sample["labels"] == ch]
        if len(b):  # GT outlines in cell coordinates: pixel i spans [i, i+1), matching col = floor(u)
            c = box_corners(b[:, :3].astype(float), b[:, 3:6].astype(float), b[:, 6].astype(float))[:, :4, :2]
            uv = np.stack([(c[..., 0] - g.x_range[0]) / cell[0], (c[..., 1] - g.y_range[0]) / cell[1]], -1)
            rec.log(f"bev/{name}/gt", rr.LineStrips2D(np.concatenate([uv, uv[:, :1]], 1), colors=(255, 255, 255)))
        else:
            rec.log(f"bev/{name}/gt", rr.Clear(recursive=False))
        k = tcls == ch
        rec.log(f"bev/{name}/centres", rr.Points2D(np.stack([cols[k] + reg[k, 0], rows[k] + reg[k, 1]], 1),
                                                   colors=(255, 0, 255), radii=0.6))
        r_idx, c_idx = np.nonzero(h > HEAT_TILE_MIN)
        if len(r_idx):
            centers = np.stack([cx[c_idx], cy[r_idx], np.full(len(r_idx), g.z_range[0])], 1)
            rec.log(f"world/targets/{name}", rr.Boxes3D(
                centers=centers, half_sizes=np.broadcast_to([cell[0] / 2, cell[1] / 2, 0.01], centers.shape),
                colors=turbo(h[r_idx, c_idx]), fill_mode="solid"))
        else:
            rec.log(f"world/targets/{name}", rr.Clear(recursive=False))
    decoded = reg_to_boxes(rows, cols, reg, cfg)
    hd = decoded[:, 6]
    quats = np.stack([np.zeros_like(hd), np.zeros_like(hd), np.sin(hd / 2), np.cos(hd / 2)], 1)
    rec.log("world/targets/decoded", rr.Boxes3D(centers=decoded[:, :3], half_sizes=decoded[:, 3:6] / 2,
                                                quaternions=quats, colors=(255, 0, 255)))


# Rerun draws at most 60,000 boxes per entity by default (a viewer setting); stay below it.
MAX_INSTANCES_PER_ENTITY = 50_000


def split_for_rerun(n: int, limit: int = MAX_INSTANCES_PER_ENTITY) -> list[slice]:
    """Slices covering range(n) with at most `limit` items each (at least one slice)."""
    return [slice(i, min(i + limit, n)) for i in range(0, max(n, 1), limit)]


def log_frame(rec: rr.RecordingStream, sample: dict, bev: torch.Tensor, grid: GridConfig, classes: tuple[str, ...],
              class_ids: tuple[int, ...], as_points: bool, parts_used: dict[int, int]) -> int:
    """Log one voxelized frame; returns the number of occupied voxels.

    Voxels go to one entity per height slice (world/voxels/zNN), which keeps each entity under Rerun's
    per-entity instance cap and lets each model input channel be toggled in the viewer. `parts_used`
    remembers how many parts each slice had, so stale parts are cleared on later frames.
    """
    nz = grid.nz
    occ = torch.nonzero(bev[0, :nz])  # (K, 3) as (iz, iy, ix): exactly the model input's non-zeros
    idx = occ[:, [2, 1, 0]]
    centers = voxel_centers(idx, grid).cpu().numpy()
    idx_np = idx.cpu().numpy()
    extras = {}
    if grid.extra_channels:
        iy, ix = occ[:, 1], occ[:, 2]
        extras = {"density": bev[0, nz, iy, ix].float().cpu().numpy(),
                  "intensity": bev[0, nz + 1, iy, ix].float().cpu().numpy()}
    half = np.array(grid.voxel_size, dtype=np.float32) / 2
    for z in range(nz):
        sel = np.flatnonzero(idx_np[:, 2] == z)
        parts = split_for_rerun(len(sel))
        if len(parts) > 1:
            rec.log(f"world/voxels/z{z:02d}", rr.Clear(recursive=False))  # data moves to /partK children
        for k, part in enumerate(parts):
            path = f"world/voxels/z{z:02d}" + (f"/part{k}" if len(parts) > 1 or parts_used.get(z, 1) > 1 else "")
            ids = sel[part]
            if len(ids) == 0:
                rec.log(path, rr.Clear(recursive=False))
                continue
            color = turbo(np.array([z / max(nz - 1, 1)]))[0]
            if as_points:
                rec.log(path, rr.Points3D(centers[ids], colors=color, radii=min(grid.voxel_size) / 2))
            else:
                rec.log(path, rr.Boxes3D(centers=centers[ids], half_sizes=np.broadcast_to(half, (len(ids), 3)),
                                         colors=color, fill_mode="solid"))
            rec.log(path, rr.AnyValues(ix=idx_np[ids, 0], iy=idx_np[ids, 1], iz=idx_np[ids, 2],
                                       **{k_: v[ids] for k_, v in extras.items()}))
        for k in range(len(parts), parts_used.get(z, 0)):
            rec.log(f"world/voxels/z{z:02d}/part{k}", rr.Clear(recursive=False))
        parts_used[z] = max(parts_used.get(z, 0), len(parts))

    p = sample["points"]
    rec.log("world/points", rr.Points3D(p[:, :3], colors=intensity_colors(p[:, 3]), radii=rr.Radius.ui_points(1.5)))
    b, labels = sample["boxes"], sample["labels"]
    for ch, name in enumerate(classes):
        m = labels == ch
        if not m.any():
            rec.log(f"world/boxes/{name}", rr.Clear(recursive=False))
            continue
        h = b[m, 6]
        quats = np.stack([np.zeros_like(h), np.zeros_like(h), np.sin(h / 2), np.cos(h / 2)], axis=1)
        rec.log(f"world/boxes/{name}", rr.Boxes3D(centers=b[m, :3], half_sizes=b[m, 3:6] / 2, quaternions=quats,
                                                  colors=BOX_TYPE_COLORS[class_ids[ch]]))
    return len(occ)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="View voxelized frames in Rerun")
    p.add_argument("root", type=Path)
    p.add_argument("--split", default="training")
    p.add_argument("--segment", help="default: first complete segment of the split")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--frame", type=int, help="frame index (default 0)")
    g.add_argument("--frames", help="range start:stop:step, e.g. 0:200:10")
    p.add_argument("--augment", action="store_true", help="show the training-time augmented sample")
    p.add_argument("--epoch", type=int, default=0, help="epoch for --augment")
    p.add_argument("--config", type=Path)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    p.add_argument("--as-points", action="store_true", help="draw voxels as points instead of boxes")
    p.add_argument("--targets", action="store_true", help="also show CenterPoint training targets")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--save", type=Path, help="write an .rrd instead of opening a window")
    out.add_argument("--web", action="store_true", help="serve in the browser instead of the desktop viewer")
    p.add_argument("--web-port", type=int, default=9090, help="web viewer port for --web (default 9090)")
    p.add_argument("--no-browser", action="store_true", help="with --web: print the URL instead of opening a browser")
    a = p.parse_args(argv)

    cfg = load_config(a.config, a.set)
    ds = WaymoDataset(a.root)
    if a.segment is None:
        complete, _ = complete_segments(ds, a.split)
        if not complete:
            raise SystemExit(f"no complete segments in {a.split}")
        a.segment = complete[0]
    seg = ds.segment(a.split, a.segment)
    ts_all = seg.timestamps
    indices = parse_frames(a.frame, a.frames, len(ts_all))

    rec = rr.RecordingStream("waymo_lidar_voxels", recording_id=f"{a.segment}-{time.time_ns()}")
    bp = blueprint(len(indices) > 1, cfg.data.classes if a.targets else ())
    if a.save:
        rec.save(a.save, default_blueprint=bp)
    elif a.web:
        url = rec.serve_grpc(default_blueprint=bp)
        rr.serve_web_viewer(web_port=a.web_port, open_browser=not a.no_browser, connect_to=url)
        print(f"web viewer: http://localhost:{a.web_port}/?url={quote(url, safe='')}")
    else:
        rec.spawn(default_blueprint=bp)
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("world/grid", rr.LineStrips3D(grid_outline(cfg.grid), colors=(120, 120, 120), radii=rr.Radius.ui_points(1)),
            static=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vox = Voxelizer(cfg.grid).to(device)
    vox(torch.zeros((1, 5), device=device), 1)  # warm up CUDA so the printed times are steady-state
    mode = "train" if a.augment else "eval"
    parts_used: dict[int, int] = {}
    builder = TargetBuilder(cfg) if a.targets else None
    print(f"{a.split}/{a.segment}: {len(indices)} frame(s), {mode} mode, {cfg.grid.in_channels} channels, {device}")
    for i in indices:
        ts = int(ts_all[i])
        sample = load_sample(a.root, a.split, a.segment, ts, cfg, mode=mode, epoch=a.epoch)
        points = collate([sample])["points"].to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        bev = vox(points, 1)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rec.set_time("frame", sequence=i)
        rec.set_time("frame_time", timestamp=np.datetime64(ts, "us"))
        n = log_frame(rec, sample, bev, cfg.grid, cfg.data.classes, cfg.data.class_ids, a.as_points, parts_used)
        if builder:
            log_targets(rec, sample, builder(sample), cfg)
        print(f"  frame {i}: {n:,} occupied voxels, {len(sample['points']):,} points, {len(sample['boxes'])} boxes, "
              f"voxelize {1000 * dt:.1f} ms")

    rec.flush()
    if a.web:
        print("Serving; press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    elif a.save:
        rec.disconnect()
        print(f"saved {a.save}; view with: uv run rerun {a.save}")


if __name__ == "__main__":
    main(sys.argv[1:])
