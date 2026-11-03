"""CenterPoint training targets: per-class Gaussian heatmaps + regression at object centres.

    targets = TargetBuilder(cfg)
    ds = FrameStream(root, "training", cfg, mode="train", transform=targets)

Cells use the voxelizer's convention at stride s = cfg.target.out_stride (column = x, row = y):
    u = (x - x_min) / (vx * s), col = floor(u);   v = (y - y_min) / (vy * s), row = floor(v)
Regression per object: [u - col, v - row, z, log l, log w, log h, sin(heading), cos(heading)].
`reg_to_boxes` is the exact inverse and is shared with decoding. See docs/detector/targets.md.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

from .config import DetectorConfig

REG_FIELDS = ("dx", "dy", "z", "log_l", "log_w", "log_h", "sin", "cos")
TARGET_KEYS = ("heatmap", "target_ind", "target_mask", "target_cls", "target_reg")


def gaussian_radius(length: float, width: float, min_overlap: float) -> float:
    """CornerNet / CenterNet / CenterPoint radius rule (kept verbatim for comparability with CenterPoint).

    Sizes are in heatmap cells. Returns the smallest of the three cases' radii.
    """
    a1, b1 = 1.0, length + width
    c1 = width * length * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4.0, 2 * (length + width)
    c2 = (1 - min_overlap) * width * length
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (length + width)
    c3 = (min_overlap - 1) * width * length
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def gaussian_kernel(radius: int) -> np.ndarray:
    """(2r+1, 2r+1) Gaussian with sigma = (2r+1)/6, peak exactly 1 at the centre."""
    d = 2 * radius + 1
    sigma = d / 6
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    return g


def draw_gaussian(heatmap: np.ndarray, row: int, col: int, radius: int) -> None:
    """Elementwise max of a Gaussian into heatmap (H, W) around (row, col), clipped at the borders."""
    g = gaussian_kernel(radius)
    h, w = heatmap.shape
    left, right = min(col, radius), min(w - col, radius + 1)
    top, bottom = min(row, radius), min(h - row, radius + 1)
    window = heatmap[row - top:row + bottom, col - left:col + right]
    np.maximum(window, g[radius - top:radius + bottom, radius - left:radius + right], out=window)


def box_cells(boxes: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(u, v) float cell coordinates of box centres, their (row, col), and an in-grid mask."""
    g, s = cfg.grid, cfg.target.out_stride
    u = (boxes[:, 0].astype(np.float64) - g.x_range[0]) / (g.voxel_size[0] * s)
    v = (boxes[:, 1].astype(np.float64) - g.y_range[0]) / (g.voxel_size[1] * s)
    col, row = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    h, w = cfg.target_out_size
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    return np.stack([u, v], 1), row, col, inside


def reg_to_boxes(rows, cols, reg, cfg: DetectorConfig):
    """Inverse of the encoding: (N,) rows, (N,) cols, (N, 8) regression -> (N, 7) boxes.

    Works on NumPy arrays or torch tensors (decode uses it on the GPU).
    """
    g, s = cfg.grid, cfg.target.out_stride
    is_torch = isinstance(reg, torch.Tensor)
    exp, atan2, stack = (torch.exp, torch.atan2, torch.stack) if is_torch else (np.exp, np.arctan2, np.stack)
    x = g.x_range[0] + (cols + reg[:, 0]) * (g.voxel_size[0] * s)
    y = g.y_range[0] + (rows + reg[:, 1]) * (g.voxel_size[1] * s)
    size = exp(reg[:, 3:6])
    heading = atan2(reg[:, 6], reg[:, 7])
    return stack([x, y, reg[:, 2], size[:, 0], size[:, 1], size[:, 2], heading], 1)


class TargetBuilder:
    """Sample -> sample + CenterPoint targets (NumPy; runs in DataLoader workers)."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.num_classes = len(cfg.data.classes)
        self.h, self.w = cfg.target_out_size
        self.cell = (cfg.grid.voxel_size[0] * cfg.target.out_stride, cfg.grid.voxel_size[1] * cfg.target.out_stride)

    def radius(self, length: float, width: float) -> int:
        t = self.cfg.target
        r = gaussian_radius(length / self.cell[0], width / self.cell[1], t.gaussian_overlap)
        return max(t.min_radius, int(r))

    def __call__(self, sample: dict) -> dict:
        boxes, labels = sample["boxes"], sample["labels"]
        K = self.cfg.target.max_objects
        heatmap = np.zeros((self.num_classes, self.h, self.w), dtype=np.float32)
        ind = np.zeros(K, dtype=np.int64)
        mask = np.zeros(K, dtype=np.uint8)
        cls = np.zeros(K, dtype=np.int64)
        reg = np.zeros((K, len(REG_FIELDS)), dtype=np.float32)

        uv, row, col, inside = box_cells(boxes, self.cfg)
        keep = np.flatnonzero(inside)
        if len(keep) > K:
            warnings.warn(f"{sample.get('segment')}@{sample.get('timestamp')}: {len(keep)} objects > "
                          f"max_objects={K}; keeping the first {K}")
            keep = keep[:K]
        for k, i in enumerate(keep):
            b = boxes[i]
            draw_gaussian(heatmap[labels[i]], int(row[i]), int(col[i]), self.radius(float(b[3]), float(b[4])))
            ind[k] = row[i] * self.w + col[i]
            mask[k] = 1
            cls[k] = labels[i]
            reg[k] = (uv[i, 0] - col[i], uv[i, 1] - row[i], b[2],
                      math.log(b[3]), math.log(b[4]), math.log(b[5]), math.sin(b[6]), math.cos(b[6]))
        return {**sample, "heatmap": heatmap, "target_ind": ind, "target_mask": mask, "target_cls": cls,
                "target_reg": reg}
