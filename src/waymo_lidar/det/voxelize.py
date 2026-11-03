"""Points -> BEV occupancy image: z slices as channels, (B, C, ny, nx).

    vox = Voxelizer(cfg.grid)
    bev = vox(batch["points"].cuda(), batch["batch_size"], dtype=torch.bfloat16)

Index convention (shared with targets and decode):
    ix = floor((x - x_min) / vx)  -> column     iy = floor((y - y_min) / vy) -> row
    iz = floor((z - z_min) / vz)  -> channel    (half-open ranges; points outside are dropped)
Row 0 is y = y_min (the car's right), column 0 is x = x_min (behind the car).
See docs/detector/voxelize.md.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import GridConfig

# log1p(points per BEV cell) is divided by this. log1p(32): cells with <= 32 points map to <= 1, which
# covers ~99% of occupied cells (measured: p99 = 30 points per cell).
DENSITY_SCALE = math.log1p(32)


def channel_names(grid: GridConfig) -> list[str]:
    names = [f"z{i}:{grid.z_range[0] + i * grid.voxel_size[2]:+.2f}m" for i in range(grid.nz)]
    return names + (["density", "intensity"] if grid.extra_channels else [])


def points_to_voxel_indices(xyz: torch.Tensor, grid: GridConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 3) metres -> ((N, 3) int64 [ix, iy, iz], (N,) bool in-range mask). Out-of-range rows are garbage.

    Computed in float64: in float32, points within ~1e-5 of a voxel boundary can floor into the
    neighbouring cell (~8 of 160k points per frame), and every component must agree on the cell.
    """
    xyz = xyz.double()
    lo = torch.tensor([grid.x_range[0], grid.y_range[0], grid.z_range[0]], dtype=torch.float64, device=xyz.device)
    hi = torch.tensor([grid.x_range[1], grid.y_range[1], grid.z_range[1]], dtype=torch.float64, device=xyz.device)
    size = torch.tensor(grid.voxel_size, dtype=torch.float64, device=xyz.device)
    mask = ((xyz >= lo) & (xyz < hi)).all(dim=1)
    idx = torch.floor((xyz - lo) / size).long()
    # Float rounding can put a point just below the upper bound at index n; clamp it back.
    n = torch.tensor([grid.nx, grid.ny, grid.nz], device=xyz.device)
    idx = torch.minimum(idx.clamp_min(0), n - 1)
    return idx, mask


def voxel_centers(idx: torch.Tensor, grid: GridConfig) -> torch.Tensor:
    """(N, 3) [ix, iy, iz] -> (N, 3) float32 voxel centres in metres."""
    lo = torch.tensor([grid.x_range[0], grid.y_range[0], grid.z_range[0]], dtype=torch.float64, device=idx.device)
    size = torch.tensor(grid.voxel_size, dtype=torch.float64, device=idx.device)
    return (lo + (idx.double() + 0.5) * size).float()


def bev_cell_centers(grid: GridConfig, stride: int = 1, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(ny // stride,) y and (nx // stride,) x centres of BEV cells, stride voxels per cell."""
    cy = grid.y_range[0] + (torch.arange(grid.ny // stride, device=device, dtype=torch.float64) + 0.5) * grid.voxel_size[1] * stride
    cx = grid.x_range[0] + (torch.arange(grid.nx // stride, device=device, dtype=torch.float64) + 0.5) * grid.voxel_size[0] * stride
    return cy.float(), cx.float()


class Voxelizer(nn.Module):
    """Binary occupancy per (z slice, y, x); optional density and max-intensity BEV channels. No parameters."""

    def __init__(self, grid: GridConfig):
        super().__init__()
        self.grid = grid
        self.nx, self.ny, self.nz, self.channels = grid.nx, grid.ny, grid.nz, grid.in_channels

    @torch.no_grad()
    def forward(self, points: torch.Tensor, batch_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """points (N, 5) [batch, x, y, z, intensity] -> (batch_size, C, ny, nx)."""
        nx, ny, nz, C = self.nx, self.ny, self.nz, self.channels
        out = torch.zeros(batch_size * C * ny * nx, dtype=dtype, device=points.device)
        idx, mask = points_to_voxel_indices(points[:, 1:4], self.grid)
        b = points[mask, 0].long()
        ix, iy, iz = idx[mask, 0], idx[mask, 1], idx[mask, 2]

        out[((b * C + iz) * ny + iy) * nx + ix] = 1  # duplicates write the same 1

        if self.grid.extra_channels:
            cell = (b * ny + iy) * nx + ix  # BEV cell over the batch
            counts = torch.zeros(batch_size * ny * nx, dtype=torch.float32, device=points.device)
            counts.scatter_add_(0, cell, torch.ones_like(cell, dtype=torch.float32))
            inten = torch.zeros(batch_size * ny * nx, dtype=torch.float32, device=points.device)
            inten.scatter_reduce_(0, cell, torch.tanh(points[mask, 4].float()), reduce="amax", include_self=False)
            out = out.view(batch_size, C, ny * nx)
            out[:, nz] = (torch.log1p(counts) / DENSITY_SCALE).view(batch_size, ny * nx).to(dtype)
            out[:, nz + 1] = inten.view(batch_size, ny * nx).to(dtype)
        return out.view(batch_size, C, ny, nx)
