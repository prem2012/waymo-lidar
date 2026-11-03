"""CenterPoint loss: Gaussian focal loss on heatmaps + masked L1 / smooth L1 regression at object centres.

    loss_fn = CenterPointLoss(cfg)
    out = loss_fn(model(bev), batch)
    out.total.backward()
    out.scalars()   # {"loss/total": ..., "loss/heatmap/vehicle": ..., "loss/reg/log_l": ..., ...} for W&B

Always computed in float32. See docs/detector/loss.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import DetectorConfig
from .targets import REG_FIELDS

DEFAULT_SMOOTH_L1_BETA = 1.0 / 9.0  # Faster R-CNN


@dataclass
class LossOutput:
    total: torch.Tensor  # scalar with gradient
    logs: dict[str, torch.Tensor]  # detached scalars, named for logging (still on the device)

    def scalars(self) -> dict[str, float]:
        """Logged values as floats, in one device->host transfer. Call only when logging: it syncs."""
        values = torch.stack([v.float() for v in self.logs.values()]).cpu().tolist()
        return dict(zip(self.logs.keys(), values))


def focal_loss(logits: torch.Tensor, gt: torch.Tensor, alpha: float, beta: float
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaussian focal loss (CornerNet / CenterNet / CenterPoint).

    Returns (total, per-class (C,), number of positives). Both losses are normalised by the batch-wide
    number of positives (cells where gt == 1), so the per-class values sum to the total.
    log p and log(1 - p) use logsigmoid: exact and stable, with no clamping of p.
    """
    x, gt = logits.float(), gt.float()
    p = torch.sigmoid(x)
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    pos_term = -(1 - p).pow(alpha) * F.logsigmoid(x) * pos
    neg_term = -(1 - gt).pow(beta) * p.pow(alpha) * F.logsigmoid(-x) * neg
    num_pos = pos.sum()
    per_class = (pos_term + neg_term).sum(dim=(0, 2, 3)) / num_pos.clamp_min(1.0)
    return per_class.sum(), per_class, num_pos


def gather_reg(reg: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """(B, F, H, W) predictions at flat indices (B, K) -> (B, K, F)."""
    b, f, h, w = reg.shape
    flat = reg.float().reshape(b, f, h * w)
    return flat.gather(2, ind.unsqueeze(1).expand(b, f, ind.shape[1])).transpose(1, 2)


def reg_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, kind: str = "l1",
             beta: float = DEFAULT_SMOOTH_L1_BETA) -> torch.Tensor:
    """Masked element loss per field: (B, K, F) -> (F,), summed over objects / number of objects.

    Same reduction as CenterPoint's RegLoss. Padded slots (mask 0) contribute nothing.
    """
    m = mask.float().unsqueeze(-1)
    target = target.float()
    if kind == "l1":
        elem = (pred - target).abs()
    elif kind == "smooth_l1":
        elem = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    else:
        raise ValueError(f"unknown regression loss {kind!r}")
    return (elem * m).sum(dim=(0, 1)) / (m.sum() + 1e-4)


class CenterPointLoss(nn.Module):
    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        lc = cfg.loss
        self.alpha, self.beta, self.reg_weight = lc.focal_alpha, lc.focal_beta, lc.reg_weight
        self.reg_kind = lc.reg_loss
        self.smooth_beta = lc.smooth_l1_beta if lc.smooth_l1_beta is not None else DEFAULT_SMOOTH_L1_BETA
        self.classes = tuple(cfg.data.classes)
        self.register_buffer("code_weights", torch.tensor(lc.code_weights, dtype=torch.float32), persistent=False)

    def forward(self, out: dict[str, torch.Tensor], batch: dict) -> LossOutput:
        dev = out["heatmap"].device
        gt = batch["heatmap"].to(dev, non_blocking=True)
        ind = batch["target_ind"].to(dev, non_blocking=True)
        mask = batch["target_mask"].to(dev, non_blocking=True)
        target = batch["target_reg"].to(dev, non_blocking=True)

        hm, hm_per_class, num_pos = focal_loss(out["heatmap"], gt, self.alpha, self.beta)
        per_field = reg_loss(gather_reg(out["reg"], ind), target, mask, self.reg_kind, self.smooth_beta)
        reg = (per_field * self.code_weights).sum()
        total = hm + self.reg_weight * reg

        logs = {"loss/total": total.detach(), "loss/heatmap": hm.detach(), "loss/reg": reg.detach(),
                "targets/num_objects": mask.float().sum()}
        logs.update({f"loss/heatmap/{c}": hm_per_class[i].detach() for i, c in enumerate(self.classes)})
        logs.update({f"loss/reg/{f}": per_field[i].detach() for i, f in enumerate(REG_FIELDS)})
        return LossOutput(total=total, logs=logs)  # no host sync here; .scalars() when logging
