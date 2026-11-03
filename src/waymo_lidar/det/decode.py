"""Model outputs -> 3D boxes: 3x3 max-pool peak picking, top-K, score threshold, reg_to_boxes.

    dets = decode(model(bev), cfg)        # list of Detections, one per frame (on the model's device)
    dets[0].boxes, dets[0].scores, dets[0].labels

See docs/detector/decode.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import DetectorConfig
from .targets import REG_FIELDS, reg_to_boxes

# Log-size clamp before exp(), so an untrained model can't produce inf/0 sizes. Real vehicle / pedestrian /
# cyclist boxes span 0.059-19.7 m (measured over all 1,000 segments), well inside [0.02, 50] m.
SIZE_MIN, SIZE_MAX = 0.02, 50.0
LOG_SIZE = slice(3, 6)  # log l, log w, log h in REG_FIELDS
assert REG_FIELDS[LOG_SIZE] == ("log_l", "log_w", "log_h")


@dataclass
class Detections:
    """One frame's detections, sorted by score (highest first). Boxes in the vehicle frame."""

    boxes: torch.Tensor  # (N, 7) float32: x, y, z, length, width, height, heading
    scores: torch.Tensor  # (N,) float32 in [0, 1]
    labels: torch.Tensor  # (N,) int64 class channel (index into data.classes)

    def __len__(self) -> int:
        return len(self.scores)

    def numpy(self) -> dict[str, np.ndarray]:
        return {"boxes": self.boxes.detach().cpu().numpy(), "scores": self.scores.detach().cpu().numpy(),
                "labels": self.labels.detach().cpu().numpy()}


def peak_mask(p: torch.Tensor) -> torch.Tensor:
    """Cells equal to the max of their 3x3 neighbourhood (ties keep every tied cell)."""
    return F.max_pool2d(p, kernel_size=3, stride=1, padding=1) == p


@torch.no_grad()
def decode(out: dict[str, torch.Tensor], cfg: DetectorConfig) -> list[Detections]:
    dc = cfg.decode
    if dc.nms != "none":
        raise NotImplementedError(f"decode.nms: {dc.nms} needs rotated BEV IoU, which comes with the IoU "
                                  f"component (det/iou.py); use decode.nms: none for now")
    p = torch.sigmoid(out["heatmap"].float())
    b, c, h, w = p.shape
    scores, idx = (p * peak_mask(p)).reshape(b, -1).topk(min(dc.topk, c * h * w), dim=1)  # sorted, descending
    labels, cell = idx // (h * w), idx % (h * w)
    rows, cols = cell // w, cell % w
    reg = out["reg"].float().reshape(b, len(REG_FIELDS), h * w)
    lo, hi = math.log(SIZE_MIN), math.log(SIZE_MAX)
    result = []
    for i in range(b):
        keep = scores[i] >= dc.score_threshold
        r = reg[i][:, cell[i][keep]].T.clone()  # (N, 8)
        r[:, LOG_SIZE] = r[:, LOG_SIZE].clamp(lo, hi)
        boxes = reg_to_boxes(rows[i][keep], cols[i][keep], r, cfg)
        result.append(Detections(boxes=boxes.float(), scores=scores[i][keep], labels=labels[i][keep]))
    return result


def perfect_outputs(batch: dict, cfg: DetectorConfig) -> dict[str, torch.Tensor]:
    """Model-shaped outputs that encode the targets exactly (for tests and the viewer).

    heatmap logits = logit(target heatmap), clipped to +-20; reg holds target_reg at each object's centre.
    Accepts collated tensors or a single sample's NumPy arrays (a batch dimension is added).
    """
    gt = torch.as_tensor(batch["heatmap"]).float()
    if gt.dim() == 3:  # single sample
        gt = gt[None]
        ind, mask, treg = (torch.as_tensor(batch[k])[None] for k in ("target_ind", "target_mask", "target_reg"))
    else:
        ind, mask, treg = (torch.as_tensor(batch[k]) for k in ("target_ind", "target_mask", "target_reg"))
    eps = 1e-12
    logits = torch.logit(gt.clamp(eps, 1 - eps)).clamp(-20.0, 20.0)
    b, _, h, w = gt.shape
    reg = torch.zeros(b, len(REG_FIELDS), h * w)
    for i in range(b):
        m = mask[i].bool()
        reg[i][:, ind[i][m]] = treg[i][m].float().T
    return {"heatmap": logits, "reg": reg.reshape(b, len(REG_FIELDS), h, w)}
