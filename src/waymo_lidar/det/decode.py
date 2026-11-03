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
    p = torch.sigmoid(out["heatmap"].float())
    b, c, h, w = p.shape
    scores, idx = (p * peak_mask(p)).reshape(b, -1).topk(min(dc.topk, c * h * w), dim=1)  # sorted, descending
    labels, cell = idx // (h * w), idx % (h * w)
    rows, cols = cell // w, cell % w
    reg = out["reg"].float().reshape(b, len(REG_FIELDS), h * w)
    lo, hi = math.log(SIZE_MIN), math.log(SIZE_MAX)
    frames = []
    for i in range(b):
        keep = scores[i] >= dc.score_threshold
        r = reg[i][:, cell[i][keep]].T.clone()  # (N, 8)
        r[:, LOG_SIZE] = r[:, LOG_SIZE].clamp(lo, hi)
        boxes = reg_to_boxes(rows[i][keep], cols[i][keep], r, cfg).float()
        frames.append((boxes, scores[i][keep], labels[i][keep]))
    if dc.nms == "bev":
        frames = _nms(frames, dc.nms_iou)
    return [Detections(boxes=bx, scores=sc, labels=lb) for bx, sc, lb in frames]


def _nms(frames, thresholds: tuple[float, ...]):
    """Rotated BEV NMS per (frame, class) for the whole batch in one pass; classes never suppress each other."""
    from .iou import nms_bev_grouped

    sizes = [len(sc) for _, sc, _ in frames]
    if sum(sizes) == 0:
        return frames
    boxes = torch.cat([bx for bx, _, _ in frames])
    scores = torch.cat([sc for _, sc, _ in frames])
    labels = torch.cat([lb for _, _, lb in frames])
    frame = torch.repeat_interleave(torch.arange(len(frames), device=scores.device),
                                    torch.as_tensor(sizes, device=scores.device))
    thr = torch.as_tensor(thresholds, device=scores.device, dtype=torch.float32)[labels]
    keep = nms_bev_grouped(boxes, scores, frame * len(thresholds) + labels, thr)
    kept = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    kept[keep] = True
    out, start = [], 0
    for n in sizes:  # per frame, still sorted by score (concatenation preserved each frame's order)
        m = kept[start:start + n]
        bx, sc, lb = boxes[start:start + n][m], scores[start:start + n][m], labels[start:start + n][m]
        out.append((bx, sc, lb))
        start += n
    return out


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
