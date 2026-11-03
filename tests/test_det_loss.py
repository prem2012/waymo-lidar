"""Loss tests (pinned stage-1 segment only)."""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from waymo_lidar.data.dataset import WaymoDataset
from waymo_lidar.det.config import build_optimizer, load_config
from waymo_lidar.det.data import collate, load_sample
from waymo_lidar.det.loss import DEFAULT_SMOOTH_L1_BETA, CenterPointLoss, focal_loss, gather_reg, reg_loss
from waymo_lidar.det.targets import REG_FIELDS, TargetBuilder

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
SEG = "10023947602400723454_1120_000_1140_000"
needs_data = pytest.mark.skipif(not (ROOT / "training" / "lidar_box" / f"{SEG}.parquet").is_file(),
                                reason="stage-1 data missing")
CFG = load_config()


def centerpoint_reference(logits, gt, alpha=2, beta=4):
    """CenterPoint's FastFocalLoss-style formula with its clamped sigmoid."""
    p = torch.clamp(torch.sigmoid(logits.float()), 1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    pos_loss = (torch.log(p) * torch.pow(1 - p, alpha) * pos).sum()
    neg_loss = (torch.log(1 - p) * torch.pow(p, alpha) * torch.pow(1 - gt, beta) * neg).sum()
    n = pos.sum()
    return -neg_loss if n == 0 else -(pos_loss + neg_loss) / n


def synthetic_heatmap(b=2, c=3, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(b, c, h, w, generator=g) ** 6  # mostly small values, like Gaussian tails
    for bi in range(b):
        for ci in range(c):
            gt[bi, ci, int(torch.randint(h, (1,), generator=g)), int(torch.randint(w, (1,), generator=g))] = 1.0
    logits = torch.randn(b, c, h, w, generator=g) * 2  # p mostly within (1e-4, 1 - 1e-4)
    return logits, gt


def batch_with(ind, mask, reg_target, heatmap):
    return {"heatmap": heatmap, "target_ind": ind, "target_mask": mask, "target_reg": reg_target}


# --- heatmap -----------------------------------------------------------------------------------------


def test_focal_matches_centerpoint_away_from_clamp():
    logits, gt = synthetic_heatmap()
    ours, per_class, n = focal_loss(logits, gt, 2, 4)
    ref = centerpoint_reference(logits, gt)
    assert n.item() == 6
    assert ours.item() == pytest.approx(ref.item(), rel=1e-5)
    assert per_class.sum().item() == pytest.approx(ours.item(), rel=1e-6)


def test_focal_perfect_prediction_is_zero():
    _, gt = synthetic_heatmap()
    gt = torch.where(gt == 1, gt, torch.zeros_like(gt))  # only peaks
    logits = torch.where(gt == 1, torch.full_like(gt, 20.0), torch.full_like(gt, -20.0))
    assert focal_loss(logits, gt, 2, 4)[0].item() < 1e-6


def test_focal_gradient_signs():
    logits, gt = synthetic_heatmap()
    logits.requires_grad_(True)
    focal_loss(logits, gt, 2, 4)[0].backward()
    g = logits.grad
    assert (g[gt == 1] < 0).all()        # descending the loss raises positive logits
    assert (g[gt < 1] >= 0).all()        # ... and lowers negative ones


def test_focal_no_objects_is_finite():
    logits = torch.randn(1, 3, 16, 16)
    loss, _, n = focal_loss(logits, torch.zeros(1, 3, 16, 16), 2, 4)
    assert n.item() == 0 and torch.isfinite(loss)


def test_focal_bf16_input_matches_fp32():
    logits, gt = synthetic_heatmap()
    a = focal_loss(logits, gt, 2, 4)[0]
    b = focal_loss(logits.to(torch.bfloat16), gt, 2, 4)[0]
    exact_on_rounded = focal_loss(logits.to(torch.bfloat16).float(), gt, 2, 4)[0]
    assert b.dtype == torch.float32 and b.item() == pytest.approx(exact_on_rounded.item(), rel=1e-6)
    assert b.item() == pytest.approx(a.item(), rel=2e-2)  # only bf16 rounding of the logits differs


# --- regression --------------------------------------------------------------------------------------


def make_reg_case():
    """2 samples, 376 x 376 maps; sample 0 has 2 objects, sample 1 has 1; 5 padded slots each."""
    reg = torch.zeros(2, 8, 376, 376)
    ind = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.zeros(2, 5, dtype=torch.uint8)
    target = torch.zeros(2, 5, 8)
    places = [(0, 0, 1000), (0, 1, 50_000), (1, 0, 141_375)]
    for b, k, i in places:
        ind[b, k], mask[b, k] = i, 1
        target[b, k] = torch.arange(8, dtype=torch.float32) * 0.1
        r, c = divmod(i, 376)
        reg[b, :, r, c] = target[b, k] + torch.tensor([0.5, -0.5, 0.05, 0, 0, 0, 0.2, -2.0])
    reg[:, :, 0, 0] = 99.0  # garbage at index 0, where padded slots point
    return reg, ind, mask, target


def test_gather_and_l1_by_hand():
    reg, ind, mask, target = make_reg_case()
    pred = gather_reg(reg, ind)
    assert pred.shape == (2, 5, 8)
    per_field = reg_loss(pred, target, mask, "l1")
    expected = torch.tensor([0.5, 0.5, 0.05, 0, 0, 0, 0.2, 2.0])  # same error for all 3 objects
    assert torch.allclose(per_field, expected * 3 / (3 + 1e-4), atol=1e-6)


def test_padded_slots_ignored():
    reg, ind, mask, target = make_reg_case()
    base = reg_loss(gather_reg(reg, ind), target, mask, "l1")
    reg[:, :, 0, 0] = -1e6
    assert torch.allclose(reg_loss(gather_reg(reg, ind), target, mask, "l1"), base)
    assert torch.allclose(reg_loss(gather_reg(reg, ind), target, mask, "smooth_l1"),
                          reg_loss(gather_reg(make_reg_case()[0], ind), target, mask, "smooth_l1"))


def test_smooth_l1_matches_torch_and_limits():
    reg, ind, mask, target = make_reg_case()
    pred = gather_reg(reg, ind)
    ours = reg_loss(pred, target, mask, "smooth_l1", beta=DEFAULT_SMOOTH_L1_BETA)
    m = mask.bool()
    ref = F.smooth_l1_loss(pred[m], target[m], beta=DEFAULT_SMOOTH_L1_BETA, reduction="none").sum(0) / (3 + 1e-4)
    assert torch.allclose(ours, ref, atol=1e-7)
    beta = DEFAULT_SMOOTH_L1_BETA
    # |e| = 0.5, 2.0 >> beta: L1 - beta/2.  |e| = 0.05 < beta: 0.5 e^2 / beta.  |e| = 0: 0.
    expected = torch.tensor([0.5 - beta / 2, 0.5 - beta / 2, 0.5 * 0.05 ** 2 / beta, 0, 0, 0, 0.2 - beta / 2, 2.0 - beta / 2])
    assert torch.allclose(ours, expected * 3 / (3 + 1e-4), atol=1e-6)
    assert DEFAULT_SMOOTH_L1_BETA == pytest.approx(1 / 9)


def test_regression_zero_when_exact():
    reg, ind, mask, target = make_reg_case()
    pred = gather_reg(reg, ind)
    for kind in ("l1", "smooth_l1"):
        assert reg_loss(target.clone(), target, mask, kind).abs().max() == 0
    assert reg_loss(pred, target, torch.zeros_like(mask), "l1").abs().max() == 0  # no objects


# --- CenterPointLoss -----------------------------------------------------------------------------------


def full_case(overrides=()):
    cfg = load_config(overrides=list(overrides))
    reg, ind, mask, target = make_reg_case()
    logits, gt = synthetic_heatmap(b=2, h=376, w=376)
    out = {"heatmap": logits, "reg": reg}
    return cfg, out, batch_with(ind, mask, target, gt)


def test_total_breakdowns_and_names():
    cfg, out, batch = full_case()
    res = CenterPointLoss(cfg)(out, batch)
    s = res.scalars()
    classes = [f"loss/heatmap/{c}" for c in cfg.data.classes]
    fields = [f"loss/reg/{f}" for f in REG_FIELDS]
    assert set(s) == {"loss/total", "loss/heatmap", "loss/reg", "targets/num_objects", *classes, *fields}
    assert sum(s[k] for k in classes) == pytest.approx(s["loss/heatmap"], rel=1e-6)
    assert sum(s[k] for k in fields) == pytest.approx(s["loss/reg"], rel=1e-6)  # code_weights all 1
    assert s["loss/total"] == pytest.approx(s["loss/heatmap"] + 0.25 * s["loss/reg"], rel=1e-6)
    assert s["targets/num_objects"] == 3
    assert all(v.requires_grad is False for v in res.logs.values()) and res.total.requires_grad is False  # no params here


def test_code_and_reg_weights():
    cfg, out, batch = full_case()
    base = CenterPointLoss(cfg)(out, batch).scalars()
    cfg2, _, _ = full_case(["loss.code_weights=[2, 1, 1, 1, 1, 1, 1, 1]"])
    doubled = CenterPointLoss(cfg2)(out, batch).scalars()
    assert doubled["loss/reg"] == pytest.approx(base["loss/reg"] + base["loss/reg/dx"], rel=1e-6)
    assert doubled["loss/reg/dx"] == pytest.approx(base["loss/reg/dx"])  # per-field logs are before weights
    cfg3, _, _ = full_case(["loss.reg_weight=0.5"])
    w = CenterPointLoss(cfg3)(out, batch).scalars()
    assert w["loss/heatmap"] == pytest.approx(base["loss/heatmap"])
    assert w["loss/total"] == pytest.approx(base["loss/heatmap"] + 0.5 * base["loss/reg"], rel=1e-6)


def test_smooth_l1_selected_by_config():
    cfg, out, batch = full_case(["loss.reg_loss=smooth_l1", "loss.smooth_l1_beta=0.2"])
    s = CenterPointLoss(cfg)(out, batch).scalars()
    assert s["loss/reg/dx"] == pytest.approx((0.5 - 0.1) * 3 / (3 + 1e-4), rel=1e-5)


@needs_data
def test_initial_loss_matches_analytic_estimate():
    tb = TargetBuilder(CFG)
    ts = WaymoDataset(ROOT).segment("training", SEG).timestamps
    batch = collate([tb(load_sample(ROOT, "training", SEG, int(t), CFG)) for t in ts[[10, 90]]])
    gt = batch["heatmap"]
    logits = torch.full_like(gt, CFG.model.heatmap_bias)
    p = 1 / (1 + math.exp(-CFG.model.heatmap_bias))
    pos = gt.eq(1)
    n = pos.sum().item()
    expected = ((1 - gt[~pos]) ** 4).sum().item() * p ** 2 * -math.log(1 - p) + n * (1 - p) ** 2 * -math.log(p)
    expected /= n
    got = focal_loss(logits, gt, 2, 4)[0].item()
    assert got == pytest.approx(expected, rel=1e-4)
    assert 5 < got < 20


@needs_data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("kind", ["l1", "smooth_l1"])
def test_overfit_one_batch(kind):
    from waymo_lidar.det.model import BEVDetector
    from waymo_lidar.det.voxelize import Voxelizer
    cfg = load_config(overrides=[f"loss.reg_loss={kind}", "optim.lr=1e-3"])
    tb = TargetBuilder(cfg)
    ts = WaymoDataset(ROOT).segment("training", SEG).timestamps
    batch = collate([tb(load_sample(ROOT, "training", SEG, int(t), cfg)) for t in ts[[20, 140]]])
    torch.manual_seed(0)
    bev = Voxelizer(cfg.grid).cuda()(batch["points"].cuda(), 2, dtype=torch.bfloat16)
    bev = bev.contiguous(memory_format=torch.channels_last)
    model = BEVDetector.from_config(cfg).cuda().to(memory_format=torch.channels_last).train()
    loss_fn = CenterPointLoss(cfg).cuda()
    opt = build_optimizer(cfg.optim, model.parameters())
    first = last = None
    for step in range(151):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(bev)
        res = loss_fn(out, batch)
        opt.zero_grad(set_to_none=True)
        res.total.backward()
        opt.step()
        if step == 0:
            first = res.scalars()
    last = res.scalars()
    assert last["loss/total"] < 0.1 * first["loss/total"]          # > 90% reduction (measured ~99%)
    assert last["loss/heatmap"] < 0.1 * first["loss/heatmap"]
    assert last["loss/reg"] < 0.2 * first["loss/reg"]
    assert all(np.isfinite(v) for v in last.values())
