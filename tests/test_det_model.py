"""Model tests. Shape tests use a small grid (+-12.8 m -> 256 x 256) for speed; one GPU test runs full size."""

import os
from pathlib import Path

import pytest
import torch
import torchvision

from waymo_lidar.det.config import load_config
from waymo_lidar.det.model import (REG_HEADS, RESNET_CHANNELS, BEVDetector, ResNetBackbone, count_parameters,
                                   summary)
from waymo_lidar.det.targets import REG_FIELDS

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
SEG = "10023947602400723454_1120_000_1140_000"
SMALL = ["grid.x_range=[-12.8, 12.8]", "grid.y_range=[-12.8, 12.8]"]  # 256 x 256: divisible by 32
BACKBONES = ["simple", "resnet18", "resnet34", "resnet50"]


def small_cfg(*overrides):
    return load_config(overrides=[*SMALL, *overrides])


def run(cfg, batch=2, seed=0):
    torch.manual_seed(seed)
    model = BEVDetector.from_config(cfg).eval()
    x = (torch.rand(batch, cfg.grid.in_channels, cfg.grid.ny, cfg.grid.nx) < 0.01).float()
    with torch.no_grad():
        return model, model(x)


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("extras", [False, True])
def test_output_shapes_every_backbone(backbone, extras):
    cfg = small_cfg(f"model.backbone={backbone}", f"grid.extra_channels={'true' if extras else 'false'}")
    _, out = run(cfg)
    h, w = cfg.target_out_size
    assert (h, w) == (64, 64)
    assert out["heatmap"].shape == (2, 3, h, w) and out["reg"].shape == (2, len(REG_FIELDS), h, w)
    assert torch.isfinite(out["heatmap"]).all() and torch.isfinite(out["reg"]).all()


@pytest.mark.parametrize("backbone", ["simple", "resnet18"])
def test_out_stride_8(backbone):
    cfg = small_cfg(f"model.backbone={backbone}", "target.out_stride=8")
    _, out = run(cfg, batch=1)
    assert out["heatmap"].shape == (1, 3, 32, 32) and out["reg"].shape == (1, 8, 32, 32)


@pytest.mark.parametrize("layers, strides", [("[1, 2, 3]", [4, 8, 16]), ("[1, 2, 3, 4]", [4, 8, 16, 32]), ("[2, 4]", [8, 32])])
def test_resnet_layers_and_reported_channels(layers, strides):
    cfg = small_cfg("model.backbone=resnet18", f"model.resnet.layers={layers}")
    bb = ResNetBackbone(cfg.grid.in_channels, cfg.model)
    assert bb.stem[0].in_channels == 24 and bb.out_strides == strides
    feats = bb(torch.zeros(1, 24, 256, 256))
    assert [f.shape[1] for f in feats] == bb.out_channels
    assert [256 // f.shape[-1] for f in feats] == strides
    _, out = run(cfg, batch=1)
    assert out["heatmap"].shape == (1, 3, 64, 64)


@pytest.mark.parametrize("name", ["resnet18", "resnet34", "resnet50"])
def test_resnet_param_count_matches_torchvision(name):
    cfg = small_cfg(f"model.backbone={name}")
    ours = count_parameters(ResNetBackbone(cfg.grid.in_channels, cfg.model))
    tv = getattr(torchvision.models, name)(weights=None)
    ref = count_parameters(tv) - count_parameters(tv.fc) - count_parameters(tv.conv1)
    ref += tv.conv1.out_channels * cfg.grid.in_channels * 7 * 7  # conv1 rebuilt for 24 input channels
    assert ours == ref
    assert RESNET_CHANNELS[name][-1] == tv.fc.in_features


def test_resnet_pretrained_conv1_adaptation():
    cfg = small_cfg("model.backbone=resnet18", "model.resnet.pretrained=true")
    try:
        bb = ResNetBackbone(cfg.grid.in_channels, cfg.model)
        ref = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    except Exception as e:  # no network / no cached weights
        pytest.skip(f"ImageNet weights unavailable: {e}")
    expected = ref.conv1.weight.mean(dim=1, keepdim=True).repeat(1, 24, 1, 1) * 3 / 24
    assert torch.allclose(bb.stem[0].weight, expected)
    assert torch.equal(bb.layers[0][0].conv1.weight, ref.layer1[0].conv1.weight)  # rest untouched
    model = BEVDetector.from_config(cfg)  # model init must not overwrite the pretrained backbone
    assert torch.equal(model.backbone.layers[0][0].conv1.weight, ref.layer1[0].conv1.weight)


@pytest.mark.parametrize("backbone", ["simple", "resnet18"])
def test_heatmap_init_probability(backbone):
    _, out = run(small_cfg(f"model.backbone={backbone}"))
    p = torch.sigmoid(out["heatmap"]).mean().item()
    assert 0.05 < p < 0.2  # bias -2.19 -> ~0.1


@pytest.mark.parametrize("backbone", ["simple", "resnet18"])
def test_every_parameter_gets_a_gradient(backbone):
    cfg = small_cfg(f"model.backbone={backbone}")
    torch.manual_seed(0)
    model = BEVDetector.from_config(cfg).train()
    x = (torch.rand(2, 24, 256, 256) < 0.01).float()
    out = model(x)
    (out["heatmap"].sigmoid().mean() + out["reg"].pow(2).mean()).backward()
    for name, prm in model.named_parameters():
        assert prm.grad is not None and torch.isfinite(prm.grad).all(), name
        assert prm.grad.abs().sum() > 0, name


def test_config_wiring_changes_parameter_count():
    base = count_parameters(BEVDetector.from_config(small_cfg()))
    wider = count_parameters(BEVDetector.from_config(small_cfg("model.simple.stage_channels=[64, 128, 384]")))
    deeper = count_parameters(BEVDetector.from_config(small_cfg("model.simple.stage_blocks=[2, 3, 5]")))
    neck = count_parameters(BEVDetector.from_config(small_cfg("model.neck_channels=64")))
    assert wider > base and deeper > base and neck < base
    head = BEVDetector.from_config(small_cfg()).head
    assert sum(head.reg[n][-1].out_channels for n, _ in REG_HEADS) == len(REG_FIELDS)


def test_same_seed_same_weights():
    a, _ = run(small_cfg(), seed=3)
    b, _ = run(small_cfg(), seed=3)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def test_summary_rows():
    cfg = small_cfg()
    rows = summary(BEVDetector.from_config(cfg), (1, 24, 256, 256))
    assert rows[-1]["part"] == "total" and rows[-1]["params"] == count_parameters(BEVDetector.from_config(cfg))
    assert any(r["shape"] == (1, 3, 64, 64) for r in rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not (ROOT / "training" / "lidar_box" / f"{SEG}.parquet").is_file(), reason="stage-1 data missing")
def test_full_size_real_frame_bf16_gpu():
    from waymo_lidar.data.dataset import WaymoDataset
    from waymo_lidar.det.data import collate, load_sample
    from waymo_lidar.det.voxelize import Voxelizer
    cfg = load_config()
    ts = int(WaymoDataset(ROOT).segment("training", SEG).timestamps[30])
    batch = collate([load_sample(ROOT, "training", SEG, ts, cfg)])
    bev = Voxelizer(cfg.grid).cuda()(batch["points"].cuda(), 1, dtype=torch.bfloat16)
    model = BEVDetector.from_config(cfg).cuda().to(memory_format=torch.channels_last).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(bev.contiguous(memory_format=torch.channels_last))
    assert out["heatmap"].shape == (1, 3, 376, 376) and out["reg"].shape == (1, 8, 376, 376)
    assert torch.isfinite(out["heatmap"].float()).all() and torch.isfinite(out["reg"].float()).all()
