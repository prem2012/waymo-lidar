"""BEV occupancy image -> CenterPoint maps. Backbone (simple or torchvision ResNet) -> neck -> head.

    model = BEVDetector.from_config(cfg)
    out = model(bev)        # {"heatmap": (B, C, H, W) logits, "reg": (B, 8, H, W) in REG_FIELDS order}

H, W = cfg.target_out_size (376 x 376 at the default stride 4). See docs/detector/model.md.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import nn

from .config import DetectorConfig, ModelConfig, load_config
from .targets import REG_FIELDS

# Regression heads, in REG_FIELDS order: (name, channels).
REG_HEADS = (("offset", 2), ("z", 1), ("size", 3), ("rot", 2))
assert sum(c for _, c in REG_HEADS) == len(REG_FIELDS)

RESNET_CHANNELS = {"resnet18": (64, 128, 256, 512), "resnet34": (64, 128, 256, 512),
                   "resnet50": (256, 512, 1024, 2048)}


def conv_bn_relu(cin: int, cout: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs + BN, identity or 1x1 projection shortcut."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (nn.Identity() if stride == 1 and cin == cout else
                         nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class SimpleBackbone(nn.Module):
    """Stride-2 stem, then one stride-2 stage of BasicBlocks per configured stage (strides 4, 8, 16, ...)."""

    def __init__(self, in_channels: int, cfg: ModelConfig):
        super().__init__()
        sc = cfg.simple
        self.stem = nn.Sequential(conv_bn_relu(in_channels, sc.stem_channels, stride=2),
                                  conv_bn_relu(sc.stem_channels, sc.stem_channels))
        stages, cin = [], sc.stem_channels
        for cout, blocks in zip(sc.stage_channels, sc.stage_blocks):
            stages.append(nn.Sequential(BasicBlock(cin, cout, stride=2), *[BasicBlock(cout, cout) for _ in range(blocks - 1)]))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.out_channels = list(sc.stage_channels)
        self.out_strides = list(cfg.stage_strides)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class ResNetBackbone(nn.Module):
    """torchvision ResNet without avgpool/fc; conv1 rebuilt for `in_channels`; returns the configured layers."""

    def __init__(self, in_channels: int, cfg: ModelConfig):
        super().__init__()
        import torchvision

        name, rc = cfg.backbone, cfg.resnet
        weights = {"resnet18": "IMAGENET1K_V1", "resnet34": "IMAGENET1K_V1", "resnet50": "IMAGENET1K_V2"}[name]
        net = getattr(torchvision.models, name)(weights=weights if rc.pretrained else None)
        old = net.conv1
        conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        if rc.pretrained:  # mean RGB kernel, replicated across inputs, rescaled so activations keep their scale
            with torch.no_grad():
                conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * 3 / in_channels)
        else:
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.ModuleList([getattr(net, f"layer{k}") for k in range(1, max(rc.layers) + 1)])
        self.keep = list(rc.layers)
        self.out_channels = [RESNET_CHANNELS[name][k - 1] for k in rc.layers]
        self.out_strides = list(cfg.stage_strides)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for k, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if k in self.keep:
                feats.append(x)
        return feats


def build_backbone(in_channels: int, cfg: ModelConfig) -> nn.Module:
    return SimpleBackbone(in_channels, cfg) if cfg.backbone == "simple" else ResNetBackbone(in_channels, cfg)


class Neck(nn.Module):
    """Bring every backbone feature map to the output stride, concatenate, fuse with a 3x3 conv."""

    def __init__(self, in_channels: list[int], in_strides: list[int], out_stride: int, channels: int):
        super().__init__()
        branches = []
        for cin, stride in zip(in_channels, in_strides):
            if stride > out_stride:  # upsample
                f = stride // out_stride
                op = nn.ConvTranspose2d(cin, channels, f, f, bias=False)
            elif stride == out_stride:
                op = nn.Conv2d(cin, channels, 1, bias=False)
            else:  # downsample
                f = out_stride // stride
                op = nn.Conv2d(cin, channels, f, f, bias=False)
            branches.append(nn.Sequential(op, nn.BatchNorm2d(channels), nn.ReLU(inplace=True)))
        self.branches = nn.ModuleList(branches)
        self.fuse = conv_bn_relu(channels * len(branches), channels)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        return self.fuse(torch.cat([b(f) for b, f in zip(self.branches, feats)], dim=1))


class CenterHead(nn.Module):
    """Shared conv, then a heatmap branch (class logits) and one branch per regression group."""

    def __init__(self, in_channels: int, num_classes: int, channels: int, heatmap_bias: float):
        super().__init__()
        self.shared = conv_bn_relu(in_channels, channels)

        def branch(cout: int) -> nn.Sequential:
            return nn.Sequential(conv_bn_relu(channels, channels), nn.Conv2d(channels, cout, 1))

        self.heatmap = branch(num_classes)
        nn.init.constant_(self.heatmap[-1].bias, heatmap_bias)
        self.reg = nn.ModuleDict({name: branch(c) for name, c in REG_HEADS})
        for b in self.reg.values():
            nn.init.zeros_(b[-1].bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.shared(x)
        return {"heatmap": self.heatmap(x), "reg": torch.cat([self.reg[name](x) for name, _ in REG_HEADS], dim=1)}


class BEVDetector(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, out_stride: int, cfg: ModelConfig):
        super().__init__()
        self.backbone = build_backbone(in_channels, cfg)
        self.neck = Neck(self.backbone.out_channels, self.backbone.out_strides, out_stride, cfg.neck_channels)
        self.head = CenterHead(cfg.neck_channels, num_classes, cfg.head_channels, cfg.heatmap_bias)
        self._init_weights(skip_pretrained=cfg.backbone != "simple" and cfg.resnet.pretrained)

    @classmethod
    def from_config(cls, cfg: DetectorConfig) -> BEVDetector:
        return cls(cfg.grid.in_channels, len(cfg.data.classes), cfg.target.out_stride, cfg.model)

    def _init_weights(self, skip_pretrained: bool) -> None:
        modules = [self.neck, self.head] + ([] if skip_pretrained else [self.backbone])
        for part in modules:
            for m in part.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        # Final 1x1 output convs get small weights (RetinaNet / CenterNet practice): with Kaiming init, the
        # heatmap conv's few output channels give large random logits that swamp heatmap_bias (measured:
        # mean sigmoid 0.42 instead of ~0.1). Their biases (heatmap_bias, 0) were set in CenterHead.
        for out_conv in [self.head.heatmap[-1], *(b[-1] for b in self.head.reg.values())]:
            nn.init.normal_(out_conv.weight, std=0.01)

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.neck(self.backbone(bev)))


# --- introspection ----------------------------------------------------------------------------------


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def summary(model: BEVDetector, in_shape: tuple[int, int, int, int], device="cpu") -> list[dict]:
    """Output shape and parameter count of the stem, each backbone stage, the neck and the head."""
    rows, x = [], torch.zeros(in_shape, device=device)
    model = model.to(device).eval()
    with torch.no_grad():
        feats = model.backbone(x)
        bb = model.backbone
        rows.append({"part": f"backbone ({type(bb).__name__})", "shape": None, "params": count_parameters(bb)})
        for i, (f, s) in enumerate(zip(feats, bb.out_strides)):
            rows.append({"part": f"  feature {i} (stride {s})", "shape": tuple(f.shape), "params": None})
        n = model.neck(feats)
        rows.append({"part": "neck", "shape": tuple(n.shape), "params": count_parameters(model.neck)})
        out = model.head(n)
        rows.append({"part": "head: heatmap", "shape": tuple(out["heatmap"].shape), "params": count_parameters(model.head.heatmap)})
        rows.append({"part": "head: reg", "shape": tuple(out["reg"].shape),
                     "params": count_parameters(model.head.reg) + count_parameters(model.head.shared)})
    rows.append({"part": "total", "shape": None, "params": count_parameters(model)})
    return rows


# --- benchmark --------------------------------------------------------------------------------------


def benchmark(cfg: DetectorConfig, batch_sizes: list[int], bev_batch: torch.Tensor | None, iters: int = 5) -> list[dict]:
    """Forward and forward+backward time and peak memory (bf16 autocast, channels_last) per batch size."""
    dev = torch.device("cuda")
    model = BEVDetector.from_config(cfg).to(dev).to(memory_format=torch.channels_last)
    results = []
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        try:
            if bev_batch is not None:
                reps = math.ceil(bs / len(bev_batch))
                x = bev_batch.repeat(reps, 1, 1, 1)[:bs].to(dev, torch.bfloat16)
            else:
                x = (torch.rand(bs, cfg.grid.in_channels, cfg.grid.ny, cfg.grid.nx, device=dev) < 0.002).to(torch.bfloat16)
            x = x.contiguous(memory_format=torch.channels_last)
            model.train()
            torch.cuda.reset_peak_memory_stats()
            for i in range(iters + 2):  # 2 warm-up iterations
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                    loss = out["heatmap"].float().mean() + out["reg"].float().mean()
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t_train = (time.perf_counter() - t0) / iters
            peak = torch.cuda.max_memory_allocated() / 2**30
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(iters + 2):
                    if i == 2:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                    out = model(x)
            torch.cuda.synchronize(); t_fwd = (time.perf_counter() - t0) / iters
            results.append({"batch": bs, "fwd_ms": 1000 * t_fwd, "train_ms": 1000 * t_train, "peak_gib": peak,
                            "heatmap": tuple(out["heatmap"].shape), "reg": tuple(out["reg"].shape)})
        except torch.OutOfMemoryError:
            results.append({"batch": bs, "oom": True})
            break
        finally:
            x = out = loss = None
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark BEVDetector backbones on the GPU (bf16)")
    p.add_argument("--backbone", nargs="+", default=["simple"], choices=["simple", "resnet18", "resnet34", "resnet50"])
    p.add_argument("--batch", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--extras", action="store_true", help="26 input channels")
    p.add_argument("--root", default="~/dev/data/waymo_v2", help="real frames for the input (stage-1 segment)")
    p.add_argument("--random-input", action="store_true", help="use random sparse occupancy instead of real frames")
    a = p.parse_args(argv)

    bev = None
    for name in a.backbone:
        cfg = load_config(overrides=[f"model.backbone={name}", f"grid.extra_channels={'true' if a.extras else 'false'}"])
        if bev is None and not a.random_input:
            from ..data.dataset import WaymoDataset
            from .data import collate, load_sample
            from .voxelize import Voxelizer
            seg = "10023947602400723454_1120_000_1140_000"
            ts = WaymoDataset(a.root).segment("training", seg).timestamps[[0, 60, 120, 180]]
            batch = collate([load_sample(a.root, "training", seg, int(t), cfg) for t in ts])
            bev = Voxelizer(cfg.grid).cuda()(batch["points"].cuda(), 4)
        model = BEVDetector.from_config(cfg)
        print(f"\n== {name}: {count_parameters(model) / 1e6:.2f}M parameters, input {cfg.grid.in_channels}×{cfg.grid.ny}×{cfg.grid.nx}")
        for r in summary(model, (1, cfg.grid.in_channels, 256, 256)):
            params = "" if r["params"] is None else f"{r['params'] / 1e6:.2f}M"
            print(f"   {r['part']:32s} {str(r['shape'] or ''):22s} {params}")
        print("   (shapes above for a 256×256 input; full size is 1504×1504)")
        for r in benchmark(cfg, a.batch, bev):
            if r.get("oom"):
                print(f"   batch {r['batch']:2d}: out of memory")
                continue
            print(f"   batch {r['batch']:2d}: forward {r['fwd_ms']:7.1f} ms | forward+backward {r['train_ms']:7.1f} ms | "
                  f"peak {r['peak_gib']:5.2f} GiB | heatmap {r['heatmap']} reg {r['reg']}")


if __name__ == "__main__":
    main()
