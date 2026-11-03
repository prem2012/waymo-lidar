"""Detector configuration: typed, validated dataclasses loaded from YAML with dotted CLI overrides.

    cfg = load_config("configs/bev_centerpoint.yaml", overrides=["train.batch_size=8"])
    cfg.grid.nx, cfg.grid.in_channels, cfg.target_out_size   # 1504, 24, (376, 376)

Every number the detector components read lives here, so it is defined once, checked at load time,
and saved with each run (``save_config``). See docs/detector/config.md for what each field does.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from ..data.types import BoxType


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types and invalid values; the message names the key."""


def _voxel_count(lo: float, hi: float, size: float, key: str) -> int:
    n = (hi - lo) / size
    if abs(n - round(n)) > 1e-6:
        raise ConfigError(f"{key}: range {hi - lo:g} m is not a whole number of {size:g} m voxels ({n:.4f})")
    return int(round(n))


# --- sections ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridConfig:
    """Voxel grid in the vehicle frame (metres). The BEV image is (ny rows, nx cols); z slices are channels."""

    x_range: tuple[float, float] = (-75.2, 75.2)
    y_range: tuple[float, float] = (-75.2, 75.2)
    z_range: tuple[float, float] = (-2.0, 4.0)
    voxel_size: tuple[float, float, float] = (0.1, 0.1, 0.25)
    extra_channels: bool = False  # add log1p(point count) and max intensity per BEV cell

    @property
    def nx(self) -> int:
        return _voxel_count(*self.x_range, self.voxel_size[0], "grid.x_range")

    @property
    def ny(self) -> int:
        return _voxel_count(*self.y_range, self.voxel_size[1], "grid.y_range")

    @property
    def nz(self) -> int:
        return _voxel_count(*self.z_range, self.voxel_size[2], "grid.z_range")

    @property
    def in_channels(self) -> int:
        return self.nz + (2 if self.extra_channels else 0)


@dataclass(frozen=True)
class DataConfig:
    classes: tuple[str, ...] = ("vehicle", "pedestrian", "cyclist")  # model channel order
    returns: tuple[int, ...] = (1, 2)
    frame_interval_train: int = 5  # use every Nth frame (consecutive 10 Hz frames are near-duplicates)
    frame_interval_eval: int = 5
    shuffle_buffer: int = 128  # frames per DataLoader worker (~2.3 MB each)
    num_workers: int = 12
    min_points_train: int = 1  # drop training boxes with fewer top-LiDAR points

    @property
    def class_ids(self) -> tuple[int, ...]:
        """Dataset type IDs (BoxType) in model channel order, e.g. (1, 2, 4)."""
        return tuple(int(BoxType[c.upper()]) for c in self.classes)


@dataclass(frozen=True)
class AugmentConfig:
    flip_y_prob: float = 0.5  # mirror across the x axis (y -> -y, heading -> -heading)
    rotation_range: float = math.pi / 4  # global yaw rotation, uniform in [-r, r] radians
    scale_range: tuple[float, float] = (0.95, 1.05)  # global scaling factor


@dataclass(frozen=True)
class TargetConfig:
    out_stride: int = 4  # heatmap cell = out_stride voxels
    max_objects: int = 500
    gaussian_overlap: float = 0.1  # CenterPoint radius rule
    min_radius: int = 2  # heatmap cells


@dataclass(frozen=True)
class SimpleBackboneConfig:
    """The `simple` backbone: stride-2 stem, then one stride-2 stage of residual blocks per entry."""

    stem_channels: int = 32
    stage_channels: tuple[int, ...] = (64, 128, 256)  # strides 4, 8, 16
    stage_blocks: tuple[int, ...] = (2, 3, 3)


@dataclass(frozen=True)
class ResNetBackboneConfig:
    """torchvision ResNet backbones (resnet18/34/50)."""

    layers: tuple[int, ...] = (1, 2, 3, 4)  # ResNet layers feeding the neck; layer k has stride 2^(k+1)
    pretrained: bool = False  # ImageNet weights, conv1 adapted to the input channels


BACKBONES = ("simple", "resnet18", "resnet34", "resnet50")


@dataclass(frozen=True)
class ModelConfig:
    backbone: Literal["simple", "resnet18", "resnet34", "resnet50"] = "simple"
    simple: SimpleBackboneConfig = field(default_factory=SimpleBackboneConfig)  # used when backbone == simple
    resnet: ResNetBackboneConfig = field(default_factory=ResNetBackboneConfig)  # used when backbone == resnet*
    neck_channels: int = 128
    head_channels: int = 64
    heatmap_bias: float = -2.19  # sigmoid(-2.19) ~= 0.1 initial foreground probability

    @property
    def stage_strides(self) -> tuple[int, ...]:
        """Strides of the feature maps the backbone hands to the neck."""
        if self.backbone == "simple":
            return tuple(2 ** (2 + i) for i in range(len(self.simple.stage_channels)))
        return tuple(2 ** (1 + k) for k in self.resnet.layers)

    @property
    def max_stride(self) -> int:
        return max(self.stage_strides)


@dataclass(frozen=True)
class LossConfig:
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    reg_weight: float = 0.25
    code_weights: tuple[float, ...] = (1.0,) * 8  # offset x, y · z · log l, w, h · sin, cos
    reg_loss: Literal["l1", "smooth_l1"] = "l1"
    smooth_l1_beta: float | None = None  # smooth_l1 only; None -> 1/9 (Faster R-CNN)


@dataclass(frozen=True)
class DecodeConfig:
    topk: int = 500
    score_threshold: float = 0.1
    nms: Literal["none", "bev"] = "none"
    nms_iou: tuple[float, ...] = (0.7, 0.5, 0.5)  # per class, used when nms == "bev"


@dataclass(frozen=True)
class EvalConfig:
    iou_thresholds: tuple[float, ...] = (0.5, 0.7, 0.8)
    distance_bins: tuple[float, ...] = (0.0, 30.0, 50.0, 75.0)  # metres, xy distance from ego
    center_distances: tuple[float, ...] = (0.5, 1.0)  # metres: centre-distance AP thresholds (empty = off)
    min_points: int = 1  # GT with fewer top-LiDAR points is ignored (don't care)


OPTIMIZERS = ("adamw", "adam", "sgd")
_ADAM_DEFAULT_BETAS = (0.9, 0.99)
_SGD_DEFAULT_MOMENTUM = 0.9


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer. Optimizer-specific fields default to None and must stay None for other optimizers."""

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 3e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] | None = None  # adam/adamw; None -> (0.9, 0.99)
    momentum: float | None = None  # sgd; None -> 0.9
    nesterov: bool | None = None  # sgd; None -> False


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    epochs: int = 12
    grad_clip: float = 35.0  # max gradient norm; 0 disables
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    eval_every: int = 2  # epochs
    seed: int = 0
    runs_dir: str = "runs"
    lr_schedule: Literal["onecycle", "cosine", "constant"] = "onecycle"
    warmup_frac: float = 0.4  # onecycle: fraction of steps spent warming up to optim.lr
    log_every: int = 50  # steps between TensorBoard scalar writes
    val_max_frames: int = 0  # validation frames per pass (0 = all)
    best_metric: str = "eval/bev_ap@0.7/mean"  # higher is better; selects best.pt


@dataclass(frozen=True)
class DetectorConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def target_out_size(self) -> tuple[int, int]:
        """(rows, cols) of the heatmap: (ny, nx) / out_stride."""
        s = self.target.out_stride
        return self.grid.ny // s, self.grid.nx // s



# --- validation -----------------------------------------------------------------------------------


def validate(cfg: DetectorConfig) -> DetectorConfig:
    g, d, a, t, m, lo, dec, ev, op, tr = (cfg.grid, cfg.data, cfg.augment, cfg.target, cfg.model, cfg.loss,
                                          cfg.decode, cfg.eval, cfg.optim, cfg.train)

    def need(ok: bool, key: str, msg: str) -> None:
        if not ok:
            raise ConfigError(f"{key}: {msg}")

    for key, (rmin, rmax) in (("grid.x_range", g.x_range), ("grid.y_range", g.y_range), ("grid.z_range", g.z_range),
                              ("augment.scale_range", a.scale_range)):
        need(rmin < rmax, key, f"min must be < max, got {(rmin, rmax)}")
    need(all(v > 0 for v in g.voxel_size), "grid.voxel_size", f"sizes must be > 0, got {g.voxel_size}")
    nx, ny, _ = g.nx, g.ny, g.nz  # raises if a range is not a whole number of voxels

    sb = m.simple
    need(len(sb.stage_channels) == len(sb.stage_blocks) > 0, "model.simple.stage_blocks",
         f"need one block count per stage ({len(sb.stage_channels)} stages, got {len(sb.stage_blocks)})")
    rl = m.resnet.layers
    need(len(rl) > 0 and list(rl) == sorted(set(rl)) and set(rl) <= {1, 2, 3, 4}, "model.resnet.layers",
         f"must be a sorted, unique, non-empty subset of [1, 2, 3, 4], got {list(rl)}")
    s = t.out_stride
    need(s >= 2 and s & (s - 1) == 0 and s <= m.max_stride, "target.out_stride",
         f"must be a power of 2 between 2 and the backbone's max stride {m.max_stride}, got {s}")
    for key, n in (("grid.x_range", nx), ("grid.y_range", ny)):
        need(n % m.max_stride == 0, key, f"{n} cells is not divisible by the backbone stride {m.max_stride}")

    valid = [b.name.lower() for b in BoxType if b != BoxType.UNKNOWN]
    for c in d.classes:
        need(c in valid, "data.classes", f"unknown class {c!r}; valid: {valid}")
    need(len(d.classes) > 0 and len(set(d.classes)) == len(d.classes), "data.classes", "must be non-empty and unique")
    need(len(d.returns) > 0 and set(d.returns) <= {1, 2} and len(set(d.returns)) == len(d.returns),
         "data.returns", f"must be a non-empty subset of [1, 2], got {list(d.returns)}")
    for key, v in (("data.frame_interval_train", d.frame_interval_train), ("data.frame_interval_eval", d.frame_interval_eval),
                   ("data.shuffle_buffer", d.shuffle_buffer), ("target.max_objects", t.max_objects),
                   ("decode.topk", dec.topk), ("train.batch_size", tr.batch_size), ("train.epochs", tr.epochs),
                   ("train.eval_every", tr.eval_every), ("model.simple.stem_channels", sb.stem_channels),
                   ("model.neck_channels", m.neck_channels), ("model.head_channels", m.head_channels)):
        need(v >= 1, key, f"must be >= 1, got {v}")
    need(d.num_workers >= 0, "data.num_workers", f"must be >= 0, got {d.num_workers}")
    need(d.min_points_train >= 0, "data.min_points_train", f"must be >= 0, got {d.min_points_train}")
    need(all(c >= 1 for c in sb.stage_channels) and all(b >= 1 for b in sb.stage_blocks), "model.simple.stage_channels",
         "channels and block counts must be >= 1")

    need(0.0 <= a.flip_y_prob <= 1.0, "augment.flip_y_prob", f"must be in [0, 1], got {a.flip_y_prob}")
    need(0.0 <= a.rotation_range <= math.pi, "augment.rotation_range", f"must be in [0, pi], got {a.rotation_range}")
    need(a.scale_range[0] > 0, "augment.scale_range", f"must be > 0, got {a.scale_range}")
    need(0.0 < t.gaussian_overlap < 1.0, "target.gaussian_overlap", f"must be in (0, 1), got {t.gaussian_overlap}")
    need(t.min_radius >= 0, "target.min_radius", f"must be >= 0, got {t.min_radius}")

    need(len(lo.code_weights) == 8, "loss.code_weights", f"need 8 weights (offset 2, z 1, size 3, rot 2), got {len(lo.code_weights)}")
    need(lo.reg_weight >= 0 and all(w >= 0 for w in lo.code_weights), "loss.code_weights", "weights must be >= 0")
    if lo.reg_loss == "l1":
        need(lo.smooth_l1_beta is None, "loss.smooth_l1_beta", "only applies to reg_loss: smooth_l1")
    elif lo.smooth_l1_beta is not None:
        need(lo.smooth_l1_beta > 0, "loss.smooth_l1_beta", f"must be > 0, got {lo.smooth_l1_beta}")
    need(len(dec.nms_iou) == len(d.classes), "decode.nms_iou",
         f"need one IoU per class ({len(d.classes)}), got {len(dec.nms_iou)}")
    need(all(0 < v <= 1 for v in dec.nms_iou), "decode.nms_iou", "values must be in (0, 1]")
    need(0 <= dec.score_threshold < 1, "decode.score_threshold", f"must be in [0, 1), got {dec.score_threshold}")

    need(len(ev.iou_thresholds) > 0 and all(0 < v <= 1 for v in ev.iou_thresholds), "eval.iou_thresholds",
         f"values must be in (0, 1], got {list(ev.iou_thresholds)}")
    need(all(d > 0 for d in ev.center_distances), "eval.center_distances",
         f"values must be > 0, got {list(ev.center_distances)}")
    need(ev.min_points >= 0, "eval.min_points", f"must be >= 0, got {ev.min_points}")
    bins = ev.distance_bins
    need(len(bins) >= 2 and bins[0] == 0 and all(b2 > b1 for b1, b2 in zip(bins, bins[1:])), "eval.distance_bins",
         f"must start at 0 and increase, got {list(bins)}")

    need(op.lr > 0, "optim.lr", f"must be > 0, got {op.lr}")
    need(op.weight_decay >= 0, "optim.weight_decay", f"must be >= 0, got {op.weight_decay}")
    if op.name in ("adam", "adamw"):
        for k in ("momentum", "nesterov"):
            need(getattr(op, k) is None, f"optim.{k}", f"only applies to sgd, not {op.name}")
        if op.betas is not None:
            need(all(0 <= b < 1 for b in op.betas), "optim.betas", f"must be in [0, 1), got {list(op.betas)}")
    else:  # sgd
        need(op.betas is None, "optim.betas", "only applies to adam/adamw, not sgd")
        if op.momentum is not None:
            need(0 <= op.momentum < 1, "optim.momentum", f"must be in [0, 1), got {op.momentum}")
        need(not (op.nesterov and op.momentum == 0), "optim.nesterov", "requires momentum > 0")
    need(tr.grad_clip >= 0, "train.grad_clip", f"must be >= 0, got {tr.grad_clip}")
    need(0 < tr.warmup_frac < 1, "train.warmup_frac", f"must be in (0, 1), got {tr.warmup_frac}")
    need(tr.log_every >= 1, "train.log_every", f"must be >= 1, got {tr.log_every}")
    need(tr.val_max_frames >= 0, "train.val_max_frames", f"must be >= 0, got {tr.val_max_frames}")
    need(tr.best_metric.startswith(("eval/", "val/")), "train.best_metric",
         f"must be an eval/... or val/... metric name, got {tr.best_metric!r}")
    return cfg


# --- loading / saving -----------------------------------------------------------------------------


def _coerce(value: Any, tp: Any, key: str) -> Any:
    """Check/convert a YAML value against a dataclass field annotation."""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], key)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{key}: must be one of {list(args)}, got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{key}: expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{key}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{key}: expected {len(args)} values, got {len(value)}")
        return tuple(_coerce(v, a, f"{key}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{key}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
        return value
    if tp is float:
        # PyYAML (YAML 1.1) reads "1e-4" as a string; only "1.0e-4" is a float. Accept both.
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise ConfigError(f"{key}: expected a number, got {value!r}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"{key}: expected a finite number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{key}: expected a string, got {value!r}")
        return value
    raise TypeError(f"{key}: unsupported annotation {tp!r}")  # programming error, not user input


def _build(cls: type, raw: dict, prefix: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{prefix or 'config'}: expected a mapping, got {raw!r}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise ConfigError(f"{prefix + '.' if prefix else ''}{unknown[0]}: unknown key; valid: {sorted(names)}")
    kwargs = {}
    for name, value in raw.items():
        key = f"{prefix}.{name}" if prefix else name
        tp = hints[name]
        kwargs[name] = _build(tp, value, key) if dataclasses.is_dataclass(tp) else _coerce(value, tp, key)
    return cls(**kwargs)


def _apply_override(raw: dict, override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override {override!r}: expected key=value")
    key, text = override.split("=", 1)
    parts = key.strip().split(".")
    node = raw
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {key}: {p} is not a section")
    node[parts[-1]] = yaml.safe_load(text)


def config_from_dict(raw: dict | None, overrides: list[str] | None = None) -> DetectorConfig:
    raw = dict(raw or {})
    raw = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
    for o in overrides or []:
        _apply_override(raw, o)
    return validate(_build(DetectorConfig, raw, ""))


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> DetectorConfig:
    """Load a YAML config (missing fields take defaults) and apply ``section.key=value`` overrides."""
    raw = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    return config_from_dict(raw, overrides)


def config_to_dict(cfg: DetectorConfig) -> dict:
    def plain(v: Any) -> Any:
        if isinstance(v, tuple):
            return [plain(x) for x in v]
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        return v

    return plain(dataclasses.asdict(cfg))


def save_config(cfg: DetectorConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))


# --- optimizer ------------------------------------------------------------------------------------


def build_optimizer(cfg: OptimConfig, params):
    """torch optimizer from the config. torch is imported here so configs load without it."""
    import torch

    if cfg.name in ("adam", "adamw"):
        cls = torch.optim.AdamW if cfg.name == "adamw" else torch.optim.Adam
        return cls(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas or _ADAM_DEFAULT_BETAS)
    momentum = _SGD_DEFAULT_MOMENTUM if cfg.momentum is None else cfg.momentum
    return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=momentum,
                           nesterov=bool(cfg.nesterov))
