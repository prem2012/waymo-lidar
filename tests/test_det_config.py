from pathlib import Path

import pytest
import yaml

from waymo_lidar.det.config import (ConfigError, DetectorConfig, build_optimizer, config_from_dict, config_to_dict,
                                    load_config, save_config, validate)

YAML = Path(__file__).parent.parent / "configs" / "bev_centerpoint.yaml"


def test_defaults_derive_grid_and_classes():
    cfg = validate(DetectorConfig())
    assert (cfg.grid.nx, cfg.grid.ny, cfg.grid.nz) == (1504, 1504, 24)
    assert cfg.grid.in_channels == 24
    assert cfg.target_out_size == (376, 376)
    assert cfg.data.class_ids == (1, 2, 4)
    assert cfg.model.max_stride == 16 and cfg.model.stage_strides == (4, 8, 16)


@pytest.mark.parametrize("backbone, layers, strides", [
    ("resnet18", "[1, 2, 3, 4]", (4, 8, 16, 32)), ("resnet50", "[1, 2, 3]", (4, 8, 16)), ("resnet34", "[2, 4]", (8, 32)),
])
def test_resnet_backbone_strides(backbone, layers, strides):
    cfg = load_config(overrides=[f"model.backbone={backbone}", f"model.resnet.layers={layers}"])
    assert cfg.model.stage_strides == strides and cfg.model.max_stride == max(strides)


def test_extra_channels():
    assert load_config(overrides=["grid.extra_channels=true"]).grid.in_channels == 26


def test_yaml_file_loads_and_matches_defaults_except_explicit_betas():
    cfg = load_config(YAML)
    assert cfg.grid.nx == 1504 and cfg.optim.betas == (0.9, 0.99)
    d_file, d_default = config_to_dict(cfg), config_to_dict(DetectorConfig())
    d_file["optim"]["betas"] = None  # the only field the YAML sets differently from the dataclass default
    assert d_file == d_default


def test_round_trip(tmp_path):
    cfg = load_config(YAML, overrides=["train.batch_size=8", "eval.iou_thresholds=[0.5, 0.7]"])
    save_config(cfg, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml") == cfg


def test_overrides_typed():
    cfg = load_config(YAML, overrides=["train.batch_size=8", "grid.extra_channels=true", "optim.lr=1e-4",
                                       "data.classes=[vehicle]", "decode.nms_iou=[0.7]"])
    assert cfg.train.batch_size == 8 and cfg.grid.extra_channels is True
    assert cfg.optim.lr == pytest.approx(1e-4) and isinstance(cfg.optim.lr, float)
    assert cfg.data.classes == ("vehicle",) and cfg.data.class_ids == (1,)


def test_int_accepted_for_float():
    assert load_config(overrides=["train.grad_clip=10"]).train.grad_clip == 10.0


def test_exponent_without_dot_is_a_float(tmp_path):
    # PyYAML parses "1e-4" as a string (YAML 1.1); both overrides and files must accept it.
    assert load_config(overrides=["optim.lr=1e-4"]).optim.lr == pytest.approx(1e-4)
    (tmp_path / "c.yaml").write_text("optim:\n  lr: 1e-4\n")
    assert load_config(tmp_path / "c.yaml").optim.lr == pytest.approx(1e-4)
    with pytest.raises(ConfigError, match="optim.lr"):
        load_config(overrides=["optim.lr=fast"])
    with pytest.raises(ConfigError, match="optim.lr"):
        load_config(overrides=["optim.lr=nan"])


@pytest.mark.parametrize("overrides, key", [
    (["grid.x_range=[-75.25, 75.2]"], "grid.x_range"),            # not a whole number of voxels
    (["grid.x_range=[-75.0, 75.0]"], "grid.x_range"),             # 1500 cells, not divisible by 16
    (["target.out_stride=3"], "target.out_stride"),
    (["data.classes=[vehicle, truck]", "decode.nms_iou=[0.7, 0.5]"], "data.classes"),
    (["data.classes=[vehicle, vehicle]", "decode.nms_iou=[0.7, 0.5]"], "data.classes"),
    (["data.returns=[3]"], "data.returns"),
    (["decode.nms_iou=[0.7]"], "decode.nms_iou"),                 # one per class
    (["loss.code_weights=[1, 1]"], "loss.code_weights"),
    (["grid.z_range=[4.0, -2.0]"], "grid.z_range"),
    (["model.simple.stage_blocks=[2, 3]"], "model.simple.stage_blocks"),
    (["model.backbone=resnet101"], "model.backbone"),
    (["model.resnet.layers=[3, 1]"], "model.resnet.layers"),
    (["model.resnet.layers=[]"], "model.resnet.layers"),
    (["model.resnet.layers=[5]"], "model.resnet.layers"),
    (["model.stage_blocks=[2, 3]"], "model.stage_blocks"),       # old flat key -> unknown key
    (["eval.distance_bins=[10, 30]"], "eval.distance_bins"),
    (["train.batch_size=0"], "train.batch_size"),
    (["train.precision=fp8"], "train.precision"),
    (["train.bogus=1"], "train.bogus"),                           # unknown key
    (["bogus.x=1"], "bogus"),                                     # unknown section
    (["train.batch_size=four"], "train.batch_size"),              # wrong type
    (["train.batch_size=4.5"], "train.batch_size"),
    (["grid.extra_channels=1"], "grid.extra_channels"),           # int is not a bool
    (["grid.voxel_size=[0.1, 0.1]"], "grid.voxel_size"),          # wrong length
    (["optim.name=rmsprop"], "optim.name"),
    (["optim.momentum=0.9"], "optim.momentum"),                   # sgd-only field with adamw
    (["optim.nesterov=true"], "optim.nesterov"),
    (["optim.name=sgd"], "optim.betas"),                          # YAML sets betas, which sgd doesn't use
    (["optim.name=sgd", "optim.betas=null", "optim.momentum=0", "optim.nesterov=true"], "optim.nesterov"),
    (["optim.lr=0"], "optim.lr"),
    (["loss.reg_loss=l2"], "loss.reg_loss"),
    (["loss.smooth_l1_beta=0.1"], "loss.smooth_l1_beta"),                      # only with smooth_l1
    (["loss.reg_loss=smooth_l1", "loss.smooth_l1_beta=0"], "loss.smooth_l1_beta"),
])
def test_validation_errors_name_the_key(overrides, key):
    with pytest.raises(ConfigError, match=key.replace("[", r"\[")):
        load_config(YAML, overrides=overrides)


def test_smooth_l1_options():
    cfg = load_config(YAML, overrides=["loss.reg_loss=smooth_l1"])
    assert cfg.loss.reg_loss == "smooth_l1" and cfg.loss.smooth_l1_beta is None
    assert load_config(YAML, overrides=["loss.reg_loss=smooth_l1", "loss.smooth_l1_beta=0.05"]).loss.smooth_l1_beta == 0.05


def test_bad_override_syntax():
    with pytest.raises(ConfigError, match="key=value"):
        load_config(overrides=["train.batch_size"])


def test_config_from_dict_rejects_non_mapping_section():
    with pytest.raises(ConfigError, match="train"):
        config_from_dict({"train": 4})


def test_saved_yaml_is_plain():
    text = yaml.safe_dump(config_to_dict(DetectorConfig()))
    assert "!!python" not in text  # loadable by any YAML reader


@pytest.mark.parametrize("overrides, cls, expect", [
    ([], "AdamW", {"lr": 3e-3, "weight_decay": 0.01, "betas": (0.9, 0.99)}),
    (["optim.name=adam", "optim.betas=[0.8, 0.9]"], "Adam", {"betas": (0.8, 0.9)}),
    (["optim.name=sgd", "optim.betas=null"], "SGD", {"momentum": 0.9, "nesterov": False}),
    (["optim.name=sgd", "optim.betas=null", "optim.momentum=0.5", "optim.nesterov=true", "optim.lr=0.1"], "SGD",
     {"momentum": 0.5, "nesterov": True, "lr": 0.1}),
])
def test_build_optimizer(overrides, cls, expect):
    torch = pytest.importorskip("torch")
    cfg = load_config(YAML, overrides=overrides)
    opt = build_optimizer(cfg.optim, torch.nn.Linear(3, 2).parameters())
    assert type(opt).__name__ == cls
    for k, v in expect.items():
        got = opt.param_groups[0][k]
        assert (tuple(got) if isinstance(got, (list, tuple)) else got) == pytest.approx(v)


def test_torch_cuda_available():
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()
