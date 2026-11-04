"""ONNX export (+ TensorRT engine when the `tensorrt` package is available) on a small-grid checkpoint."""

import json

import numpy as np
import onnx
import pytest
import torch

from waymo_lidar.det import export as E
from waymo_lidar.det.config import load_config
from waymo_lidar.det.model import BEVDetector
from waymo_lidar.det.train import load_model, save_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SMALL = ["grid.x_range=[-25.6, 25.6]", "grid.y_range=[-25.6, 25.6]"]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("ck") / "ck.pt"
    cfg = load_config(overrides=SMALL)
    torch.manual_seed(0)
    model = BEVDetector.from_config(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(path, model, opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0), cfg, 3, 42, 0.0)
    return path


def test_load_model_enables_cudnn_benchmark(ckpt):
    torch.backends.cudnn.benchmark = False
    load_model(ckpt)
    assert torch.backends.cudnn.benchmark


@pytest.mark.parametrize("precision,elem", [("fp16", onnx.TensorProto.FLOAT16), ("fp32", onnx.TensorProto.FLOAT)])
def test_export_onnx(tmp_path, ckpt, precision, elem):
    path, info = E.export_onnx(ckpt, tmp_path / f"m_{precision}.onnx", precision)
    m = onnx.load(str(path))  # loads the external .data weights too
    onnx.checker.check_model(m)
    io = {v.name: v for v in [*m.graph.input, *m.graph.output]}
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    assert dims(io["bev"]) == info["input"]["shape"] == [1, 24, 512, 512]
    assert dims(io["heatmap"]) == info["outputs"]["heatmap"]["shape"] == [1, 3, 128, 128]
    assert dims(io["reg"]) == info["outputs"]["reg"]["shape"] == [1, 8, 128, 128]
    assert all(v.type.tensor_type.elem_type == elem for v in io.values())
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["precision"] == precision and meta["epoch"] == 3 and meta["step"] == 42
    assert meta["outputs"]["reg"]["fields"] == ["dx", "dy", "z", "log_l", "log_w", "log_h", "sin", "cos"]
    assert meta["config"]["grid"]["x_range"] == [-25.6, 25.6]


def test_tensorrt_engine_matches_pytorch(tmp_path, ckpt):
    pytest.importorskip("tensorrt")  # run with: uv run --with tensorrt-cu13 pytest tests/test_det_export.py
    path, _ = E.export_onnx(ckpt, tmp_path / "m.onnx", "fp32")
    engine = E.build_engine(path, tmp_path / "m.plan")
    assert (tmp_path / "m.plan").stat().st_size > 0
    r = E.check_engine(engine, ckpt)
    assert r["max_abs_sigmoid_heatmap"] < 1e-3 and r["max_abs_reg"] < 1e-2, r
    assert r["p50_ms"] > 0
    # The input is written on the caller's (default) stream behind a ~50 ms GPU stall; a runner that doesn't wait
    # for that stream reads the buffer before the write lands and returns the outputs for an empty grid.
    runner = E.TrtRunner(engine)
    model, cfg, _ = load_model(ckpt)
    model.eval()
    g = cfg.grid
    gen = torch.Generator(device="cuda").manual_seed(1)
    src = (torch.rand((1, 24, g.ny, g.nx), device="cuda", generator=gen) < 0.02).float()
    with torch.no_grad():
        ref = model(src.contiguous(memory_format=torch.channels_last))["heatmap"].float().cpu().numpy()
        x = torch.zeros_like(src)
        torch.cuda.synchronize()
        torch.cuda._sleep(100_000_000)  # GPU cycles on the current stream
        x.copy_(src)
        got = runner(x)["heatmap"].float().cpu().numpy()
    np.testing.assert_allclose(got, ref, atol=0.05)
