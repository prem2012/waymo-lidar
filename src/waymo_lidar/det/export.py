"""Export the detector network to ONNX (and optionally build + check a TensorRT engine).

    uv run wl export runs/baseline/checkpoints/best.pt --out exports/baseline_fp16.onnx
    uv run --with tensorrt-cu13 wl export runs/baseline/checkpoints/best.pt --out exports/baseline_fp16.onnx --engine

Only the network is exported: (1, C, ny, nx) BEV grid -> heatmap logits (1, classes, H, W) and reg (1, 8, H, W).
Voxelization and decoding stay outside (they're ~1 ms and need custom scatter / top-K logic). A JSON file next
to the ONNX records the grid, classes, output stride and reg layout a deployment needs. See
docs/detector/export.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import config_to_dict
from .targets import REG_FIELDS
from .train import load_model

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16}  # bf16: TensorRT 11.4 has no bf16 ConvTranspose
INPUT, OUTPUTS = "bev", ("heatmap", "reg")


class _Network(torch.nn.Module):
    """The detector with a (heatmap, reg) tuple output, as ONNX graphs want."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(bev)
        return out["heatmap"], out["reg"]


def export_onnx(ckpt: str | Path, out: str | Path, precision: str = "fp16", device: str = "cuda") -> tuple[Path, dict]:
    """Writes <out> (+ <out>.data with the weights) and <out minus .onnx>.json; returns (onnx path, metadata).

    The whole graph is in `precision` (weights, activations, outputs): TensorRT's strongly typed builder takes
    precision from the ONNX types, so an fp16 export is an fp16 engine.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg, meta = load_model(ckpt, device)
    dtype = PRECISIONS[precision]
    net = _Network(copy.deepcopy(model).to(dtype)).eval()
    g = cfg.grid
    shape = (1, g.in_channels, g.ny, g.nx)
    torch.onnx.export(net, (torch.zeros(shape, device=device, dtype=dtype),), str(out),
                      input_names=[INPUT], output_names=list(OUTPUTS), dynamo=True)
    s = cfg.target.out_stride
    info = {
        "checkpoint": str(ckpt), "epoch": meta["epoch"], "step": meta["step"], "precision": precision,
        "input": {"name": INPUT, "shape": list(shape), "layout": "NCHW: z slices (+ extra BEV channels) x y x x",
                  "dtype": precision},
        "outputs": {"heatmap": {"shape": [1, len(cfg.data.classes), g.ny // s, g.nx // s], "activation": "sigmoid"},
                    "reg": {"shape": [1, len(REG_FIELDS), g.ny // s, g.nx // s], "fields": list(REG_FIELDS)}},
        "classes": list(cfg.data.classes), "out_stride": s, "config": config_to_dict(cfg),
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return out, info


def build_engine(onnx_path: str | Path, plan_path: str | Path, workspace_gib: int = 8):
    """Strongly typed TensorRT engine from the ONNX file (needs the `tensorrt` package). Returns the engine."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed (see the TensorRT log above)")
    Path(plan_path).write_bytes(bytes(plan))
    return trt.Runtime(logger).deserialize_cuda_engine(plan)


class TrtRunner:
    """Runs a TensorRT engine on torch CUDA tensors (on a non-default stream, as TensorRT prefers)."""

    def __init__(self, engine):
        import tensorrt as trt

        to_torch = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16}
        self.ctx = engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.bufs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            self.bufs[name] = torch.empty(tuple(engine.get_tensor_shape(name)), device="cuda",
                                          dtype=to_torch[engine.get_tensor_dtype(name)])
            self.ctx.set_tensor_address(name, self.bufs[name].data_ptr())

    def __call__(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        self.stream.wait_stream(torch.cuda.current_stream())  # `bev` may still be being written on the caller's stream
        with torch.cuda.stream(self.stream):
            self.bufs[INPUT].copy_(bev)  # dtype cast + NCHW-contiguous
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(self.stream)
        return {k: self.bufs[k] for k in OUTPUTS}

    def time_ms(self, iters: int = 200, warmup: int = 30) -> tuple[float, float]:
        """p50 / p95 of the engine alone (input already in place)."""
        with torch.cuda.stream(self.stream):
            for _ in range(warmup):
                self.ctx.execute_async_v3(self.stream.cuda_stream)
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
            for a, b in ev:
                a.record()
                self.ctx.execute_async_v3(self.stream.cuda_stream)
                b.record()
        self.stream.synchronize()
        ms = np.array([a.elapsed_time(b) for a, b in ev])
        return float(np.median(ms)), float(np.percentile(ms, 95))


@torch.no_grad()
def check_engine(engine, ckpt: str | Path, seed: int = 0) -> dict:
    """Compare the engine with the PyTorch model (strict fp32) on a synthetic occupancy grid, and time it.

    This checks that the engine computes the exported graph. Accuracy on real data (AP) is a separate check:
    on real frames fp16 differs more in raw outputs (background cells), with AP unchanged (see export.md).
    """
    model, cfg, _ = load_model(ckpt, "cuda")
    model.eval()
    g = cfg.grid
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bev = (torch.rand((1, g.in_channels, g.ny, g.nx), device="cuda", generator=gen) < 0.01).float()
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        ref = model(bev.contiguous(memory_format=torch.channels_last))
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    runner = TrtRunner(engine)
    got = runner(bev)
    p50, p95 = runner.time_ms()
    return {
        "max_abs_sigmoid_heatmap": float((torch.sigmoid(got["heatmap"].float()) - torch.sigmoid(ref["heatmap"])).abs().max()),
        "max_abs_reg": float((got["reg"].float() - ref["reg"]).abs().max()),
        "p50_ms": p50, "p95_ms": p95,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="wl export", description="Export the detector network to ONNX / TensorRT")
    p.add_argument("ckpt", type=Path)
    p.add_argument("--out", type=Path, help="ONNX path (default: <run>/export/<ckpt stem>_<precision>.onnx)")
    p.add_argument("--precision", choices=sorted(PRECISIONS), default="fp16")
    p.add_argument("--engine", action="store_true",
                   help="also build a TensorRT engine (<out>.plan) and check it against PyTorch; needs `tensorrt`")
    a = p.parse_args(argv)
    out = a.out or a.ckpt.parent.parent / "export" / f"{a.ckpt.stem}_{a.precision}.onnx"
    t = time.perf_counter()
    onnx_path, info = export_onnx(a.ckpt, out, a.precision)
    print(f"wrote {onnx_path} (+ .data, .json) in {time.perf_counter() - t:.0f} s: input {info['input']['shape']} "
          f"{a.precision} -> heatmap {info['outputs']['heatmap']['shape']}, reg {info['outputs']['reg']['shape']}")
    if not a.engine:
        print(f"TensorRT: trtexec --onnx={onnx_path} --stronglyTyped --saveEngine={onnx_path.with_suffix('.plan')}")
        return
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        raise SystemExit("--engine needs TensorRT's Python package, e.g. `uv run --with tensorrt-cu13 wl export ...`")
    t = time.perf_counter()
    engine = build_engine(onnx_path, onnx_path.with_suffix(".plan"))
    print(f"built {onnx_path.with_suffix('.plan')} in {time.perf_counter() - t:.0f} s")
    r = check_engine(engine, a.ckpt)
    print(f"engine vs PyTorch fp32: max |Δ sigmoid(heatmap)| {r['max_abs_sigmoid_heatmap']:.2e}, "
          f"max |Δ reg| {r['max_abs_reg']:.2e}; latency p50 {r['p50_ms']:.2f} ms, p95 {r['p95_ms']:.2f} ms")


if __name__ == "__main__":
    main(sys.argv[1:])
