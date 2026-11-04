# Export: ONNX and TensorRT

Part of the [detector](README.md). Code: `src/waymo_lidar/det/export.py`; tests:
`tests/test_det_export.py`.

## Usage

```bash
# ONNX (fp16 by default) -> runs/baseline/export/best_fp16.onnx (+ .onnx.data weights, .json metadata)
uv run wl export runs/baseline/checkpoints/best.pt
uv run wl export runs/baseline/checkpoints/best.pt --precision fp32 --out exports/baseline_fp32.onnx

# Also build a TensorRT engine (.plan) here and check it against PyTorch. TensorRT isn't a project
# dependency (it's platform-specific), so add it for the one command:
uv run --with tensorrt-cu13 wl export runs/baseline/checkpoints/best.pt --engine
```

**What's exported:** the network only. The input is a `(1, C, ny, nx)` BEV grid, `(1, 24, 1504, 1504)` by
default. The outputs are `heatmap` logits `(1, classes, ny/4, nx/4)` and `reg` `(1, 8, ny/4, nx/4)`.
[Voxelization](voxelize.md) and [decoding](decode.md) stay outside: together they're ~1 ms, and they need a
scatter and top-K that are simpler to run as they are. The `.json` next to the ONNX holds what a deployment
needs to do those two steps: input/output shapes and dtypes, class order, output stride, the `reg` channel
layout (`dx, dy, z, log_l, log_w, log_h, sin, cos`), and the full config (grid, decode settings).

**Precision is baked into the ONNX.** TensorRT 10+ builds *strongly typed* engines: every layer's
precision comes from the ONNX types, so an fp16 export is an fp16 engine. `--precision` is `fp16` or
`fp32`. bf16 isn't offered: TensorRT 11.4 has no bf16 implementation of the neck's `ConvTranspose`, so the
engine build fails.

**`--engine`** builds `<out>.plan` with the TensorRT Python API, then runs it on a synthetic occupancy grid
against the PyTorch model in strict fp32 (TF32 off). It reports the largest differences and the latency.
Engines are specific to the GPU and TensorRT version that built them. Build on the target.

## On a Jetson Orin

Copy **both** `best_fp16.onnx` and `best_fp16.onnx.data`, keeping them in the same directory, then build
with JetPack's `trtexec`:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=best_fp16.onnx --stronglyTyped --saveEngine=best_fp16.plan
/usr/src/tensorrt/bin/trtexec --loadEngine=best_fp16.plan --useCudaGraph      # latency
```

Set the power mode first, e.g. `sudo nvpmodel -m 0 && sudo jetson_clocks` for MAXN. Estimated from the
model's cost (386 GFLOPs per frame) and Orin's specs, not measured: ~10–15 ms for the network on AGX Orin
64 GB, and ~13–19 ms per frame with voxelize and decode.

## Benchmark (RTX 4090)

Network only, batch 1, `runs/baseline` best checkpoint, TensorRT 11.4 / PyTorch 2.15. AP is on 40 frames of
one validation segment, so differences under ~0.01 are noise.

| Variant | p50 | BEV AP@0.7 | BEV AP@0.5 |
|---|---:|---:|---:|
| **TensorRT fp16** | **2.47 ms** | 0.331 | 0.663 |
| TensorRT fp32 (TF32 on or off) | 6.3–6.4 ms | 0.333 | 0.666 |
| PyTorch bf16 autocast, `cudnn.benchmark` (what `wl predict` / `wl eval` run) | 4.67 ms | 0.341 | 0.659 |
| PyTorch bf16 autocast, without `cudnn.benchmark` | 5.6 ms | | |
| PyTorch fp32 with TF32 (PyTorch's default) | 11.5 ms | | |
| PyTorch strict fp32 | 14.4 ms (`cudnn.benchmark`) / 41 ms | 0.333 | 0.666 |

Add ~0.65 ms (voxelize) and ~0.4 ms (decode) for a full frame. Batching doesn't raise throughput: one
1504² frame already fills the GPU.

On real frames, fp16 raw outputs differ from fp32 by up to ~0.05 in heatmap probability and ~2 in `reg`.
These differences are in background cells the decoder discards: detections and AP don't change beyond noise.
The synthetic-grid check in `--engine` is tighter (≤ 4e-4 / 2e-2). It confirms the engine computes the
graph; it says nothing about data accuracy.

## Running an engine from Python

```python
import tensorrt as trt
from waymo_lidar.det.export import TrtRunner
engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(open("best_fp16.plan", "rb").read())
run = TrtRunner(engine)
out = run(bev)              # bev: (1, C, ny, nx) CUDA tensor, any dtype/layout -> {"heatmap", "reg"}
dets = decode(out, cfg)     # the usual decode
```

`TrtRunner` runs on its own CUDA stream. It first makes that stream wait for the caller's stream, because
`bev` may still be being written there. A test stalls the caller's stream to make sure the runner waits.
The output buffers are reused, so copy them before the next call if you keep them.
