# Detector (Phase 2)

A single-frame LiDAR 3D detector: top-LiDAR points → binary occupancy voxels → a 2D CNN over the
bird's-eye view (BEV) → a CenterPoint head. It is built one component at a time, and each component
gets its own doc in this folder when it's built.

Related docs: [dataset.md](../dataset.md) (the data) and
[analytics-viewer-architecture.md](../analytics-viewer-architecture.md) (Phase 1, including the shared
`waymo_lidar.data` layer this detector reads from).

## Design

| Choice | Value | Why |
|---|---|---|
| Frame | Vehicle frame at the frame timestamp, for points and boxes: x forward, y left, z up, origin on the ground and most likely below the rear axle ([definition](../dataset.md#coordinate-frames)). The top LiDAR sits at (+1.43, 0, +2.18) m. | Boxes are labelled in this frame, and points are moved into it with rolling-shutter correction, so the two line up with no further transform. |
| Input | top LiDAR, single frame, returns 1 + 2 | The 75 m sensor covers the grid. |
| Grid | x, y ∈ [−75.2, 75.2) m at 0.1 m → 1504 × 1504; z ∈ [−2, 4) m at 0.25 m → 24 slices | Covers the top LiDAR's range and the height of trucks, with margin for hills. The grid is centred on the vehicle origin, not the sensor: along x it spans −76.6 … +73.8 m from the top LiDAR, and z ∈ [−2, 4) is 2 m below to 4 m above the ground under the car. |
| Input tensor | (24, 1504, 1504): binary occupancy, z slices as image channels; optionally + 2 BEV channels (log1p point count, max intensity) | Ordinary 2D convolutions work on it (as in PIXOR). The extras add the density and reflectance that binary occupancy drops. |
| Backbone | selectable: `simple` (2D residual CNN: stem stride 2, stages at strides 4 / 8 / 16) or torchvision `resnet18` / `resnet34` / `resnet50`; a shared neck merges every level back to stride 4 | Downsampling early keeps memory manageable at 1504². The standard ResNets are there to compare against. |
| Head | CenterPoint at stride 4 (376 × 376, 0.4 m cells): per-class heatmap + offset, z, log size, sin/cos heading | Anchor-free and simple to decode. A 0.4 m cell is fine enough for pedestrians. |
| Classes | vehicle, pedestrian, cyclist | The three classes Waymo ranks. |
| Metrics | BEV AP and 3D AP at IoU 0.5 / 0.7 / 0.8, per class and distance bin | Chosen for this project. Ground-truth boxes with 0 top-LiDAR points don't count. |

## Components

| Component | Code | Doc | Status |
|---|---|---|---|
| Config | `det/config.py`, `configs/bev_centerpoint.yaml` | [config.md](config.md) | **done** |
| Data stream | `det/data.py` | [data.md](data.md) | **done** |
| Voxelizer + voxel viewer | `det/voxelize.py`, `det/viz_voxels.py` | [voxelize.md](voxelize.md) | **done** |
| Targets | `det/targets.py` | [targets.md](targets.md) | **done** |
| Model | `det/model.py` | [model.md](model.md) | **done** |
| Loss | `det/loss.py` | [loss.md](loss.md) | **done** |
| Decode | `det/decode.py` | | planned |
| IoU | `det/iou.py` | | planned |
| Evaluation | `det/eval.py` | | planned |
| Training | `det/train.py` | | planned |
| Prediction + viewer | `det/predict.py`, `viz/stream.py` | | planned |
