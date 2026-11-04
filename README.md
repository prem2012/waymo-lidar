# waymo-lidar

LiDAR 3D object detection on the [Waymo Open Dataset](https://waymo.com/open) (Perception v2, Parquet):
data tooling, a dataset analytics and Rerun viewer, and a single-frame BEV CenterPoint detector with
training, evaluation, prediction viewing and ONNX / TensorRT export.

```bash
uv sync
./download.sh training 2                   # needs a registered Waymo account, see docs/dataset.md
uv run wl --help
```

| Doc | What's in it |
|---|---|
| [docs/dataset.md](docs/dataset.md) | The dataset: access, layout, labels, coordinate frames, gotchas |
| [docs/analytics-viewer-architecture.md](docs/analytics-viewer-architecture.md) | The shared data layer, analytics page and 3D viewer |
| [docs/detector/README.md](docs/detector/README.md) | The detector. Quick start for training, TensorBoard, Rerun predictions and export |

## License and data terms

The code in this repository is released under the [MIT License](LICENSE). That covers this code only,
**not the Waymo Open Dataset**, which Waymo publishes under the
[Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms) (March 2025
version). Using the data with this code means registering at [waymo.com/open](https://waymo.com/open)
and accepting that agreement yourself.

**Attribution** (required by Section 1 of the agreement for software made using the dataset):

> This software was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset
> License Agreement for Non-Commercial Use, available at
> [waymo.com/open/terms](https://waymo.com/open/terms), and your access and use of such work are
> governed by the terms and conditions therein.

```bibtex
@misc{waymo_open_dataset,
  title   = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year    = {2019-2025}
}
```

What that means for this repository:

- **No dataset files are included.** You download them yourself (see [docs/dataset.md](docs/dataset.md#access)).
  The docs contain only small extracts used for illustration: summary statistics, metrics, segment
  names. Don't commit or share dataset files or bulk extracts of them, such as point clouds, labels,
  or `.rrd` recordings. Sharing them is allowed only with people who have registered and accepted the
  agreement.
- **Trained models are not included either.** Checkpoints and exported networks (`runs/`, `*.pt`,
  `*.onnx`, `*.plan`) are ignored by git. Under Section 4, models trained on the dataset may be used
  for non-commercial purposes only, which means research, teaching, publication and personal
  experimentation. They may **not** be used to operate a vehicle or assist in operating one, in
  production systems, or for primarily commercial purposes. That includes the TensorRT / Jetson Orin
  export path described in [docs/detector/export.md](docs/detector/export.md): it is for latency
  research, not deployment. Publishing trained weights is allowed (Section 2(c)) only with a prominent
  notice that they're released under the agreement, a copy of the agreement, and the attribution above.

This is a summary for convenience, not legal advice. The agreement on Waymo's site is the binding text.
