# Waymo Open Dataset — Perception v2.0.1

What the data is, how it is stored, and what labels exist. Everything here was checked against the
actual files in `gs://waymo_open_dataset_v_2_0_1` (November 2026) or against Waymo's own definitions;
where the two disagree, the data wins and the difference is noted under [Gotchas](#gotchas).

For how this project reads the data, see [architecture.md](architecture.md).

---

## At a glance

| | |
|---|---|
| Content | 20 s driving segments with LiDAR, cameras, poses and labels |
| Segments | 798 training · 202 validation · 150 test (16 `testing` + 134 `testing_location`) |
| Frames | 10 Hz, ~198 per segment · 158,081 training · 39,987 validation |
| Sensors | 5 LiDARs (1 mid-range top + 4 short-range), 5 cameras |
| 3D boxes | 12.1 M (training + validation): vehicle, pedestrian, cyclist, sign |
| Other labels | 2D camera boxes, LiDAR semantic segmentation, camera panoptic segmentation, human keypoints, 2D↔3D associations |
| Format | Apache Parquet, one file per segment per component |
| Size | ~474 GiB training · ~120 GiB validation (all components) |
| LiDAR-only size | ~148 GiB training · ~37 GiB validation (what this project downloads) |
| Locations | San Francisco, Phoenix, other |
| License | Non-commercial; register at [waymo.com/open](https://waymo.com/open) |

---

## Access

1. Register at [waymo.com/open](https://waymo.com/open) and accept the license with a Google account.
2. `gcloud auth login` with that account. The buckets are readable only after step 1.
3. Data lives at `gs://waymo_open_dataset_v_2_0_1/<split>/<component>/<segment>.parquet`.

This project's [`download.sh`](../download.sh) fetches the LiDAR-only components for N segments of a split,
and can be stopped and restarted (it skips files already downloaded):

```bash
./download.sh training 2       # first 2 training segments
./download.sh validation all   # every validation segment
```

---

## Splits

| Split | Segments | Frames | Labels |
|---|---:|---:|---|
| `training` | 798 | 158,081 | yes |
| `validation` | 202 | 39,987 | yes |
| `testing` | 16 | 3,101 | **none** (label tables exist but have 0 rows) |
| `testing_location` | 134 | 26,546 | **none** |

Every component has exactly one file per segment in every split. Test splits are for leaderboard
submissions; use `validation` for local evaluation.

---

## Storage layout

```
<split>/
  lidar/                <segment>.parquet
  lidar_box/            <segment>.parquet
  ...one folder per component...
```

A **segment name** looks like `10017090168044687777_6380_000_6400_000`. It appears to be a run ID
followed by the start and end of the 20 s window within that run (6380.000 → 6400.000 s).

Each file is a table. Rows are identified by **key columns**, and components are combined by joining
on them:

| Key column | Type | Meaning |
|---|---|---|
| `key.segment_context_name` | string | segment name (same as the file name) |
| `key.frame_timestamp_micros` | int64 | frame time, microseconds since Unix epoch |
| `key.laser_name` | int8 | LiDAR ID (see [LiDARs](#lidars)) |
| `key.camera_name` | int8 | camera ID (see [Cameras](#cameras)) |
| `key.laser_object_id` | string | 3D object ID, stable across frames of a segment (tracking ID) |
| `key.camera_object_id` | string | 2D object ID, stable across frames of a segment |

Data columns are named `[<Component>Component].<field>`, e.g. `[LiDARBoxComponent].box.center.x`.
Arrays (range images, matrices) are stored as a flat `values` list plus a `shape` column, row-major.
4×4 transforms are a 16-element row-major list.

---

## Components

Sizes are totals over the split, in GiB. "Per" is what one row describes.

### LiDAR

| Component | Per | Contents | Train | Val |
|---|---|---|---:|---:|
| `lidar` | frame × LiDAR | range images, returns 1 and 2 | 127.8 | 32.2 |
| `lidar_pose` | frame (top LiDAR only) | per-pixel vehicle pose, for rolling-shutter correction | 19.2 | 4.8 |
| `lidar_calibration` | segment × LiDAR | extrinsic, beam inclinations | <0.01 | <0.01 |
| `lidar_box` | frame × object | 3D boxes | 0.61 | 0.15 |
| `lidar_segmentation` | labelled frame (top LiDAR) | per-pixel semantic class + instance ID | 0.60 | 0.15 |
| `lidar_hkp` | frame × object | 3D human keypoints | 0.01 | <0.01 |
| `lidar_camera_projection` | frame × LiDAR | where each LiDAR pixel lands in the cameras | 55.4 | 13.9 |
| `lidar_camera_synced_box` | frame × object | 3D box shifted to the camera capture time | 0.23 | 0.06 |

### Camera

| Component | Per | Contents | Train | Val |
|---|---|---|---:|---:|
| `camera_image` | frame × camera | JPEG image, pose, velocity, rolling-shutter timing | 268.3 | 68.2 |
| `camera_calibration` | segment × camera | intrinsics, distortion, extrinsic, image size | 0.01 | <0.01 |
| `camera_box` | frame × camera × object | 2D boxes | 0.13 | 0.04 |
| `camera_segmentation` | labelled frame × camera | panoptic segmentation PNG | 1.35 | 0.19 |
| `camera_hkp` | frame × camera × object | 2D (+3D) human keypoints | 0.03 | 0.01 |
| `projected_lidar_box` | frame × camera × object | 3D boxes projected to 2D image boxes | 0.25 | 0.06 |
| `camera_to_lidar_box_association` | frame × camera × object | camera object ID ↔ LiDAR object ID | 0.01 | <0.01 |

### Shared

| Component | Per | Contents | Train | Val |
|---|---|---|---:|---:|
| `vehicle_pose` | frame | 4×4 `world_from_vehicle` | 0.03 | 0.01 |
| `stats` | frame | time of day, location, weather, object counts | 0.02 | <0.01 |

This project downloads `lidar`, `lidar_pose`, `lidar_calibration`, `lidar_box`, `vehicle_pose` and `stats`.

---

## Coordinate frames

| Frame | Definition |
|---|---|
| **Vehicle** | **x forward, y left, z up** (right-handed), origin at ground level (box bottoms sit at z ≈ 0). Boxes and decoded points are in this frame. |
| **Sensor** | Each LiDAR and camera has its own frame; the calibration `extrinsic` is the 4×4 `vehicle_from_sensor`. |
| **World** | A fixed global frame. `vehicle_pose` gives `world_from_vehicle` per frame. Translations are large (thousands of metres), so subtract a reference pose before working in float32. |

**Box heading** is the rotation about +z that turns +x onto the box's front face, in [-π, π).
Heading 0 means the box faces the same way as the ego vehicle.

---

## LiDARs

| ID | Name | Range image (H × W) | Max range | Inclination | Points / frame (return 1) |
|---:|---|---|---:|---|---:|
| 1 | `TOP` | 64 × 2650 | 75 m | −17.7° … +2.2°, non-uniform | ~137,000 |
| 2 | `FRONT` | 200 × 600 | 20 m | −90° … +30°, uniform | ~3,600 |
| 3 | `SIDE_LEFT` | 200 × 600 | 20 m | −90° … +30°, uniform | ~4,200 |
| 4 | `SIDE_RIGHT` | 200 × 600 | 20 m | −90° … +30°, uniform | ~4,000 |
| 5 | `REAR` | 200 × 600 | 20 m | −90° … +30°, uniform | ~4,500 |

Point counts are averages over the stage-1 segments. The short-range LiDARs mainly cover the area right
around the car that the top LiDAR can't see.

### Range images

Each LiDAR delivers a **range image** per frame instead of a point list. Pixel (row, col) is one laser
firing; `shape` is `[H, W, 4]`:

| Channel | Meaning |
|---|---|
| 0 | range in metres; **≤ 0 (stored as −1) = no return** |
| 1 | intensity (unitless, heavy-tailed; median ~0.1, retroreflectors > 1) |
| 2 | elongation (pulse stretching; high on slanted or partial hits) |
| 3 | in a no-label zone: `1` = yes, `-1` = no |

- **Two returns.** One pulse can produce two echoes. Return 2 is the second one along the same ray,
  typically from vegetation, glass or object edges. Return 2 has about 7% as many points as return 1 for the top LiDAR, and
  about 1–3% for the short-range ones.
- **Rows** are beam elevations: row 0 is the **highest** beam. The top LiDAR's 64 inclinations are in
  `beam_inclination.values` (stored ascending, so reverse them to index by row). The others only give
  `min`/`max`, so the beams are evenly spaced.
- **Columns** are azimuth, sweeping a full 360°: column 0 is at +π (behind the sensor) and angles
  decrease to the right, corrected by the sensor's yaw in its extrinsic.
- **Rolling shutter (top LiDAR).** A sweep takes ~0.1 s, so the car moves during it. `lidar_pose` gives
  each pixel's `world_from_vehicle` pose as `[roll, pitch, yaw, x, y, z]`, with rotation
  Rz(yaw)·Ry(pitch)·Rx(roll). Correction: sensor → vehicle (extrinsic) → world (pixel pose) → vehicle
  at frame time (inverse `vehicle_pose`). The return 1 pose also applies to return 2.
- **No-label zones (NLZ)** are regions annotators skipped. Boxes are never placed there, so treat NLZ
  points as "unknown" rather than background. They are rare: under 0.1% of points in our segments.

The full decode is in [`range_image.py`](../src/waymo_lidar/data/range_image.py).

### LiDAR → camera projection

`lidar_camera_projection` has the same `[H, W]` grid as `lidar`, with 6 channels:
`[camera_1, x_1, y_1, camera_2, x_2, y_2]`. Each pixel gets up to two camera projections, chosen in
the order FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT. Camera ID `0` means no projection.
The values are stored as float32 but are integers.

---

## Cameras

| ID | Name | Resolution |
|---:|---|---|
| 1 | `FRONT` | 1920 × 1280 |
| 2 | `FRONT_LEFT` | 1920 × 1280 |
| 3 | `FRONT_RIGHT` | 1920 × 1280 |
| 4 | `SIDE_LEFT` | 1920 × 886 |
| 5 | `SIDE_RIGHT` | 1920 × 886 |

The camera enum also defines `REAR_LEFT`, `REAR` and `REAR_RIGHT` (6–8), but those cameras aren't used
in the perception dataset. Other details:

- **Images** are JPEG (~400 KB each).
- **Intrinsics** use a pinhole model with OpenCV-style distortion: `f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3`.
- **Rolling shutter** reads left to right (`rolling_shutter_direction = 2`). Per-image `shutter`,
  `camera_trigger_time` and `camera_readout_done_time` let you compute each column's capture time.

---

## Labels

### 3D boxes (`lidar_box`)

Boxes are upright: 7 degrees of freedom, with no pitch or roll. They're in the vehicle frame, and each
object keeps the same `key.laser_object_id` across the segment, so the boxes also work as tracks.

| Field | Meaning |
|---|---|
| `box.center.{x,y,z}` | box centre (m). z is the centre, not the bottom. |
| `box.size.{x,y,z}` | length (along heading), width, height (m) |
| `box.heading` | yaw (rad), [-π, π) |
| `type` | class, see below |
| `num_lidar_points_in_box` | points from all LiDARs inside the box |
| `num_top_lidar_points_in_box` | points from the top LiDAR only |
| `speed.{x,y,z}`, `acceleration.{x,y,z}` | object motion (m/s, m/s²) |
| `difficulty_level.detection` | `2` = LEVEL_2, **null = not set** |
| `difficulty_level.tracking` | same, for tracking |

| ID | Class | Training boxes | Validation boxes | Share |
|---:|---|---:|---:|---:|
| 1 | `VEHICLE` | 4,778,641 | 1,245,056 | 49.9% |
| 2 | `PEDESTRIAN` | 2,229,448 | 542,324 | 22.9% |
| 3 | `SIGN` | 2,591,779 | 623,879 | 26.6% |
| 4 | `CYCLIST` | 53,344 | 13,340 | 0.6% |
| | **Total** | **9,653,212** | **2,424,599** | |

`0` (`UNKNOWN`) is defined but never used. Cyclists are rare, about 1 in 180 boxes, so plan for class
imbalance. Shares are over training + validation.

### Difficulty and evaluation

These rules are from the official
[3D detection challenge](https://waymo.com/open/challenges/3d-detection/):

- **LEVEL_1**: more than 5 points, and not marked LEVEL_2.
- **LEVEL_2**: 1–5 points, or marked LEVEL_2 in the data.
- **0-point boxes are not evaluated.**
- L2 metrics count both L1 and L2 ground truth; the headline number is **mAPH / L2**.
- The ranked classes are vehicle, pedestrian and cyclist. Signs are labelled but not ranked.
- A detection matches a box at 3D IoU **0.7** for vehicles and **0.5** for pedestrians and cyclists.
- **APH** is average precision weighted by heading accuracy, so heading errors cost points.

In our stage-1 segments, 9–14% of vehicle, pedestrian and sign boxes have 0 points (vehicles 9.3%,
pedestrians 13.6%, signs 11.0%).

### 2D camera boxes (`camera_box`)

Axis-aligned image boxes: `box.center.{x,y}` and `box.size.{x,y}` in pixels, plus `type` and
`difficulty_level`. There are three classes: vehicle 7.70 M, pedestrian 2.17 M and cyclist 0.06 M boxes
(training + validation). **There are no signs.** Camera object IDs are separate from LiDAR object IDs.

### Derived box components

| Component | What it is |
|---|---|
| `projected_lidar_box` | each 3D box projected into each camera as a 2D box (vehicle and sign in our sample) |
| `lidar_camera_synced_box` | each 3D box moved to the moment its `most_visible_camera_name` camera captured the box centre, using the object's velocity. This is the ground truth for camera-only 3D detection. |
| `camera_to_lidar_box_association` | links `camera_object_id` ↔ `laser_object_id` for **pedestrians and cyclists** |

### LiDAR segmentation (`lidar_segmentation`)

Per-pixel labels for the **top LiDAR**, on the same 64 × 2650 grid, with 2 int32 channels:
`[instance_id, semantic_class]`. Instance IDs ≤ 0 mean no instance.

- **Coverage:** every training and validation segment has it, but only for a subset of frames. In our
  sample segment that was 30 of 198 frames, 0.5 s apart.
- **Classes** (23):

| ID | Class | ID | Class | ID | Class |
|---:|---|---:|---|---:|---|
| 0 | undefined | 8 | sign | 16 | tree trunk |
| 1 | car | 9 | traffic light | 17 | curb |
| 2 | truck | 10 | pole | 18 | road |
| 3 | bus | 11 | construction cone | 19 | lane marker |
| 4 | other vehicle | 12 | bicycle | 20 | other ground |
| 5 | motorcyclist | 13 | motorcycle | 21 | walkable |
| 6 | bicyclist | 14 | building | 22 | sidewalk |
| 7 | pedestrian | 15 | vegetation | | |

### Camera panoptic segmentation (`camera_segmentation`)

Each label is a uint16 PNG at image resolution, where each pixel holds
`semantic_class × panoptic_label_divisor + instance_id`; the divisor was 1000 in our sample.
`instance_id_to_global_id_mapping` links instances across cameras and over time.

- **Coverage:** about 696 of 798 training segments, but only about 20 of 202 validation segments.
  These counts are estimated from file sizes, since empty files are about 10 KB. Labelled frames are
  sparse: our sample had 20 frames × 5 cameras.
- **Classes** (29): undefined, ego vehicle, car, truck, bus, other large vehicle, bicycle, motorcycle,
  trailer, pedestrian, cyclist, motorcyclist, bird, ground animal, construction cone/pole, pole,
  pedestrian object, sign, traffic light, building, road, lane marker, road marker, sidewalk,
  vegetation, sky, ground, dynamic, static (IDs 0–28 in that order).

### Human keypoints (`lidar_hkp`, `camera_hkp`)

Human keypoints, as lists per object: `type`, location (3D in metres in the vehicle frame for
LiDAR; 2D pixels plus optional 3D for camera) and `is_occluded`.

- **Types** (14): nose (1), left/right shoulder, elbow, wrist, hip, knee, ankle (5–10, 13–18),
  forehead (19), head centre (20).
- **Sparse coverage:**

| | Training | Validation |
|---|---|---|
| `lidar_hkp` | 302 segments, 8,125 objects | 80 segments, 1,873 objects |
| `camera_hkp` | 439 segments, 146,002 objects | 114 segments, 27,524 objects |

---

## Scene metadata (`stats`)

One row per frame. `time_of_day`, `location` and `weather` are constant within a segment.
`lidar_object_counts` and `camera_object_counts` hold the number of boxes per class **in that frame**.

| | Training | Validation | `testing` | `testing_location` |
|---|---|---|---|---|
| Day / Night / Dawn-Dusk | 647 / 79 / 72 | 160 / 19 / 23 | 9 / 5 / 2 | 96 / 22 / 16 |
| SF / Phoenix / other | 409 / 284 / 105 | 88 / 93 / 21 | 3 / 2 / 11 | 110 / 23 / 1 |
| Sunny / rain | 793 / 5 | 201 / 1 | 16 / 0 | 134 / 0 |

Almost every segment is in dry weather; there's too little rain to evaluate on.

---

## Gotchas

Things that differ from what the docs or a first guess would suggest:

1. **`num_lidar_points_in_box` is about 3% higher than what you decode from range images.** The count
   comes from the raw point cloud, and some points are lost when it is turned into a range image. Our
   decode matches it with correlation 0.9999 at a median ratio of 0.97, so use it as a sanity check,
   not for exact equality.
2. **Detection difficulty is null or 2**, never 0 or 1. Fill nulls before casting to int; NumPy turns
   NaN into an arbitrary integer.
3. **`stats` object counts are per frame.** `dataset.proto` says "unique objects in the segment", but
   in the data the counts add up to exactly the number of `lidar_box` rows.
4. **Camera↔LiDAR associations cover pedestrians and cyclists.** `label.proto` says only pedestrians.
5. **Intensity isn't calibrated across LiDARs.** Median intensity differs about 10× between `side_left`
   and `side_right`. Normalise per LiDAR, or give the model the LiDAR ID as a feature.
6. **Max range is truncated** to 75 m for the top LiDAR and 20 m for the others. There are no points
   beyond that, even though boxes exist out to ~80 m.
7. **Range images use −1 for "no return"**. Mask on `range > 0`, not `range != 0`.
8. **Inclinations are stored bottom-to-top, but rows run top-to-bottom.** Reverse them before indexing.
9. **World poses are large numbers.** Re-centre them before converting to float32, or points jitter by
   centimetres.
10. **Test splits have no labels**, and validation has little camera segmentation (about 20 segments).

---

## Sources

- Data: `gs://waymo_open_dataset_v_2_0_1`. Sizes, counts, schemas and coverage were measured directly.
- Definitions: [`dataset.proto`](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto),
  [`label.proto`](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/label.proto),
  [`segmentation.proto`](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto),
  [`camera_segmentation.proto`](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/camera_segmentation.proto),
  [`keypoint.proto`](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/keypoint.proto)
- Evaluation rules: [Waymo 3D detection challenge](https://waymo.com/open/challenges/3d-detection/)
