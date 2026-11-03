#!/usr/bin/env bash
# Download LiDAR-only components of Waymo Open Dataset Perception v2 (Parquet).
# Usage: ./download.sh <split> <num_segments|all> [dest_dir]
#   ./download.sh training 2            # stage 1: ~400 frames
#   ./download.sh validation 1
#   ./download.sh training all          # stage 2
set -euo pipefail

BUCKET="gs://waymo_open_dataset_v_2_0_1"
SPLIT="${1:?split: training|validation|testing}"
N="${2:?num segments or 'all'}"
DEST="${3:-$HOME/dev/data/waymo_v2}"

# Components a LiDAR detector needs (no cameras):
#   lidar             range images (the bulk of the bytes)
#   lidar_calibration extrinsics + beam inclinations
#   lidar_pose        per-pixel pose for top-lidar range image -> points
#   vehicle_pose      ego pose per frame
#   lidar_box         3D box labels
#   stats             per-segment metadata: time of day, weather, location, object counts
COMPONENTS=(lidar lidar_calibration lidar_pose vehicle_pose lidar_box stats)

# Segment list comes from the small lidar_box component.
mapfile -t SEGS < <(gsutil ls "$BUCKET/$SPLIT/lidar_box/" | xargs -n1 basename | sort)
[[ "$N" == "all" ]] || SEGS=("${SEGS[@]:0:$N}")
echo "Split=$SPLIT  segments=${#SEGS[@]}  dest=$DEST"

for c in "${COMPONENTS[@]}"; do
  mkdir -p "$DEST/$SPLIT/$c"
  printf "%s\n" "${SEGS[@]}" | sed "s|^|$BUCKET/$SPLIT/$c/|" \
    | gsutil -m cp -n -I "$DEST/$SPLIT/$c/"   # -n: skip existing, so reruns resume
done
du -sh "$DEST/$SPLIT"/*
