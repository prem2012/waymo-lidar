"""Enum values and naming used across the Waymo Open Dataset v2 Parquet files."""

from enum import IntEnum


class Laser(IntEnum):
    TOP = 1
    FRONT = 2
    SIDE_LEFT = 3
    SIDE_RIGHT = 4
    REAR = 5


LASER_NAMES: dict[int, str] = {l.value: l.name.lower() for l in Laser}


class BoxType(IntEnum):
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    SIGN = 3
    CYCLIST = 4


BOX_TYPE_NAMES: dict[int, str] = {t.value: t.name.lower() for t in BoxType}

# One color per class, shared by the analytics charts and the 3D viewer.
BOX_TYPE_COLORS: dict[int, tuple[int, int, int]] = {
    BoxType.UNKNOWN: (150, 150, 150),
    BoxType.VEHICLE: (42, 120, 214),
    BoxType.PEDESTRIAN: (235, 120, 40),
    BoxType.SIGN: (150, 90, 200),
    BoxType.CYCLIST: (30, 160, 90),
}

DIFFICULTY_NAMES: dict[int, str] = {0: "unknown", 1: "L1", 2: "L2"}

SPLITS = ("training", "validation", "testing")

# Range image channels.
CH_RANGE, CH_INTENSITY, CH_ELONGATION, CH_NLZ = 0, 1, 2, 3
