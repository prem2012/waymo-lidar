"""Colour maps shared by the Rerun viewers."""

from __future__ import annotations

import numpy as np

_TURBO_COEFFS = np.array([
    [0.13572138, 0.09140261, 0.10667330],
    [4.61539260, 2.19418839, 12.64194608],
    [-42.66032258, 4.84296658, -60.58204836],
    [132.13108234, -14.18503333, 110.36276771],
    [-152.94239396, 4.27729857, -89.90310912],
    [59.28637943, 2.82956604, 27.34824973],
])


def turbo(x: np.ndarray) -> np.ndarray:
    """Turbo colour map (polynomial approximation): x in [0, 1] -> (N, 3) uint8."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)[:, None]
    rgb = sum(_TURBO_COEFFS[i] * x ** i for i in range(6))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def intensity_colors(intensity: np.ndarray) -> np.ndarray:
    """Intensity is heavy-tailed (retroreflectors >> asphalt), so colour it on a log scale."""
    return turbo((np.log10(np.maximum(intensity, 1e-3)) + 2.5) / 3.0)
