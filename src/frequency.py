"""Frequency-domain features shared with the validated Colab model."""

from __future__ import annotations

import numpy as np
from PIL import Image

FFT_BINS = 32
FFT_FEATURE_DIM = FFT_BINS * 3


def fft_radial_features(image: Image.Image, bins: int = FFT_BINS) -> np.ndarray:
    """Return native-resolution RGB FFT log-magnitude radial profiles.

    The implementation intentionally matches the final DINOv2 SID Colab:
    32 radial bins are averaged independently for each RGB channel.
    """

    if bins <= 0:
        raise ValueError("bins must be a positive integer")

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _ = array.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt(
        (yy - (height - 1) / 2.0) ** 2
        + (xx - (width - 1) / 2.0) ** 2
    )
    radius = radius / max(float(radius.max()), 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    features: list[float] = []

    for channel in range(3):
        magnitude = np.log1p(
            np.abs(np.fft.fftshift(np.fft.fft2(array[:, :, channel])))
        )
        for index in range(bins):
            if index == bins - 1:
                mask = (radius >= edges[index]) & (radius <= edges[index + 1])
            else:
                mask = (radius >= edges[index]) & (radius < edges[index + 1])
            features.append(float(magnitude[mask].mean()) if mask.any() else 0.0)

    result = np.asarray(features, dtype=np.float32)
    expected_shape = (bins * 3,)
    if result.shape != expected_shape or not np.isfinite(result).all():
        raise ValueError(
            f"Invalid FFT feature vector: expected {expected_shape}, got {result.shape}"
        )
    return result
