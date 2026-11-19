"""Convert a CUDA power map to the pinned ResNet-50 input layout."""

import math

import cupy as cp
import numpy as np

IMAGE_SIZE = 224
MODEL_INPUT_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)

# Normalization used by microsoft/resnet-50 at the pinned serving revision.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

_preprocess = cp.ElementwiseKernel(
    "raw float32 power, int32 height, int32 width, float32 scale",
    "raw float32 pixels",
    rf"""
    const int x = i % {IMAGE_SIZE};
    const int y = i / {IMAGE_SIZE};
    const float sx = fminf(width - 1, fmaxf(0.0f, (x + 0.5f) * width / {IMAGE_SIZE}.0f - 0.5f));
    const float sy = fminf(height - 1, fmaxf(0.0f, (y + 0.5f) * height / {IMAGE_SIZE}.0f - 0.5f));
    const int x0 = (int)sx;
    const int y0 = (int)sy;
    const int x1 = x0 + (x0 + 1 < width);
    const int y1 = y0 + (y0 + 1 < height);
    const float dx = sx - x0;
    const float dy = sy - y0;

    const float a = saturating_power(power[y0 * width + x0], scale);
    const float b = saturating_power(power[y0 * width + x1], scale);
    const float c = saturating_power(power[y1 * width + x0], scale);
    const float d = saturating_power(power[y1 * width + x1], scale);
    const float top = a + dx * (b - a);
    const float bottom = c + dx * (d - c);
    const float intensity = top + dy * (bottom - top);

    pixels[i] = (intensity - {IMAGE_MEAN[0]}f) / {IMAGE_STD[0]}f;
    pixels[{IMAGE_SIZE * IMAGE_SIZE} + i] = (intensity - {IMAGE_MEAN[1]}f) / {IMAGE_STD[1]}f;
    pixels[{2 * IMAGE_SIZE * IMAGE_SIZE} + i] = (intensity - {IMAGE_MEAN[2]}f) / {IMAGE_STD[2]}f;
    """,
    "exercise7_preprocess_power_map",
    preamble=r"""
    __device__ float saturating_power(float power, float scale) {
        if (power <= 0.0f) return 0.0f;
        // Evaluate p/(p+s) without overflowing when p is very large.
        if (power >= scale) return 1.0f / (1.0f + scale / power);
        const float ratio = power / scale;
        return ratio / (1.0f + ratio);
    }
    """,
)


def preprocess_power_map(power: cp.ndarray, scale: float, *, out: cp.ndarray) -> cp.ndarray:
    """Scale, bilinearly resize, and normalize a map on the current CUDA stream.

    The positive scale is fixed across windows. Power at the scale maps to
    0.5, and larger values approach 1. The caller owns the output buffer and
    orders input and output reuse with the corresponding CUDA events.
    """
    if not isinstance(power, cp.ndarray) or power.ndim != 2 or min(power.shape) == 0:
        raise ValueError("power must be a nonempty 2D CUDA array")
    if power.dtype != cp.float32 or not power.flags.c_contiguous:
        raise ValueError("power must be contiguous float32")
    if not isinstance(out, cp.ndarray) or out.shape != MODEL_INPUT_SHAPE:
        raise ValueError(f"out must be a CUDA array with shape {MODEL_INPUT_SHAPE}")
    if out.dtype != cp.float32 or not out.flags.c_contiguous:
        raise ValueError("out must be contiguous float32")
    if power.device.id != out.device.id or power.device.id != cp.cuda.Device().id:
        raise ValueError("power and out must be on the current CUDA device")
    if not math.isfinite(scale) or not 0 < scale <= np.finfo(np.float32).max:
        raise ValueError("scale must be a positive finite float32 value")
    scale = np.float32(scale)
    if scale == 0:
        raise ValueError("scale underflows float32")

    height, width = power.shape
    _preprocess(power, height, width, scale, out, size=IMAGE_SIZE * IMAGE_SIZE)
    return out
