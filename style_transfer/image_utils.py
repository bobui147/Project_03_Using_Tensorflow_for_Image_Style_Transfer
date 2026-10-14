"""Image loading, conversion, and saving helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tensorflow as tf


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image_path(path: str | Path) -> Path:
    """Return a resolved image path or raise a useful validation error."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {resolved}")
    if resolved.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Định dạng ảnh không hỗ trợ: {resolved.suffix}. Hỗ trợ: {supported}")
    return resolved


def load_image(path: str | Path, max_dim: int = 512) -> tf.Tensor:
    """Load a local image as a float tensor shaped ``[1, H, W, 3]`` in ``[0, 1]``."""

    image_path = validate_image_path(path)
    raw = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    longest_side = tf.reduce_max(shape)
    scale = tf.minimum(tf.cast(max_dim, tf.float32) / longest_side, 1.0)
    new_shape = tf.maximum(tf.cast(tf.round(shape * scale), tf.int32), 1)
    image = tf.image.resize(image, new_shape, antialias=True)
    return image[tf.newaxis, ...]


def clip_image(image: tf.Tensor) -> tf.Tensor:
    """Clamp a float image tensor to the displayable range."""

    return tf.clip_by_value(image, 0.0, 1.0)


def tensor_to_pil(image: Any) -> Image.Image:
    """Convert a TensorFlow/NumPy image tensor to an RGB Pillow image."""

    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("Chỉ hỗ trợ batch có đúng một ảnh.")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Tensor ảnh phải có shape [H, W, 3], nhận {array.shape}.")
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def save_image(image: Any, output_path: str | Path) -> Path:
    """Save an image tensor and return its resolved output path."""

    output = Path(output_path).expanduser().resolve()
    if output.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError("File output phải có đuôi .png, .jpg, .jpeg, .bmp hoặc .webp.")
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image).save(output)
    return output

