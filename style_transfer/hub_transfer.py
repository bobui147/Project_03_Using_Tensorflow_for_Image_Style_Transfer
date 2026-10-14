"""Fast arbitrary style transfer through the official TensorFlow Hub model."""

from __future__ import annotations

import tensorflow as tf


DEFAULT_HUB_MODEL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"


def run_fast_style_transfer(
    content_image: tf.Tensor,
    style_image: tf.Tensor,
    model_url: str = DEFAULT_HUB_MODEL,
) -> tf.Tensor:
    """Apply a pretrained arbitrary-style model to a pair of images."""

    try:
        import tensorflow_hub as hub
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu tensorflow-hub. Hãy chạy: python -m pip install -r requirements.txt"
        ) from exc
    model = hub.load(model_url)
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    if not isinstance(outputs, (tuple, list)) or not outputs:
        raise RuntimeError("TensorFlow Hub model không trả về ảnh kết quả hợp lệ.")
    return tf.clip_by_value(outputs[0], 0.0, 1.0)

