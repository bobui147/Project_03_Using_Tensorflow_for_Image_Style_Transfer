"""VGG19 feature extractor for neural style transfer."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


DEFAULT_STYLE_LAYERS = (
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
)
DEFAULT_CONTENT_LAYERS = ("block5_conv2",)


def gram_matrix(features: tf.Tensor) -> tf.Tensor:
    """Calculate normalized channel correlations for style representation."""

    result = tf.linalg.einsum("bijc,bijd->bcd", features, features)
    shape = tf.shape(features)
    locations = tf.cast(shape[1] * shape[2], tf.float32)
    return result / tf.maximum(locations, 1.0)


def build_vgg_layers(
    layer_names: Sequence[str], weights: str | None = "imagenet"
) -> tf.keras.Model:
    """Build a frozen VGG19 model returning the requested intermediate activations."""

    vgg = tf.keras.applications.VGG19(include_top=False, weights=weights)
    vgg.trainable = False
    known_layers = {layer.name for layer in vgg.layers}
    missing = [name for name in layer_names if name not in known_layers]
    if missing:
        raise ValueError(f"Layer VGG19 không tồn tại: {', '.join(missing)}")
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs, name="vgg19_style_extractor")


class StyleContentModel(tf.keras.Model):
    """Extract VGG19 content features and Gram-matrix style features."""

    def __init__(
        self,
        style_layers: Sequence[str] = DEFAULT_STYLE_LAYERS,
        content_layers: Sequence[str] = DEFAULT_CONTENT_LAYERS,
        weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        self.style_layers = tuple(style_layers)
        self.content_layers = tuple(content_layers)
        self.num_style_layers = len(self.style_layers)
        self.vgg = build_vgg_layers(self.style_layers + self.content_layers, weights=weights)
        self.vgg.trainable = False

    def call(self, inputs: tf.Tensor) -> dict[str, dict[str, tf.Tensor]]:
        inputs = tf.cast(inputs, tf.float32)
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs * 255.0)
        outputs = self.vgg(preprocessed)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        style_outputs = outputs[: self.num_style_layers]
        content_outputs = outputs[self.num_style_layers :]
        style = {
            name: gram_matrix(value)
            for name, value in zip(self.style_layers, style_outputs, strict=True)
        }
        content = {
            name: value
            for name, value in zip(self.content_layers, content_outputs, strict=True)
        }
        return {"content": content, "style": style}

