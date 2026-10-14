"""Iterative optimization engine for the original neural style-transfer method."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from .config import TransferConfig
from .image_utils import clip_image
from .model import StyleContentModel


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    total_loss: float
    style_loss: float
    content_loss: float
    variation_loss: float


ProgressCallback = Callable[[StepMetrics, tf.Tensor], None]


def _mean_squared_losses(
    outputs: Mapping[str, tf.Tensor], targets: Mapping[str, tf.Tensor]
) -> tf.Tensor:
    losses = [tf.reduce_mean(tf.square(outputs[name] - target)) for name, target in targets.items()]
    if not losses:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(losses) / tf.cast(len(losses), tf.float32)


def calculate_losses(
    outputs: Mapping[str, Mapping[str, tf.Tensor]],
    style_targets: Mapping[str, tf.Tensor],
    content_targets: Mapping[str, tf.Tensor],
    image: tf.Tensor,
    config: TransferConfig,
) -> dict[str, tf.Tensor]:
    """Calculate weighted style, content, variation, and total losses."""

    raw_style = _mean_squared_losses(outputs["style"], style_targets)
    raw_content = _mean_squared_losses(outputs["content"], content_targets)
    raw_variation = tf.reduce_sum(tf.image.total_variation(image))
    style_loss = raw_style * config.style_weight
    content_loss = raw_content * config.content_weight
    variation_loss = raw_variation * config.total_variation_weight
    return {
        "total": style_loss + content_loss + variation_loss,
        "style": style_loss,
        "content": content_loss,
        "variation": variation_loss,
    }


def run_classic_style_transfer(
    content_image: tf.Tensor,
    style_image: tf.Tensor,
    config: TransferConfig,
    *,
    extractor: StyleContentModel | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[tf.Tensor, list[StepMetrics]]:
    """Optimize one generated image to match content and style targets."""

    config.validate()
    tf.keras.utils.set_random_seed(config.seed)
    feature_extractor = extractor or StyleContentModel()
    feature_extractor.trainable = False

    style_targets = feature_extractor(style_image)["style"]
    content_targets = feature_extractor(content_image)["content"]
    if config.init == "noise":
        initial = tf.random.uniform(tf.shape(content_image), seed=config.seed)
    else:
        initial = tf.identity(content_image)
    generated = tf.Variable(initial, trainable=True, dtype=tf.float32, name="generated_image")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        epsilon=config.epsilon,
    )

    @tf.function(reduce_retracing=True)
    def train_step() -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            outputs = feature_extractor(generated)
            losses = calculate_losses(
                outputs,
                style_targets,
                content_targets,
                generated,
                config,
            )
        gradient = tape.gradient(losses["total"], generated)
        optimizer.apply_gradients([(gradient, generated)])
        generated.assign(clip_image(generated))
        return losses

    history: list[StepMetrics] = []
    for step in range(1, config.steps + 1):
        losses = train_step()
        metrics = StepMetrics(
            step=step,
            total_loss=float(losses["total"].numpy()),
            style_loss=float(losses["style"].numpy()),
            content_loss=float(losses["content"].numpy()),
            variation_loss=float(losses["variation"].numpy()),
        )
        history.append(metrics)
        if progress_callback is not None:
            progress_callback(metrics, generated)
    return tf.identity(generated), history

