"""Configuration objects shared by the style-transfer pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransferConfig:
    """Hyperparameters for iterative VGG19 neural style transfer."""

    steps: int = 400
    learning_rate: float = 0.02
    beta_1: float = 0.99
    epsilon: float = 0.1
    content_weight: float = 1e4
    style_weight: float = 1e-2
    total_variation_weight: float = 30.0
    init: str = "content"
    seed: int = 42

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps phải lớn hơn 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate phải lớn hơn 0.")
        if not 0 <= self.beta_1 < 1:
            raise ValueError("beta_1 phải nằm trong [0, 1).")
        if self.epsilon <= 0:
            raise ValueError("epsilon phải lớn hơn 0.")
        if min(self.content_weight, self.style_weight, self.total_variation_weight) < 0:
            raise ValueError("Các trọng số loss không được âm.")
        if self.init not in {"content", "noise"}:
            raise ValueError("init phải là 'content' hoặc 'noise'.")


def validate_image_size(max_dim: int) -> int:
    """Validate and return a maximum image dimension."""

    if max_dim < 32:
        raise ValueError("max_dim phải từ 32 trở lên để dùng VGG19.")
    return max_dim

