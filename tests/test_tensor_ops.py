from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from style_transfer.image_utils import tensor_to_pil  # noqa: E402
from style_transfer.model import gram_matrix  # noqa: E402


def test_gram_matrix_shape_and_values() -> None:
    features = tf.ones((1, 2, 3, 4), dtype=tf.float32)
    gram = gram_matrix(features)
    assert tuple(gram.shape) == (1, 4, 4)
    np.testing.assert_allclose(gram.numpy(), np.ones((1, 4, 4)), atol=1e-6)


def test_tensor_to_pil_converts_single_image_batch() -> None:
    image = tf.ones((1, 8, 10, 3), dtype=tf.float32) * 0.5
    converted = tensor_to_pil(image)
    assert converted.mode == "RGB"
    assert converted.size == (10, 8)

