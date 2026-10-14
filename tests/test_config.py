from __future__ import annotations

import pytest

from style_transfer.config import TransferConfig, validate_image_size


def test_default_config_is_valid() -> None:
    TransferConfig().validate()


@pytest.mark.parametrize(
    "config",
    [
        TransferConfig(steps=0),
        TransferConfig(learning_rate=0),
        TransferConfig(style_weight=-1),
        TransferConfig(init="invalid"),
    ],
)
def test_invalid_config_raises(config: TransferConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_validate_image_size() -> None:
    assert validate_image_size(512) == 512
    with pytest.raises(ValueError):
        validate_image_size(31)

