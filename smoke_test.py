"""Run two optimization steps with random VGG19 weights and synthetic images."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from style_transfer.config import TransferConfig
from style_transfer.console import configure_utf8_console
from style_transfer.runtime import configure_runtime_cache


def _make_content(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    return np.stack((x / size, y / size, ((x // 8 + y // 8) % 2)), axis=-1)


def _make_style(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    waves = (np.sin(x / 3.0) + 1.0) / 2.0
    return np.stack((waves, (y % 12) / 12.0, 1.0 - waves), axis=-1)


def main() -> int:
    configure_utf8_console()
    configure_runtime_cache()
    try:
        from style_transfer.engine import run_classic_style_transfer
        from style_transfer.image_utils import load_image, save_image
        from style_transfer.model import StyleContentModel
    except ImportError as exc:
        print(f"Thiếu dependency để chạy smoke test: {exc}")
        return 1

    with tempfile.TemporaryDirectory(prefix="style-transfer-smoke-") as temp_dir:
        temp = Path(temp_dir)
        content_path = temp / "content.png"
        style_path = temp / "style.png"
        output_path = temp / "output.png"
        Image.fromarray(np.uint8(np.clip(_make_content(), 0, 1) * 255)).save(content_path)
        Image.fromarray(np.uint8(np.clip(_make_style(), 0, 1) * 255)).save(style_path)

        content = load_image(content_path, max_dim=64)
        style = load_image(style_path, max_dim=64)
        extractor = StyleContentModel(weights=None)
        result, history = run_classic_style_transfer(
            content,
            style,
            TransferConfig(steps=2),
            extractor=extractor,
        )
        save_image(result, output_path)
        if result.shape != content.shape or len(history) != 2 or not output_path.is_file():
            raise RuntimeError("Smoke test không tạo đúng output.")
        print(f"Smoke test thành công: shape={tuple(result.shape)}, steps={len(history)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

