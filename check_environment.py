"""Check whether the local environment is ready for TensorFlow style transfer."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

from style_transfer.console import configure_utf8_console
from style_transfer.runtime import configure_runtime_cache


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "CHƯA CÀI"


def main() -> int:
    configure_utf8_console()
    configure_runtime_cache()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for package in ("tensorflow", "tensorflow-hub", "numpy", "Pillow", "matplotlib"):
        print(f"{package}: {_version(package)}")

    try:
        import tensorflow as tf
    except ImportError:
        print("\nMôi trường chưa sẵn sàng: thiếu TensorFlow.")
        print("Chạy: python -m pip install -r requirements.txt")
        return 1

    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")
    print("\nTensorFlow devices:")
    for device in devices:
        print(f"- {device.device_type}: {device.name}")
    if gpus:
        print(f"\nSẵn sàng chạy GPU ({len(gpus)} GPU).")
    else:
        print("\nKhông thấy GPU; project vẫn chạy được bằng CPU nhưng phương pháp classic sẽ chậm hơn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

