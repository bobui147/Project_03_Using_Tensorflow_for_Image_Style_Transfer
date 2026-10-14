"""Download the two public sample images used by TensorFlow tutorials."""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

from style_transfer.console import configure_utf8_console


SAMPLES = {
    "samples/content/labrador.jpg": (
        "https://storage.googleapis.com/download.tensorflow.org/example_images/"
        "YellowLabradorLooking_new.jpg"
    ),
    "samples/style/kandinsky.jpg": (
        "https://storage.googleapis.com/download.tensorflow.org/example_images/"
        "Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"
    ),
}


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Đã có, bỏ qua: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "TensorFlow-Style-Transfer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Đã tải: {destination}")


def main() -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Tải content/style images mẫu.")
    parser.add_argument("--force", action="store_true", help="Tải lại nếu file đã tồn tại.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for relative_path, url in SAMPLES.items():
        download(url, root / relative_path, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

