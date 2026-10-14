"""Project-local runtime cache configuration."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_cache() -> Path:
    """Keep library caches writable and contained inside this project."""

    project_root = Path(__file__).resolve().parent.parent
    runtime_root = project_root / ".runtime"
    matplotlib_cache = runtime_root / "matplotlib"
    tensorflow_hub_cache = runtime_root / "tensorflow_hub"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    tensorflow_hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("TFHUB_CACHE_DIR", str(tensorflow_hub_cache))
    return runtime_root

